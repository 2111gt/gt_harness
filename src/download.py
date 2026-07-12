"""
Automatic download of AI model weights (and missing Python deps) for GT Diagnostic Harness.

What gets ensured
-----------------
1. Repair / install Python packages with **prebuilt wheels** when possible
   (avoids MSVC/CMake failures on Windows)
2. Granite 4.1 8B GGUF → models/ (Hugging Face LFS)
3. SentenceTransformer embeddings (all-MiniLM-L6-v2) → HF cache
4. Granite TS Pulse weights → HF cache (optional; statistical fallback if API missing)

Environment overrides
---------------------
GT_NO_DOWNLOAD=1          Skip all network downloads / pip installs
GT_GGUF_REPO              HF repo for GGUF (default ibm-granite/granite-4.1-8b-GGUF)
GT_GGUF_FILE              Filename inside the repo
GT_GGUF_PATH              Use an existing local GGUF instead of downloading
GT_TSPULSE_MODEL          HF id for TS Pulse
GT_TSPULSE_REVISION       HF revision
GT_EMBEDDING_MODEL        SentenceTransformer model name
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import (
    DEFAULT_EMBEDDING_MODEL,
    MODELS_DIR,
    ensure_directories,
    find_gguf_model,
    setup_logging,
)

logger = setup_logging()

# Process-lifetime caches so startup never re-runs failed pip repair loops.
# (Previously each ensure_* called repair_environment / ensure_granite_tsfm,
# which re-ran multi-second pip installs every launch and looked like a hang.)
_REPAIR_STATUS: Optional[Dict[str, str]] = None
_TSFM_STATUS: Optional[str] = None
_EMBED_FAIL_MSG: Optional[str] = None
_PIP_TRIED: set = set()
# Last ensure_tspulse failure detail (for diagnostics / UI)
_TSPULSE_LAST_ERROR: Optional[str] = None

# Official IBM Granite 4.1 8B GGUF repo (filenames use hyphen before quant tag)
DEFAULT_GGUF_REPO = "ibm-granite/granite-4.1-8b-GGUF"
DEFAULT_GGUF_FILE = "granite-4.1-8b-Q4_K_M.gguf"  # ~5.3 GB

GGUF_FALLBACK_FILES: Tuple[str, ...] = (
    "granite-4.1-8b-Q4_K_M.gguf",
    "granite-4.1-8b-Q4_K_S.gguf",
    "granite-4.1-8b-Q3_K_M.gguf",
    "granite-4.1-8b-Q2_K.gguf",
)

DEFAULT_TSPULSE_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
DEFAULT_TSPULSE_REVISION = "tspulse-block-ad"

# Prebuilt CPU wheels for llama-cpp-python (Windows/Linux without local C++ toolchain)
LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# Prefer the stack that works with tsfm_public (TS Pulse) + ST + Gradio on this app:
# transformers 4.57.x + hub 0.36 + sentence-transformers 5.x (verified encode + TSPulse).
# Do NOT force hub 1.x / transformers 5.x — that breaks tsfm_public and BertConfig.
_COMPAT_STACK: Tuple[str, ...] = (
    "huggingface_hub>=0.34.0,<1.0",
    "transformers>=4.57.0,<5.0",
    "tokenizers>=0.21.0",
    "sentence-transformers>=3.0.0",
    "chromadb>=0.5.0",
)


def downloads_enabled() -> bool:
    """False when GT_NO_DOWNLOAD is set (1/true/yes)."""
    flag = (os.environ.get("GT_NO_DOWNLOAD") or "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _pip(
    args: Sequence[str],
    *,
    timeout: int = 1800,
) -> Tuple[bool, str]:
    from .utils import run_hidden_subprocess

    cmd = [sys.executable, "-m", "pip", *args]
    try:
        # Hidden console: pip under Textual TUI otherwise can hang on Windows
        proc = run_hidden_subprocess(cmd, timeout=timeout)
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode == 0:
            return True, "ok"
        return False, out[-800:]
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _pip_install(
    *packages: str,
    extra_index_url: Optional[str] = None,
    prefer_binary: bool = True,
    upgrade: bool = False,
    only_binary: Optional[str] = None,
) -> Tuple[bool, str]:
    args: List[str] = ["install"]
    if upgrade:
        args.append("--upgrade")
    if prefer_binary:
        args.append("--prefer-binary")
    if only_binary:
        args.extend(["--only-binary", only_binary])
    if extra_index_url:
        args.extend(["--extra-index-url", extra_index_url])
    args.extend(packages)
    return _pip(args)


def repair_environment(*, force: bool = False) -> Dict[str, str]:
    """
    Fix package breakage only when imports actually fail.

    Runs **at most once per process** (unless ``force=True``) so TUI startup
    never thrash-installs packages for minutes.
    Gradio is optional (TUI does not need it) and is never installed here.
    """
    global _REPAIR_STATUS
    if _REPAIR_STATUS is not None and not force:
        return dict(_REPAIR_STATUS)

    status: Dict[str, str] = {}
    if not downloads_enabled():
        status["repair"] = "skipped (downloads disabled)"
        _REPAIR_STATUS = dict(status)
        return status

    hub_ok = False
    try:
        import huggingface_hub  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401

        hub_ok = True
        status["huggingface_hub"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["huggingface_hub"] = f"broken: {exc}"

    st_ok = False
    try:
        from .compat import ensure_ml_compat

        ensure_ml_compat()
        from sentence_transformers import SentenceTransformer  # noqa: F401

        st_ok = True
        status["sentence_transformers"] = "ok"
        status["transformers"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["sentence_transformers"] = f"broken: {exc}"

    tsfm_ok = _can_import("tsfm_public") or _can_import("granite_tsfm")
    status["tsfm"] = "ok" if tsfm_ok else "missing"
    # Gradio is unused by the TUI — report only, never block repair on it
    status["gradio"] = "ok" if _can_import("gradio") else "optional-missing"

    # Only reinstall when hub or ST cannot be imported (needed for RAG)
    if hub_ok and st_ok:
        status["compat_stack"] = "healthy (no reinstall)"
        logger.info("Package stack healthy for RAG; skipping reinstall")
        _REPAIR_STATUS = dict(status)
        return status

    # One attempt only — if pip "succeeds" but imports stay broken, do not loop
    logger.info("Repairing package stack once (hub/ST) …")
    ok, detail = _pip_install(*_COMPAT_STACK, upgrade=True)
    status["compat_stack"] = "repaired" if ok else f"failed: {detail[:300]}"

    # Re-probe after single repair
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        status["sentence_transformers"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["sentence_transformers"] = f"still-broken: {exc}"
        logger.warning(
            "Embeddings stack still broken after one repair; "
            "using keyword/memory RAG (no more pip retries this session)."
        )

    logger.info("Repair status: %s", status)
    _REPAIR_STATUS = dict(status)
    return status


def ensure_llama_cpp() -> str:
    """
    Install llama-cpp-python using **prebuilt wheels** when possible.

    On Windows without MSVC, source builds fail with CMAKE_C_COMPILER errors.
    We try the official CPU wheel index first, never force a source build.

    If a wheel is installed but crashes with illegal instruction (0xc000001d),
    try an older wheel that may target a broader CPU instruction set.
    """
    if _can_import("llama_cpp"):
        return "ok"

    if not downloads_enabled():
        return "missing (downloads disabled)"

    # Prefer wheels known to ship win_amd64 binaries
    candidates = [
        "llama-cpp-python",
        "llama-cpp-python==0.3.2",
        "llama-cpp-python==0.2.90",
        "llama-cpp-python==0.2.27",
    ]
    last_detail = ""
    for spec in candidates:
        logger.info(
            "Installing %s from prebuilt wheel index (no C++ compiler) …",
            spec,
        )
        ok, detail = _pip_install(
            spec,
            extra_index_url=LLAMA_CPP_WHEEL_INDEX,
            prefer_binary=True,
            upgrade=True,
        )
        last_detail = detail
        if ok and _can_import("llama_cpp"):
            logger.info("Installed %s (prebuilt wheel)", spec)
            return f"installed ({spec})"

        ok2, detail2 = _pip_install(
            spec,
            prefer_binary=True,
            only_binary=":all:",
            upgrade=True,
        )
        last_detail = detail2 or detail
        if ok2 and _can_import("llama_cpp"):
            logger.info("Installed %s (binary-only PyPI)", spec)
            return f"installed ({spec}, pypi binary)"

    # Do NOT fall through to a source build — it needs MSVC/CMake
    msg = (
        "failed: no prebuilt wheel for this platform/Python. "
        "Install Visual C++ Build Tools to compile from source "
        f"(with AVX disabled if you hit 0xc000001d). Details: {last_detail[:250]}"
    )
    logger.warning("Could not install llama-cpp-python: %s", msg)
    return msg


def ensure_granite_tsfm() -> str:
    """
    Best-effort install of granite-tsfm / tsfm_public.

    Never installs the unrelated native ``tsfm`` package that requires MSVC.
    Always re-probes imports first so a stale "failed" cache cannot block a
    package that is actually importable later in the same process.
    """
    global _TSFM_STATUS

    # 1) Prefer live import check over any cached failure
    live = _probe_tsfm_status()
    if live.startswith("ok") or live.startswith("installed"):
        _TSFM_STATUS = live
        return _TSFM_STATUS

    # 2) If we already tried pip this process and import still fails, keep detail
    if "granite-tsfm" in _PIP_TRIED and _TSFM_STATUS and "failed" in str(_TSFM_STATUS):
        # Still re-probe once more in case user installed mid-session
        live2 = _probe_tsfm_status()
        if live2.startswith("ok"):
            _TSFM_STATUS = live2
            return _TSFM_STATUS
        return f"{_TSFM_STATUS} | re-probe: {live2}"

    if not downloads_enabled():
        _TSFM_STATUS = f"missing (downloads disabled); {live}"
        return _TSFM_STATUS

    _PIP_TRIED.add("granite-tsfm")
    logger.info("Installing granite-tsfm once (prefer binary) …")
    ok, detail = _pip_install("granite-tsfm", prefer_binary=True)
    _invalidate_import_caches()
    live_after = _probe_tsfm_status()
    if live_after.startswith("ok"):
        logger.info("Installed / verified granite-tsfm → tsfm_public")
        _TSFM_STATUS = "installed" if ok else live_after
        return _TSFM_STATUS

    # Optional alternate package name used by some releases (one try)
    ok2, detail2 = _pip_install("ibm-granite-tsfm", prefer_binary=True)
    _invalidate_import_caches()
    live_after2 = _probe_tsfm_status()
    if live_after2.startswith("ok"):
        _TSFM_STATUS = "installed (ibm-granite-tsfm)"
        return _TSFM_STATUS

    msg = (
        f"failed (statistical fallback): pip={detail or detail2 or 'n/a'!s}; "
        f"import={live_after2}"
    )
    logger.warning("granite-tsfm unavailable: %s", msg)
    _TSFM_STATUS = msg
    return _TSFM_STATUS


def _invalidate_import_caches() -> None:
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:
        pass
    # Drop failed partial imports so the next probe can succeed
    for name in list(sys.modules):
        if name == "tsfm_public" or name.startswith("tsfm_public."):
            try:
                del sys.modules[name]
            except Exception:
                pass


def _probe_tsfm_status() -> str:
    """
    Live check: can we import tsfm_public and TSPulseForReconstruction?
    Returns a short status string starting with 'ok' on success.
    """
    try:
        import tsfm_public  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"tsfm_public import failed: {type(exc).__name__}: {exc}"

    loader, err = _tspulse_loader_class_with_error()
    if loader is not None:
        return "ok"
    return f"tsfm_public present but TSPulse class missing: {err}"


def ensure_python_packages(
    packages: Optional[Sequence[Tuple[str, str]]] = None,
    *,
    upgrade: bool = False,
    repair: bool = False,
) -> Dict[str, str]:
    """
    Best-effort pip install for missing packages needed by models.

    Parameters
    ----------
    repair :
        If True, run ``repair_environment`` once (process-cached). Default
        False so normal TUI startup does not thrash pip.
    """
    status: Dict[str, str] = {}
    if repair and downloads_enabled():
        status.update({f"repair:{k}": v for k, v in repair_environment().items()})

    # Core packages with careful install paths
    if packages is None:
        # llama-cpp-python is optional — we use llama-cli binary when py-bind fails
        if _can_import("llama_cpp"):
            status["llama-cpp-python"] = "ok"
        else:
            status["llama-cpp-python"] = "optional (llama-cli used if present)"

        # Probe ST usability (import package != usable SentenceTransformer)
        try:
            from .compat import ensure_ml_compat

            ensure_ml_compat()
            from sentence_transformers import SentenceTransformer  # noqa: F401

            status["sentence-transformers"] = "ok"
        except Exception as exc:  # noqa: BLE001
            status["sentence-transformers"] = f"unusable: {exc}"

        status["chromadb"] = "ok" if _can_import("chromadb") else "missing"
        status["torch"] = "ok" if _can_import("torch") else "missing"
        # Do not pip-install tsfm at startup — statistical anomalies work offline
        if _can_import("tsfm_public") or _can_import("granite_tsfm"):
            status["granite-tsfm"] = "ok"
        else:
            status["granite-tsfm"] = "missing (statistical anomaly fallback)"
        return status

    # Explicit list path (used by targeted callers)
    for pip_name, import_name in packages:
        if _can_import(import_name):
            status[pip_name] = "ok"
            continue
        if not downloads_enabled():
            status[pip_name] = "missing (downloads disabled)"
            continue
        if pip_name in {"llama-cpp-python", "llama_cpp"}:
            status[pip_name] = ensure_llama_cpp()
            continue
        if pip_name in {"granite-tsfm", "tsfm"}:
            status[pip_name] = ensure_granite_tsfm()
            continue
        if pip_name in _PIP_TRIED:
            status[pip_name] = "failed (already tried this session)"
            continue
        _PIP_TRIED.add(pip_name)
        logger.info("Installing missing package once: %s …", pip_name)
        ok, detail = _pip_install(pip_name, upgrade=upgrade)
        status[pip_name] = "installed" if ok and _can_import(import_name) else f"failed: {detail[:300]}"
        if not ok:
            logger.warning("Could not install %s: %s", pip_name, detail[:300])
    return status


def ensure_gguf(
    *,
    repo: Optional[str] = None,
    filename: Optional[str] = None,
    force: bool = False,
) -> Tuple[Optional[Path], str]:
    """
    Ensure a Granite GGUF exists under models/ (download if missing).

    Returns (path_or_None, status_message).
    """
    ensure_directories()
    existing = find_gguf_model()
    if existing is not None and not force:
        return existing, f"GGUF already present: {existing.name}"

    if not downloads_enabled():
        return None, "No GGUF found and downloads disabled (GT_NO_DOWNLOAD)."

    repo = repo or os.environ.get("GT_GGUF_REPO") or DEFAULT_GGUF_REPO
    preferred = filename or os.environ.get("GT_GGUF_FILE") or DEFAULT_GGUF_FILE
    candidates = [preferred] + [f for f in GGUF_FALLBACK_FILES if f != preferred]

    if not _can_import("huggingface_hub"):
        ensure_python_packages([("huggingface_hub", "huggingface_hub")], repair=False)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None, "huggingface_hub not available; cannot download GGUF."

    last_err: Optional[str] = None
    for fname in candidates:
        dest = MODELS_DIR / fname
        if dest.is_file() and dest.stat().st_size > 1_000_000 and not force:
            return dest.resolve(), f"GGUF already present: {dest.name}"
        logger.info(
            "Downloading GGUF %s from %s (this can take a while, multi‑GB) …",
            fname,
            repo,
        )
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=str(MODELS_DIR),
            )
            p = Path(path)
            if p.is_file():
                logger.info("Downloaded GGUF → %s (%.1f GB)", p, p.stat().st_size / 1e9)
                return p.resolve(), f"Downloaded GGUF: {p.name}"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            logger.warning("GGUF download failed for %s: %s", fname, exc)
            continue

    return None, f"GGUF download failed: {last_err}"


_EMBED_CACHE: Dict[str, Any] = {}
_TSPULSE_CACHE: Dict[str, Any] = {}


def ensure_embeddings(model_name: Optional[str] = None) -> Tuple[Any, str]:
    """
    Download / load SentenceTransformer embedding model into local HF cache.

    Returns (model_or_None, status). Cached after first successful load.
    On failure, remembers the error for the process — no pip repair loops.
    """
    global _EMBED_FAIL_MSG
    name = model_name or os.environ.get("GT_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    if name in _EMBED_CACHE:
        return _EMBED_CACHE[name], f"Embedding model ready (cached): {name}"
    if _EMBED_FAIL_MSG is not None:
        return None, _EMBED_FAIL_MSG

    if not downloads_enabled() and not _can_import("sentence_transformers"):
        _EMBED_FAIL_MSG = "sentence-transformers missing and downloads disabled."
        return None, _EMBED_FAIL_MSG

    SentenceTransformer = None  # type: ignore[assignment]
    try:
        from .compat import ensure_ml_compat

        ensure_ml_compat()
        from sentence_transformers import SentenceTransformer as _ST

        SentenceTransformer = _ST
    except ImportError:
        if downloads_enabled() and "sentence-transformers" not in _PIP_TRIED:
            _PIP_TRIED.add("sentence-transformers")
            ensure_python_packages(
                [("sentence-transformers", "sentence_transformers")],
                repair=False,
            )
            try:
                from sentence_transformers import SentenceTransformer as _ST

                SentenceTransformer = _ST
            except Exception as exc:  # noqa: BLE001
                _EMBED_FAIL_MSG = f"sentence-transformers not usable: {exc}"
                return None, _EMBED_FAIL_MSG
        else:
            _EMBED_FAIL_MSG = "sentence-transformers not installed."
            return None, _EMBED_FAIL_MSG
    except Exception as exc:  # noqa: BLE001 — version skew / broken stack
        if downloads_enabled() and _REPAIR_STATUS is None:
            logger.info("Embeddings import broken (%s); one repair attempt …", exc)
            repair_environment()
            try:
                from .compat import ensure_ml_compat

                ensure_ml_compat()
                from sentence_transformers import SentenceTransformer as _ST

                SentenceTransformer = _ST
            except Exception as exc2:  # noqa: BLE001
                _EMBED_FAIL_MSG = f"sentence-transformers not usable: {exc2}"
                return None, _EMBED_FAIL_MSG
        else:
            _EMBED_FAIL_MSG = f"sentence-transformers not usable: {exc}"
            return None, _EMBED_FAIL_MSG

    if SentenceTransformer is None:
        _EMBED_FAIL_MSG = "sentence-transformers not available."
        return None, _EMBED_FAIL_MSG

    try:
        logger.info("Loading embedding model '%s' …", name)
        try:
            from .device import torch_device

            dev = torch_device()
            model = SentenceTransformer(name, device=dev)
            dev_note = f" on {dev}"
        except TypeError:
            model = SentenceTransformer(name)
            dev_note = ""
        _EMBED_CACHE[name] = model
        return model, f"Embedding model ready: {name}{dev_note}"
    except Exception as exc:  # noqa: BLE001
        _EMBED_FAIL_MSG = f"Embedding load failed: {exc}"
        logger.warning("%s", _EMBED_FAIL_MSG)
        return None, _EMBED_FAIL_MSG


def ensure_tspulse(
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    *,
    load_weights: bool = True,
) -> Tuple[Any, str, str]:
    """
    Download / load Granite TS Pulse weights.

    Returns (model_or_None, mode, status) where mode is 'tspulse' or 'statistical'.
    Cached after first successful load.
    """
    global _TSPULSE_LAST_ERROR

    mid = model_id or os.environ.get("GT_TSPULSE_MODEL") or DEFAULT_TSPULSE_MODEL
    rev = revision or os.environ.get("GT_TSPULSE_REVISION") or DEFAULT_TSPULSE_REVISION
    cache_key = f"{mid}::{rev}"

    if load_weights and cache_key in _TSPULSE_CACHE:
        return _TSPULSE_CACHE[cache_key], "tspulse", f"TS Pulse ready (cached): {mid}"

    # Fast path: do not pip-install or download snapshots during deferred startup
    if not load_weights:
        loader, err = _tspulse_loader_class_with_error()
        if loader is None:
            return (
                None,
                "statistical",
                f"TS Pulse deferred (loader not ready: {err})",
            )
        return None, "statistical", f"TS Pulse package present for {mid} (load deferred)"

    # Probe / install package, then resolve class with real error detail
    pkg_status = ensure_granite_tsfm()
    loader, load_err = _tspulse_loader_class_with_error()
    if loader is None:
        # One more hard reset of import caches then retry
        _invalidate_import_caches()
        pkg_status = ensure_granite_tsfm()
        loader, load_err = _tspulse_loader_class_with_error()
    if loader is None:
        msg = (
            f"TS Pulse API not importable (pkg={pkg_status}; {load_err}); "
            "statistical fallback. Try: python -m pip install -U granite-tsfm"
        )
        _TSPULSE_LAST_ERROR = msg
        logger.warning("%s", msg)
        return None, "statistical", msg
    # Success path clears sticky failure detail
    _TSPULSE_LAST_ERROR = None

    if downloads_enabled():
        try:
            _snapshot_repo(mid, rev)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TS Pulse snapshot optional fail: %s", exc)

    last_err: Optional[Exception] = None
    # Prefer default revision first (anomaly-specific rev often missing)
    for kwargs in ({}, {"revision": rev} if rev else {}):
        try:
            logger.info("Loading TS Pulse %s %s …", mid, kwargs or "(default)")
            model = loader.from_pretrained(mid, **kwargs)
            model.eval()
            # Prefer CUDA when available
            try:
                from .device import device_status_summary, move_module_to_device, torch_device

                dev = torch_device()
                model = move_module_to_device(model, dev)
                dev_note = f" on {dev}"
                logger.info("TS Pulse device: %s", device_status_summary())
            except Exception as dev_exc:  # noqa: BLE001
                dev_note = ""
                logger.debug("TS Pulse device placement skipped: %s", dev_exc)
            _TSPULSE_CACHE[cache_key] = model
            return (
                model,
                "tspulse",
                f"TS Pulse ready: {mid} {kwargs or '(default rev)'}{dev_note}",
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("TS Pulse load attempt failed: %s", exc)

    return (
        None,
        "statistical",
        f"TS Pulse unavailable ({last_err}); using statistical detector",
    )


def _tspulse_loader_class() -> Any:
    """Resolve TSPulse class or None."""
    cls, _err = _tspulse_loader_class_with_error()
    return cls


def _tspulse_loader_class_with_error() -> Tuple[Any, str]:
    """
    Resolve TSPulse / TTM class from installed packages (granite-tsfm → tsfm_public).

    Returns (class_or_None, error_detail).
    """
    errors: List[str] = []
    # Prefer package root export, then modeling module (both work on granite-tsfm 0.3.x)
    for path in (
        ("tsfm_public.models.tspulse", "TSPulseForReconstruction"),
        ("tsfm_public.models.tspulse.modeling_tspulse", "TSPulseForReconstruction"),
        ("tsfm_public.models.tspulse", "TSPulseModel"),
        ("tsfm_public.models.tinytimemixer", "TinyTimeMixerForPrediction"),
    ):
        mod_name, cls_name = path
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls, "ok"
            errors.append(f"{mod_name}: no attr {cls_name}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            logger.warning("TS Pulse import %s.%s failed: %s", mod_name, cls_name, exc)
            continue
    return None, "; ".join(errors) if errors else "no candidate modules tried"


def _snapshot_repo(repo_id: str, revision: Optional[str] = None) -> str:
    """Download a full HF model snapshot into the hub cache."""
    if not downloads_enabled():
        return "skipped"
    if not _can_import("huggingface_hub"):
        ensure_python_packages([("huggingface_hub", "huggingface_hub")], repair=False)
    try:
        from huggingface_hub import snapshot_download

        kwargs: Dict[str, Any] = {"repo_id": repo_id}
        if revision:
            kwargs["revision"] = revision
        path = snapshot_download(**kwargs)
        return f"cached at {path}"
    except Exception as exc:  # noqa: BLE001
        if revision:
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(repo_id=repo_id)
                return f"cached at {path} (default rev; wanted {revision})"
            except Exception as exc2:  # noqa: BLE001
                return f"failed: {exc2}"
        return f"failed: {exc}"


def ensure_all_models(
    *,
    download_gguf: bool = True,
    download_embeddings: bool = True,
    download_tspulse: bool = True,
    install_packages: bool = True,
    load_weights: bool = True,
) -> Dict[str, str]:
    """
    One-shot bootstrap used at app startup.

    Parameters
    ----------
    load_weights :
        If False, only ensure packages + files on disk (no from_pretrained /
        SentenceTransformer init). Callers then load once — avoids double work.

    Returns a flat status dictionary for logging / UI.
    """
    ensure_directories()
    status: Dict[str, str] = {}

    if not downloads_enabled():
        status["downloads"] = "disabled (GT_NO_DOWNLOAD)"
        existing = find_gguf_model()
        status["gguf"] = f"present: {existing.name}" if existing else "missing"
        return status

    status["downloads"] = "enabled"

    if install_packages:
        # Probe only — no pip thrash (repair=False). Real installs happen
        # only when a component is actually missing and downloads enabled.
        pkg_status = ensure_python_packages(repair=False)
        for k, v in pkg_status.items():
            status[f"pkg:{k}"] = v

    if download_gguf:
        path, msg = ensure_gguf()
        status["gguf"] = msg
        if path:
            status["gguf_path"] = str(path)
        # Official CPU llama.cpp binary (fallback when python bind hits illegal instruction)
        try:
            from .llama_cli_backend import ensure_llama_cli_binary

            cli = ensure_llama_cli_binary()
            status["llama_cli"] = str(cli) if cli else "missing"
        except Exception as exc:  # noqa: BLE001
            status["llama_cli"] = f"failed: {exc}"

    if download_embeddings:
        if load_weights:
            _model, msg = ensure_embeddings()
            status["embeddings"] = msg
        else:
            status["embeddings"] = "deferred (load once with load_models)"

    if download_tspulse:
        if load_weights:
            _m, mode, msg = ensure_tspulse(load_weights=True)
            status["tspulse"] = msg
            status["tspulse_mode"] = mode
        else:
            # No pip / no HF snapshot during deferred ensure
            _m, mode, msg = ensure_tspulse(load_weights=False)
            status["tspulse"] = msg
            status["tspulse_mode"] = mode

    logger.info(
        "Model ensure status: %s",
        {k: (v[:120] + "…") if len(str(v)) > 120 else v for k, v in status.items()},
    )
    return status
