"""
Model loaders for GT Diagnostic Harness.

1. Main LLM: IBM Granite 4.1 8B (GGUF) via llama-cpp-python
2. Time-series anomaly: Granite TS Pulse (anomaly detection revision)
   with a statistical fallback when granite-tsfm / weights are unavailable

Design notes for beginners
--------------------------
- Heavy imports are inside functions / try blocks so importing this module
  does not crash if optional packages are not installed yet.
- ``ModelBundle`` holds whatever successfully loaded so the rest of the app
  can query capabilities and degrade gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import DEFAULT_EMBEDDING_MODEL, find_gguf_model, setup_logging

logger = setup_logging()

# (step_id, step_index, total_steps, frac_0_1, message)
LoadProgressCb = Callable[[str, int, int, float, str], None]

# Hugging Face model id for TS Pulse anomaly detection revision.
# Users can override with env GT_TSPULSE_MODEL / GT_TSPULSE_REVISION.
TSPULSE_MODEL_ID = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_REVISION = "tspulse-block-ad"
TSPULSE_FALLBACK_IDS = (
    "ibm-granite/granite-timeseries-tspulse-r1",
    "ibm-granite/granite-timeseries-ttm-r2",
)

# Classification head (dual-head revision). Fine-tune later; scaffold loads untrained head.
TSPULSE_CLF_REVISION = "tspulse-block-dualhead-512-p16-r1"
DEFAULT_CLF_LABELS: Tuple[str, ...] = (
    "normal",
    "cold_spot",
    "hets",
    "combustion_dynamics",
)


def _default_threads() -> int:
    import os

    return max(2, min(8, os.cpu_count() or 4))


def _clf_enabled() -> bool:
    """Classification head on by default; set GT_TSPULSE_CLF_ENABLE=0 to skip."""
    import os

    flag = (os.environ.get("GT_TSPULSE_CLF_ENABLE") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _clf_labels_from_env() -> List[str]:
    import os

    raw = (os.environ.get("GT_TSPULSE_CLF_LABELS") or "").strip()
    if not raw:
        return list(DEFAULT_CLF_LABELS)
    labels = [x.strip() for x in raw.split(",") if x.strip()]
    return labels if labels else list(DEFAULT_CLF_LABELS)


def _clf_channels_from_env() -> int:
    import os

    try:
        return max(1, int(os.environ.get("GT_TSPULSE_CLF_CHANNELS") or "16"))
    except ValueError:
        return 16


@dataclass
class LLMConfig:
    """Settings for llama.cpp Granite chat (CPU or CUDA when available)."""

    n_ctx: int = 2048  # smaller context → faster prompt eval on CPU
    n_threads: int = 0  # 0 → auto via _default_threads()
    # -1 → auto: prefer CUDA offload when available (see device.preferred_n_gpu_layers)
    n_gpu_layers: int = -1
    temperature: float = 0.2
    max_tokens: int = 512  # room for Reasoning / Hypotheses / Self-review / Final
    top_p: float = 0.9
    chat_format: Optional[str] = None  # auto / chatml / etc.

    def resolved_threads(self) -> int:
        return self.n_threads if self.n_threads and self.n_threads > 0 else _default_threads()

    def resolved_n_gpu_layers(self) -> int:
        if self.n_gpu_layers is not None and self.n_gpu_layers >= 0:
            return int(self.n_gpu_layers)
        from .device import preferred_n_gpu_layers

        return preferred_n_gpu_layers()


@dataclass
class ModelBundle:
    """
    Container for loaded runtime models.

    Attributes
    ----------
    llm : object or None
        llama_cpp.Llama instance
    tspulse : object or None
        TS Pulse reconstruction model (anomaly residuals)
    tspulse_mode : str
        'tspulse' | 'statistical' | 'unavailable'
    tspulse_clf : object or None
        TS Pulse classification head (signature labels)
    tspulse_clf_labels : list of str
        Class names matching classifier output dims
    tspulse_clf_trained : bool
        True when loaded from a fine-tuned checkpoint path
    embedding_model_name : str
        SentenceTransformer model name used by RAG
    status : dict
        Human-readable load status per component
    """

    llm: Any = None
    tspulse: Any = None
    tspulse_mode: str = "unavailable"
    tspulse_clf: Any = None
    tspulse_clf_labels: List[str] = field(default_factory=lambda: list(DEFAULT_CLF_LABELS))
    tspulse_clf_trained: bool = False
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    status: Dict[str, str] = field(default_factory=dict)
    gguf_path: Optional[str] = None


# Module-level singleton (lazy)
_BUNDLE: Optional[ModelBundle] = None


def get_bundle(force_reload: bool = False, auto_download: bool = True) -> ModelBundle:
    """Return (and lazily create) the process-wide ModelBundle."""
    global _BUNDLE
    if _BUNDLE is None or force_reload:
        _BUNDLE = load_models(auto_download=auto_download)
    return _BUNDLE


def load_models(
    gguf_path: Optional[str] = None,
    llm_config: Optional[LLMConfig] = None,
    load_llm: bool = True,
    load_tspulse: bool = True,
    auto_download: bool = True,
    progress: Optional[LoadProgressCb] = None,
) -> ModelBundle:
    """
    Load LLM and TS Pulse components.

    Parameters
    ----------
    gguf_path : str, optional
        Path to Granite 4.1 8B GGUF file.
    llm_config : LLMConfig, optional
        llama.cpp generation settings.
    load_llm / load_tspulse : bool
        Allow skipping a component (useful in tests).
    auto_download : bool
        If True (default), install missing packages and download model
        weights when not already present. Set False in unit tests or
        offline CI (or set env GT_NO_DOWNLOAD=1).
    progress :
        Optional callback(step_id, step_index, total_steps, frac, message)
        for TUI status during multi-stage load.
    """
    bundle = ModelBundle()
    cfg = llm_config or LLMConfig()
    # steps: packages(0), llm(1), tspulse(2), embeddings(3) — rag is caller-side
    total = 4
    nominal = (8.0, 5.0, 20.0, 12.0)

    def _p(step_i: int, detail: str = "", *, within: float = 0.15) -> None:
        if progress is None:
            return
        labels = (
            ("packages", "Packages / downloads"),
            ("llm", "Bind Granite GGUF"),
            ("tspulse", "Load TS Pulse"),
            ("embeddings", "Load embeddings"),
        )
        sid, label = labels[step_i]
        done = sum(nominal[:step_i]) + within * nominal[step_i]
        frac = min(0.99, done / sum(nominal))
        remaining = sum(nominal[step_i + 1 :]) + (1.0 - within) * nominal[step_i]
        msg = f"{label}" + (f" — {detail}" if detail else "")
        msg += f" · ~{int(remaining)}s left · step {step_i + 1}/{total}"
        try:
            progress(sid, step_i, total, frac, msg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("load progress failed: %s", exc)

    _p(0, "checking GGUF / llama-cli")
    if auto_download:
        try:
            from .download import ensure_gguf, downloads_enabled
            from .llama_cli_backend import ensure_llama_cli_binary

            # Fast path only: local GGUF file + CLI binary.
            # Do NOT probe sentence-transformers/torch/tsfm here (multi-second imports
            # and historical pip repair thrash made the TUI look hung).
            bits: Dict[str, str] = {}
            if load_llm:
                path, msg = ensure_gguf()
                bits["gguf"] = msg
                if path:
                    bits["gguf_path"] = str(path)
                try:
                    cli = ensure_llama_cli_binary()
                    bits["llama_cli"] = str(cli) if cli else "missing"
                except Exception as exc:  # noqa: BLE001
                    bits["llama_cli"] = f"failed: {exc}"
            bits["downloads"] = "enabled" if downloads_enabled() else "disabled"
            bits["heavy"] = "deferred (TS Pulse + embeddings on first diagnosis)"
            bundle.status["download"] = "; ".join(f"{k}={v}" for k, v in bits.items())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-download step failed: %s", exc)
            bundle.status["download"] = f"failed: {exc}"
    _p(0, "files ready", within=1.0)

    if load_llm:
        _p(1, "binding GGUF runner")
        from .device import device_status_summary

        bundle.status["device"] = device_status_summary()
        # Apply resolved thread + GPU layer counts
        cfg = LLMConfig(
            n_ctx=cfg.n_ctx,
            n_threads=cfg.resolved_threads(),
            n_gpu_layers=cfg.resolved_n_gpu_layers(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            chat_format=cfg.chat_format,
        )
        bundle.llm, bundle.gguf_path, msg = _load_llama(gguf_path, cfg)
        bundle.status["llm"] = msg
        _p(1, msg[:80], within=1.0)
    else:
        bundle.status["llm"] = "skipped"
        _p(1, "skipped", within=1.0)

    # Defer heavy weight loads so TUI reaches "Ready" in seconds.
    # TS Pulse recon + classifier + embeddings load on first diagnosis / RAG.
    if load_tspulse:
        _p(2, "anomaly engine (deferred)")
        bundle.tspulse = None
        bundle.tspulse_mode = "statistical"
        bundle.status["tspulse"] = (
            "deferred — loads on first diagnosis if package available"
        )
        # Classifier head: optional; scaffold from dual-head revision until fine-tuned
        bundle.tspulse_clf = None
        bundle.tspulse_clf_labels = list(_clf_labels_from_env())
        bundle.tspulse_clf_trained = False
        if _clf_enabled():
            bundle.status["tspulse_clf"] = (
                "deferred — classification head loads on first diagnosis "
                "(untrained until GT_TSPULSE_CLF_PATH is set)"
            )
        else:
            bundle.status["tspulse_clf"] = "disabled (set GT_TSPULSE_CLF_ENABLE=1)"
        _p(2, bundle.status["tspulse"][:80], within=1.0)
    else:
        bundle.tspulse_mode = "statistical"
        bundle.status["tspulse"] = "skipped (statistical fallback)"
        bundle.status["tspulse_clf"] = "skipped"
        _p(2, "skipped", within=1.0)

    _p(3, "embeddings (deferred)")
    emb_name = bundle.embedding_model_name
    bundle.status["embeddings"] = (
        f"deferred — '{emb_name}' loads on first RAG query"
    )
    _p(3, bundle.status["embeddings"][:80], within=1.0)

    logger.info("Model load status: %s", bundle.status)
    return bundle


def _load_llama(
    gguf_path: Optional[str],
    cfg: LLMConfig,
) -> Tuple[Any, Optional[str], str]:
    """
    Load Granite GGUF.

    CUDA preference
    ---------------
    When ``n_gpu_layers > 0`` (auto when CUDA is available):
      1. Try llama-cpp-python with GPU offload first
      2. Fall back to llama-cli with ``-ngl``
    When CPU-only: prefer llama-cli (stable on older CPUs), then python binding.
    """
    path = find_gguf_model(gguf_path)
    if path is None:
        return (
            None,
            None,
            "No GGUF found under models/. Auto-download failed or was disabled "
            "(set GT_NO_DOWNLOAD unset, or place a .gguf / set GT_GGUF_PATH).",
        )

    n_gpu = cfg.resolved_n_gpu_layers() if hasattr(cfg, "resolved_n_gpu_layers") else int(
        cfg.n_gpu_layers or 0
    )
    # cfg may already have resolved non-negative n_gpu_layers from load_models
    if cfg.n_gpu_layers is not None and cfg.n_gpu_layers >= 0:
        n_gpu = int(cfg.n_gpu_layers)

    prefer_gpu = n_gpu > 0
    errors: List[str] = []

    def _try_python() -> Tuple[Optional[Any], str]:
        try:
            try:
                from llama_cpp import Llama  # type: ignore
            except ImportError:
                from .download import ensure_python_packages

                ensure_python_packages([("llama-cpp-python", "llama_cpp")], repair=False)
                from llama_cpp import Llama  # type: ignore

            kwargs: Dict[str, Any] = {
                "model_path": str(path),
                "n_ctx": cfg.n_ctx,
                "n_threads": cfg.resolved_threads(),
                "n_gpu_layers": n_gpu,
                "verbose": False,
            }
            if cfg.chat_format:
                kwargs["chat_format"] = cfg.chat_format
            llm = Llama(**kwargs)
            tag = f"GPU n_gpu_layers={n_gpu}" if n_gpu > 0 else "CPU"
            return llm, f"Loaded GGUF via llama-cpp-python ({tag}): {path.name}"
        except Exception as exc:  # noqa: BLE001
            msg = f"llama-cpp-python: {exc}"
            logger.warning("llama-cpp-python GGUF load failed: %s", exc)
            return None, msg

    def _try_cli() -> Tuple[Optional[Any], str]:
        try:
            from .llama_cli_backend import ensure_llama_cli_binary, try_load_llama_cli

            cli = ensure_llama_cli_binary(prefer_cuda=prefer_gpu)
            if cli is None:
                return None, "llama-cli binary not available"
            runner, msg = try_load_llama_cli(
                path,
                n_ctx=cfg.n_ctx,
                n_threads=cfg.resolved_threads(),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                n_gpu_layers=n_gpu,
            )
            return runner, msg
        except Exception as exc:  # noqa: BLE001
            logger.warning("llama-cli path failed: %s", exc)
            return None, f"llama-cli: {exc}"

    order = (_try_python, _try_cli) if prefer_gpu else (_try_cli, _try_python)
    for loader in order:
        obj, msg = loader()
        if obj is not None:
            return obj, str(path), msg
        errors.append(msg)

    return (
        None,
        str(path),
        "GGUF load failed (cli + python): " + " | ".join(errors),
    )


def _load_tspulse(auto_download: bool = True) -> Tuple[Any, str, str]:
    """
    Load Granite TS Pulse for anomaly detection.

    Always prefers ``download.ensure_tspulse`` (weights under models/tspulse/).
    Falls back to a direct local load if the helper raises.
    """
    # Primary path — shared with download / tests (finds TSPulseForReconstruction)
    try:
        from .download import ensure_tspulse, _tspulse_loader_class

        if auto_download:
            return ensure_tspulse(load_weights=True)

        # Offline: only load if package already importable + local weights present
        loader = _tspulse_loader_class()
        if loader is None:
            return (
                None,
                "statistical",
                "TS Pulse package (tsfm_public) not importable; using statistical detector",
            )
        return ensure_tspulse(load_weights=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_tspulse failed: %s", exc)

    import os

    from .download import (
        download_hf_repo_to_local,
        tspulse_local_dir,
        _local_model_ready,
        _resolve_local_model_root,
        _tspulse_loader_class,
    )

    model_id = os.environ.get("GT_TSPULSE_MODEL", TSPULSE_MODEL_ID)
    revision = os.environ.get("GT_TSPULSE_REVISION", TSPULSE_REVISION)
    loader = _tspulse_loader_class()
    if loader is None:
        return (
            None,
            "statistical",
            "TS Pulse API not importable; using statistical detector. "
            "Install: pip install granite-tsfm",
        )

    local_dir = tspulse_local_dir(model_id, revision)
    last_err: Optional[Exception] = None
    try:
        if auto_download:
            root, _msg = download_hf_repo_to_local(
                model_id, local_dir, revision=revision
            )
        elif _local_model_ready(local_dir):
            root = _resolve_local_model_root(local_dir)
        else:
            return (
                None,
                "statistical",
                f"TS Pulse weights missing under {local_dir}; using statistical detector",
            )
        model = loader.from_pretrained(str(root), local_files_only=True)
        model.eval()
        try:
            from .device import move_module_to_device, torch_device

            model = move_module_to_device(model, torch_device())
        except Exception:
            pass
        return model, "tspulse", f"Loaded TS Pulse from {root}"
    except Exception as exc:  # noqa: BLE001
        last_err = exc

    return (
        None,
        "statistical",
        f"TS Pulse weights unavailable ({last_err}); using statistical detector",
    )


def _maybe_load_tspulse_classifier(bundle: ModelBundle) -> None:
    """Lazy-load classification head once when enabled and not yet attempted."""
    if bundle.tspulse_clf is not None:
        return
    if not _clf_enabled():
        bundle.status["tspulse_clf"] = "disabled (set GT_TSPULSE_CLF_ENABLE=1)"
        return

    st = str(bundle.status.get("tspulse_clf", "")).lower()
    recon_st = str(bundle.status.get("tspulse", "")).lower()
    if "disabled" in st or st == "skipped" or "skipped" in st:
        return
    # Only load when startup deferred the head (or recon), not on pure statistical test bundles
    should = "deferred" in st or "deferred" in recon_st
    if not should:
        return
    if "failed" in st and "deferred" not in st:
        retries = int(bundle.status.get("_tspulse_clf_retries", 0) or 0)
        if retries >= 2:
            return

    retries = int(bundle.status.get("_tspulse_clf_retries", 0) or 0)
    bundle.status["_tspulse_clf_retries"] = retries + 1
    model, labels, trained, msg = _load_tspulse_classifier()
    bundle.tspulse_clf_labels = labels
    bundle.tspulse_clf_trained = trained
    if model is not None:
        bundle.tspulse_clf = model
        bundle.status["tspulse_clf"] = msg
        logger.info("TS Pulse classifier: %s", msg)
    else:
        bundle.status["tspulse_clf"] = msg
        logger.warning("TS Pulse classifier not loaded: %s", msg)


def _load_tspulse_classifier() -> Tuple[Any, List[str], bool, str]:
    """
    Load TSPulseForClassification.

    - If GT_TSPULSE_CLF_PATH points at a fine-tuned directory → trained=True
    - Else scaffold from dual-head HF revision with DEFAULT/env labels (trained=False)
    """
    import os
    from pathlib import Path

    labels = _clf_labels_from_env()
    n_ch = _clf_channels_from_env()
    local = (os.environ.get("GT_TSPULSE_CLF_PATH") or "").strip()
    model_id = os.environ.get("GT_TSPULSE_MODEL", TSPULSE_MODEL_ID)
    revision = os.environ.get("GT_TSPULSE_CLF_REVISION", TSPULSE_CLF_REVISION)

    try:
        from tsfm_public.models.tspulse import TSPulseForClassification
    except Exception as exc:  # noqa: BLE001
        return None, labels, False, f"TSPulseForClassification import failed: {exc}"

    try:
        if local and Path(local).exists():
            model = TSPulseForClassification.from_pretrained(local)
            model.eval()
            # Prefer labels stored alongside checkpoint if present
            meta = Path(local) / "gt_labels.json"
            if meta.is_file():
                import json

                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data:
                        labels = [str(x) for x in data]
                    elif isinstance(data, dict) and "labels" in data:
                        labels = [str(x) for x in data["labels"]]
                except Exception:
                    pass
            n_out = int(getattr(model.config, "num_targets", len(labels)) or len(labels))
            if len(labels) != n_out:
                labels = (labels + [f"class_{i}" for i in range(n_out)])[:n_out]
            return (
                model,
                labels,
                True,
                f"Loaded fine-tuned classifier from {local} ({n_out} classes)",
            )

        # Scaffold: download dual-head revision into models/tspulse_clf/ then load
        from .download import download_hf_repo_to_local, tspulse_clf_local_dir

        clf_dir = tspulse_clf_local_dir(model_id, revision)
        try:
            root, dl_msg = download_hf_repo_to_local(
                model_id, clf_dir, revision=revision
            )
            logger.info("TS Pulse classifier weights: %s", dl_msg)
            model = TSPulseForClassification.from_pretrained(
                str(root),
                local_files_only=True,
                num_targets=len(labels),
                num_input_channels=n_ch,
            )
        except Exception as local_exc:  # noqa: BLE001
            logger.warning(
                "Local classifier snapshot failed (%s); hub fallback", local_exc
            )
            model = TSPulseForClassification.from_pretrained(
                model_id,
                revision=revision,
                num_targets=len(labels),
                num_input_channels=n_ch,
            )
            try:
                download_hf_repo_to_local(model_id, clf_dir, revision=revision)
            except Exception:
                pass
        model.eval()
        try:
            from .device import move_module_to_device, torch_device

            model = move_module_to_device(model, torch_device())
        except Exception:
            pass
        return (
            model,
            labels,
            False,
            f"Scaffold classifier loaded from models/tspulse_clf ({revision}, "
            f"{len(labels)} classes, {n_ch} channels) — UNTRAINED head; "
            f"set GT_TSPULSE_CLF_PATH after fine-tune",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classifier load failed: %s", exc)
        return None, labels, False, f"classifier load failed: {exc}"


def classify_signature(
    df: pd.DataFrame,
    columns: Sequence[str],
    bundle: Optional[ModelBundle] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run TS Pulse classification head on the last context window of ``df``.

    Returns None if classifier disabled/unavailable. Probs are **not reliable**
    until the head is fine-tuned (``tspulse_clf_trained``).
    """
    bundle = bundle or get_bundle()
    if bundle.tspulse_clf is None:
        return None

    import torch

    model = bundle.tspulse_clf
    labels = list(bundle.tspulse_clf_labels or DEFAULT_CLF_LABELS)
    cfg = getattr(model, "config", None)
    context_length = int(
        getattr(cfg, "context_length", None)
        or getattr(cfg, "seq_len", None)
        or 512
    )
    n_ch_model = int(getattr(cfg, "num_input_channels", None) or _clf_channels_from_env())

    cols = [c for c in columns if c in df.columns][:n_ch_model]
    if not cols:
        return {
            "enabled": True,
            "trained": bundle.tspulse_clf_trained,
            "error": "no numeric columns for classifier",
        }

    # Build (T, C) float matrix, pad/truncate channels to model width
    series = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").astype(float).ffill().bfill().fillna(0.0)
        series.append(s.to_numpy(dtype=np.float32))
    mat = np.column_stack(series)  # (T, C_data)
    t_len, c_data = mat.shape
    if c_data < n_ch_model:
        pad = np.zeros((t_len, n_ch_model - c_data), dtype=np.float32)
        mat = np.concatenate([mat, pad], axis=1)
    elif c_data > n_ch_model:
        mat = mat[:, :n_ch_model]

    # Last context_length rows (pad if short)
    if t_len >= context_length:
        window = mat[-context_length:]
    else:
        window = np.zeros((context_length, n_ch_model), dtype=np.float32)
        window[-t_len:] = mat

    # Per-channel z-score for stability
    mu = window.mean(axis=0, keepdims=True)
    sigma = window.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    window = ((window - mu) / sigma).astype(np.float32)

    x = torch.as_tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, T, C)
    model.eval()
    torch.manual_seed(0)
    with torch.inference_mode():
        out = model(past_values=x, return_loss=False, return_dict=True)
    logits = getattr(out, "prediction_outputs", None)
    if logits is None:
        logits = getattr(out, "logits", None)
    if logits is None:
        return {
            "enabled": True,
            "trained": bundle.tspulse_clf_trained,
            "error": "no prediction_outputs on classifier",
        }

    logits_np = logits.detach().cpu().float().numpy().reshape(-1)
    # Softmax
    logits_np = logits_np - logits_np.max()
    exp = np.exp(logits_np)
    probs = exp / max(float(exp.sum()), 1e-12)
    n = min(len(labels), len(probs))
    labels = labels[:n]
    probs = probs[:n]
    order = list(np.argsort(-probs))
    top_i = int(order[0]) if order else 0
    prob_map = {labels[i]: float(np.round(probs[i], 6)) for i in range(n)}
    ranking = [(labels[i], float(np.round(probs[i], 6))) for i in order]

    return {
        "enabled": True,
        "trained": bool(bundle.tspulse_clf_trained),
        "labels": labels,
        "probs": prob_map,
        "ranking": ranking,
        "top_label": labels[top_i] if labels else None,
        "top_prob": float(np.round(probs[top_i], 6)) if n else None,
        "context_length": context_length,
        "channels_used": cols,
        "note": (
            "Fine-tuned checkpoint"
            if bundle.tspulse_clf_trained
            else "UNTRAINED classification head (scaffold only) — fine-tune and set GT_TSPULSE_CLF_PATH"
        ),
    }


def llm_available(bundle: Optional[ModelBundle] = None) -> bool:
    """True when a real GGUF runner is loaded (python bind or llama-cli)."""
    b = bundle or get_bundle()
    return b is not None and b.llm is not None


OFFLINE_DRAFT_MARKERS = (
    "rule-based — llama.cpp GGUF not active",
    "offline fallback — install Granite GGUF",
    "Offline draft only",
    "no generative model was loaded",
)


def is_offline_draft(text: str) -> bool:
    """True if text is the deterministic offline/rule-based draft (not model-backed)."""
    t = text or ""
    return any(m in t for m in OFFLINE_DRAFT_MARKERS)


def generate_llm(
    bundle: ModelBundle,
    system_prompt: str,
    user_prompt: str,
    config: Optional[LLMConfig] = None,
    *,
    role: str = "analyst",
) -> str:
    """
    Run a single chat completion with Granite GGUF
    (llama-cpp-python or official llama-cli backend).

    If the LLM is not loaded, returns a deterministic offline draft so the
    rest of the pipeline (and unit tests) still work.

    Parameters
    ----------
    role :
        ``analyst`` → structured offline draft from the user prompt.
        ``reflect`` is not handled here when offline — callers should skip
        a second offline pass (see analysis.run_diagnosis).
    """
    cfg = config or LLMConfig()
    if bundle.llm is None:
        # Only produce the analyst offline draft. Reflection is handled
        # deterministically in analysis so final_report stays clean.
        return _offline_llm_draft(system_prompt, user_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        # Prefer chat completion API when available (python bind + LlamaCliRunner)
        if hasattr(bundle.llm, "create_chat_completion"):
            out = bundle.llm.create_chat_completion(
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
            )
            return out["choices"][0]["message"]["content"].strip()

        # Raw completion fallback
        prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        out = bundle.llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=["User:", "System:"],
        )
        return out["choices"][0]["text"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed")
        return _offline_llm_draft(system_prompt, user_prompt) + f"\n\n_(LLM error: {exc})_"


def _offline_llm_draft(system_prompt: str, user_prompt: str) -> str:
    """
    Deterministic structured draft when llama.cpp cannot run the GGUF.

    Uses anomaly summary, operator context, and top channels from the
    analyst prompt. Clear about CPU/llama.cpp issues when GGUF is present
    but unloadable (Windows illegal-instruction).
    """
    severity_line = _first_matching_line(user_prompt, ("severity:",))
    mode_line = _first_matching_line(user_prompt, ("mode:",))
    summary_block = _extract_section(user_prompt, "## Anomaly summary", "## ")
    flag_block = _extract_section(user_prompt, "### Flagged points", "## ")
    context_block = _extract_section(user_prompt, "## Operator / process context", "## ")
    rag_block = _extract_section(user_prompt, "## Retrieved knowledge & prior cases", "## ")

    top_channel = None
    if severity_line and "top_channel=" in severity_line:
        top_channel = severity_line.split("top_channel=")[-1].strip(" )")

    # Tailor failure modes to top channel name
    modes = [
        "1. Combustion / exhaust temperature spreads (fuel nozzle imbalance, TC fault)",
        "2. Vibration / bearing channels (unbalance, alignment, sensor jump)",
        "3. Compressor efficiency and IGV / bleed valve state",
    ]
    tc = (top_channel or "").upper()
    if "EGT" in tc or "SPREAD" in tc or "EXHAUST" in tc:
        modes = [
            "1. **Primary:** Exhaust temperature spread — fuel nozzle imbalance, clogged nozzle, "
            "or thermocouple fault (confirm true spread vs single-TC bias)",
            "2. Concurrent vibration rise — mechanical insult vs measurement correlation with load/fuel",
            "3. Compressor / IGV contribution if CDP/CDT moved with the event",
        ]
    elif "VIB" in tc:
        modes = [
            "1. **Primary:** Bearing / shaft vibration — unbalance, alignment, coupling, oil film",
            "2. Process-coupled excitation if load or EGT moved with vibration",
            "3. Sensor/probe mounting or cable fault if only one channel jumped",
        ]

    ctx = (context_block or "").strip()
    if ctx and ctx.lower() not in {"(none provided)", ""}:
        ctx_line = f"Operator notes: {ctx[:400]}"
    else:
        ctx_line = "No additional operator context provided."

    rag_note = ""
    if rag_block and "No relevant" not in rag_block:
        rag_note = "Prior cases / process maps were retrieved and should guide checks (see RAG section)."

    lines = [
        "## GT Diagnostic Draft (rule-based — llama.cpp GGUF not active on this CPU)",
        "",
        "## Reasoning",
        "Anomaly screening completed on the uploaded historian window. "
        "Severity is driven by robust z-scores (or TS Pulse residuals when available) "
        f"across sensor channels. {ctx_line}",
    ]
    if mode_line:
        lines.append(f"- {mode_line.strip()}")
    if severity_line:
        lines.append(f"- {severity_line.strip()}")
    if rag_note:
        lines.append(f"- {rag_note}")
    if summary_block:
        lines.append("Evidence snippets:")
        for ln in summary_block.strip().splitlines()[:8]:
            ln = ln.strip()
            if ln:
                lines.append(f"- {ln[:240]}")
    if flag_block:
        lines.append("Top flagged points:")
        for ln in flag_block.strip().splitlines()[:12]:
            ln = ln.strip()
            if ln.startswith("-"):
                lines.append(ln[:240])
    lines.extend(
        [
            "",
            "## Initial hypotheses",
            *modes,
            "",
            "## Self-review",
            "- Instrumentation validation is required before any hardware work "
            "(false positives from TC/probe faults are common).",
            "- Rankings above are heuristic without a generative model — treat confidence as Med/Low.",
            "- Missing load/ambient context would weaken process-coupling claims.",
            "",
            "## Final diagnosis",
            "### Executive summary",
            "See severity and top channel above; prioritize the #1 hypothesis after instrument checks.",
            "",
            "### Immediate actions",
            "1. Validate instrumentation on the top flagged channel(s) before hardware work",
            "2. Compare this window to the last stable baseload period at similar load/ambient",
            "3. Capture SOE / alarms / fuel-transfer events around the peak anomaly rows",
            "",
            "### Longer-term checks",
            "1. Trend the top channels across prior starts and load ramps",
            "2. Save validated findings with **Save & Learn** so RAG improves next time",
            "",
            "### Data quality caveats",
            "Generative Granite chat is inactive here (llama.cpp prebuilt wheel hit a CPU "
            "illegal-instruction error, or GGUF was not loadable). Scoring and process-map "
            "RAG remain valid.",
        ]
    )
    return "\n".join(lines)


def _md_section(text: str, heading: str) -> str:
    """Extract markdown section body under ``## heading`` (local, no circular import)."""
    import re

    if not text or not heading:
        return ""
    pat = re.compile(
        rf"^(#{{1,3}})\s*{re.escape(heading)}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return ""
    rest = text[m.end() :]
    nxt = re.search(r"^#{1,3}\s+\S", rest, re.MULTILINE)
    return (rest[: nxt.start()] if nxt else rest).strip()


def offline_reflect(
    draft: str,
    *,
    mode_key: str,
    anomaly_summary: str,
    severity: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Package a final report from the draft.

    If the draft already has a ``## Final diagnosis`` section (LLM structured
    output), prefer that as the body so operators see the model’s conclusion.
    """
    sev = severity or {}
    label = sev.get("label") or "n/a"
    level = sev.get("level") or "n/a"
    score = sev.get("severity")
    top = sev.get("top_channel")
    score_txt = f"{score}" if score is not None else "n/a"

    body = draft or ""
    for marker in (
        "## GT Diagnostic Draft (rule-based — llama.cpp GGUF not active on this CPU)",
        "## GT Diagnostic Draft (offline fallback — install Granite GGUF for full LLM)",
        "## GT Diagnostic Draft",
    ):
        body = body.replace(marker, "").strip()

    # Prefer explicit Final diagnosis section from structured LLM output
    final_sec = _md_section(body, "Final diagnosis")
    if final_sec:
        body_block = final_sec
    else:
        body_block = body if body else "_No draft content._"

    # Pull reasoning trail into the packaged report when present
    reasoning = _md_section(body, "Reasoning")
    hyps = _md_section(body, "Initial hypotheses")

    lines = [
        "## GT Diagnostic Final Report",
        "",
        f"**Mode:** {mode_key}",
        f"**Severity:** {label} (level=`{level}`, score={score_txt}, top_channel=`{top}`)",
        "",
        "### Anomaly summary",
        anomaly_summary or "_No anomaly summary._",
        "",
        "### Final diagnosis",
        body_block,
    ]
    if reasoning:
        lines.extend(["", "### Reasoning (from model)", reasoning])
    if hyps:
        lines.extend(["", "### Initial hypotheses (from model)", hyps])
    lines.extend(
        [
            "",
            "### Packaging notes",
            "- Severity label and top channel attached for operator triage.",
            "- Full reasoning / self-review also shown in the report sections below.",
        ]
    )
    return "\n".join(lines)


def _first_matching_line(text: str, prefixes: Sequence[str]) -> str:
    for line in (text or "").splitlines():
        low = line.strip().lower()
        for p in prefixes:
            if p.lower() in low:
                return line.strip()
    return ""


def _extract_section(text: str, start_heading: str, next_heading_prefix: str) -> str:
    """Return text between start_heading and the next markdown heading prefix."""
    if not text or start_heading not in text:
        return ""
    start = text.find(start_heading)
    rest = text[start + len(start_heading) :]
    # Find next ## heading
    idx = rest.find("\n## ")
    if idx >= 0:
        return rest[:idx].strip()
    return rest.strip()


def detect_anomalies(
    df: pd.DataFrame,
    bundle: Optional[ModelBundle] = None,
    numeric_columns: Optional[Sequence[str]] = None,
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Run anomaly detection on a multivariate sensor CSV frame.

    Prefers TS Pulse residual scoring when the model is loaded; otherwise
    uses robust statistical (median/MAD) z-scores — always deterministic
    for the same inputs.
    """
    bundle = bundle or get_bundle()
    if df is None or df.empty:
        return {
            "mode": "empty",
            "rows": 0,
            "columns": [],
            "anomalies": [],
            "summary": "No data provided.",
            "column_scores": {},
        }

    num_cols = list(numeric_columns) if numeric_columns else _infer_numeric_columns(df)
    if not num_cols:
        return {
            "mode": "empty",
            "rows": len(df),
            "columns": [],
            "anomalies": [],
            "summary": "No numeric sensor columns found.",
            "column_scores": {},
        }

    # Always compute statistical baseline (also used if TS Pulse fails mid-run)
    stats_result = _statistical_anomalies(df, num_cols, z_threshold=z_threshold)

    # Lazy-load TS Pulse recon on first diagnosis when startup deferred it.
    # Do NOT auto-load when tests inject a statistical-only bundle (no "deferred").
    if bundle.tspulse is None and (bundle.tspulse_mode or "") != "tspulse":
        try:
            st = str(bundle.status.get("tspulse", "")).lower()
            retries = int(bundle.status.get("_tspulse_retries", 0) or 0)
            package_ok = False
            try:
                import tsfm_public  # noqa: F401

                package_ok = True
            except Exception:
                package_ok = False
            false_neg = (
                package_ok
                and retries < 3
                and (
                    "failed permanently" in st
                    or "not importable" in st
                    or "statistical fallback): ok" in st
                )
            )
            should_try = ("deferred" in st and retries < 3) or false_neg
            if "deferred" not in st and not false_neg:
                should_try = False
            if should_try:
                bundle.status["_tspulse_retries"] = retries + 1
                logger.info("Lazy-loading TS Pulse for anomaly detection …")
                model, mode, msg = _load_tspulse(auto_download=True)
                if model is not None and mode == "tspulse":
                    bundle.tspulse = model
                    bundle.tspulse_mode = "tspulse"
                    bundle.status["tspulse"] = msg
                    logger.info("Lazy-loaded TS Pulse: %s", msg)
                else:
                    bundle.tspulse_mode = "statistical"
                    bundle.status["tspulse"] = msg
                    logger.warning("TS Pulse not available: %s", msg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TS Pulse lazy load failed: %s", exc)
            bundle.status["tspulse"] = f"lazy load error: {exc}"

    # Lazy-load classification head (scaffold or fine-tuned checkpoint)
    _maybe_load_tspulse_classifier(bundle)

    result: Dict[str, Any]
    if bundle.tspulse is not None and bundle.tspulse_mode == "tspulse":
        try:
            pulse = _tspulse_anomalies(df, num_cols, bundle.tspulse)
            if pulse.get("reconstruction_channels", 0) > 0 and pulse.get("column_scores"):
                pulse["statistical_backup"] = {
                    "anomaly_count": len(stats_result["anomalies"]),
                    "column_scores": stats_result["column_scores"],
                }
                result = pulse
            else:
                logger.warning(
                    "TS Pulse returned no reconstruction residuals; using statistical"
                )
                result = dict(stats_result)
                result["mode"] = "statistical"
        except Exception as exc:  # noqa: BLE001
            logger.warning("TS Pulse inference failed (using statistical): %s", exc)
            result = dict(stats_result)
            result["mode"] = "statistical"
            result["summary"] = str(result.get("summary", "")) + f" (TS Pulse error: {exc})"
            result["tspulse_error"] = str(exc)
    else:
        result = dict(stats_result)
        result["mode"] = "statistical"

    # Signature classification (optional second head)
    try:
        clf = classify_signature(df, num_cols, bundle=bundle)
        if clf:
            result["classification"] = clf
            top = clf.get("top_label")
            conf = clf.get("top_prob")
            trained = clf.get("trained")
            note = "trained" if trained else "UNTRAINED head — train later"
            result["summary"] = (
                str(result.get("summary", ""))
                + f" | Signature clf: {top} ({conf:.2%}, {note})"
                if isinstance(conf, (int, float))
                else str(result.get("summary", "")) + f" | Signature clf: {top} ({note})"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Signature classification failed: %s", exc)
        result["classification"] = {"enabled": False, "error": str(exc)}

    return result


def _infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Pick numeric columns, skipping obvious index/time fields when possible."""
    skip = {"index", "time", "timestamp", "datetime", "date", "t", "id"}
    cols: List[str] = []
    for c in df.columns:
        if str(c).strip().lower() in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


def _statistical_anomalies(
    df: pd.DataFrame,
    columns: Sequence[str],
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Robust z-score anomaly detection (median / MAD).

    For each numeric column:
    - z = 0.6745 * (x - median) / MAD
    - Flag points with |z| >= threshold
    """
    anomalies: List[Dict[str, Any]] = []
    column_scores: Dict[str, float] = {}

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").astype(float)
        valid = series.dropna()
        if valid.empty:
            column_scores[col] = 0.0
            continue
        median = float(np.median(valid.values))
        mad = float(np.median(np.abs(valid.values - median)))
        # Avoid divide-by-zero; use std if MAD is 0
        if mad < 1e-12:
            std = float(valid.std(ddof=0)) or 1.0
            z = (series - median) / std
        else:
            z = 0.6745 * (series - median) / mad

        z = z.fillna(0.0)
        abs_z = z.abs()
        column_scores[col] = float(abs_z.max()) if len(abs_z) else 0.0

        flag_idx = np.where(abs_z.values >= z_threshold)[0]
        # Cap per-column flags to keep reports readable
        for i in flag_idx[:50]:
            anomalies.append(
                {
                    "row": int(i),
                    "column": col,
                    "value": float(series.iloc[i]) if not pd.isna(series.iloc[i]) else None,
                    "score": float(abs_z.iloc[i]),
                    "method": "robust_zscore",
                }
            )

    anomalies.sort(key=lambda a: a["score"], reverse=True)
    top_cols = sorted(column_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
    summary = (
        f"Statistical anomaly scan over {len(columns)} channels, {len(df)} rows. "
        f"Flagged {len(anomalies)} point(s). "
        f"Top channels by max |z|: "
        + ", ".join(f"{c}={s:.2f}" for c, s in top_cols)
    )
    return {
        "mode": "statistical",
        "rows": len(df),
        "columns": list(columns),
        "anomalies": anomalies[:200],
        "summary": summary,
        "column_scores": column_scores,
        "z_threshold": z_threshold,
    }


def _tspulse_anomalies(
    df: pd.DataFrame,
    columns: Sequence[str],
    model: Any,
    context_length: int = 512,
) -> Dict[str, Any]:
    """
    Score anomalies using TS Pulse reconstruction residuals
    (``TSPulseForReconstruction``: past_values → reconstruction_outputs).

    Deterministic for fixed model weights + fixed CSV (eval + inference_mode,
    fixed float32 pipeline, rounded scores). Reconstruction failures raise —
    never silently fall back to |x-mean| while still reporting mode=tspulse.
    """
    import torch

    anomalies: List[Dict[str, Any]] = []
    column_scores: Dict[str, float] = {}
    model.eval()

    # Deterministic torch / numpy for residual math
    torch.manual_seed(0)
    np.random.seed(0)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    cfg = getattr(model, "config", None)
    if cfg is not None:
        context_length = int(
            getattr(cfg, "context_length", None)
            or getattr(cfg, "seq_len", None)
            or context_length
        )

    # Stable column order
    cols = list(columns)
    recon_ok = 0
    recon_fail: List[str] = []

    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        series = series.ffill().bfill().fillna(0.0)
        values = series.to_numpy(dtype=np.float32, copy=True)
        if len(values) < 8:
            column_scores[col] = 0.0
            continue

        # Population stats (ddof=0) for determinism across pandas/numpy versions
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma < 1e-12:
            sigma = 1.0
        normed = ((values - mu) / sigma).astype(np.float32)

        # Single full-series window: pad/trim to fixed context_length once
        # (avoids multi-window accumulation order noise on short historian snips)
        if len(normed) >= context_length:
            window_in = normed[-context_length:]
            valid_len = context_length
            offset = len(normed) - context_length
        else:
            window_in = np.zeros(context_length, dtype=np.float32)
            window_in[: len(normed)] = normed
            valid_len = len(normed)
            offset = 0

        try:
            residual = _tspulse_residual(model, window_in, torch)
            recon_ok += 1
        except Exception as exc:  # noqa: BLE001
            recon_fail.append(f"{col}: {exc}")
            logger.warning("TS Pulse reconstruction failed for %s: %s", col, exc)
            raise RuntimeError(
                f"TS Pulse reconstruction failed for channel {col}: {exc}"
            ) from exc

        # Map residuals back onto original series length
        scores = np.zeros(len(normed), dtype=np.float32)
        n = min(valid_len, len(residual))
        scores[offset : offset + n] = residual[:n]
        # Round for cross-process stability of floats
        max_score = float(np.round(float(scores.max()) if len(scores) else 0.0, 6))
        column_scores[col] = max_score

        thr = float(np.round(float(scores.mean() + 3.0 * (float(scores.std()) or 1e-3)), 6))
        for i in np.where(scores >= thr)[0][:50]:
            anomalies.append(
                {
                    "row": int(i),
                    "column": col,
                    "value": float(values[i]),
                    "score": float(np.round(float(scores[i]), 6)),
                    "method": "tspulse_residual",
                }
            )

    if recon_ok == 0:
        raise RuntimeError(
            "TS Pulse produced no reconstruction residuals: " + "; ".join(recon_fail)
        )

    anomalies.sort(key=lambda a: (-a["score"], a["column"], a["row"]))
    top_cols = sorted(column_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    summary = (
        f"TS Pulse anomaly scan over {len(cols)} channels, {len(df)} rows. "
        f"Flagged {len(anomalies)} point(s). Top residual channels: "
        + ", ".join(f"{c}={s:.3f}" for c, s in top_cols)
    )
    return {
        "mode": "tspulse",
        "rows": len(df),
        "columns": cols,
        "anomalies": anomalies[:200],
        "summary": summary,
        "column_scores": column_scores,
        "reconstruction_channels": recon_ok,
    }


def _tspulse_residual(model: Any, chunk: np.ndarray, torch_mod: Any) -> np.ndarray:
    """
    Reconstruction residual for one fixed-length 1-D window.

    TSPulse applies stochastic masking even in eval; we seed torch before each
    forward so the same window always yields the same residual (deterministic
    scores for a fixed CSV).

    Raises on failure — callers must not claim mode=tspulse with mean-fallback.
    """
    # Prefer CUDA when the model (or torch) is on GPU
    try:
        from .device import torch_device

        dev = torch_device()
    except Exception:
        dev = "cpu"
    try:
        p = next(model.parameters())
        dev = str(p.device)
    except Exception:
        pass

    x = torch_mod.as_tensor(chunk, dtype=torch_mod.float32).view(1, -1, 1)
    try:
        x = x.to(dev)
    except Exception:
        pass
    last_err: Optional[Exception] = None

    # Stable seed from window content so identical series → identical masks
    seed = int(np.abs(chunk.astype(np.float64).sum() * 1e6)) % (2**31 - 1)
    seed = (seed + 17 * len(chunk) + 7919) % (2**31 - 1)

    # Prefer official TSPulseForReconstruction API
    for call in (
        lambda: model(past_values=x, return_loss=False),
        lambda: model(past_values=x),
        lambda: model(x),
    ):
        try:
            torch_mod.manual_seed(seed)
            if torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(seed)
            with torch_mod.inference_mode():
                out = call()
            recon = _extract_prediction(out)
            if recon is None:
                raise RuntimeError("no reconstruction/prediction tensor in model output")
            pred_np = recon.detach().cpu().float().numpy().reshape(-1)
            n = min(len(chunk), len(pred_np))
            if n <= 0:
                raise RuntimeError("empty reconstruction tensor")
            return np.abs(chunk[:n] - pred_np[:n]).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    raise RuntimeError(f"TS Pulse residual failed: {last_err}")


def _extract_prediction(out: Any) -> Any:
    """Pull a tensor from various HF / TSPulse model output types."""
    if out is None:
        return None
    if hasattr(out, "reconstruction_outputs") and out.reconstruction_outputs is not None:
        return out.reconstruction_outputs
    if hasattr(out, "prediction_outputs") and out.prediction_outputs is not None:
        return out.prediction_outputs
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    return out if hasattr(out, "detach") else None
