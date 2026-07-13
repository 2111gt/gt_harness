"""
Utility helpers for GT Diagnostic Harness.

This module is intentionally free of heavy ML imports so it stays fast
and easy to test. Beginners: start here to learn project layout and paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------

# src/utils.py → parents[1] is the project root (gt_harness/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

MODELS_DIR: Path = PROJECT_ROOT / "models"
# Project-local HF weights (same idea as GGUF under models/)
TSPULSE_MODELS_DIR: Path = MODELS_DIR / "tspulse"
TSPULSE_CLF_MODELS_DIR: Path = MODELS_DIR / "tspulse_clf"
KNOWLEDGE_DIR: Path = PROJECT_ROOT / "knowledge"
SAVED_CASES_DIR: Path = PROJECT_ROOT / "saved_cases"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
SAMPLES_DIR: Path = PROJECT_ROOT / "samples"

# Default GGUF filename under models/ (IBM HF naming uses hyphen before quant).
# Auto-downloaded from ibm-granite/granite-4.1-8b-GGUF on first launch if missing.
DEFAULT_GGUF_NAME = "granite-4.1-8b-Q4_K_M.gguf"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chroma collection name for knowledge + saved cases
RAG_COLLECTION_NAME = "gt_knowledge"


def ensure_directories() -> None:
    """Create required folders if they do not exist."""
    for path in (
        MODELS_DIR,
        TSPULSE_MODELS_DIR,
        TSPULSE_CLF_MODELS_DIR,
        KNOWLEDGE_DIR,
        SAVED_CASES_DIR,
        CHROMA_DIR,
        SAMPLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run_hidden_subprocess(
    cmd: Sequence[str],
    *,
    timeout: Optional[float] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """
    Run a child process without attaching a console window.

    Critical for Textual/TUI on Windows: llama-cli / pip children that inherit
    the TUI console often hang waiting on stdin or console APIs. CLI mode
    works because nothing has taken over the terminal.
    """
    kwargs: Dict[str, Any] = {
        "args": list(cmd),
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
        "cwd": cwd,
        "env": env,
    }
    if input_text is not None:
        kwargs["input"] = input_text
    else:
        kwargs["stdin"] = subprocess.DEVNULL

    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000 — no console for the child
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
        except Exception:
            pass

    return subprocess.run(**kwargs)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure a simple console logger for the harness.

    Returns
    -------
    logging.Logger
        Logger named "gt_harness".
    """
    logger = logging.getLogger("gt_harness")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def configure_tui_logging() -> Path:
    """
    Route all library/harness logs to a file and silence console noise.

    Textual owns the terminal; StreamHandlers / tqdm / HF progress bars
    otherwise paint over the TUI.
    """
    # Progress bars / tokenizer / HF chatter
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "gt_harness.log"

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    # Root: file only (remove stream handlers)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)

    # Harness logger: file only
    gh = logging.getLogger("gt_harness")
    for h in list(gh.handlers):
        gh.removeHandler(h)
    gh.addHandler(file_handler)
    gh.setLevel(logging.INFO)
    gh.propagate = False

    # Quiet noisy third-party loggers (still go to file via root if propagate)
    for name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "torch",
        "tsfm_public",
        "chromadb",
        "urllib3",
        "filelock",
        "asyncio",
        "markdown_it",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = True

    # Silence tqdm WITHOUT replacing the class with a lambda.
    # Hugging Face hub does `class tqdm(old_tqdm):` — if old_tqdm is a function,
    # import crashes with: TypeError: function() argument 'code' must be code, not str
    # which then breaks tsfm_public / sentence-transformers / TS Pulse loading.
    _silence_tqdm_safely()

    return log_path


def _silence_tqdm_safely() -> None:
    """Force tqdm disabled via env + subclass hook; never assign a lambda to tqdm.tqdm."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        import tqdm
        import tqdm.std as tqdm_std

        # Restore real class if a previous broken patch left a function/lambda
        real = getattr(tqdm_std, "tqdm", None)
        if real is not None and isinstance(real, type):
            if not isinstance(getattr(tqdm, "tqdm", None), type):
                tqdm.tqdm = real  # type: ignore[attr-defined]
            try:
                import tqdm.auto as tqdm_auto

                if not isinstance(getattr(tqdm_auto, "tqdm", None), type):
                    tqdm_auto.tqdm = real  # type: ignore[attr-defined]
            except Exception:
                pass

            # Prefer a disabled subclass over replacing with a non-class callable
            if not getattr(real, "_gt_harness_disabled_subclass", False):

                class _QuietTqdm(real):  # type: ignore[misc,valid-type]
                    _gt_harness_disabled_subclass = True

                    def __init__(self, *args, **kwargs):
                        kwargs["disable"] = True
                        super().__init__(*args, **kwargs)

                _QuietTqdm._gt_harness_disabled_subclass = True  # type: ignore[attr-defined]
                tqdm.tqdm = _QuietTqdm  # type: ignore[attr-defined]
                try:
                    import tqdm.auto as tqdm_auto

                    tqdm_auto.tqdm = _QuietTqdm  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    import tqdm.std as tqdm_std2

                    tqdm_std2.tqdm = _QuietTqdm  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass


def mode_label(mode: str) -> str:
    """Normalize UI radio values to internal mode keys."""
    m = (mode or "").strip().lower()
    # New primary labels
    if m in {"alerts", "alert"} or m.startswith("alert"):
        return "alerts"
    if "trip" in m or m in {"trips/event", "trips_event", "trip/event"}:
        return "trips_event"
    # Legacy labels (older cases / CLI)
    if "event" in m or "investigat" in m:
        return "trips_event"
    if "routine" in m:
        return "alerts"
    return "alerts"


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string (microseconds for stable ordering)."""
    return datetime.now(timezone.utc).isoformat()


def new_case_id() -> str:
    """Generate a short unique case id (safe for filenames)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"case_{stamp}_{uuid.uuid4().hex[:8]}"


def safe_filename(name: str, max_len: int = 80) -> str:
    """Sanitize a string so it is safe as a Windows/Linux filename."""
    cleaned = re.sub(r"[^\w.\-]+", "_", name.strip())
    cleaned = cleaned.strip("._") or "file"
    return cleaned[:max_len]


def read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Read a whole text file; returns empty string if missing."""
    try:
        return path.read_text(encoding=encoding)
    except FileNotFoundError:
        return ""
    except OSError:
        return ""


def write_json(path: Path, data: Any, indent: int = 2) -> Path:
    """Write JSON with UTF-8 encoding; create parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path) -> Optional[Any]:
    """Load JSON from disk or return None if unavailable/invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def list_knowledge_files(extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List documents under knowledge/ for RAG indexing.

    Parameters
    ----------
    extensions : list of str, optional
        File suffixes to include (default: .md .txt .json).
    """
    if extensions is None:
        extensions = [".md", ".txt", ".json"]
    if not KNOWLEDGE_DIR.exists():
        return []
    files: List[Path] = []
    for ext in extensions:
        files.extend(sorted(KNOWLEDGE_DIR.rglob(f"*{ext}")))
    return files


def list_saved_case_files() -> List[Path]:
    """List JSON case files under saved_cases/."""
    if not SAVED_CASES_DIR.exists():
        return []
    return sorted(SAVED_CASES_DIR.glob("*.json"))


def find_gguf_model(explicit: Optional[str] = None) -> Optional[Path]:
    """
    Locate a GGUF model file for llama.cpp.

    Search order
    ------------
    1. ``explicit`` path or env ``GT_GGUF_PATH``
    2. ``models/DEFAULT_GGUF_NAME``
    3. First ``*.gguf`` under models/
    """
    candidates: List[Path] = []
    env_path = explicit or os.environ.get("GT_GGUF_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(MODELS_DIR / DEFAULT_GGUF_NAME)
    if MODELS_DIR.exists():
        candidates.extend(sorted(MODELS_DIR.glob("*.gguf")))

    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def format_kv_block(data: Dict[str, Any], title: str = "") -> str:
    """Pretty-print a dict as a markdown-friendly key/value block."""
    lines: List[str] = []
    if title:
        lines.append(f"### {title}")
    for key, value in data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 4000) -> str:
    """Truncate long text with an ellipsis marker."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n… [truncated] …"
