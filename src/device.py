"""
Device / CUDA preference helpers.

Policy
------
Prefer CUDA when it is available and the user has not forced CPU.
Override with env:

- ``GT_FORCE_CPU=1`` — never use GPU
- ``GT_N_GPU_LAYERS=N`` — llama.cpp offload layers (0 = CPU, 99 ≈ all)
- ``GT_TORCH_DEVICE=cuda|cpu|auto`` — torch device for TS Pulse / embeddings
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Optional

from .utils import setup_logging

logger = setup_logging()


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def force_cpu() -> bool:
    """User requested CPU-only."""
    return _env_truthy("GT_FORCE_CPU") or _env_truthy("GT_NO_GPU")


@lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    """True if PyTorch can see a CUDA device."""
    if force_cpu():
        return False
    try:
        import torch

        ok = bool(torch.cuda.is_available())
        if ok:
            try:
                name = torch.cuda.get_device_name(0)
                logger.info("CUDA available (torch): %s", name)
            except Exception:
                logger.info("CUDA available (torch)")
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch CUDA probe failed: %s", exc)
        return False


@lru_cache(maxsize=1)
def nvidia_smi_available() -> bool:
    """True if nvidia-smi runs (driver present). Does not guarantee CUDA toolkit."""
    if force_cpu():
        return False
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        proc = subprocess.run(
            [smi, "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        return proc.returncode == 0 and bool((proc.stdout or "").strip())
    except Exception:
        return False


def cuda_likely_available() -> bool:
    """Prefer torch CUDA; fall back to nvidia-smi as a soft signal."""
    if force_cpu():
        return False
    if torch_cuda_available():
        return True
    return nvidia_smi_available()


def preferred_n_gpu_layers() -> int:
    """
    Layers to offload to GPU for llama.cpp / llama-cpp-python.

    - ``GT_N_GPU_LAYERS`` wins if set (even under GT_FORCE_CPU if you set 0)
    - else ``0`` when ``GT_FORCE_CPU``
    - else ``99`` when CUDA is likely available
    - else ``0`` (CPU)
    """
    raw = (os.environ.get("GT_N_GPU_LAYERS") or "").strip()
    if raw != "":
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("Invalid GT_N_GPU_LAYERS=%r — using auto", raw)
    if force_cpu():
        return 0
    if cuda_likely_available():
        return 99
    return 0


def torch_device() -> str:
    """
    Device string for torch / SentenceTransformer.

    ``GT_TORCH_DEVICE=cuda|cpu|auto`` (default auto).
    """
    if force_cpu():
        return "cpu"
    raw = (os.environ.get("GT_TORCH_DEVICE") or "auto").strip().lower()
    if raw in {"cpu", "cuda"}:
        if raw == "cuda" and not torch_cuda_available():
            logger.warning("GT_TORCH_DEVICE=cuda but CUDA not available — using cpu")
            return "cpu"
        return raw
    return "cuda" if torch_cuda_available() else "cpu"


def move_module_to_device(model: Any, device: Optional[str] = None) -> Any:
    """Best-effort ``model.to(device)``; returns model (possibly unchanged)."""
    if model is None:
        return model
    dev = device or torch_device()
    if dev == "cpu":
        return model
    try:
        if hasattr(model, "to"):
            model = model.to(dev)
            logger.info("Moved model to %s", dev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not move model to %s: %s", dev, exc)
    return model


def device_status_summary() -> str:
    """One-line status for logs / UI."""
    if force_cpu():
        return "cpu (GT_FORCE_CPU)"
    parts = []
    if torch_cuda_available():
        try:
            import torch

            parts.append(f"torch.cuda={torch.cuda.get_device_name(0)}")
        except Exception:
            parts.append("torch.cuda=yes")
    else:
        parts.append("torch.cuda=no")
    if nvidia_smi_available():
        parts.append("nvidia-smi=yes")
    layers = preferred_n_gpu_layers()
    parts.append(f"n_gpu_layers={layers}")
    parts.append(f"torch_device={torch_device()}")
    return "; ".join(parts)
