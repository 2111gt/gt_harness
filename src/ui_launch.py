"""
Interchangeable TUI backends: Textual (Python) vs ratatui (Rust).

Selection order
---------------
1. Explicit ``--ui textual|ratatui``
2. Environment ``GT_UI=textual|ratatui``
3. Default: ``textual``

Ratatui launch looks for a prebuilt binary under ``tui_ratatui/target/...``
or runs ``cargo run --release`` when Cargo is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .utils import PROJECT_ROOT, setup_logging

logger = setup_logging()

VALID_UIS = ("textual", "ratatui")


def resolve_ui(explicit: Optional[str] = None) -> str:
    """Return normalized UI backend name."""
    raw = (explicit or os.environ.get("GT_UI") or "textual").strip().lower()
    if raw in ("text", "python", "tui"):
        raw = "textual"
    if raw in ("rust", "rat"):
        raw = "ratatui"
    if raw not in VALID_UIS:
        raise ValueError(
            f"Unknown UI backend {raw!r}. Choose one of: {', '.join(VALID_UIS)}"
        )
    return raw


def ratatui_binary_candidates() -> List[Path]:
    """Possible locations for the compiled ratatui frontend."""
    root = PROJECT_ROOT / "tui_ratatui"
    names = ["gt_harness_ratatui.exe", "gt_harness_ratatui"]
    dirs = [
        root / "target" / "release",
        root / "target" / "debug",
        PROJECT_ROOT / "bin",
    ]
    out: List[Path] = []
    for d in dirs:
        for n in names:
            out.append(d / n)
    return out


def find_ratatui_binary() -> Optional[Path]:
    for p in ratatui_binary_candidates():
        if p.is_file():
            return p
    # PATH
    which = shutil.which("gt_harness_ratatui")
    if which:
        return Path(which)
    return None


def find_cargo() -> Optional[str]:
    return shutil.which("cargo")


def launch_textual(*, auto_download: bool = True) -> int:
    from .tui_app import run_tui

    logger.info("UI backend: textual")
    return int(run_tui(auto_download=auto_download) or 0)


def launch_ratatui(
    *,
    auto_download: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    """
    Start the Rust ratatui frontend.

    The binary shells back into this project via ``python app.py --json-once``.
    """
    logger.info("UI backend: ratatui")
    env = os.environ.copy()
    env["GT_HARNESS_ROOT"] = str(PROJECT_ROOT)
    env["GT_HARNESS_PYTHON"] = sys.executable
    demo = PROJECT_ROOT / "samples" / "gt_sensors_demo.csv"
    if demo.is_file():
        env["GT_DEFAULT_CSV"] = str(demo.resolve())
    if not auto_download:
        env["GT_NO_DOWNLOAD"] = "1"
    # Always start ratatui with project root as cwd so relative paths work
    cwd = str(PROJECT_ROOT)

    bin_path = find_ratatui_binary()
    if bin_path is not None:
        cmd = [str(bin_path)]
        if extra_args:
            cmd.extend(extra_args)
        logger.info("Starting ratatui binary: %s (cwd=%s)", bin_path, cwd)
        return int(subprocess.call(cmd, cwd=cwd, env=env))

    cargo = find_cargo()
    crate = PROJECT_ROOT / "tui_ratatui"
    if cargo and (crate / "Cargo.toml").is_file():
        cmd = [cargo, "run", "--release", "--manifest-path", str(crate / "Cargo.toml")]
        if extra_args:
            cmd.append("--")
            cmd.extend(extra_args)
        logger.info("Building/running ratatui via cargo (first run may take a few minutes)…")
        # cargo run uses crate dir; env still carries GT_HARNESS_ROOT + DEFAULT_CSV
        return int(subprocess.call(cmd, cwd=str(crate), env=env))

    msg = (
        "Ratatui UI is not available yet.\n\n"
        "Ratatui is a Rust frontend. To enable it:\n"
        "  1. Install Rust: https://rustup.rs  (or: winget install Rustlang.Rustup)\n"
        "  2. From the project root:\n"
        "       cd tui_ratatui\n"
        "       cargo build --release\n"
        "  3. Launch:\n"
        "       python app.py --ui ratatui\n"
        "       # or:  set GT_UI=ratatui && python app.py\n\n"
        "Fallback:  python app.py --ui textual\n"
    )
    print(msg, file=sys.stderr)
    return 3


def launch_ui(
    ui: Optional[str] = None,
    *,
    auto_download: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    backend = resolve_ui(ui)
    if backend == "ratatui":
        return launch_ratatui(auto_download=auto_download, extra_args=extra_args)
    return launch_textual(auto_download=auto_download)
