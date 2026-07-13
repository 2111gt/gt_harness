"""
UI backends: Textual TUI (default) and modern desktop GUI.

Selection order
---------------
1. Explicit ``--ui textual|gui``
2. Environment ``GT_UI=textual|gui``
3. Default: ``textual``
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from .utils import setup_logging

logger = setup_logging()

VALID_UIS = ("textual", "gui")


def resolve_ui(explicit: Optional[str] = None) -> str:
    """Return normalized UI backend name."""
    raw = (explicit or os.environ.get("GT_UI") or "textual").strip().lower()
    if raw in ("text", "python", "tui", "terminal"):
        raw = "textual"
    if raw in ("desktop", "window", "ctk", "tk", "customtkinter", "gui-app"):
        raw = "gui"
    # Legacy aliases (ratatui removed) → modern GUI
    if raw in ("rust", "rat", "ratatui"):
        logger.warning("UI backend %r was removed; using gui instead", raw)
        raw = "gui"
    if raw not in VALID_UIS:
        raise ValueError(
            f"Unknown UI backend {raw!r}. Choose one of: {', '.join(VALID_UIS)}"
        )
    return raw


def launch_textual(*, auto_download: bool = True) -> int:
    from .tui_app import run_tui

    logger.info("UI backend: textual")
    return int(run_tui(auto_download=auto_download) or 0)


def launch_gui(*, auto_download: bool = True) -> int:
    from .gui_app import run_gui

    logger.info("UI backend: gui (CustomTkinter)")
    return int(run_gui(auto_download=auto_download) or 0)


def launch_ui(
    ui: Optional[str] = None,
    *,
    auto_download: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> int:
    del extra_args  # reserved for future CLI passthrough
    backend = resolve_ui(ui)
    if backend == "gui":
        return launch_gui(auto_download=auto_download)
    return launch_textual(auto_download=auto_download)
