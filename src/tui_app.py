"""
GT Diagnostic Harness — Textual TUI (terminal user interface).

Replaces Gradio. Same diagnosis pipeline: CSV → anomaly → RAG → LLM → Save & Learn.
Shows a stepped progress bar with ETA while work runs.

Threading notes
---------------
Workers must not block on ``call_from_thread`` for every progress tick (that
can stall under heavy UI layout). Progress is delivered with ``post_message``,
which is thread-safe and non-blocking.

Subprocesses (llama-cli, pip) must use CREATE_NO_WINDOW on Windows — see
``utils.run_hidden_subprocess`` — otherwise they hang when Textual owns the
console (CLI mode works; TUI mode freezes).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Blur, Focus, Paste
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Rule,
    Select,
    Static,
    TextArea,
)
from rich.console import Group, RenderableType
from rich.text import Text

from .analysis import diagnosis_step_plan, get_rag, run_diagnosis, score_severity
from .models import get_bundle, load_models
from .tools import cases_history_markdown, save_case
from .utils import (
    PROJECT_ROOT,
    SAMPLES_DIR,
    configure_tui_logging,
    ensure_directories,
    setup_logging,
)

logger = setup_logging()

DEFAULT_CSV = str((SAMPLES_DIR / "gt_sensors_demo.csv").resolve())


def list_sample_csv_options() -> List[Tuple[str, str]]:
    """
    Build (label, absolute path) pairs for the sample scenario picker.

    Includes root demo CSV plus scenario folders under samples/.
    """
    opts: List[Tuple[str, str]] = []
    if not SAMPLES_DIR.is_dir():
        return opts
    demo = SAMPLES_DIR / "gt_sensors_demo.csv"
    if demo.is_file():
        opts.append(("Demo · gt_sensors_demo.csv", str(demo.resolve())))
    for sub in sorted(SAMPLES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for csv in sorted(sub.glob("*.csv")):
            opts.append((f"{sub.name} · {csv.name}", str(csv.resolve())))
    return opts


def normalize_csv_path(raw: str) -> str:
    """
    Clean paths pasted or dropped into a terminal.

    Windows Terminal / Explorer often insert quoted paths or ``file:///`` URIs
    when you drag a file onto the window (drop = paste path text, not true GUI DnD).
    """
    text = (raw or "").strip()
    if not text:
        return ""
    # Multi-line paste: take first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    # Strip matching quotes (Windows drag-drop classic)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    # file:// URI
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        path = unquote(parsed.path or "")
        # Windows: file:///C:/Users/... → /C:/Users/... → C:/Users/...
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        text = path
    # Normalize separators for display on Windows; resolve relative to project root
    try:
        p = Path(text)
        if p.is_file():
            text = str(p.resolve())
        else:
            from .utils import PROJECT_ROOT, SAMPLES_DIR

            for cand in (
                PROJECT_ROOT / text,
                SAMPLES_DIR / Path(text).name,
            ):
                if cand.is_file():
                    text = str(cand.resolve())
                    break
            else:
                # scenario subfolders: samples/<scenario>/<file>
                name = Path(text).name
                if SAMPLES_DIR.is_dir() and name:
                    for sub in SAMPLES_DIR.iterdir():
                        if sub.is_dir():
                            hit = sub / name
                            if hit.is_file():
                                text = str(hit.resolve())
                                break
    except Exception:
        pass
    return text.strip()


def _extract_dropped_paths(text: str) -> List[str]:
    """
    Parse one or more file paths from terminal drag-drop / paste text.

    Windows Terminal inserts quoted paths when you drop a file onto the window
    while a focusable widget is active.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    # Prefer Windows-aware token split (quoted paths with spaces)
    tokens: List[str] = []
    if os.name == "nt":
        tokens = re.findall(r'(?:[^\s"]|"(?:\\.|[^"])*")+', raw)
    else:
        import shlex

        try:
            tokens = shlex.split(raw)
        except ValueError:
            tokens = raw.split()
    if not tokens:
        tokens = [raw]
    paths: List[str] = []
    for tok in tokens:
        p = normalize_csv_path(tok)
        if not p:
            continue
        try:
            path = Path(p)
            if path.is_file():
                paths.append(str(path.resolve()))
            elif p.lower().endswith(".csv"):
                # Allow not-yet-visible / network path; still show it
                paths.append(p)
        except Exception:
            continue
    # Fallback: whole string as one path
    if not paths:
        one = normalize_csv_path(raw)
        if one:
            paths.append(one)
    return paths


class CsvDropZone(Widget, can_focus=True, can_focus_children=False):
    """
    Visual drop target: click to focus, then drag a CSV onto the *terminal window*.

    Terminals cannot do true OS DnD into a pixel region; they paste the path into
    the focused widget. This zone accepts that paste and shows an icon + filename
    *in the drop box* while also filling the path field.
    """

    DEFAULT_CSS = """
    CsvDropZone {
        height: 7;
        min-height: 6;
        border: tall $accent;
        background: $panel;
        content-align: center middle;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    CsvDropZone:focus {
        border: tall $success;
        background: $boost;
    }
    """

    path: reactive[str] = reactive("")
    label: reactive[str] = reactive("Drop CSV here")

    class PathDropped(Message):
        """Posted when a file path is accepted by the drop zone."""

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def render(self) -> RenderableType:
        if self.path:
            name = Path(self.path).name
            parent = str(Path(self.path).parent)
            # Simple “icon” that works without Nerd Fonts
            icon = "📄" if name.lower().endswith(".csv") else "📎"
            return Group(
                Text(f"{icon}  {name}", style="bold cyan", justify="center"),
                Text(parent, style="dim", justify="center"),
                Text("✓ path set — ready to Run diagnosis", style="green", justify="center"),
            )
        return Group(
            Text("📥  DROP CSV HERE", style="bold", justify="center"),
            Text("1) Click this box   2) Drag file onto the terminal window", style="dim", justify="center"),
            Text("Path appears here  ·  also fills the field below  ·  or Browse…", style="dim", justify="center"),
        )

    def on_focus(self, event: Focus) -> None:
        self.border_title = "DROP TARGET (focused — drop CSV now)"

    def on_blur(self, event: Blur) -> None:
        self.border_title = "DROP TARGET"

    def on_mount(self) -> None:
        self.border_title = "DROP TARGET"
        # Seed display from default path if present
        if DEFAULT_CSV and Path(DEFAULT_CSV).is_file():
            self.path = DEFAULT_CSV

    def on_paste(self, event: Paste) -> None:
        paths = _extract_dropped_paths(event.text or "")
        if not paths:
            return
        event.prevent_default()
        event.stop()
        chosen = paths[0]
        # Prefer a .csv if multiple dropped
        for p in paths:
            if p.lower().endswith(".csv"):
                chosen = p
                break
        self.path = chosen
        self.post_message(self.PathDropped(chosen))

    def set_path(self, path: str) -> None:
        """Update the visual path (e.g. after Browse)."""
        self.path = normalize_csv_path(path) or ""


# Startup phases (model load) — nominal seconds for ETA display
LOAD_STEPS: Tuple[Tuple[str, str, float], ...] = (
    ("packages", "Packages / downloads", 8.0),
    ("llm", "Bind Granite GGUF (llama-cli)", 5.0),
    ("tspulse", "Load TS Pulse anomaly model", 20.0),
    ("embeddings", "Load embeddings (RAG)", 12.0),
    ("rag", "Warm knowledge index", 3.0),
)


class PipelineProgress(Message):
    """Non-blocking progress event from a worker thread."""

    def __init__(
        self,
        step_id: str,
        step_i: int,
        total: int,
        frac: float,
        message: str,
    ) -> None:
        self.step_id = step_id
        self.step_i = step_i
        self.total = total
        self.frac = frac
        self.message = message
        super().__init__()


class ModelsReady(Message):
    def __init__(self, bundle: Any, err: Optional[str]) -> None:
        self.bundle = bundle
        self.err = err
        super().__init__()


class DiagnosisDone(Message):
    def __init__(self, result: Any, err: Optional[str]) -> None:
        self.result = result
        self.err = err
        super().__init__()


class LiveOutput(Message):
    """Stream a partial diagnosis chunk into the live feed."""

    def __init__(self, section: str, text: str) -> None:
        self.section = section
        self.text = text
        super().__init__()


class StatusBar(Static):
    """Single-line status under the progress bar."""


class StepList(Static):
    """Checklist of pipeline steps with ✓ / → / · markers."""

    def set_steps(
        self,
        steps: Sequence[Tuple[str, str, float]],
        active: int = -1,
        *,
        done_all: bool = False,
    ) -> None:
        lines: List[str] = []
        for i, (_sid, label, _nom) in enumerate(steps):
            if done_all or (active >= 0 and i < active):
                mark = "✓"
            elif i == active:
                mark = "→"
            else:
                mark = "·"
            lines.append(f" {mark}  {i + 1}. {label}")
        self.update("\n".join(lines) if lines else "")


class GTDiagnosticTUI(App[None]):
    """Full-screen modern terminal workspace for GT Diagnostic Harness."""

    TITLE = "GT Diagnostic Harness"
    SUB_TITLE = "Local · offline-capable gas turbine diagnostics"
    # Industrial dark workspace: command strip + setup / results cockpit
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }
    /* ── Top command chrome ───────────────────────────────── */
    #command-strip {
        height: auto;
        min-height: 3;
        dock: top;
        layout: horizontal;
        background: $boost;
        border-bottom: tall $accent;
        padding: 0 1;
        align: left middle;
    }
    #brand {
        width: auto;
        min-width: 18;
        color: $accent;
        text-style: bold;
        padding: 0 1 0 0;
        content-align: left middle;
    }
    #status-chips {
        width: 1fr;
        height: auto;
        min-height: 1;
        color: $text-muted;
        content-align: left middle;
        padding: 0 1;
    }
    #command-actions {
        width: auto;
        height: auto;
        layout: horizontal;
        align: right middle;
    }
    #command-actions Button {
        margin: 0 0 0 1;
        min-width: 12;
    }
    /* ── Main split ───────────────────────────────────────── */
    #main {
        height: 1fr;
        layout: horizontal;
        padding: 0 0;
    }
    #left {
        width: 38%;
        min-width: 34;
        max-width: 56;
        border: round $accent 60%;
        background: $surface;
        padding: 0 1 1 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    #right {
        width: 1fr;
        border: round $primary 50%;
        background: $panel;
        padding: 0 1 0 1;
        layout: vertical;
    }
    .pane-heading {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $boost;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    .section-title {
        text-style: bold;
        color: $secondary;
        margin-top: 1;
        margin-bottom: 0;
    }
    .section-hint {
        color: $text-muted;
        margin-bottom: 0;
        height: auto;
    }
    #csv-row {
        height: auto;
        margin: 0 0 1 0;
    }
    #csv-path {
        width: 1fr;
        border: tall $surface;
    }
    #btn-browse {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    #sample-select {
        width: 100%;
        margin-bottom: 1;
    }
    #btn-row {
        height: auto;
        margin: 1 0;
        layout: horizontal;
    }
    #btn-row Button {
        margin-right: 1;
    }
    Button {
        margin-right: 0;
    }
    TextArea {
        height: 6;
        border: tall $surface;
    }
    #context {
        height: 7;
    }
    #corrections {
        height: 5;
    }
    #history {
        height: 10;
        border: tall $surface;
        background: $panel;
        padding: 0 1;
    }
    /* ── Results / proof plots ────────────────────────────── */
    #results-heading {
        height: 1;
        text-style: bold;
        color: $primary;
        background: $boost;
        padding: 0 1;
        margin: 0 0 0 0;
    }
    #report-scroll {
        height: 1fr;
        min-height: 6;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    #evidence-title {
        height: auto;
        color: $warning;
        text-style: bold;
        margin: 1 0 0 0;
        padding: 0 1;
        background: $boost;
    }
    #evidence-plots {
        height: auto;
        max-height: 48;
        min-height: 8;
        padding: 0 1 1 1;
        background: $background;
        border: tall $warning;
        margin: 0 0 1 0;
        overflow-y: auto;
        overflow-x: auto;
        scrollbar-gutter: stable;
        color: $text;
        text-style: none;
    }
    #report-label {
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0 1;
        background: $boost;
        margin-top: 0;
    }
    #report {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }
    #live-panel {
        height: 12;
        min-height: 8;
        max-height: 16;
        border-top: heavy $accent;
        background: $surface;
        layout: vertical;
        padding: 0;
    }
    #live-title {
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0 1;
        background: $boost;
    }
    #live-log {
        height: 1fr;
        min-height: 6;
        padding: 0 1;
        scrollbar-gutter: stable;
    }
    /* ── Progress dock ────────────────────────────────────── */
    #progress-panel {
        height: auto;
        max-height: 9;
        border-top: tall $primary;
        padding: 0 1 0 1;
        background: $boost;
    }
    #progress-title {
        text-style: bold;
        color: $accent;
        margin-top: 0;
        height: 1;
    }
    #step-list {
        height: auto;
        max-height: 3;
        color: $text-muted;
        margin: 0;
        overflow-y: auto;
    }
    #progress-bar {
        width: 100%;
        height: 1;
        margin: 0;
        color: $success;
    }
    #status {
        height: auto;
        max-height: 2;
        color: $text-muted;
    }
    /* Severity accent states on chips */
    #status-chips.sev-critical {
        color: $error;
        text-style: bold;
    }
    #status-chips.sev-high {
        color: $warning;
        text-style: bold;
    }
    #status-chips.sev-elevated {
        color: $warning;
    }
    #status-chips.sev-normal {
        color: $success;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+r", "run", "Run diagnosis"),
        Binding("ctrl+s", "save", "Save & Learn"),
        Binding("ctrl+h", "refresh_history", "History"),
        Binding("ctrl+o", "browse_csv", "Browse CSV"),
        Binding("ctrl+n", "new_session", "New session"),
    ]

    def __init__(self, *, auto_download: bool = True) -> None:
        super().__init__()
        self.auto_download = auto_download
        self._bundle = None
        self._last_result = None
        self._busy = False
        self._models_ready = False
        self._plan_steps: List[Tuple[str, str, float]] = list(LOAD_STEPS)
        self._live_chunks: List[str] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # Always-visible modern command chrome
        with Horizontal(id="command-strip"):
            yield Static("GT HARNESS", id="brand")
            yield Static(
                "MODE · Alerts   ·  SEV · —   ·  ENGINE · —   ·  MODELS · loading…",
                id="status-chips",
            )
            with Horizontal(id="command-actions"):
                yield Button("Run diagnosis", variant="primary", id="btn-run")
                yield Button("Save & Learn", variant="success", id="btn-save")
                yield Button("New session", id="btn-new", variant="warning")
        with Horizontal(id="main"):
            with VerticalScroll(id="left"):
                yield Static("SETUP  ·  input & learn", classes="pane-heading")
                yield Label("Sensor CSV", classes="section-title")
                yield Label(
                    "Drop onto box · Browse / Ctrl+O · paste path · or pick a sample",
                    classes="section-hint",
                )
                yield CsvDropZone(id="csv-drop")
                with Horizontal(id="csv-row"):
                    yield Input(
                        value=DEFAULT_CSV,
                        placeholder=r"Full path after drop / Browse…",
                        id="csv-path",
                    )
                    yield Button("Browse…", id="btn-browse", variant="default")
                yield Label("Sample scenarios", classes="section-title")
                sample_opts = list_sample_csv_options()
                yield Select(
                    sample_opts if sample_opts else [("No samples found", "")],
                    prompt="Load a sample CSV…",
                    id="sample-select",
                    allow_blank=True,
                )
                yield Label("Diagnostic mode", classes="section-title")
                with RadioSet(id="mode"):
                    yield RadioButton("Alerts", value=True, id="mode-alerts")
                    yield RadioButton("Trips/Event", id="mode-trips")
                yield Label("Context / process maps / SOE", classes="section-title")
                yield TextArea(
                    "Alarm text, trip first-outs, SOE notes, process-map excerpts…",
                    id="context",
                )
                with Horizontal(id="btn-row"):
                    yield Button("Refresh history", id="btn-hist")
                    yield Button("Quit", variant="error", id="btn-quit")
                yield Rule()
                yield Label("User corrections (Save & Learn)", classes="section-title")
                yield TextArea("", id="corrections")
                yield Label("Session history", classes="section-title")
                yield Markdown("_Loading history…_", id="history")
            with Vertical(id="right"):
                yield Static(
                    "RESULTS  ·  Final → Proof plots → Reasoning → Evidence",
                    id="results-heading",
                    classes="pane-heading",
                )
                with VerticalScroll(id="report-scroll"):
                    yield Label(
                        "Proof plots — channels the detector blames (▲ = flagged)",
                        id="evidence-title",
                    )
                    yield Static(
                        "Awaiting diagnosis…\n\n"
                        "Top anomaly channels will plot here as visual proof\n"
                        "alongside the written report (▲ marks flagged samples).",
                        id="evidence-plots",
                    )
                    yield Label("Write-up", id="report-label")
                    yield Markdown(
                        "_Modern workspace ready._\n\n"
                        "1. Pick a **sample** or drop a CSV  \n"
                        "2. Choose **Alerts** or **Trips/Event**  \n"
                        "3. **Run diagnosis** (Ctrl+R) from the command bar  \n\n"
                        "After a run you get:\n"
                        "- **Proof plots** of issue channels (panel above)\n"
                        "- **Final report** + reasoning trail\n"
                        "- **Live stream** of each pipeline step below\n",
                        id="report",
                    )
                with Vertical(id="live-panel"):
                    yield Label(
                        "LIVE PIPELINE  ·  streams as each step finishes",
                        id="live-title",
                    )
                    yield RichLog(
                        id="live-log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                        auto_scroll=True,
                        max_lines=2000,
                    )
        with Vertical(id="progress-panel"):
            yield Label("Progress", id="progress-title")
            yield StepList(id="step-list")
            yield ProgressBar(total=100, show_eta=False, id="progress-bar")
            yield StatusBar("Starting…", id="status")
        yield Footer()

    def on_mount(self) -> None:
        ensure_directories()
        self._set_status_chips(models="loading…")
        self._set_plan(list(LOAD_STEPS), title="Startup · loading models")
        self._update_progress(
            step_i=0,
            total=len(LOAD_STEPS),
            frac=0.0,
            message="Loading models (first run can take a few minutes)…  [logs → logs/gt_harness.log]",
        )
        try:
            log = self.query_one("#live-log", RichLog)
            log.clear()
            log.write("[bold]Live workspace ready.[/] Pipeline steps stream here during Run diagnosis.")
            log.write("Startup: loading models…")
        except Exception as exc:  # noqa: BLE001
            logger.debug("live-log init: %s", exc)
        # Separate group so diagnosis can still queue cleanly after load
        self.load_models_worker()

    def _set_status_chips(
        self,
        *,
        mode: Optional[str] = None,
        severity: Optional[str] = None,
        level: Optional[str] = None,
        engine: Optional[str] = None,
        models: Optional[str] = None,
    ) -> None:
        """Update the modern status chip strip under the brand."""
        try:
            chips = self.query_one("#status-chips", Static)
        except Exception:
            return
        mode_s = mode if mode is not None else self._selected_mode()
        sev_s = severity if severity is not None else "—"
        eng_s = engine if engine is not None else "—"
        mod_s = models if models is not None else ("ready" if self._models_ready else "…")
        text = (
            f"MODE · {mode_s}   ·  SEV · {sev_s}   ·  "
            f"ENGINE · {eng_s}   ·  MODELS · {mod_s}"
        )
        chips.update(text)
        # Severity color class
        for cls in ("sev-critical", "sev-high", "sev-elevated", "sev-normal", "sev-mild"):
            chips.remove_class(cls)
        lvl = (level or "").lower()
        if lvl in {"critical", "high", "elevated", "normal", "mild"}:
            chips.add_class(f"sev-{lvl}" if lvl != "mild" else "sev-elevated")

    # ── Progress UI helpers ──────────────────────────────────────────────

    def _set_plan(
        self,
        steps: Sequence[Tuple[str, str, float]],
        *,
        title: str = "Progress",
    ) -> None:
        self._plan_steps = list(steps)
        self.query_one("#progress-title", Label).update(title)
        self.query_one("#step-list", StepList).set_steps(self._plan_steps, active=0)
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=100, progress=0)

    def _update_progress(
        self,
        *,
        step_i: int,
        total: int,
        frac: float,
        message: str,
        done: bool = False,
    ) -> None:
        steps = self._plan_steps or list(LOAD_STEPS)
        active = min(max(step_i, 0), max(len(steps) - 1, 0))
        self.query_one("#step-list", StepList).set_steps(
            steps, active=active if not done else len(steps), done_all=done
        )
        pct = 100.0 if done else max(0.0, min(100.0, float(frac) * 100.0))
        bar = self.query_one("#progress-bar", ProgressBar)
        try:
            bar.update(progress=pct)
        except Exception:
            try:
                bar.progress = pct  # type: ignore[attr-defined]
            except Exception:
                pass
        self.query_one("#status", StatusBar).update(message)

    def _emit_progress(
        self,
        step_id: str,
        step_i: int,
        total: int,
        frac: float,
        message: str,
    ) -> None:
        """Thread-safe, non-blocking progress (from workers)."""
        self.post_message(
            PipelineProgress(step_id, step_i, total, frac, message)
        )

    def on_pipeline_progress(self, event: PipelineProgress) -> None:
        done = event.step_id == "done" or event.frac >= 1.0
        self._update_progress(
            step_i=event.step_i if not done else event.total,
            total=event.total,
            frac=1.0 if done else event.frac,
            message=event.message,
            done=done,
        )
        # One live line per progress tick (compact — full sections come via LiveOutput)
        try:
            log = self.query_one("#live-log", RichLog)
            log.write(f"[dim]▸ {event.message}[/dim]")
        except Exception:
            pass

    def on_live_output(self, event: LiveOutput) -> None:
        """Append streamed diagnosis sections to the bottom live pane."""
        text = (event.text or "").strip()
        if not text:
            return
        self._live_chunks.append(text)
        self._append_live(event.section, text)

    def _append_live(self, section: str, text: str) -> None:
        """Write to #live-log; never raise into the worker path."""
        try:
            log = self.query_one("#live-log", RichLog)
        except Exception as exc:  # noqa: BLE001
            logger.warning("live-log widget missing: %s", exc)
            return
        try:
            header = f"[bold yellow]── {section} ──[/]" if section else ""
            if header:
                log.write(header)
            # Cap very long chunks so the UI stays responsive mid-run
            lines = text.splitlines()
            limit = 120 if section in {"draft", "reflection", "final"} else 60
            for line in lines[:limit]:
                # Escape accidental markup closers from model text
                safe = line.replace("[/", "\\[/")
                log.write(safe if safe else " ")
            if len(lines) > limit:
                log.write(f"[dim]… ({len(lines) - limit} more lines — full text in report when done)[/]")
            log.write("")
        except Exception as exc:  # noqa: BLE001
            logger.warning("live-log write failed: %s", exc)

    def on_models_ready(self, event: ModelsReady) -> None:
        # Do not name helpers _on_* — Textual treats those as message handlers
        # and would call them as _on_models_ready(event) only.
        self._apply_models_ready(event.bundle, event.err)

    def on_diagnosis_done(self, event: DiagnosisDone) -> None:
        self._apply_diagnosis_done(event.result, event.err)

    # ── CSV path: paste / browse (terminals rarely support true OS drag-drop) ─

    def _set_csv_path(self, raw: str, *, notify: bool = True) -> bool:
        path = normalize_csv_path(raw)
        if not path:
            return False
        self.query_one("#csv-path", Input).value = path
        # Mirror into the drop-zone visual (icon + name where you dropped)
        try:
            zone = self.query_one("#csv-drop", CsvDropZone)
            zone.set_path(path)
        except Exception:
            pass
        exists = Path(path).is_file()
        if notify:
            if exists:
                self.notify(f"CSV path set: {Path(path).name}", severity="information")
            else:
                self.notify(f"Path set (file not found yet): {path}", severity="warning")
        return exists

    def on_csv_drop_zone_path_dropped(self, event: CsvDropZone.PathDropped) -> None:
        """Path accepted by the drop-zone widget (path already shown in the box)."""
        self._set_csv_path(event.path, notify=True)
        try:
            self.query_one("#csv-path", Input).focus()
        except Exception:
            pass

    def on_paste(self, event: Paste) -> None:
        """
        App-wide paste fallback. Prefer the focused drop zone's own handler;
        if paste lands elsewhere, still capture CSV paths.
        """
        # If drop zone is focused, let CsvDropZone.on_paste handle it
        try:
            zone = self.query_one("#csv-drop", CsvDropZone)
            if zone.has_focus:
                return
        except Exception:
            pass
        paths = _extract_dropped_paths(event.text or "")
        if not paths:
            cleaned = normalize_csv_path(event.text or "")
            paths = [cleaned] if cleaned else []
        if not paths:
            return
        chosen = paths[0]
        for p in paths:
            if p.lower().endswith(".csv"):
                chosen = p
                break
        looks_csv = chosen.lower().endswith(".csv")
        looks_path = ":" in chosen or chosen.startswith("\\\\") or chosen.startswith("/")
        if looks_csv or (looks_path and Path(chosen).suffix):
            event.prevent_default()
            event.stop()
            self._set_csv_path(chosen)
            try:
                self.query_one("#csv-drop", CsvDropZone).focus()
            except Exception:
                pass

    @on(Input.Changed, "#csv-path")
    def on_csv_path_changed(self, event: Input.Changed) -> None:
        # Auto-strip quotes after paste into the field itself
        raw = event.value or ""
        cleaned = normalize_csv_path(raw)
        if cleaned and cleaned != raw:
            # Avoid feedback loop: only rewrite when quotes/URI present
            if raw.strip() != cleaned:
                event.input.value = cleaned

    @on(Button.Pressed, "#btn-browse")
    def on_browse_pressed(self) -> None:
        self.action_browse_csv()

    def action_browse_csv(self) -> None:
        """Open a native file dialog (reliable alternative to drag-and-drop)."""
        self.browse_csv_worker()

    @work(thread=True, exclusive=True, group="dialog")
    def browse_csv_worker(self) -> None:
        path = ""
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            initial = SAMPLES_DIR if SAMPLES_DIR.is_dir() else PROJECT_ROOT
            start_dir = str(initial)
            path = filedialog.askopenfilename(
                title="Select gas turbine sensor CSV",
                initialdir=start_dir,
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Browse dialog failed")
            self.call_from_thread(
                self.notify, f"Browse failed: {exc}", severity="error"
            )
            return
        if path:
            self.call_from_thread(self._set_csv_path, path)

    # ── Model load ───────────────────────────────────────────────────────

    @work(thread=True, exclusive=True, group="pipeline")
    def load_models_worker(self) -> None:
        import time

        t0 = time.monotonic()
        total = len(LOAD_STEPS)
        nominal = sum(s[2] for s in LOAD_STEPS)

        def report(step_i: int, detail: str = "", *, within: float = 0.1) -> None:
            sid, label, nom = LOAD_STEPS[step_i]
            done_nom = sum(s[2] for s in LOAD_STEPS[:step_i]) + within * nom
            frac = min(0.99, done_nom / max(nominal, 1e-6))
            elapsed = time.monotonic() - t0
            remaining = sum(s[2] for s in LOAD_STEPS[step_i + 1 :]) + (1.0 - within) * nom
            if frac > 0.1 and elapsed > 2.0:
                eta = max(0.0, (elapsed / frac) * (1.0 - frac))
            else:
                eta = remaining
            msg = f"{label}" + (f" — {detail}" if detail else "")
            msg += f" · ~{int(eta)}s left · step {step_i + 1}/{total} · elapsed {int(elapsed)}s"
            self._emit_progress(sid, step_i, total, frac, msg)

        try:
            report(0, "checking packages")
            bundle = load_models(
                auto_download=self.auto_download,
                progress=self._emit_progress,
            )
            # Do NOT warm embeddings/TS Pulse here — that was the multi-minute hang.
            # RAG + TS Pulse load on first diagnosis instead (keyword RAG works until then).
            report(4, "ready (heavy models deferred)")
            elapsed = time.monotonic() - t0
            self._emit_progress(
                "done",
                total,
                total,
                1.0,
                f"Ready in {elapsed:.0f}s · LLM bound · TS Pulse/embeddings load on first run",
            )
            self.post_message(ModelsReady(bundle, None))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model load failed")
            self.post_message(ModelsReady(None, str(exc)))

    def _apply_models_ready(self, bundle: Any, err: Optional[str]) -> None:
        if err:
            self._models_ready = False
            self.query_one("#status", StatusBar).update(f"Model load error: {err}")
            self._set_status_chips(models="error")
            self.notify(err, severity="error")
            return
        self._bundle = bundle
        self._models_ready = True
        st = bundle.status if bundle else {}
        llm = str(st.get("llm", ""))[:70]
        emb = str(st.get("embeddings", ""))[:40]
        ts = str(st.get("tspulse", ""))[:40]
        clf = str(st.get("tspulse_clf", ""))[:40]
        llm_short = "gguf" if "ok" in llm.lower() or "llama" in llm.lower() else (llm[:24] or "n/a")
        self.query_one("#status", StatusBar).update(
            f"Ready · LLM: {llm} · TS: {ts} · Clf: {clf} · Emb: {emb}"
        )
        self.query_one("#progress-title", Label).update("Ready · idle")
        self._set_status_chips(models=llm_short, severity="—", engine="—")
        self.action_refresh_history()

    def _selected_mode(self) -> str:
        try:
            rs = self.query_one("#mode", RadioSet)
            pressed = rs.pressed_button
            if pressed and "trip" in (pressed.id or ""):
                return "Trips/Event"
        except Exception:
            pass
        return "Alerts"

    @on(Select.Changed, "#sample-select")
    def on_sample_select_changed(self, event: Select.Changed) -> None:
        """Load a shipped sample CSV from the modern scenario picker."""
        val = event.value
        # Select.BLANK / NULL are falsey sentinels for “no selection”
        if val is None or val is Select.BLANK or val is Select.NULL or val == "":
            return
        path = str(val)
        if path and Path(path).is_file():
            self._set_csv_path(path, notify=True)
            self._set_status_chips(mode=self._selected_mode())

    @on(RadioSet.Changed, "#mode")
    def on_mode_changed(self, event: RadioSet.Changed) -> None:
        self._set_status_chips(mode=self._selected_mode())

    @on(Button.Pressed, "#btn-run")
    def on_run_pressed(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#btn-save")
    def on_save_pressed(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#btn-new")
    def on_new_session_pressed(self) -> None:
        self.action_new_session()

    @on(Button.Pressed, "#btn-hist")
    def on_hist_pressed(self) -> None:
        self.action_refresh_history()

    @on(Button.Pressed, "#btn-quit")
    def on_quit_pressed(self) -> None:
        self.exit()

    def action_new_session(self) -> None:
        """Clear report pane + live output for a fresh diagnosis session."""
        if self._busy:
            self.notify("Cannot start a new session while diagnosis is running", severity="warning")
            return
        self._last_result = None
        self._live_chunks = []
        idle_report = (
            "_New session — report cleared._\n\n"
            "Keys: **Ctrl+R** run · **Ctrl+N** new session · **Ctrl+S** save · **Q** quit\n\n"
            "Set CSV path (drop / Browse), choose mode, then **Run diagnosis**.\n"
            "Live output streams below; the full write-up appears here when complete."
        )
        try:
            self.query_one("#report", Markdown).update(idle_report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("clear report failed: %s", exc)
        try:
            self.query_one("#evidence-plots", Static).update(
                "_New session — proof plots cleared. Run a diagnosis to regenerate._"
            )
            self.query_one("#evidence-title", Label).update(
                "Proof plots — channels the detector blames (▲ = flagged)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("clear evidence plots: %s", exc)
        try:
            log = self.query_one("#live-log", RichLog)
            log.clear()
            log.write("[bold]New session[/] — live output cleared.")
            log.write("Ready for next Run diagnosis.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("clear live-log failed: %s", exc)
        try:
            self.query_one("#progress-title", Label).update("Ready · idle")
            self.query_one("#status", StatusBar).update(
                "New session — reports cleared · models still loaded"
            )
            self._plan_steps = []
            self.query_one("#step-list", StepList).set_steps([], active=-1, done_all=False)
            self.query_one("#progress-bar", ProgressBar).update(total=100, progress=0)
            self._set_status_chips(
                mode=self._selected_mode(),
                severity="—",
                level="normal",
                engine="—",
                models="ready" if self._models_ready else "…",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("clear progress UI: %s", exc)
        self.notify("New session — report and live output cleared", severity="information")

    def action_run(self) -> None:
        if self._busy:
            self.notify("Already running…", severity="warning")
            return
        if not self._models_ready and self._bundle is None:
            self.notify("Models still loading — wait for Ready status", severity="warning")
            return
        csv_path = normalize_csv_path(self.query_one("#csv-path", Input).value)
        if csv_path:
            self.query_one("#csv-path", Input).value = csv_path
        if not csv_path:
            self.notify("Enter a CSV path (or Browse… / Ctrl+O)", severity="error")
            return
        if not Path(csv_path).is_file():
            self.notify(f"CSV not found: {csv_path}", severity="error")
            return
        ctx = self.query_one("#context", TextArea).text
        mode = self._selected_mode()
        import os

        # Match analysis default: reflection on when LLM bound unless GT_FULL_REFLECTION=0
        env = (os.environ.get("GT_FULL_REFLECTION") or "").strip().lower()
        if env in {"0", "false", "no", "off"}:
            full = False
        elif env in {"1", "true", "yes", "on"}:
            full = True
        else:
            from .models import llm_available

            full = llm_available(self._bundle) if self._bundle is not None else True
        plan = diagnosis_step_plan(full_reflection=full)
        self._set_plan(plan, title=f"Diagnosis · {mode}")
        self._update_progress(
            step_i=0,
            total=len(plan),
            frac=0.0,
            message=f"Starting diagnosis ({mode})…",
        )
        self.query_one("#report", Markdown).update(
            "_Diagnosing…_\n\n"
            "- Progress bar + step checklist below\n"
            "- **Live output** at the bottom streams each step as it finishes\n"
            "- Full write-up (Final + Reasoning + Self-review + Evidence) appears here when complete\n\n"
            "The LLM step (Granite GGUF on CPU) is usually the longest."
        )
        self._live_chunks = []
        try:
            log = self.query_one("#live-log", RichLog)
            log.clear()
            log.write(f"[bold cyan]Diagnosis started[/] mode={mode}")
            log.write(f"CSV: {csv_path}")
            log.write("[dim]Waiting for first step… (CSV → anomaly → RAG → LLM)[/]")
            log.write("")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clear live-log: %s", exc)
            self.notify(f"Live log unavailable: {exc}", severity="warning")
        self._busy = True
        self.run_diagnosis_worker(csv_path, mode, ctx)

    @work(thread=True, exclusive=True, group="pipeline")
    def run_diagnosis_worker(self, csv_path: str, mode: str, context: str) -> None:
        def live_cb(section: str, text: str) -> None:
            self.post_message(LiveOutput(section, text))

        try:
            bundle = self._bundle or get_bundle(auto_download=self.auto_download)
            logger.info("TUI diagnosis start mode=%s csv=%s", mode, csv_path)
            result = run_diagnosis(
                csv_file=csv_path,
                mode=mode,
                context=context or "",
                bundle=bundle,
                rag=get_rag(),
                progress=self._emit_progress,
                live=live_cb,
            )
            logger.info("TUI diagnosis finished mode=%s", result.mode)
            self.post_message(DiagnosisDone(result, None))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Diagnosis failed")
            self.post_message(DiagnosisDone(None, str(exc)))

    def _apply_diagnosis_done(self, result: Any, err: Optional[str]) -> None:
        self._busy = False
        if err:
            self.query_one("#status", StatusBar).update(f"Error: {err}")
            self.query_one("#report", Markdown).update(f"**Diagnosis failed**\n\n`{err}`")
            try:
                self.query_one("#evidence-plots", Static).update(
                    f"_Diagnosis failed — no proof plots._\n`{err}`"
                )
            except Exception:
                pass
            self.notify(err, severity="error")
            return
        self._last_result = result
        sev = score_severity(result.anomaly)
        # Proof plots panel (monospace ASCII — better than Markdown for charts)
        try:
            plots = ""
            if getattr(result, "evidence", None) is not None:
                plots = (result.evidence.ascii_art or "").strip()
            if not plots:
                plots = (
                    "No proof plots for this run "
                    "(no scored channels or non-numeric data)."
                )
            self.query_one("#evidence-plots", Static).update(plots)
            ch_n = 0
            if getattr(result, "evidence", None) and result.evidence.channels:
                ch_n = len(result.evidence.channels)
                names = ", ".join(c.name for c in result.evidence.channels[:4])
                self.query_one("#evidence-title", Label).update(
                    f"Proof plots — {ch_n} channel(s): {names}  (▲ = flagged)"
                )
            else:
                self.query_one("#evidence-title", Label).update(
                    "Proof plots — no channels selected"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("evidence plot UI update failed: %s", exc)
        # Operator-first markdown: Final → plots (also embedded) → reasoning → evidence
        try:
            body = result.to_display_markdown()
        except Exception:
            body = result.to_markdown()
        header = (
            f"**Severity: {sev['label']}** · "
            f"level=`{sev['level']}` · score=`{sev['severity']}` · "
            f"top=`{sev['top_channel']}` · engine=`{result.anomaly.get('mode')}`\n\n"
            f"_Scroll: Final report → Proof plots (panel above + section 2) → "
            f"Reasoning → Evidence._\n\n"
            "---\n\n"
        )
        body = header + body
        # Soft cap to keep Textual Markdown responsive
        if len(body) > 60000:
            body = body[:60000] + "\n\n… _(report truncated for display; use Save & Learn for full JSON)_"
        self.query_one("#report", Markdown).update(body)
        try:
            log = self.query_one("#live-log", RichLog)
            log.write("[bold green]✓ Full report + proof plots written above — scroll the report pane[/]")
            log.write(
                f"Severity {sev['severity']} ({sev['level']}) · engine={result.anomaly.get('mode')}"
            )
            if getattr(result, "evidence", None) and result.evidence.channels:
                log.write(
                    "[cyan]Proof channels:[/] "
                    + ", ".join(
                        f"{c.name}({c.score:.2f})" for c in result.evidence.channels
                    )
                )
        except Exception:
            pass
        # Scroll report pane to top so Final report + plots are visible
        try:
            self.query_one("#report-scroll", VerticalScroll).scroll_home(animate=False)
        except Exception:
            pass
        # Also write a plain-text copy so the full report is easy to re-read outside the TUI
        report_path = None
        try:
            from .utils import PROJECT_ROOT, ensure_directories

            ensure_directories()
            out_dir = PROJECT_ROOT / "logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "last_diagnosis_report.md"
            # Prefer full report with plots; also drop a .txt plots sidecar
            report_path.write_text(body, encoding="utf-8")
            plots_path = out_dir / "last_evidence_plots.txt"
            plots_txt = ""
            if getattr(result, "evidence", None) is not None:
                plots_txt = (result.evidence.ascii_art or "").strip()
            if plots_txt:
                plots_path.write_text(plots_txt + "\n", encoding="utf-8")
            logger.info("Wrote full report to %s", report_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not write report file: %s", exc)
        self.query_one("#progress-title", Label).update(
            "Diagnosis complete — scroll report + proof plots"
        )
        path_hint = f" · saved {report_path.name}" if report_path else ""
        self.query_one("#status", StatusBar).update(
            f"Done · severity={sev['severity']} ({sev['level']}) · "
            f"proof plots + Final + Reasoning in right pane{path_hint}"
        )
        self._set_status_chips(
            mode=result.mode if getattr(result, "mode", None) else self._selected_mode(),
            severity=f"{sev['level']} ({sev['severity']})",
            level=str(sev.get("level") or ""),
            engine=str(result.anomaly.get("mode") or "n/a"),
            models="ready",
        )
        self.notify(
            f"Diagnosis complete — {sev['label']}. Proof plots + write-up in the "
            f"report pane (also logs/last_diagnosis_report.md).",
            severity="information",
        )

    def action_save(self) -> None:
        if self._last_result is None:
            self.notify("Run a diagnosis first", severity="warning")
            return
        result = self._last_result
        corr = self.query_one("#corrections", TextArea).text
        sev = score_severity(result.anomaly)
        clf = (result.anomaly or {}).get("classification") or {}
        try:
            case = save_case(
                mode=result.mode,
                context=result.context_used,
                anomaly_summary=str(result.anomaly.get("summary", "")),
                analysis=result.draft,
                reflection=result.reflection,
                final_report=result.final_report,
                user_corrections=corr or "",
                severity=sev,
                metadata={
                    "severity": sev,
                    "anomaly_mode": result.anomaly.get("mode"),
                    "classification": clf,
                    "signature_label": clf.get("top_label"),
                    "signature_trained": clf.get("trained"),
                },
                rag=get_rag(),
                reindex=True,
            )
            self.notify(f"Saved {case['case_id']}", severity="information")
            self.query_one("#status", StatusBar).update(
                f"Saved {case['case_id']} · score={sev['severity']}"
            )
            self.action_refresh_history()
        except Exception as exc:  # noqa: BLE001
            self.notify(str(exc), severity="error")

    def action_refresh_history(self) -> None:
        try:
            md = cases_history_markdown(limit=20)
        except Exception as exc:  # noqa: BLE001
            md = f"_History error: {exc}_"
        self.query_one("#history", Markdown).update(md)


def run_tui(*, auto_download: bool = True) -> int:
    """Entry used by app.py."""
    ensure_directories()
    log_path = configure_tui_logging()
    logger.info("TUI starting; console logs redirected to %s", log_path)
    app = GTDiagnosticTUI(auto_download=auto_download)
    app.run()
    return 0
