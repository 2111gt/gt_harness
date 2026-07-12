"""
Evidence / proof plots for GT Diagnostic Harness.

Turns the channels the anomaly engine ranks highest into high-resolution
terminal charts with markers on flagged samples — visual proof next to the
LLM write-up.

Rendering
---------
Default is a **braille canvas** (2×4 sub-pixels per cell) for smooth, readable
lines without extra dependencies. Optional plotext polish when
``GT_USE_PLOTEXT=1``. Series auto-zoom around the flagged event window so
spikes stay sharp on long historian CSVs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import setup_logging

logger = setup_logging()

# High-visibility flag glyph (single cell, overlaid on braille)
_FLAG = "▲"

# Braille: 2 columns × 4 rows of dots per character cell
# Dot bit layout (Unicode braille):
#  1 4
#  2 5
#  3 6
#  7 8
_BRAILLE_DOTS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)
_BRAILLE_BASE = 0x2800

# Defaults tuned for modern wide terminals / TUI results pane
_DEFAULT_WIDTH = 96
_DEFAULT_HEIGHT = 16
_DEFAULT_MAX_POINTS = 240
_DEFAULT_MAX_CHANNELS = 4


@dataclass
class ChannelEvidence:
    """One sensor channel selected as visual proof of the issue."""

    name: str
    score: float
    values: List[float]
    flag_rows: List[int]
    method: str = ""
    reason: str = ""
    # Original series indices represented by values[0] .. values[-1]
    window_start: int = 0
    window_end: int = 0
    full_n: int = 0

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class EvidenceBundle:
    """Plot-ready evidence extracted from a diagnosis run."""

    channels: List[ChannelEvidence] = field(default_factory=list)
    ascii_art: str = ""
    title: str = "Proof plots — channels implicated by the anomaly engine"
    note: str = ""

    def is_empty(self) -> bool:
        return not self.channels


def select_issue_channels(
    anomaly: Dict[str, Any],
    *,
    max_channels: int = _DEFAULT_MAX_CHANNELS,
) -> List[Tuple[str, float, List[int], str]]:
    """
    Pick channels that best support the diagnosis.

    Ranking uses column_scores; flag row indices come from point anomalies.
    Returns list of (name, score, flag_rows, method_hint).
    """
    scores = dict(anomaly.get("column_scores") or {})
    points = list(anomaly.get("anomalies") or [])

    flag_map: Dict[str, List[int]] = {}
    method_map: Dict[str, str] = {}
    for a in points:
        col = str(a.get("column") or "")
        if not col:
            continue
        try:
            row = int(a.get("row"))
        except (TypeError, ValueError):
            continue
        flag_map.setdefault(col, []).append(row)
        if col not in method_map and a.get("method"):
            method_map[col] = str(a.get("method"))

    for col, rows in flag_map.items():
        if col not in scores:
            col_pts = [a for a in points if str(a.get("column")) == col]
            try:
                scores[col] = max(float(a.get("score") or 0) for a in col_pts) if col_pts else 1.0
            except (TypeError, ValueError):
                scores[col] = 1.0

    if not scores:
        return []

    ordered = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    out: List[Tuple[str, float, List[int], str]] = []
    for name, sc in ordered[: max(1, max_channels)]:
        rows = sorted(set(flag_map.get(name, [])))
        out.append((name, float(sc), rows, method_map.get(name, "")))
    return out


def build_evidence_bundle(
    df: Optional[pd.DataFrame],
    anomaly: Dict[str, Any],
    *,
    max_channels: int = _DEFAULT_MAX_CHANNELS,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> EvidenceBundle:
    """
    Extract top issue channels from ``df`` + anomaly result and render plots.
    """
    if df is None or getattr(df, "empty", True):
        return EvidenceBundle(
            note="No sensor frame available for proof plots.",
            ascii_art="_No sensor data to plot._",
        )

    picks = select_issue_channels(anomaly, max_channels=max_channels)
    if not picks:
        return EvidenceBundle(
            note="No channels scored high enough to plot.",
            ascii_art="_No high-scoring channels to plot as evidence._",
        )

    channels: List[ChannelEvidence] = []
    for name, score, flag_rows, method in picks:
        if name not in df.columns:
            match = next((c for c in df.columns if str(c).lower() == name.lower()), None)
            if match is None:
                continue
            name = str(match)
        series = pd.to_numeric(df[name], errors="coerce").astype(float)
        values = series.to_numpy(dtype=float)
        if len(values) == 0 or np.all(np.isnan(values)):
            continue

        # Zoom onto the event window so spikes stay high-resolution
        win_vals, win_flags, w0, w1 = _extract_event_window(values, flag_rows)
        vals_ds, flags_ds = _downsample_with_flags(
            win_vals, win_flags, max_points=max_points
        )
        reason = _reason_for_channel(name, score, flag_rows, method, anomaly)
        channels.append(
            ChannelEvidence(
                name=name,
                score=score,
                values=vals_ds.tolist(),
                flag_rows=flags_ds,
                method=method or str(anomaly.get("mode") or ""),
                reason=reason,
                window_start=w0,
                window_end=w1,
                full_n=len(values),
            )
        )

    if not channels:
        return EvidenceBundle(
            note="Selected channels were missing or non-numeric.",
            ascii_art="_Could not build proof plots from selected channels._",
        )

    use_plotext = (os.environ.get("GT_USE_PLOTEXT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    art = ""
    if use_plotext:
        art = _render_plotext(channels, width=width, height=height)
    if not art:
        art = _render_hires(channels, width=width, height=height)

    engine = str(anomaly.get("mode") or "n/a")
    title = "Proof plots — channels implicated by the anomaly engine"
    header = (
        f"{title}\n"
        f"Engine: {engine} · ranked by anomaly score · "
        f"{_FLAG} = flagged sample · braille hi-res · auto-zoom on event\n"
    )
    full = header + "\n" + art
    note = (
        f"Showing {len(channels)} channel(s) the detector ranked highest as "
        "high-resolution visual proof (zoomed to anomaly window when flags exist)."
    )
    return EvidenceBundle(
        channels=channels,
        ascii_art=full.strip() + "\n",
        title=title,
        note=note,
    )


def evidence_to_markdown(bundle: EvidenceBundle) -> str:
    """Markdown section embedding fenced monospaced plots."""
    if bundle.is_empty() and not (bundle.ascii_art or "").strip():
        return (
            "## Proof plots (sensor evidence)\n\n"
            "_No channels selected for plotting._\n"
        )
    body = bundle.ascii_art or "_n/a_"
    return (
        f"## Proof plots (sensor evidence)\n\n"
        f"{bundle.note}\n\n"
        f"```text\n{body.rstrip()}\n```\n"
    )


def _reason_for_channel(
    name: str,
    score: float,
    flag_rows: Sequence[int],
    method: str,
    anomaly: Dict[str, Any],
) -> str:
    bits = [f"score={score:.3f}"]
    if flag_rows:
        bits.append(f"{len(flag_rows)} flagged")
        bits.append(f"@row {flag_rows[0]}")
        if len(flag_rows) > 1:
            bits.append(f"→{flag_rows[-1]}")
    if method:
        bits.append(method)
    clf = anomaly.get("classification") or {}
    if clf.get("enabled") and clf.get("top_label"):
        bits.append(f"sig={clf.get('top_label')}")
    return " · ".join(bits)


def _extract_event_window(
    values: np.ndarray,
    flag_rows: Sequence[int],
    *,
    pad_frac: float = 0.20,
    min_frac: float = 0.25,
) -> Tuple[np.ndarray, List[int], int, int]:
    """
    Zoom the series onto the flagged event so the anomaly is large and readable.

    Returns (window_values, flags_relative_to_window, start, end_exclusive).
    """
    n = len(values)
    flags = sorted({int(r) for r in flag_rows if 0 <= int(r) < n})
    if n <= 48 or not flags:
        return values.astype(float), flags, 0, n

    f0, f1 = flags[0], flags[-1]
    span = max(f1 - f0 + 1, 1)
    pad = max(int(span * pad_frac), int(n * 0.05), 8)
    # Never zoom so tight that context disappears
    min_len = max(int(n * min_frac), 32)
    half = max(pad, (min_len - span) // 2)
    w0 = max(0, f0 - half)
    w1 = min(n, f1 + half + 1)
    if w1 - w0 < min_len:
        # Expand symmetrically
        need = min_len - (w1 - w0)
        w0 = max(0, w0 - need // 2)
        w1 = min(n, w1 + (need - need // 2))
    # If still the whole series, just return it
    win = values[w0:w1].astype(float)
    rel = [f - w0 for f in flags if w0 <= f < w1]
    return win, rel, w0, w1


def _downsample_with_flags(
    values: np.ndarray,
    flag_rows: Sequence[int],
    *,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> Tuple[np.ndarray, List[int]]:
    """
    Downsample series for the canvas.

    Uses min/max envelope per bucket (preserves spike peaks) instead of mean.
    """
    n = len(values)
    flags = sorted({int(r) for r in flag_rows if 0 <= int(r) < n})
    if n <= max_points:
        return values.astype(float), flags

    # Even number of buckets: each pair stores min then max for envelope look
    n_buckets = max_points // 2
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    out_chunks: List[float] = []
    new_flags: List[int] = []
    flag_set = set(flags)
    for i in range(n_buckets):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = min(n, a + 1)
        chunk = values[a:b]
        finite = chunk[np.isfinite(chunk)]
        if len(finite) == 0:
            out_chunks.extend([np.nan, np.nan])
        else:
            # Preserve spike polarity: order min/max by first occurrence of extremum
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            imin = int(np.nanargmin(chunk)) if np.any(np.isfinite(chunk)) else 0
            imax = int(np.nanargmax(chunk)) if np.any(np.isfinite(chunk)) else 0
            if imin <= imax:
                out_chunks.extend([vmin, vmax])
            else:
                out_chunks.extend([vmax, vmin])
        base = len(out_chunks) - 2
        if any(a <= f < b for f in flag_set):
            # Mark the higher of the two envelope points as flag carrier
            new_flags.append(base + 1)
    return np.asarray(out_chunks, dtype=float), new_flags


def _render_hires(
    channels: Sequence[ChannelEvidence],
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> str:
    blocks: List[str] = []
    for i, ch in enumerate(channels):
        blocks.append(_plot_one_braille(ch, width=width, height=height))
        if i < len(channels) - 1:
            blocks.append("")  # spacer between charts
    return "\n".join(blocks).rstrip() + "\n"


def _fmt_y(v: float) -> str:
    """Compact, readable Y-axis label."""
    av = abs(v)
    if av >= 1000 or (av > 0 and av < 0.01):
        return f"{v:9.3g}"
    if av >= 100:
        return f"{v:9.1f}"
    if av >= 10:
        return f"{v:9.2f}"
    return f"{v:9.3f}"


def _plot_one_braille(
    ch: ChannelEvidence,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> str:
    """
    High-definition line chart using Unicode braille (2×4 dots per cell).

    Effective resolution ≈ (plot_w*2) × (plot_h*4) sub-pixels.
    """
    vals = np.asarray(ch.values, dtype=float)
    n = len(vals)
    if n == 0:
        return f"  {ch.name}: (empty)"

    label_w = 10  # y labels
    # Leave room for left labels + right margin
    plot_w = max(40, min(width - label_w - 2, 100))
    # Character rows for braille canvas (each row = 4 sub-rows)
    plot_h = max(8, min(height, 24))

    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return f"  {ch.name}: (non-finite)"
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0
        ymin = ymin - 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    # Sub-pixel canvas
    sub_w = plot_w * 2
    sub_h = plot_h * 4
    canvas = np.zeros((sub_h, sub_w), dtype=bool)
    flag_cols: set = set()

    # Map series → dense x
    if n == 1:
        xs_v = np.array([vals[0]] * sub_w, dtype=float)
        x_idx = np.arange(sub_w)
    else:
        x_idx = np.linspace(0, n - 1, sub_w)
        safe = np.nan_to_num(vals, nan=float(np.nanmean(finite)))
        xs_v = np.interp(x_idx, np.arange(n), safe)

    # Flag columns in sub-pixel x
    for fr in ch.flag_rows:
        if n <= 1:
            col = 0
        else:
            col = int(round(fr / max(n - 1, 1) * (sub_w - 1)))
        flag_cols.add(max(0, min(sub_w - 1, col)))

    # Draw polyline in sub-pixels
    prev: Optional[Tuple[int, int]] = None
    for sx, v in enumerate(xs_v):
        if not np.isfinite(v):
            continue
        rel = (v - ymin) / (ymax - ymin)
        sy = int(round((1.0 - rel) * (sub_h - 1)))
        sy = max(0, min(sub_h - 1, sy))
        if prev is not None:
            _bresenham(canvas, prev[0], prev[1], sx, sy)
        else:
            canvas[sy, sx] = True
        prev = (sx, sy)

    # Emphasize flag x-columns with a vertical hairline (subtle)
    for fcx in flag_cols:
        for sy in range(sub_h):
            if sy % 2 == 0:  # dashed vertical
                # only lighten empty cells so line stays dominant
                pass
        # Mark peak at flag
        if 0 <= fcx < sub_w:
            # find top-most set pixel near this column
            col_hits = np.where(canvas[:, fcx])[0]
            if len(col_hits):
                canvas[int(col_hits[0]), fcx] = True

    # Encode braille
    cells = [[" " for _ in range(plot_w)] for _ in range(plot_h)]
    for cy in range(plot_h):
        for cx in range(plot_w):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    sy = cy * 4 + dy
                    sx = cx * 2 + dx
                    if sy < sub_h and sx < sub_w and canvas[sy, sx]:
                        bits |= _BRAILLE_DOTS[dy][dx]
            if bits:
                cells[cy][cx] = chr(_BRAILLE_BASE + bits)

    # Overlay flag markers at the peak cell of each flag column
    for fcx in flag_cols:
        cell_x = fcx // 2
        if not (0 <= cell_x < plot_w):
            continue
        # Peak row in this sub-column
        col_hits = np.where(canvas[:, fcx])[0]
        if len(col_hits):
            cell_y = int(col_hits[0]) // 4
        else:
            # place at interpolated value height
            v = xs_v[fcx] if fcx < len(xs_v) else ymax
            rel = (v - ymin) / (ymax - ymin)
            cell_y = int(round((1.0 - rel) * (plot_h - 1)))
        cell_y = max(0, min(plot_h - 1, cell_y))
        cells[cell_y][cell_x] = _FLAG

    # ── Assemble text frame ──────────────────────────────────────────
    lines: List[str] = []
    title = f"┏ {ch.name}"
    meta = f"  {ch.reason}"
    if ch.full_n and (ch.window_end - ch.window_start) < ch.full_n:
        meta += f"  ·  window rows {ch.window_start}–{ch.window_end - 1} of {ch.full_n}"
    header = title + meta
    # Don't truncate mid-name; wrap meta if needed
    if len(header) > max(width, 80):
        lines.append(title)
        lines.append("┃ " + meta.strip()[: max(width - 2, 40)])
    else:
        lines.append(header)
    lines.append("┣" + "━" * min(max(width - 1, plot_w + label_w), 100))

    # Y ticks: top, 1/4, mid, 3/4, bottom
    tick_rows = {0, plot_h // 4, plot_h // 2, (3 * plot_h) // 4, plot_h - 1}

    for r in range(plot_h):
        y_val = ymax - (r / max(plot_h - 1, 1)) * (ymax - ymin)
        if r in tick_rows:
            label = _fmt_y(y_val)
            axis = "┤"
        else:
            label = " " * 9
            axis = "│"
        row_chars = "".join(cells[r])
        lines.append(f"{label}{axis}{row_chars}")

    # X axis with sample-index ticks
    axis_line = " " * label_w + "└" + "─" * plot_w
    lines.append(axis_line)

    # X tick labels under axis (start / mid / end of window)
    x0 = ch.window_start
    x1 = max(ch.window_end - 1, ch.window_start)
    mid = (x0 + x1) // 2
    tick_line = [" "] * (label_w + 1 + plot_w)
    for label, frac in ((str(x0), 0.0), (str(mid), 0.5), (str(x1), 1.0)):
        pos = label_w + 1 + int(frac * (plot_w - 1))
        for i, ch_c in enumerate(label):
            p = pos + i - len(label) // 2
            if 0 <= p < len(tick_line):
                tick_line[p] = ch_c
    lines.append("".join(tick_line).rstrip())

    n_flags = len(ch.flag_rows)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    foot = (
        f"{'':9}  sample index →  ·  {n} pts drawn  ·  "
        f"{_FLAG} = {n_flags} flag(s)  ·  "
        f"min={vmin:.4g}  max={vmax:.4g}  range={vmax - vmin:.4g}"
    )
    lines.append(foot)
    lines.append("┗" + "━" * min(max(width - 1, 60), 100))
    return "\n".join(lines)


def _bresenham(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a continuous line on a boolean canvas (inclusive endpoints)."""
    h, w = canvas.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= y < h and 0 <= x < w:
            canvas[y, x] = True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _render_plotext(
    channels: Sequence[ChannelEvidence],
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> str:
    """Optional richer charts via plotext; empty string if unavailable."""
    try:
        import plotext as plt  # type: ignore
    except Exception:
        return ""

    try:
        parts: List[str] = []
        for ch in channels:
            plt.clear_figure()
            plt.plotsize(max(width, 80), max(height, 14) + 4)
            y = [v if np.isfinite(v) else None for v in ch.values]
            x = list(range(len(y)))
            plt.plot(x, y, marker="braille")
            if ch.flag_rows:
                fx = [i for i in ch.flag_rows if 0 <= i < len(y)]
                fy = [ch.values[i] for i in fx]
                if fx:
                    plt.scatter(fx, fy, marker="fhd")
            plt.title(f"{ch.name}  ({ch.reason})")
            plt.xlabel("sample (window)")
            plt.theme("clear")
            parts.append(plt.build())
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"
    except Exception as exc:  # noqa: BLE001
        logger.debug("plotext render failed, using braille: %s", exc)
        return ""
