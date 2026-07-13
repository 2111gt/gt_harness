# Proof Plots

After anomaly detection, the harness ranks sensor channels and builds **visual proof** of the issue.

## What is plotted

- Top channels by anomaly `column_scores`
- Auto-zoom to the flagged event window when point anomalies exist
- Flag markers on anomalous samples (▲)

## Titles and axes

Plots use **tag + common language name**, for example:

```text
EGT_spread_C  —  Exhaust gas temperature spread  ·  score=21.500
```

| Axis | Label style |
|------|-------------|
| **X** | `X: Sample index  (rows start–end)` |
| **Y** | `Y: TAG  (Common name)  [unit]` |

TC1–TC27 map to exhaust thermocouples; vib/EGT/load/etc. have fixed common names in `src/evidence_plots.py`.

## Outputs

| Surface | Format |
|---------|--------|
| Desktop GUI → **Proof plots** tab | Matplotlib **PNG** (dark industrial theme) |
| Textual TUI / markdown report | Braille / ASCII charts |
| Disk | `logs/evidence_plots/proof_combined.png`, `proof_01_*.png` |
| ASCII sidecar | `logs/last_evidence_plots.txt` |

Dependencies for PNG: `matplotlib`, `Pillow` (in `requirements.txt`).

## Code

- Selection + ASCII: `src/evidence_plots.py`
- Wired after anomaly step: `src/analysis.py` → `build_evidence_bundle`
- GUI display: `src/gui_app.py`
