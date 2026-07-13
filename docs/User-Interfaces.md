# User Interfaces

Same diagnosis engine; two frontends.

## Textual TUI (default)

```powershell
python app.py
python app.py --ui textual
```

| Key | Action |
|-----|--------|
| **Ctrl+R** | Run diagnosis |
| **Ctrl+S** | Save & Learn |
| **Ctrl+O** | Browse CSV |
| **Ctrl+N** | New session |
| **Q** | Quit |

Features: command strip, status chips, sample picker, drop-zone path paste, live pipeline log, braille/ASCII proof plots in the report pane, progress + ETA.

## Desktop GUI (CustomTkinter)

```powershell
python app.py --ui gui
# or
$env:GT_UI = "gui"
python app.py
```

Requires: `pip install customtkinter`

| Area | Contents |
|------|----------|
| **Setup** | CSV path, Browse, sample dropdown, mode, context, corrections |
| **Run / Save / New** | Primary actions |
| **Report tab** | Full markdown-style write-up |
| **Proof plots tab** | High-quality **matplotlib PNG** charts |
| **Live log tab** | Streamed pipeline steps |
| **Footer** | Progress bar + status |

## Selection order

1. `--ui textual|gui`
2. Env `GT_UI=textual|gui`
3. Default: **textual**

Legacy `GT_UI=ratatui` is mapped to **gui** (ratatui frontend was removed).

## Automation (no UI)

```powershell
python app.py --cli-once samples/gt_sensors_demo.csv --mode Alerts
python app.py --json-once samples/gt_sensors_demo.csv --mode Alerts
```

JSON bridge: see `src/bridge.py` (`schema_version`, severity, evidence plots paths, report text).
