# GT Harness — ratatui frontend (Rust)

Interchangeable TUI for the same Python diagnosis engine used by the Textual UI.

```text
┌ command chips ─────────────────────────────────────────────┐
│ SETUP (CSV, mode, context)  │  RESULTS (plots | report)    │
└ status / keys ─────────────────────────────────────────────┘
```

## Swap between UIs

From project root:

```powershell
# Textual (Python) — default
python app.py
python app.py --ui textual

# ratatui (Rust)
python app.py --ui ratatui
$env:GT_UI = "ratatui"; python app.py
```

## Build

Requires [Rust](https://rustup.rs) (`winget install Rustlang.Rustup`).

**Windows without Visual Studio:** use the GNU toolchain + MinGW (WinLibs), then:

```powershell
rustup default stable-x86_64-pc-windows-gnu
cd tui_ratatui
cargo build --release
# binary: target\release\gt_harness_ratatui.exe
```

With MSVC Build Tools installed, the default `stable-x86_64-pc-windows-msvc` toolchain is fine.

`python app.py --ui ratatui` uses the release binary if present, otherwise `cargo run --release`.

## How it talks to Python

On **Run** (F5 / Ctrl+R) the frontend executes:

```text
python app.py --json-once <csv> --mode Alerts|Trips/Event --context "…"
```

Stdout is one JSON document (`src/bridge.py`). Proof plots and the full write-up are shown in the Results panes.

Env vars (set automatically by the Python launcher):

| Variable | Purpose |
|----------|---------|
| `GT_HARNESS_ROOT` | Project root (where `app.py` lives) |
| `GT_HARNESS_PYTHON` | Python executable |
| `GT_NO_DOWNLOAD=1` | Offline engine |

## Keys (navigation mode — default)

| Key | Action |
|-----|--------|
| **↑↓** / **j k** | Move sample list |
| **Enter** / **Space** | Load highlighted sample into CSV path |
| **r** / **F5** | Run diagnosis |
| **m** / **F6** | Toggle Alerts ↔ Trips/Event |
| **e** | Edit CSV path (type, Enter confirm, Esc cancel) |
| **c** | Edit context notes |
| **1** / **2** | Results: plots / report |
| **PgUp** / **PgDn** | Scroll results |
| **q** / **Esc** | Quit |

**Important:** typing is off by default so keys like `r` do not corrupt the path. Press **e** only when you need to type a path.

## Architecture

- **Single engine:** Python (`run_diagnosis`, evidence plots, GGUF, TS Pulse, RAG)
- **Two skins:** Textual (`src/tui_app.py`) and ratatui (`tui_ratatui/`)
- **Swap point:** `src/ui_launch.py` + `app.py --ui` / `GT_UI`
