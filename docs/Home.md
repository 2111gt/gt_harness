# GT Diagnostic Harness Wiki

**GT Harness** is a **fully local** gas-turbine diagnostic application: sensor CSV → anomaly detection (Granite **TS Pulse** or statistical fallback) → knowledge **RAG** → local **Granite GGUF** LLM → structured report + **proof plots**.

| | |
|--|--|
| **Repository** | [2111gt/gt_harness](https://github.com/2111gt/gt_harness) |
| **Default UI** | Textual terminal TUI |
| **Desktop UI** | CustomTkinter GUI (`--ui gui`) |
| **License** | MIT (sample knowledge is educational, not OEM docs) |

> Engineering **decision-support** only. Not an OEM-certified procedure or protection system.

## Wiki pages

| Page | Contents |
|------|----------|
| [Getting Started](Getting-Started) | Clone, install, download models, first run |
| [User Interfaces](User-Interfaces) | Textual TUI vs desktop GUI |
| [Models and Downloads](Models-and-Downloads) | GGUF, TS Pulse under `models/`, llama-cli |
| [Proof Plots](Proof-Plots) | ASCII (TUI) and high-quality PNG (GUI) |
| [Environment Variables](Environment-Variables) | `GT_*` configuration reference |
| [Architecture](Architecture) | Pipeline and project layout |
| [Troubleshooting](Troubleshooting) | Common issues on Windows |

## Quick start

```powershell
git clone https://github.com/2111gt/gt_harness.git
cd gt_harness
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --download-only
python app.py              # Textual TUI
# or
python app.py --ui gui     # desktop GUI
```

## What you need on disk

- **Python 3.11+** (Windows supported)
- **Network once** for GGUF + TS Pulse + embeddings (or offline after cache)
- **~6+ GB** free for Granite 8B Q4 GGUF; more for optional larger models
- Optional **NVIDIA CUDA** stack for GPU preference (see [Models and Downloads](Models-and-Downloads))

## Contributing / local development

See the repo `README.md` and `tests/` for unit tests:

```powershell
python -m unittest discover -s tests -v
```
