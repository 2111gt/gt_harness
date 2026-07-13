# Architecture

## Pipeline

```text
1. Load sensor CSV
2. Anomaly detection
   - Prefer IBM Granite TS Pulse (reconstruction residuals)
   - Fallback: robust z-score (median/MAD)
   - Optional signature classifier head (scaffold until fine-tuned)
3. RAG over knowledge/*.md + saved_cases/*.json
   - ChromaDB + SentenceTransformer (keyword memory fallback)
4. Local LLM: Granite GGUF
   - llama-cpp-python and/or official llama-cli
   - Prefer CUDA when available
5. Structured report sections:
   - Reasoning · Initial hypotheses · Self-review · Final diagnosis
6. Proof plots of top issue channels (ASCII + PNG)
7. Optional Save & Learn → re-index case into RAG
```

## Project layout

```text
gt_harness/
  app.py                 # CLI / UI entry
  requirements.txt
  src/
    analysis.py          # Orchestration
    models.py            # LLM + TS Pulse loaders
    tools.py             # CSV, RAG, Save & Learn
    evidence_plots.py    # Proof plots (ASCII + matplotlib)
    tui_app.py           # Textual TUI
    gui_app.py           # CustomTkinter GUI
    download.py          # Models → models/
    device.py            # CUDA preference
    bridge.py            # --json-once
    llama_cli_backend.py
    ui_launch.py
    utils.py
  knowledge/             # Process maps (RAG)
  samples/               # Demo + scenario CSVs
  models/                # GGUF, llama-cli, tspulse, tspulse_clf
  tests/
  logs/                  # last report, evidence PNGs
```

## Entry points

| Command | Role |
|---------|------|
| `python app.py` | Textual TUI |
| `python app.py --ui gui` | Desktop GUI |
| `python app.py --cli-once CSV` | Batch diagnosis |
| `python app.py --json-once CSV` | Machine-readable JSON |
| `python app.py --download-only` | Fetch models and exit |

## Design constraints (Windows)

- Subprocesses (llama-cli, pip) use **CREATE_NO_WINDOW** so the Textual console does not hang.
- Do not replace `tqdm` with a lambda (breaks huggingface_hub / TS Pulse).
- GGUF path must be absolute when llama-cli runs with cwd under `llama-cpp-bin/`.
