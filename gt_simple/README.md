# GT Simple

One-file local gas-turbine diagnostic mini-app (simplified from the full **gt_harness** tree).

This folder lives inside the main monorepo:

```text
gt_harness/
  … full app …
  gt_simple/          ← this mini-app
    app.py
```

## What it does

1. **CSV + context** input  
2. **TS Pulse recon** anomalies (or statistical fallback)  
3. **TS Pulse dual-head / signature** classifier (or heuristic)  
4. **RAG** over `knowledge/` + `saved_cases/`  
5. **LLM pass 1** draft → **pass 2** critique/refine  
6. **Save & Learn** JSON flywheel re-indexed into RAG  

## Run

```powershell
cd gt_simple   # or: cd path\to\gt_harness\gt_simple

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Browser UI
streamlit run app.py

# CLI
python app.py --cli samples\gt_sensors_demo.csv --context "exhaust spread" --save

# Rebuild vector index
python app.py --rebuild-rag

# Fast offline draft (no GGUF generate)
$env:GT_SIMPLE_OFFLINE="1"
python app.py --cli samples\gt_sensors_demo.csv --no-tspulse --save
```

### Models

If `gt_simple/models/` has no GGUF, the app **automatically uses the parent harness** `../models/` (shared Granite GGUF, llama-cli, TS Pulse). You can also set:

```powershell
$env:GT_GGUF_PATH = "..\models\granite-4.1-8b-Q4_K_M.gguf"
```

## Layout

```text
gt_simple/
  app.py              # entire app (single file)
  requirements.txt
  models/             # optional local weights; falls back to ../models
  knowledge/          # process maps
  saved_cases/        # flywheel
  samples/
  chroma_db/
  logs/
```

## Env

| Variable | Meaning |
|----------|---------|
| `GT_SIMPLE_OFFLINE=1` | Skip GGUF; rule-based draft + package |
| `GT_SIMPLE_LLM_TIMEOUT` | Seconds per llama-cli call (default 900) |
| `GT_SIMPLE_EMBED=hash` | Default local hash embeddings (safe). `onnx` / `st` opt-in |
| `GT_GGUF_PATH` | Override GGUF path |
| `GT_TSPULSE_MODEL` / `GT_TSPULSE_REVISION` | Recon model |
| `GT_TSPULSE_CLF_REVISION` | Classifier revision |

Engineering **decision-support** only — not OEM protection software.
