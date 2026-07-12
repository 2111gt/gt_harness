# Claude rebuild brief — GT Diagnostic Harness

Use this document on another computer when you only have **Claude Sonnet (web)** + **VS Code**.  
Do **not** ask Claude to write the whole app in one shot. Build **slices A→H** in order.

---

## Architecture (paste into every new Claude chat)

```text
Build a LOCAL gas-turbine diagnostic harness (NOT medical OBD, NOT Gradio).

Pipeline:
1) Load sensor CSV
2) Anomaly detection: IBM Granite TS Pulse (tsfm_public TSPulseForReconstruction)
   with statistical robust z-score (median/MAD) fallback
3) RAG over knowledge/*.md and saved_cases/*.json
   (ChromaDB + SentenceTransformer all-MiniLM-L6-v2; keyword/memory if embeddings fail)
4) Local LLM: Granite 4.1 8B GGUF — try llama-cpp-python; on illegal instruction use official
   llama-cli/llama-completion from models/llama-cpp-bin (subprocess, CREATE_NO_WINDOW on Windows,
   absolute GGUF path, prompt via temp file -f)
5) Structured LLM sections exactly:
   ## Reasoning
   ## Initial hypotheses
   ## Self-review
   ## Final diagnosis
6) Optional second reflection pass via env GT_FULL_REFLECTION; single pass OK for speed
7) UI: Textual TUI only — CSV path, Browse, drop-zone (click then drag file onto terminal =
   paste path), Alerts vs Trips/Event, progress bar+ETA, live output stream, full report,
   New session (clears report+live), Save & Learn
8) Entry: python app.py | --cli-once CSV | --download-only | --no-download

Hard rules:
- Never replace tqdm.tqdm with a lambda (breaks huggingface_hub + TS Pulse)
- Defer TS Pulse/embeddings until first diagnosis; re-probe imports (no permanent false failure)
- GGUF under project models/; TS Pulse weights in HF hub cache
- Deterministic anomaly scores where possible (seed TS Pulse residuals)
- Decision-support disclaimer only — not OEM protection software

Layout:
gt_harness/
  app.py, requirements.txt, README.md
  src/ __init__.py analysis.py models.py tools.py download.py llama_cli_backend.py
       tui_app.py utils.py compat.py
  knowledge/ samples/ models/ tests/ scripts/
```

---

## Build order (one slice per Claude turn)

| Slice | Deliver | Verify |
|-------|---------|--------|
| **A** Skeleton | `app.py` argparse, `src/utils.py` paths/logging/dirs, `requirements.txt`, empty packages | `python app.py --help` |
| **B** CSV + stats anomalies | `load_sensor_csv`, robust z-score, demo CSV with spike | Load demo → severity + top channel |
| **C** Offline diagnosis | `run_diagnosis` with `llm=None` rule-based draft/final markdown | Offline report has sections |
| **D** RAG | knowledge md, Chroma or memory RAG, save_case JSON | Query returns hits |
| **E** GGUF LLM | llama-cli backend + generate_llm | Non-empty model text (not offline markers) |
| **F** TS Pulse | ensure_tspulse, lazy load, residual mode | `mode=tspulse` or clear statistical status |
| **G** TUI | Textual full UI | `python app.py`; run + New session |
| **H** Tests | unittest for analysis + TUI structure | `python -m unittest discover -s tests -v` |
| **I** (optional) | Scenario CSVs TC1–TC27 cold spot, HETS, dynamics + process maps | Files load in app |

Prompt template each time:

> Implement **Slice X only**. Do not rewrite working files unless required. Complete file contents. Windows CPU. No Gradio. No lambda tqdm.

---

## Dependencies (starting point)

```
textual>=0.80.0
rich>=13.0.0
pandas>=2.0.0
numpy>=1.24.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
transformers>=4.57.0,<5.0
huggingface-hub>=0.34.0,<1.0
torch>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
granite-tsfm>=0.2.0
# llama-cpp-python optional; prefer official llama-cli on older CPUs
```

---

## Env vars

| Variable | Meaning |
|----------|---------|
| `GT_NO_DOWNLOAD=1` | No pip/HF downloads |
| `GT_FULL_REFLECTION=0/1` | Skip / force second LLM pass |
| `GT_GGUF_PATH` | Absolute path to GGUF |
| `GT_TSPULSE_MODEL` | HF id (default ibm-granite/granite-timeseries-tspulse-r1) |
| `GT_TSPULSE_REVISION` | Optional HF revision |
| `GT_SKIP_HEAVY_MODELS=1` | Skip slow model tests |

---

## Model locations (after first successful run)

| Asset | Typical path |
|-------|----------------|
| Granite GGUF | `gt_harness/models/granite-4.1-8b-Q4_K_M.gguf` |
| llama-cli bin | `gt_harness/models/llama-cpp-bin/` |
| TS Pulse weights | `%USERPROFILE%\.cache\huggingface\hub\models--ibm-granite--granite-timeseries-tspulse-r1\` |

---

## Workflow on the new PC

1. Install Python 3.11+, open VS Code, create `Documents\gt_harness`.
2. Open Claude (Project if possible); paste this whole brief.
3. “Implement Slice A only.”
4. Save files → run verify → paste errors → fix → next slice.
5. Keep `NOTES.md`: what passed, OS, Python version, model paths.

When Claude context is full: new chat + re-paste this brief + list of existing files + last error.

---

## Success criteria

- [ ] Offline path works without GGUF  
- [ ] `--cli-once samples/gt_sensors_demo.csv` exits 0  
- [ ] TUI runs; live progress + full report; New session clears both  
- [ ] LLM path works via llama-cli if python bind fails  
- [ ] TS Pulse or honest statistical fallback (not stuck false “not importable”)  

---

## Disclaimer language for UI/README

Engineering decision-support only. Not an OEM-certified procedure or safety/protection system.
