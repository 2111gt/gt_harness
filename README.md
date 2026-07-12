# GT Diagnostic Harness

Fully **local** gas turbine (GT) diagnostic application:

| Layer | Technology |
|-------|------------|
| UI | **Modern Textual TUI** — command strip, status chips, sample picker, proof plots, live pipeline, progress + ETA |
| Anomaly detection | Granite **TS Pulse** (anomaly revision) + statistical fallback |
| Main LLM | **Granite 4.1 8B** GGUF via llama-cpp-python or official **llama-cli** (CPU) |
| RAG | **ChromaDB** + **SentenceTransformer** (`all-MiniLM-L6-v2`) |
| Flywheel | **Save & Learn** → `saved_cases/*.json` re-indexed into RAG |
| Report packaging | Single LLM draft by default + structured packaging (optional second pass) |

> Engineering decision-support only. Not an OEM-certified procedure or safety system.

## Project layout

```text
gt_harness/
├── app.py                 # TUI / --cli-once entry
├── src/
│   ├── tui_app.py         # Textual UI + progress panel
│   ├── models.py          # LLM + TS Pulse loaders & anomaly scoring
│   ├── tools.py           # CSV, RAG, Save & Learn
│   ├── analysis.py        # Pipeline + stepped progress/ETA
│   ├── llama_cli_backend.py
│   └── utils.py
├── models/                # Granite GGUF + llama-cpp-bin
├── knowledge/             # Process maps & GT background (RAG)
├── saved_cases/           # Flywheel JSON cases
├── chroma_db/             # Persistent vector index
├── samples/               # Demo sensor CSV
├── tests/
├── requirements.txt
└── README.md
```

## Quick start

```bash
cd gt_harness
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

This opens a **terminal UI** in your console — not a browser.

### Swap TUI backends (Textual ↔ ratatui)

Same diagnosis engine; two interchangeable frontends:

| Backend | Launch | Notes |
|---------|--------|--------|
| **Textual** (default) | `python app.py` or `python app.py --ui textual` | Pure Python |
| **ratatui** (Rust) | `python app.py --ui ratatui` or `$env:GT_UI='ratatui'; python app.py` | Build once: `cd tui_ratatui && cargo build --release` |

The ratatui UI calls `python app.py --json-once …` under the hood (see `src/bridge.py`, `tui_ratatui/README.md`).

| Key | Action |
|-----|--------|
| **Ctrl+R** | Run diagnosis |
| **Ctrl+S** | Save & Learn |
| **Q** | Quit |

While loading models or running a diagnosis, the **bottom status panel** shows:

- checklist of steps (CSV → anomaly → RAG → LLM → finalize)
- progress bar (% complete)
- current step text, elapsed time, and estimated time remaining

After a run, the right pane shows **proof plots**: ASCII charts of the channels the anomaly engine ranked highest, with **▲** on flagged samples — visual evidence next to the written diagnosis. The same plots are embedded in the markdown report and written to `logs/last_evidence_plots.txt`.

Optional flags:

```bash
python app.py --download-only
python app.py --no-download
python app.py --cli-once samples/gt_sensors_demo.csv --save
```

### Demo without GGUF

1. Start the app.
2. Upload `samples/gt_sensors_demo.csv`.
3. Choose **Trips/Event**, paste a short note about exhaust spread.
4. Click **Run diagnosis** — statistical/TS Pulse anomalies + knowledge RAG still run; LLM uses an offline draft if no GGUF is present.
5. Enter corrections → **Save & Learn**.

### Automatic model download

On startup (unless `--no-download` or `GT_NO_DOWNLOAD=1`), the app will:

1. `pip install` missing packages (`llama-cpp-python`, `chromadb`, `sentence-transformers`, `granite-tsfm`, …)
2. Download **Granite 4.1 8B GGUF** (~5 GB) into `models/`
3. Cache **TS Pulse** + **all-MiniLM-L6-v2** embeddings via Hugging Face

```bash
python app.py --download-only   # fetch everything, then exit
python app.py                   # download if needed, then open UI
python app.py --no-download     # offline / air-gapped
```

First GGUF download needs disk space and a network connection; later runs are fully local.

### TS Pulse

If TS Pulse cannot load, **robust z-score** anomaly detection is used automatically.

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
# or:
python -m unittest discover -s tests -v
```

Tests drive **real** shipped functions (`detect_anomalies`, `score_severity`, `save_case`, `run_diagnosis`) with injected lightweight `ModelBundle`s — no mocked scoring of the unit under test.

## Typical workflow

1. Export historian CSV (numeric sensor columns + optional timestamp).
2. Select **Alerts** or **Trips/Event**.
3. Paste process-map / operator context.
4. **Run diagnosis** → watch progress/ETA → anomaly table, RAG, diagnosis report.
5. Validate findings → **Save & Learn** for continuous improvement.

## Performance notes

Diagnosis is dominated by the **Granite 8B GGUF** step. Defaults:

- **Prefer CUDA when available** (torch TS Pulse/embeddings + GGUF `n_gpu_layers`)
- Single LLM pass unless `GT_FULL_REFLECTION=1`
- `max_tokens` / `n_ctx` sized for structured reports
- Skip GGUF smoke generate unless `GT_LLAMA_SMOKE=1`

On a laptop with an **NVIDIA RTX**, install a **CUDA** build of PyTorch and ideally **llama-cpp-python** with CUDA (or place a CUDA llama.cpp build under `models/llama-cpp-bin/`). Without CUDA packages, the app stays on CPU automatically.

`--cli-once` prints `[load …%]` / `[diag …%]` progress lines on **stderr**.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GT_GGUF_PATH` | Absolute path to Granite GGUF |
| `GT_N_GPU_LAYERS` | llama.cpp GPU layers (`0`=CPU, `99`≈all). Auto: `99` if CUDA else `0` |
| `GT_FORCE_CPU` / `GT_NO_GPU` | `1` = never use GPU |
| `GT_TORCH_DEVICE` | `auto` (default), `cuda`, or `cpu` for TS Pulse / embeddings |
| `GT_LLAMA_CPP_ZIP_URL` | Override llama.cpp binary zip (e.g. CUDA build URL) |
| `GT_TSPULSE_MODEL` | HF model id override |
| `GT_TSPULSE_REVISION` | HF revision (anomaly detection) |
| `GT_FULL_REFLECTION` | Default: second LLM self-review when GGUF is loaded. Set `0` to skip (faster). |
| `GT_TSPULSE_CLF_ENABLE` | `1` (default) load classification head; `0` disable |
| `GT_TSPULSE_CLF_PATH` | Fine-tuned classifier checkpoint dir (after training) |
| `GT_TSPULSE_CLF_LABELS` | Comma labels, default `normal,cold_spot,hets,combustion_dynamics` |
| `GT_TSPULSE_CLF_CHANNELS` | Input channel width for scaffold head (default `16`) |
| `GT_TSPULSE_CLF_REVISION` | HF dual-head revision (default `tspulse-block-dualhead-512-p16-r1`) |
| `GT_LLAMA_SMOKE` | `1` = smoke-test GGUF on bind (slower startup) |
| `GT_NO_DOWNLOAD` | `1` = offline; no pip/HF downloads |

## License

MIT — sample knowledge is educational, not OEM documentation.
