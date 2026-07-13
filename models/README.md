# Models directory

On first launch the app **automatically downloads** missing AI models **into this folder** (and related subfolders).

```bash
python app.py                  # auto-download + UI
python app.py --download-only  # download everything, then exit
python app.py --no-download    # offline: do not fetch anything
```

## Layout

```text
models/
  granite-4.1-8b-Q4_K_M.gguf     # Main LLM (GGUF)
  llama-cpp-bin/                 # llama-cli / llama-completion binaries
  tspulse/                       # TS Pulse reconstruction weights (HF snapshot)
    ibm-granite--granite-timeseries-tspulse-r1/
      tspulse-block-ad/          # (or other GT_TSPULSE_REVISION)
  tspulse_clf/                   # Classifier scaffold dual-head revision
    .../
```

Large weight files are **gitignored** — re-download with `python app.py --download-only` on a new machine.

## Main LLM (Granite 4.1 8B GGUF)

- Auto source: `ibm-granite/granite-4.1-8b-GGUF` → `granite-4.1-8b-Q4_K_M.gguf` (~5 GB)
- Via [llama.cpp](https://github.com/ggerganov/llama.cpp) / `llama-cpp-python`
- Override: `GT_GGUF_PATH`, `GT_GGUF_REPO`, `GT_GGUF_FILE`

## TS Pulse (anomaly model)

IBM Granite **TS Pulse** (`ibm-granite/granite-timeseries-tspulse-r1`) is downloaded **into `models/tspulse/`** on first use (same idea as the GGUF — project-local, not only the user HF cache).

- Override model/revision: `GT_TSPULSE_MODEL`, `GT_TSPULSE_REVISION`
- Override folder: `GT_TSPULSE_LOCAL_DIR`
- Classifier scaffold: `models/tspulse_clf/` (`GT_TSPULSE_CLF_REVISION`)
- Fine-tuned head: `GT_TSPULSE_CLF_PATH`

If unavailable, robust statistical anomaly detection is used.

## Embeddings

`all-MiniLM-L6-v2` (SentenceTransformer) is still downloaded via Hugging Face for ChromaDB RAG (default HF hub cache unless you set `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME`).
