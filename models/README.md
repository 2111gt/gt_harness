# Models directory

On first launch the app **automatically downloads** missing AI models here / into the Hugging Face cache.

```bash
python app.py                  # auto-download + UI
python app.py --download-only  # download everything, then exit
python app.py --no-download    # offline: do not fetch anything
```

## Main LLM (Granite 4.1 8B GGUF)

- Auto source: `ibm-granite/granite-4.1-8b-GGUF` → `granite-4.1-8b-Q4_K_M.gguf` (~5 GB)
- Via [llama.cpp](https://github.com/ggerganov/llama.cpp) / `llama-cpp-python`
- Override: `GT_GGUF_PATH`, `GT_GGUF_REPO`, `GT_GGUF_FILE`

## TS Pulse

IBM Granite **TS Pulse** (`ibm-granite/granite-timeseries-tspulse-r1`) is downloaded on first use.  
If unavailable, robust statistical anomaly detection is used.

## Embeddings

`all-MiniLM-L6-v2` (SentenceTransformer) is auto-downloaded for ChromaDB RAG.
