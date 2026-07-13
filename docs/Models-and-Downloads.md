# Models and Downloads

## Project `models/` layout

```text
models/
  granite-4.1-8b-Q4_K_M.gguf     # Main LLM (GGUF)
  llama-cpp-bin/                 # llama-cli / llama-completion
  tspulse/                       # TS Pulse reconstruction (HF snapshot)
    ibm-granite--granite-timeseries-tspulse-r1/
      <revision>/                # e.g. tspulse-block-ad
        config.json
        model.safetensors
  tspulse_clf/                   # Classifier dual-head scaffold revision
```

Large weights are **gitignored**. Re-download with:

```powershell
python app.py --download-only
```

## LLM (Granite GGUF)

| Item | Value |
|------|--------|
| Default file | `models/granite-4.1-8b-Q4_K_M.gguf` |
| Source | `ibm-granite/granite-4.1-8b-GGUF` |
| Override | `GT_GGUF_PATH`, `GT_GGUF_REPO`, `GT_GGUF_FILE` |

Backend order prefers **CUDA when available** (llama-cpp-python with `n_gpu_layers`, else llama-cli with `-ngl`), else CPU llama-cli.

## TS Pulse (not GGUF)

TS Pulse is a **Hugging Face directory**, not a single GGUF.

**Minimum files per revision folder:**

| File | Required? |
|------|-----------|
| `config.json` | **Yes** |
| `model.safetensors` (or weight shards / `.bin`) | **Yes** |
| README, images | Optional |

Default model: `ibm-granite/granite-timeseries-tspulse-r1`  
Default anomaly revision: `tspulse-block-ad` (`GT_TSPULSE_REVISION`)

App downloads into:

```text
models/tspulse/<org--repo>/<revision>/
```

Override folder: `GT_TSPULSE_LOCAL_DIR`.

### Another TS Pulse revision

```powershell
$env:GT_TSPULSE_REVISION = "your-revision-name"
python app.py --download-only
```

Or use `hf download ... --local-dir models/tspulse/...` (see repo `models/README.md`).

Classifier scaffold revision: `GT_TSPULSE_CLF_REVISION` → `models/tspulse_clf/`.  
Fine-tuned head: `GT_TSPULSE_CLF_PATH`.

## Embeddings

`all-MiniLM-L6-v2` for RAG (SentenceTransformers). Default store is the user HF hub cache unless you set `HF_HOME` / related env vars.

## CUDA preference

See [Environment Variables](Environment-Variables). Install a CUDA build of PyTorch and (ideally) CUDA llama-cpp-python for GPU offload. Without them the app stays on CPU automatically.
