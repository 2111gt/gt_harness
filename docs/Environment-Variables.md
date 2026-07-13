# Environment Variables

| Variable | Purpose |
|----------|---------|
| `GT_UI` | `textual` (default) or `gui` |
| `GT_NO_DOWNLOAD` | `1` = offline; no pip/HF downloads |
| `GT_GGUF_PATH` | Absolute path to Granite GGUF |
| `GT_GGUF_REPO` / `GT_GGUF_FILE` | HF GGUF source overrides |
| `GT_N_GPU_LAYERS` | llama.cpp GPU layers (`0`=CPU, `99`≈all). Auto: 99 if CUDA else 0 |
| `GT_FORCE_CPU` / `GT_NO_GPU` | Force CPU path |
| `GT_TORCH_DEVICE` | `auto` / `cuda` / `cpu` for TS Pulse & embeddings |
| `GT_LLAMA_CPP_ZIP_URL` | Override llama.cpp binary zip URL |
| `GT_LLAMA_SMOKE` | `1` = smoke-test GGUF on bind |
| `GT_FULL_REFLECTION` | `0` skip second LLM pass; `1` force |
| `GT_TSPULSE_MODEL` | HF id for TS Pulse |
| `GT_TSPULSE_REVISION` | Anomaly revision (default `tspulse-block-ad`) |
| `GT_TSPULSE_LOCAL_DIR` | Override local TS Pulse folder |
| `GT_TSPULSE_CLF_ENABLE` | `1` (default) load classifier head |
| `GT_TSPULSE_CLF_PATH` | Fine-tuned classifier directory |
| `GT_TSPULSE_CLF_LABELS` | Comma-separated labels |
| `GT_TSPULSE_CLF_CHANNELS` | Input channel width for scaffold head |
| `GT_TSPULSE_CLF_REVISION` | Dual-head HF revision |
| `GT_USE_PLOTEXT` | `1` = optional plotext terminal polish |

PowerShell examples:

```powershell
$env:GT_UI = "gui"
$env:GT_FORCE_CPU = "1"
$env:GT_N_GPU_LAYERS = "99"
$env:GT_TSPULSE_REVISION = "tspulse-block-ad"
```
