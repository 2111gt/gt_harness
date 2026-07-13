# Troubleshooting

## Models still loading / first run slow

First run downloads GGUF (~5 GB), may install packages, and warms embeddings/TS Pulse on first diagnosis. Use `--download-only` on a good network, then run offline with `--no-download` if needed.

## TUI hangs on Windows when CLI works

Ensure llama-cli / pip children use hidden console (`run_hidden_subprocess`). Update to latest `src/utils.py` / `llama_cli_backend.py` from the repo.

## CSV not found

- Prefer absolute paths or samples from the project tree.
- GUI and bridge resolve paths relative to project root and under `samples/`.
- Sample picker sets full paths automatically.

## TS Pulse not in `models/` folder (old installs)

Current code downloads TS Pulse into **`models/tspulse/`**. Older runs used only `%USERPROFILE%\.cache\huggingface\hub\...`. Re-run:

```powershell
python app.py --download-only
```

## Statistical fallback instead of TS Pulse

- Install `granite-tsfm` / ensure `tsfm_public` imports.
- Confirm `models/tspulse/.../config.json` + `model.safetensors` exist.
- Check logs for lazy-load errors.

## GPU not used

- Install **CUDA** PyTorch and (optionally) CUDA llama-cpp-python.
- Check status for `n_gpu_layers` / `torch.cuda`.
- Force CPU: `GT_FORCE_CPU=1`. Force layers: `GT_N_GPU_LAYERS=99`.
- 6 GB laptop GPUs cannot fully host 30B-class models.

## Desktop GUI missing

```powershell
pip install customtkinter matplotlib Pillow
python app.py --ui gui
```

## Illegal instruction / GGUF fails with llama-cpp-python

App prefers official **llama-cli** binary under `models/llama-cpp-bin/` on many Windows CPUs. Delete a broken wheel setup and let the app re-fetch the CLI zip, or set `GT_GGUF_PATH` to a valid GGUF.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Logs

- App log: `logs/gt_harness.log`
- Last report: `logs/last_diagnosis_report.md`
- Proof PNGs: `logs/evidence_plots/`
