# Getting Started

## 1. Get the code

```powershell
git clone https://github.com/2111gt/gt_harness.git
cd gt_harness
```

If you cannot clone, copy the project folder (or a zip of the source) to the machine. Do **not** require `models/*.gguf` or `models/tspulse/**` weights in the zip — they re-download.

## 2. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Desktop GUI also needs:

```powershell
pip install customtkinter matplotlib Pillow
```

(These are listed in `requirements.txt`.)

## 3. Download models

```powershell
python app.py --download-only
```

This fetches (when missing):

| Asset | Location |
|-------|----------|
| Granite 4.1 8B GGUF | `models/granite-4.1-8b-Q4_K_M.gguf` |
| llama-cli (Windows CPU build by default) | `models/llama-cpp-bin/` |
| TS Pulse weights | `models/tspulse/...` |
| Embeddings | Hugging Face hub cache (SentenceTransformers) |

Offline later:

```powershell
python app.py --no-download
```

## 4. Run

| UI | Command |
|----|---------|
| Terminal TUI (default) | `python app.py` |
| Desktop GUI | `python app.py --ui gui` |
| One-shot CLI | `python app.py --cli-once samples/gt_sensors_demo.csv --mode Alerts` |

## 5. First diagnosis

1. Use sample **`samples/gt_sensors_demo.csv`** (or a cold_spot / hets / comb_dyn scenario).
2. Choose **Alerts** or **Trips/Event**.
3. Optionally paste SOE / alarm context.
4. **Run diagnosis** (TUI: **Ctrl+R**; GUI: button).
5. Review **Final report** + **Proof plots**.
6. Optionally **Save & Learn** to write a case under `saved_cases/`.

## Hardware notes (laptop)

- **32 GB RAM**: Granite **8B** Q4/Q5 is the practical default.
- **RTX 1000 Ada (6 GB)**: cannot host 30B fully; CUDA helps only with a CUDA torch/llama stack and models that fit.
- Prefer CUDA when available (`GT_N_GPU_LAYERS` auto when CUDA is detected); force CPU with `GT_FORCE_CPU=1`.
