# Deploying the Space

The active Hugging Face deployment is a **Gradio Space** with hardware-aware extraction:

- **CPU Basic:** llama.cpp with the base MiniCPM-V GGUF model.
- **ZeroGPU / GPU:** Transformers vision with the fine-tuned MiniCPM-V checkpoint.

This workflow is intentionally fixed:

1. The Space must stay a Gradio Space, not a Docker Space.
2. Runtime extraction should use `EXTRACTOR_BACKEND=auto` unless a lane is being forced for testing.
3. ZeroGPU extraction calls must run behind `@spaces.GPU`; CPU Basic llama.cpp calls must not require ZeroGPU.
4. Model files must not be committed to the Space git repo.
5. When the fine-tuned model is ready, replace only the model variables for the active lanes.
6. The llama.cpp lane is automatic on CPU Basic and can still be enabled explicitly with environment variables.

Do not change this architecture unless the project intentionally gives up hardware-aware deployment. To swap models, change `ZEROGPU_MODEL_ID` (or `DEFAULT_HF_REPO` in `src/model_paths.py`) and optional `LLAMACPP_*` for the llama.cpp lane.

## 1. Space Metadata

The Space `build-small-hackathon/blood-test-explainer` must use **`sdk: gradio`** so ZeroGPU is available. The top of `README.md` must stay:

```yaml
---
title: Blood Test Explainer
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.10.13"
app_file: app.py
pinned: false
---
```

ZeroGPU is Gradio-only on Hugging Face. It is not available for Docker Spaces, which is why the previous Docker + `llama-server` deployment was replaced.

## 2. Default Model Serving

Leave `EXTRACTOR_BACKEND` unset or set it to:

```text
EXTRACTOR_BACKEND=auto
```

On **CPU Basic**, `auto` detects `cpu-basic` and selects llama.cpp vision with the base GGUF model:

```text
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

On **ZeroGPU/GPU**, `auto` selects the fine-tuned Transformers repo:

```text
ZEROGPU_MODEL_ID=build-small-hackathon/blood-test-minicpmv-4_6-medreason
```

Hub: [build-small-hackathon/blood-test-minicpmv-4_6-medreason](https://huggingface.co/build-small-hackathon/blood-test-minicpmv-4_6-medreason)

`ZEROGPU_MODEL_ID` is optional when it matches the code default in `src/model_paths.py`. Use `openbmb/MiniCPM-V-4.6` only for base-model baselines.

The backend lives in:

```text
src/extraction/zerogpu_transformers.py
```

It uses:

```python
@spaces.GPU(duration=120)
transformers.AutoModelForImageTextToText
```

This is the correct runtime for PDF/image blood-test uploads on ZeroGPU because the GPU is allocated only inside the decorated worker.

Aliases `zerogpu` and `zero-gpu` force the Transformers path in `src/extraction/factory.py`; `auto` is hardware-aware.

## 3. llama.cpp Lane

The app ships a second extraction lane for CPU Basic, hackathon badges, and GGUF deployment experiments. It is automatic on CPU Basic.

### Why it exists

- **Llama Champion badge** — inference through `llama-cpp-python` over GGUF inside `@spaces.GPU`.
- **Fine-tuned GGUF swap** — deploy a quantized model without changing the Gradio app structure.
- **Text-only fallback** — lighter lane for plain-text lab exports when vision is not needed.

For normal CPU Basic PDF/image uploads, keep `EXTRACTOR_BACKEND=auto` and let the code select llama.cpp vision.

### Force vision llama.cpp on the Space

Set these variables in the Space settings:

```bash
EXTRACTOR_BACKEND=llamacpp-gpu
LLAMACPP_VISION=1
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
LLAMACPP_CHAT_HANDLER=MiniCPMv26ChatHandler
```

Implementation: `src/extraction/llamacpp_gpu.py` with shared vision loading in `src/extraction/llamacpp_vision.py`.

Without `LLAMACPP_VISION=1`, the llama.cpp lane accepts `.txt` / `.csv` only and rejects PDF/image uploads.

### Enable text-only llama.cpp

```bash
EXTRACTOR_BACKEND=llamacpp-gpu
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
```

## 4. Swapping or Retraining the Model

To publish a newer fine-tune:

1. Upload the Transformers checkpoint to a Hugging Face model repo.
2. Optionally convert/quantize to GGUF (+ mmproj) for the llama.cpp lane.
3. Keep the same Gradio + ZeroGPU architecture.
4. Point extraction at the new repo:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-transformers-repo>
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
LLAMACPP_MMPROJ_FILE=<mmproj-file>.gguf
```

Or update `DEFAULT_HF_REPO` in `src/model_paths.py` so local runs and the Space pick it up without an env override.

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the ZeroGPU deployment.

## 5. Why This Architecture

The Docker path failed on free CPU hardware with `OOMKilled` during build. ZeroGPU is only available for Gradio SDK Spaces, so the deployment must be a Gradio Space to use the free dynamic GPU resource.

This architecture keeps:

- Free ZeroGPU eligibility.
- A CPU Basic fallback that uses llama.cpp + base GGUF instead of Transformers.
- No external hosted inference API calls.
- The fine-tuned Transformers runtime on ZeroGPU for PDF/image lab reports.
- A llama.cpp / GGUF lane for CPU Basic, badges, and fine-tuned GGUF deployment.
- A clean model swap by changing `ZEROGPU_MODEL_ID` / `DEFAULT_HF_REPO` and optional `LLAMACPP_*` variables.

## 6. Local Development

Default (Transformers vision):

```bash
pip install -r requirements.txt
python app.py
```

Optional llama.cpp vision:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=llamacpp-gpu LLAMACPP_VISION=1 python app.py
```

For quick UI-only work, continue using the static reference report without triggering extraction.
