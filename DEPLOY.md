# Deploying the Space with ZeroGPU

The active Hugging Face deployment is a **Gradio ZeroGPU Space**.

This workflow is intentionally fixed:

1. The Space must stay a Gradio Space, not a Docker Space.
2. Runtime extraction should use the Transformers backend on ZeroGPU.
3. The extraction call must run behind `@spaces.GPU` so Hugging Face allocates ZeroGPU only while the model is needed.
4. Model files must not be committed to the Space git repo.
5. When the fine-tuned model is ready, replace only the model variables for the active lanes.

Do not change this architecture unless the project intentionally gives up ZeroGPU. The intended future model-serving change is inserting the fine-tuned Transformers repository into `ZEROGPU_MODEL_ID`, and optionally inserting the fine-tuned GGUF repository into `LLAMACPP_*` for CPU fallback.

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

## 2. Model Serving

The active ZeroGPU model path is the official OpenBMB Transformers repo:

```text
EXTRACTOR_BACKEND=auto
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

The backend lives in:

```text
src/extraction/zerogpu_transformers.py
```

It uses:

```python
@spaces.GPU(duration=120)
transformers.AutoModelForImageTextToText
```

This is the correct runtime for a ZeroGPU Space because the GPU is allocated only inside the
decorated worker. A normal app-level `torch.cuda.is_available()` check may be false before the
worker starts, so `auto` also checks Hugging Face's `ACCELERATOR` runtime variable for values such
as `zero-a10g`.

The CPU fallback model path is the official OpenBMB GGUF repo running through llama.cpp:

```text
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
```

## 3. Future Fine-Tuned Model

When the fine-tuned model is ready:

1. Upload the fine-tuned Transformers checkpoint to a Hugging Face model repo.
2. Optionally convert/quantize the fine-tuned model to GGUF for CPU fallback.
3. Keep the same Gradio + ZeroGPU/CUDA Transformers + CPU llama.cpp architecture.
4. Change only these variables:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-transformers-repo>
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the ZeroGPU deployment.

## 4. Why This Architecture

The Docker path failed on free CPU hardware with `OOMKilled` during build. ZeroGPU is only available for Gradio SDK Spaces, so the deployment must be a Gradio Space to use the free dynamic GPU resource.

This architecture keeps:

- Free ZeroGPU eligibility.
- No external hosted inference API calls.
- The official OpenBMB Transformers runtime on ZeroGPU.
- A CPU `llama.cpp` / GGUF fallback when the Space is not on ZeroGPU or CUDA.
- A clean future swap to a fine-tuned model by changing only `ZEROGPU_MODEL_ID` and optional `LLAMACPP_*` variables.

## 5. Local Development

Local development can run the same backend, although the model may be slow or too large without a local GPU:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=auto python app.py
```

For quick UI-only work, continue using the static reference report without triggering extraction.
