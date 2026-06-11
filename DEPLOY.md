# Deploying the Space with ZeroGPU

The active Hugging Face deployment is a **Gradio ZeroGPU Space**.

This workflow is intentionally fixed:

1. The Space must stay a Gradio Space, not a Docker Space.
2. Runtime extraction should use the `llamacpp-gpu` backend when we are targeting the Llama Champion badge.
3. The extraction call must run behind `@spaces.GPU` so Hugging Face allocates ZeroGPU only while the model is needed.
4. Model files must not be committed to the Space git repo.
5. When the fine-tuned GGUF model is ready, only replace the `LLAMACPP_*` model variables.

Do not change this architecture unless the project intentionally gives up ZeroGPU or the llama.cpp backend proves incompatible with ZeroGPU. The intended future model-serving change is inserting the fine-tuned GGUF repository path into the existing `LLAMACPP_*` variables.

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

The badge-target model path is the official OpenBMB GGUF repo running through llama.cpp:

```text
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
EXTRACTOR_BACKEND=auto
```

The backend lives in:

```text
src/extraction/llamacpp_gpu.py
```

It uses:

```python
llama_cpp.Llama(...)
@spaces.GPU(duration=120)
```

This is a valid hackathon badge option because the submitted app remains a Gradio ZeroGPU Space,
but the model runtime is `llama.cpp` over GGUF rather than a hosted inference API.

The safe fallback backend is the official OpenBMB Transformers model:

```text
EXTRACTOR_BACKEND=zerogpu
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

Use the fallback only if `llama-cpp-python` cannot load MiniCPM-V 4.6 on ZeroGPU.

## 3. Future Fine-Tuned Model

When the fine-tuned model is ready:

1. Convert/quantize the fine-tuned model to GGUF.
2. Upload the fine-tuned GGUF model and compatible mmproj file to a Hugging Face model repo.
3. Keep the same Gradio + ZeroGPU + llama.cpp architecture.
4. Change only these variables:

```bash
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
LLAMACPP_MMPROJ_FILE=<compatible-mmproj>.gguf
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the ZeroGPU deployment.

## 4. Why This Architecture

The Docker path failed on free CPU hardware with `OOMKilled` during build. ZeroGPU is only available for Gradio SDK Spaces, so the deployment must be a Gradio Space to use the free dynamic GPU resource.

This architecture keeps:

- Free ZeroGPU eligibility.
- No external hosted inference API calls.
- A valid `llama.cpp` / GGUF runtime path for the Llama Champion badge, if `llama-cpp-python` is compatible.
- A clean future swap to a fine-tuned GGUF model by changing only `LLAMACPP_*` variables.

## 5. Local Development

Local development can run the same backend, although the model may be slow or too large without a local GPU:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=auto python app.py
```

For quick UI-only work, continue using the static reference report without triggering extraction.
