# Runbook — Adaptive Extraction + Fine-Tuned Model Swap

The active deployment path is now **Gradio with adaptive extraction**.

This replaced the Docker + `llama-server` path because ZeroGPU is only available for Gradio SDK Spaces. The Docker build was also failing on free CPU hardware with `OOMKilled`.

## Active Architecture

| Area | Current choice |
|---|---|
| Space SDK | `gradio` |
| Hardware | Adaptive: CUDA when available, CPU otherwise |
| Auto backend | Transformers on ZeroGPU/CUDA, llama.cpp on CPU |
| Force llama.cpp | `EXTRACTOR_BACKEND=llamacpp-gpu` |
| Force Transformers | `EXTRACTOR_BACKEND=zerogpu` or `EXTRACTOR_BACKEND=transformers` |
| llama.cpp variables | `LLAMACPP_GGUF_REPO`, `LLAMACPP_MODEL_FILE` |
| Transformers variables | `ZEROGPU_MODEL_ID`, `ZEROGPU_MAX_NEW_TOKENS`, `ZEROGPU_QUANTIZE` |
| Extraction backends | `src/extraction/auto.py`, `src/extraction/llamacpp_gpu.py`, `src/extraction/zerogpu_transformers.py` |
| Report enrichment | `src/report_pipeline.py` + `kb/cbc_knowledge_graph.json` |

Do not switch the Space back to Docker unless the project intentionally gives up ZeroGPU.

## Backend Selection

`EXTRACTOR_BACKEND`:

- `auto`: uses the Transformers backend when Hugging Face reports a ZeroGPU accelerator such as `ACCELERATOR=zero-a10g`, when `ZERO_GPU=TRUE` is set, or when CUDA is visible; otherwise uses the CPU llama.cpp backend. This runtime signal matters because CUDA is only visible inside a `@spaces.GPU` worker. CPU fallback after a Transformers failure is opt-in with `AUTO_FALLBACK_TO_LLAMACPP=1`.
- `llamacpp-gpu`: force the GGUF llama.cpp backend.
- `zerogpu` / `transformers`: force the OpenBMB Transformers backend.
- `api`: hosted OpenBMB endpoint for development fallback only.
- `local` / `server` / `llamacpp`: local experimental backends, not the active HF Space path.

## HF Space Requirements

`README.md` frontmatter must stay:

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

Install dependencies from `requirements.txt`, including:

```text
spaces
torch
transformers
llama-cpp-python
```

The Space installs both runtime lanes so `EXTRACTOR_BACKEND=auto` can choose at runtime. ZeroGPU or
CUDA hardware uses the official OpenBMB Transformers path. CPU hardware uses the prebuilt
`llama-cpp-python` wheel and avoids a source build.

The active llama.cpp path now uses the official prebuilt CPU manylinux wheel for `llama-cpp-python`:

```text
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28/llama_cpp_python-0.3.28-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

This avoids both the CUDA runtime mismatch that was causing the Space to abort on
`libcudart.so.12` and the slow source build that was timing out on Hugging Face. The CPU fallback
is PDF/text-only, so it keeps the Space on a simpler llama.cpp route without mmproj or an image
encoder.

The Transformers backend uses `@spaces.GPU(duration=120)` for GPU generation. The llama.cpp CPU
fallback keeps a longer duration budget because CPU inference is slower.

## Current Model

Badge-target defaults:

```bash
EXTRACTOR_BACKEND=auto
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

No model files are committed to the Space repo.

## Fine-Tuned Model Swap

When the fine-tuned model is ready:

1. Upload the fine-tuned Transformers checkpoint to a Hugging Face model repo for the CUDA lane.
2. Convert/quantize it to GGUF for the CPU llama.cpp lane.
3. Keep the same Gradio adaptive architecture.
4. Change only:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-transformers-repo>
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the Space deployment.

## Local Development

For UI-only work:

```bash
python app.py
```

For local extraction testing with the same backend:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=auto python app.py
```

Local machines without a suitable GPU may be slow or may not have enough memory for full model inference. In that case, test UI/report rendering locally and test extraction on the HF Space.

## Verification

Run before pushing:

```bash
python3 -m py_compile app.py src/*.py src/extraction/*.py
.venv/bin/python -m pytest tests/test_report_pipeline.py
```

Then verify the Space build uses Gradio, not Docker, and that ZeroGPU/CUDA hardware selects Transformers while CPU hardware selects llama.cpp.
