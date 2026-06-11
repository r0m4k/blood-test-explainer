# Runbook — ZeroGPU Extraction + Fine-Tuned Model Swap

The active deployment path is now **Gradio ZeroGPU**.

This replaced the Docker + `llama-server` path because ZeroGPU is only available for Gradio SDK Spaces. The Docker build was also failing on free CPU hardware with `OOMKilled`.

## Active Architecture

| Area | Current choice |
|---|---|
| Space SDK | `gradio` |
| Hardware | ZeroGPU |
| Badge-target runtime | `llama.cpp` through a `llama-cpp-python` build with MiniCPM-V 4.6 support |
| Badge-target backend | `EXTRACTOR_BACKEND=auto` or `EXTRACTOR_BACKEND=llamacpp-gpu` |
| Fallback backend | `EXTRACTOR_BACKEND=zerogpu` with Transformers |
| Model variables | `LLAMACPP_GGUF_REPO`, `LLAMACPP_MODEL_FILE`, `LLAMACPP_MMPROJ_FILE` |
| Extraction backends | `src/extraction/llamacpp_gpu.py`, `src/extraction/zerogpu_transformers.py` |
| Report enrichment | `src/report_pipeline.py` + `kb/cbc_knowledge_graph.json` |

Do not switch the Space back to Docker unless the project intentionally gives up ZeroGPU.

## Backend Selection

`EXTRACTOR_BACKEND`:

- `auto`: default badge-target path, runs GGUF through `llama.cpp` inside `@spaces.GPU`.
- `llamacpp-gpu`: explicit alias for the same badge-target path.
- `zerogpu`: force the ZeroGPU Transformers fallback backend.
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
llama-cpp-python
```

The Space build intentionally excludes the heavier Transformers fallback stack from the default
runtime requirements so the Hugging Face build stays fast. Add backend-specific extras only when
you are explicitly working on that backend locally.

Both ZeroGPU backends use `@spaces.GPU(duration=120)` for the model generation call.

## Current Model

Badge-target defaults:

```bash
EXTRACTOR_BACKEND=auto
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

This is the official OpenBMB GGUF model path. No model files are committed to the Space repo.

Fallback variables if llama.cpp is incompatible on ZeroGPU:

```bash
EXTRACTOR_BACKEND=zerogpu
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

## Fine-Tuned Model Swap

When the fine-tuned model is ready:

1. Convert/quantize it to GGUF.
2. Upload it and the compatible mmproj to a Hugging Face model repo.
3. Keep the same Gradio + ZeroGPU + llama.cpp architecture.
4. Change only:

```bash
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
LLAMACPP_MMPROJ_FILE=<compatible-mmproj>.gguf
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the ZeroGPU deployment.

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

Local machines without a suitable GPU may be slow or may not have enough memory for full model inference. In that case, test UI/report rendering locally and test extraction on the HF ZeroGPU Space.

## Verification

Run before pushing:

```bash
python3 -m py_compile app.py src/*.py src/extraction/*.py
.venv/bin/python -m pytest tests/test_report_pipeline.py
```

Then verify the Space build uses Gradio, not Docker, and that ZeroGPU can be selected in the Space hardware panel.
