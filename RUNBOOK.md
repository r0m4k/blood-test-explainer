# Runbook — Extraction Backends + Fine-Tuned Model Swap

The active deployment path is **Gradio with Transformers vision on ZeroGPU**.

This replaced the Docker + `llama-server` path because ZeroGPU is only available for Gradio SDK Spaces. The Docker build was also failing on free CPU hardware with `OOMKilled`.

## Active Architecture

| Area | Current choice |
|---|---|
| Space SDK | `gradio` |
| Default extraction | Transformers MiniCPM-V 4.6 (`EXTRACTOR_BACKEND=transformers`) |
| ZeroGPU worker | `@spaces.GPU` in `src/extraction/zerogpu_transformers.py` |
| Optional llama.cpp lane | `EXTRACTOR_BACKEND=llamacpp-gpu` (+ `LLAMACPP_VISION=1` for PDF/images) |
| Transformers variables | `ZEROGPU_MODEL_ID`, `ZEROGPU_MAX_NEW_TOKENS`, `ZEROGPU_DOWNSAMPLE_MODE` |
| llama.cpp variables | `LLAMACPP_GGUF_REPO`, `LLAMACPP_MODEL_FILE`, `LLAMACPP_MMPROJ_FILE`, `LLAMACPP_VISION` |
| Extraction backends | `src/extraction/factory.py`, `src/extraction/zerogpu_transformers.py`, `src/extraction/llamacpp_gpu.py` |
| Report enrichment | `src/report_pipeline.py` + `kb/cbc_knowledge_graph.json` |

Do not switch the Space back to Docker unless the project intentionally gives up ZeroGPU.

## Backend Selection

`EXTRACTOR_BACKEND` is read in `src/extraction/factory.py`:

| Value | Behavior |
|---|---|
| `transformers` (default) | OpenBMB MiniCPM-V through Transformers vision |
| `auto`, `zerogpu`, `zero-gpu` | Same as `transformers` |
| `llamacpp-gpu`, `llama-champion` | GGUF through `llama-cpp-python` |
| `local`, `server` | Local `llama-server` HTTP backend |
| `llamacpp` | In-process local GGUF + mmproj |
| `api`, `openbmb`, `hosted` | Disabled |

### Default path (production)

```bash
EXTRACTOR_BACKEND=transformers
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

This is what the HF Space should use for PDF/image blood-test uploads.

### Optional llama.cpp path

The llama.cpp lane is **opt-in**. It is not selected automatically.

**Why keep it:**

- Target the hackathon **Llama Champion** badge (`llama-cpp-python` + GGUF inside `@spaces.GPU`).
- Provide a second deployment lane for fine-tuned **GGUF** weights.
- Support a lighter text-only GGUF path for `.txt` / `.csv` without loading mmproj.

**Vision llama.cpp** — same PDF/image pipeline as Transformers:

```bash
EXTRACTOR_BACKEND=llamacpp-gpu
LLAMACPP_VISION=1
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
LLAMACPP_CHAT_HANDLER=MiniCPMv26ChatHandler
```

**Text-only llama.cpp** — no mmproj, `.txt` / `.csv` only:

```bash
EXTRACTOR_BACKEND=llamacpp-gpu
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
```

Implementation: `src/extraction/llamacpp_gpu.py` and shared vision helpers in `src/extraction/llamacpp_vision.py`.

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
transformers[torch]==5.7.0
llama-cpp-python
```

Transformers runs on ZeroGPU through `@spaces.GPU(duration=120)` (or longer for cold starts). The optional llama.cpp lane uses `@spaces.GPU(duration=600)` because GGUF inference can be slower.

On Linux x86_64 Spaces, `llama-cpp-python` comes from the prebuilt CPU manylinux wheel:

```text
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28/llama_cpp_python-0.3.28-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

This avoids both the CUDA runtime mismatch that was causing the Space to abort on `libcudart.so.12` and the slow source build that was timing out on Hugging Face.

## Current Model Defaults

Primary lane:

```bash
EXTRACTOR_BACKEND=transformers
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

Optional llama.cpp lane:

```bash
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

No model files are committed to the Space repo.

## Fine-Tuned Model Swap

When the fine-tuned model is ready:

1. Upload the fine-tuned Transformers checkpoint to a Hugging Face model repo for the primary lane.
2. Optionally convert/quantize it to GGUF (+ mmproj) for the llama.cpp lane.
3. Keep the same Gradio architecture.
4. Change only:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-transformers-repo>
LLAMACPP_GGUF_REPO=<owner>/<fine-tuned-minicpm-v-gguf-repo>
LLAMACPP_MODEL_FILE=<fine-tuned-model>.gguf
LLAMACPP_MMPROJ_FILE=<mmproj-file>.gguf
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the Space deployment.

## Local Development

For UI-only work:

```bash
python app.py
```

For local extraction with Transformers (default):

```bash
pip install -r requirements.txt
python app.py
```

For local extraction with llama.cpp vision:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=llamacpp-gpu LLAMACPP_VISION=1 python app.py
```

Local machines without a suitable GPU may be slow or may not have enough memory for full model inference. In that case, test UI/report rendering locally and test extraction on the HF Space.

## Verification

Run before pushing:

```bash
python3 -m py_compile app.py src/*.py src/extraction/*.py
.venv/bin/python -m pytest tests/test_report_pipeline.py tests/test_llamacpp_gpu.py
```

Then verify the Space build uses Gradio, not Docker, and that the default backend extracts PDF/image uploads through Transformers.
