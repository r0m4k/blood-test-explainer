# Deployment Log

## 2026-06-10 — Switch from Docker Space to Gradio ZeroGPU

Decision: use **Gradio ZeroGPU** as the active Hugging Face Space architecture.

Why:

- The Docker Space build failed on free CPU hardware with `OOMKilled`.
- Hugging Face ZeroGPU is available only for Gradio SDK Spaces, not Docker Spaces.
- The project needs a free dynamic GPU path for demoability.

What changed:

- `README.md` metadata changed from `sdk: docker` to `sdk: gradio`.
- `DEPLOY.md` and `RUNBOOK.md` now describe Gradio + ZeroGPU + Transformers as the active path.
- `src/extraction/zerogpu_transformers.py` adds the official OpenBMB MiniCPM-V Transformers backend.
- `src/extraction/factory.py` initially resolved `auto` to the ZeroGPU Transformers backend.
- Docker-only files (`Dockerfile`, `start.sh`, `.dockerignore`) were removed from the active deployment.

Current model:

```bash
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

Future fine-tuned model:

Only change this variable:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-model>
```

Do not reintroduce Docker or `llama-server` while the project is targeting ZeroGPU. Do not commit model files to the Space repository.

## 2026-06-11 — Add llama.cpp badge path on ZeroGPU

Decision: keep the Space as **Gradio ZeroGPU**, but target the hackathon llama.cpp badge with the
`auto` / `llamacpp-gpu` backend.

Why:

- ZeroGPU requires Gradio SDK, so Docker is still not the right deployment surface.
- The llama.cpp badge can still be targeted from a Gradio Space if inference runs through
  `llama-cpp-python` over GGUF inside `@spaces.GPU`.
- The official OpenBMB GGUF repo stays outside the Space git repo and is downloaded through the
  Hugging Face cache at runtime.

Current badge-target defaults:

```bash
EXTRACTOR_BACKEND=auto
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

Fallback if `llama-cpp-python` is incompatible with MiniCPM-V 4.6 on ZeroGPU:

```bash
EXTRACTOR_BACKEND=zerogpu
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

Future fine-tuned model:

Only change the `LLAMACPP_*` variables to point at the fine-tuned GGUF repo/files.
