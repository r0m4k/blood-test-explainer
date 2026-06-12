# Deployment Log

## 2026-06-13 — Route ZeroGPU to Transformers, CPU to llama.cpp

Decision: keep `EXTRACTOR_BACKEND=auto`, but make ZeroGPU select the official OpenBMB
Transformers backend instead of relying on app-level CUDA visibility.

Why:
- On ZeroGPU, CUDA is allocated only inside a `@spaces.GPU` worker, so
  `torch.cuda.is_available()` can be false in normal Gradio app code.
- The app was therefore selecting the CPU llama.cpp fallback even while the Space hardware was
  configured as ZeroGPU.
- The intended runtime behavior is now explicit: `ACCELERATOR=zero-a10g`, `ZERO_GPU=TRUE`, or
  visible CUDA selects Transformers; CPU-only runtime selects llama.cpp.

Space variables:

```bash
EXTRACTOR_BACKEND=auto
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

CPU fallback after a Transformers failure is now opt-in with:

```bash
AUTO_FALLBACK_TO_LLAMACPP=1
```

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

The app now also surfaces backend load errors directly in the UI and falls back to the
Transformers ZeroGPU backend when the llama.cpp GGUF load path fails, so a runtime mismatch does
not collapse the whole extraction flow.

We also upgraded the llama-cpp-python binding to a MiniCPM-V 4.6-capable build, which is the
actual fix for the earlier `Failed to load model from file` failure.
