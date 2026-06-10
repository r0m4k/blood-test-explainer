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
- `src/extraction/factory.py` now resolves `auto` to the ZeroGPU backend.
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
