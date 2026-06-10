# Runbook — ZeroGPU Extraction + Fine-Tuned Model Swap

The active deployment path is now **Gradio ZeroGPU**.

This replaced the Docker + `llama-server` path because ZeroGPU is only available for Gradio SDK Spaces. The Docker build was also failing on free CPU hardware with `OOMKilled`.

## Active Architecture

| Area | Current choice |
|---|---|
| Space SDK | `gradio` |
| Hardware | ZeroGPU |
| Model runtime | Official OpenBMB Transformers path |
| Default backend | `EXTRACTOR_BACKEND=auto`, resolving to `zerogpu` |
| Model variable | `ZEROGPU_MODEL_ID` |
| Extraction backend | `src/extraction/zerogpu_transformers.py` |
| Report enrichment | `src/report_pipeline.py` + `kb/cbc_knowledge_graph.json` |

Do not switch the Space back to Docker unless the project intentionally gives up ZeroGPU.

## Backend Selection

`EXTRACTOR_BACKEND`:

- `auto`: default, uses ZeroGPU Transformers.
- `zerogpu`: force the ZeroGPU Transformers backend.
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
transformers[torch]>=5.7.0
torch
torchvision
av
accelerate
```

The backend uses `@spaces.GPU(duration=120)` for the model generation call.

## Current Model

Current default:

```bash
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

This is the official OpenBMB model. No model files are committed to the Space repo.

## Fine-Tuned Model Swap

When the fine-tuned model is ready:

1. Upload it to a Hugging Face model repo.
2. Keep the same Gradio + ZeroGPU + Transformers architecture.
3. Change only:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-model>
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
EXTRACTOR_BACKEND=zerogpu python app.py
```

Local machines without a suitable GPU may be slow or may not have enough memory for full model inference. In that case, test UI/report rendering locally and test extraction on the HF ZeroGPU Space.

## Verification

Run before pushing:

```bash
python3 -m py_compile app.py src/*.py src/extraction/*.py
.venv/bin/python -m pytest tests/test_report_pipeline.py
```

Then verify the Space build uses Gradio, not Docker, and that ZeroGPU can be selected in the Space hardware panel.
