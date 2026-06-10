# Deploying the Space with ZeroGPU

The active Hugging Face deployment is a **Gradio ZeroGPU Space**.

This workflow is intentionally fixed:

1. The Space must stay a Gradio Space, not a Docker Space.
2. Runtime extraction must use the official OpenBMB MiniCPM-V Transformers path.
3. The extraction call must run behind `@spaces.GPU` so Hugging Face allocates ZeroGPU only while the model is needed.
4. Model files must not be committed to the Space git repo.
5. When the fine-tuned model is ready, only replace the model repo path in `ZEROGPU_MODEL_ID`.

Do not change this architecture unless the project intentionally gives up ZeroGPU. The only intended future model-serving change is inserting the fine-tuned model repository path into `ZEROGPU_MODEL_ID`.

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

The current model is the official OpenBMB Transformers model:

```text
openbmb/MiniCPM-V-4.6
```

The backend lives in:

```text
src/extraction/zerogpu_transformers.py
```

It uses:

```python
AutoProcessor.from_pretrained(model_id)
AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
@spaces.GPU(duration=120)
```

The app selects this backend by default through:

```text
EXTRACTOR_BACKEND=zerogpu
```

or through `auto`, which resolves to the same ZeroGPU backend.

## 3. Future Fine-Tuned Model

When the fine-tuned model is ready:

1. Upload the fine-tuned model to a Hugging Face model repo.
2. Keep the same Gradio + ZeroGPU + Transformers architecture.
3. Change only this variable:

```bash
ZEROGPU_MODEL_ID=<owner>/<fine-tuned-minicpm-v-model>
```

Do not add model files to the Space git repo. Do not reintroduce Docker or `llama-server` for the ZeroGPU deployment.

## 4. Why This Architecture

The Docker path failed on free CPU hardware with `OOMKilled` during build. ZeroGPU is only available for Gradio SDK Spaces, so the deployment must be a Gradio Space to use the free dynamic GPU resource.

This architecture keeps:

- Free ZeroGPU eligibility.
- No external hosted inference API calls.
- Official OpenBMB model loading through Transformers.
- A clean future swap to a fine-tuned model by changing only `ZEROGPU_MODEL_ID`.

## 5. Local Development

Local development can run the same backend, although the model may be slow or too large without a local GPU:

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=zerogpu python app.py
```

For quick UI-only work, continue using the static reference report without triggering extraction.
