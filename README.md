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
startup_duration_timeout: 1h
---

# Blood Test Explainer

Blood test results often arrive as dense PDFs, scans, photos, or lab documents filled with abbreviations, reference ranges, units, and flags. For many people, the result is anxiety rather than understanding: they can see that something is high or low, but they do not know what it means, what questions to ask, or what practical next steps might support better health.

Blood Test Explainer turns an uploaded blood test into a clear, interactive health dashboard. The goal is to extract the important markers, organize them into a readable visual experience, explain each result in plain language, and help the user prepare for a better conversation with a clinician.

The project focuses on education and personal clarity, not diagnosis. It should help people understand their lab report, notice which markers may deserve attention, and explore general lifestyle ideas such as food, movement, sleep, and supplement topics that may be worth discussing with a qualified professional.

The final experience should feel calm, trustworthy, and useful:

- Upload a blood test document, image, scan, or PDF.
- See extracted markers, values, units, and reference ranges.
- Review results in a polished interactive interface.
- Understand what each marker generally reflects.
- Get practical lifestyle-oriented suggestions for supporting specific markers.
- Generate thoughtful questions to bring to a doctor or healthcare provider.

The long-term vision is to make medical paperwork less intimidating and help people move from confusion to informed action.

## First App Version

The first version focuses only on extraction: upload a lab report and convert it into structured raw values such as marker name, value, unit, reference range, status, source snippet, and confidence.

## Current Pipeline

The app now runs extraction and deterministic knowledge-graph enrichment:

1. The extractor reads an uploaded image, PDF, or text document and returns patient context plus raw lab values.
2. `src.report_pipeline.build_health_report` resolves marker aliases against `kb/cbc_knowledge_graph.json`, selects age/sex-aware reference context, and merges marker explanations, importance, and food/exercise/supplement guidance.
3. `app.py` renders the enriched report as the final health-report UI.

The knowledge graph is educational context, not diagnosis. The lab-provided reference range remains the primary comparison when it is available.

## Extraction Backends

The default path is **Transformers vision** (OpenBMB MiniCPM-V 4.6). It handles PDFs, scans, and photos through the same document pipeline in `src/document_processing.py`.

| `EXTRACTOR_BACKEND` | Used for | PDF / image uploads |
|---|---|---|
| `transformers` (default), `auto`, `zerogpu` | Normal app + HF Space | Yes |
| `llamacpp-gpu` + `LLAMACPP_VISION=1` | Opt-in llama.cpp vision lane | Yes |
| `llamacpp-gpu` (vision off) | Text-only GGUF lane | No — `.txt` / `.csv` only |
| `local` / `server` | Local `llama-server` experiments | Yes |
| `llamacpp` | Local in-process GGUF + mmproj | Yes |

Backend selection lives in `src/extraction/factory.py`.

## Running with llama.cpp

The app does **not** use llama.cpp by default. Enable it only when you need the optional GGUF lane.

### Why keep llama.cpp?

1. **Hackathon badge** — the project can target the **Llama Champion** badge by running inference through `llama-cpp-python` over GGUF inside `@spaces.GPU`.
2. **Fine-tuned GGUF swap** — after fine-tuning, you can point `LLAMACPP_*` at a quantized GGUF repo without changing the Gradio app.
3. **Lighter text-only lane** — without `LLAMACPP_VISION=1`, the llama.cpp path skips mmproj and works for plain-text lab exports (`.txt` / `.csv`).
4. **Local offline experiments** — `EXTRACTOR_BACKEND=local` (external `llama-server`) or `llamacpp` (in-process GGUF + mmproj) for off-grid development.

For normal PDF/image blood-test uploads, keep the default Transformers backend.

### Vision llama.cpp (PDFs and images)

Use the same vision document pipeline as Transformers, but route inference through llama.cpp:

```bash
pip install -r requirements.txt

export EXTRACTOR_BACKEND=llamacpp-gpu
export LLAMACPP_VISION=1
export LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
export LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
export LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
export LLAMACPP_CHAT_HANDLER=MiniCPMv26ChatHandler   # override if your wheel needs a different handler

python app.py
```

On Hugging Face Spaces, set the same variables in **Settings → Repository secrets / Variables**, then restart the Space. Generation runs inside `@spaces.GPU` in `src/extraction/llamacpp_gpu.py`.

### Text-only llama.cpp (no vision)

For `.txt` / `.csv` uploads only:

```bash
export EXTRACTOR_BACKEND=llamacpp-gpu
export LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
export LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf

python app.py
```

PDF or image uploads fail with a clear error unless `LLAMACPP_VISION=1` is set.

### Local llama-server (advanced)

When pip `llama-cpp-python` is too old for MiniCPM-V 4.6 vision, run a separate server:

```bash
llama-server -m model.gguf --mmproj mmproj.gguf --port 8080
EXTRACTOR_BACKEND=local python app.py
```

See `src/extraction/local_server.py`.

## Hugging Face Space Deployment

The Hugging Face Space is deployed as a **Gradio Space** with Transformers extraction on ZeroGPU.

Default Space variables:

```bash
EXTRACTOR_BACKEND=transformers
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

Optional llama.cpp badge lane (not enabled in the default deployment):

```bash
EXTRACTOR_BACKEND=llamacpp-gpu
LLAMACPP_VISION=1
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

When the fine-tuned models are ready, replace `ZEROGPU_MODEL_ID` for the primary lane and the `LLAMACPP_*` variables for the optional GGUF lane. Do not commit model files to the Space git repo.

This workflow should not be changed back to Docker unless the project intentionally gives up ZeroGPU.

## Local Setup

Default (Transformers vision):

```bash
pip install -r requirements.txt
python app.py
```

Explicit backend:

```bash
EXTRACTOR_BACKEND=transformers python app.py
```

The Space runtime also installs `llama-cpp-python` so the optional llama.cpp lane can be enabled without code changes. On Linux x86_64 Spaces it uses the prebuilt CPU manylinux wheel:

```text
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28/llama_cpp_python-0.3.28-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

That avoids both the CUDA runtime mismatch that was causing the Space to abort on `libcudart.so.12` and the slow source build that was timing out on Hugging Face.
