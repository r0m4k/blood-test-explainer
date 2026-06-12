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

## Hugging Face Space Deployment

The Hugging Face Space is intentionally deployed as a **Gradio ZeroGPU Space**. This is the active
deployment path.

The badge-target backend runs the official OpenBMB MiniCPM-V 4.6 GGUF through `llama.cpp`
inside a ZeroGPU-managed function, using a `llama-cpp-python` build that includes MiniCPM-V 4.6
mtmd support. The deterministic knowledge-graph
enrichment and UI rendering stay in normal Gradio/Python code.

This workflow should not be further changed back to Docker unless the project intentionally gives up
ZeroGPU. When the fine-tuned GGUF model is ready, only replace the model variables:

```bash
EXTRACTOR_BACKEND=auto
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
LLAMACPP_MMPROJ_FILE=mmproj-model-f16.gguf
```

The future fine-tuned deployment should keep the same Gradio + ZeroGPU + `llama.cpp` architecture
and only insert the fine-tuned GGUF repository/path into the `LLAMACPP_*` variables. If
`llama-cpp-python` proves incompatible with MiniCPM-V 4.6 on ZeroGPU, the fallback backend is
`EXTRACTOR_BACKEND=zerogpu` with `ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6`.

## Local Setup

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=auto python app.py
```

The Space runtime is intentionally slim. The Transformers fallback stack is kept out of the
default Space dependency set so the build stays faster and more deterministic. The active
llama.cpp path uses the official prebuilt CPU manylinux wheel for `llama-cpp-python`:

```text
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28/llama_cpp_python-0.3.28-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

That avoids both the CUDA runtime mismatch that was causing the Space to abort on
`libcudart.so.12` and the slow source build that was timing out on Hugging Face. Install any
extra backend-specific packages only when you actually need to run that backend locally.
