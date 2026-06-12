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

The Hugging Face Space is intentionally deployed as a **Gradio Space** with adaptive extraction.
This is the active deployment path.

With `EXTRACTOR_BACKEND=auto`, the app checks CUDA availability at runtime. If CUDA is visible, it
uses the official OpenBMB MiniCPM-V 4.6 Transformers path. If CUDA is not visible, it falls back to
the CPU `llama.cpp` GGUF path. The deterministic knowledge-graph enrichment and UI rendering stay
in normal Gradio/Python code.

This workflow should not be further changed back to Docker unless the project intentionally gives up
ZeroGPU. When the fine-tuned models are ready, only replace the model variables:

```bash
EXTRACTOR_BACKEND=auto
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
LLAMACPP_GGUF_REPO=openbmb/MiniCPM-V-4.6-gguf
LLAMACPP_MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
```

The future fine-tuned deployment should keep the same Gradio adaptive architecture and only insert
the fine-tuned Transformers repo into `ZEROGPU_MODEL_ID` and the fine-tuned GGUF repo/path into the
`LLAMACPP_*` variables.

## Local Setup

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=auto python app.py
```

The Space runtime installs both extraction lanes so `auto` can choose at runtime. The CPU llama.cpp
path uses the official prebuilt CPU manylinux wheel for `llama-cpp-python`:

```text
https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.28/llama_cpp_python-0.3.28-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

That avoids both the CUDA runtime mismatch that was causing the Space to abort on
`libcudart.so.12` and the slow source build that was timing out on Hugging Face. The CPU fallback is
PDF/text-only, so the extractor does not load an image encoder or mmproj on CPU.
