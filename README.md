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

The app uses the official OpenBMB Transformers model path for MiniCPM-V 4.6 and allocates a ZeroGPU
worker only for the extraction call through `@spaces.GPU`. The deterministic knowledge-graph
enrichment and UI rendering stay in normal Gradio/Python code.

This workflow should not be further changed back to Docker unless the project intentionally gives up
ZeroGPU. When the fine-tuned model is ready, only replace the model variable:

```bash
ZEROGPU_MODEL_ID=openbmb/MiniCPM-V-4.6
```

The future fine-tuned deployment should keep the same Gradio + ZeroGPU + Transformers architecture
and only insert the fine-tuned model repository path into `ZEROGPU_MODEL_ID`.

## Local Setup

```bash
pip install -r requirements.txt
EXTRACTOR_BACKEND=zerogpu python app.py
```
