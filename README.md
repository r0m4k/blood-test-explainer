---
title: Blood Test Explainer
emoji: 📊
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 6.17.3
python_version: '3.13'
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

Local setup:

```bash
pip install -r requirements.txt
export OPENBMB_API_KEY="your-openbmb-token"
python app.py
```

You can also create a local `.env` file:

```text
OPENBMB_API_KEY=your-openbmb-token
OPENBMB_API_URL=http://35.203.155.71:8003/v1/chat/completions
OPENBMB_MODEL=MiniCPM-V-4.6
```

Hugging Face Space setup:

- Add `OPENBMB_API_KEY` as a Space secret.
- Optionally set `OPENBMB_API_URL` and `OPENBMB_MODEL` as Space variables.
- Default endpoint is MiniCPM-V 4.6 through an OpenAI-compatible chat completions API.

This API-backed extractor is temporary. The extraction layer is isolated so it can later be replaced by a local fine-tuned OpenBMB model running through `llama.cpp`.
