from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr

from src.local_env import load_local_env
from src.openbmb_client import DEFAULT_API_URL, DEFAULT_MODEL, OpenBMBExtractor


load_local_env()

TABLE_HEADERS = [
    "marker",
    "value",
    "unit",
    "reference_range",
    "status",
    "confidence",
    "source_text",
]


def extract_lab_values(
    uploaded_file: str | None,
    api_url: str,
    model: str,
    api_key_override: str,
    max_pages: int,
) -> tuple[list[list[Any]], str, str, str]:
    if not uploaded_file:
        return [], "Upload a lab document first.", "{}", ""

    extractor = OpenBMBExtractor(
        api_url=api_url,
        model=model,
        api_key=api_key_override.strip() or None,
    )

    try:
        result = extractor.extract(uploaded_file, max_pages=int(max_pages))
    except Exception as error:
        return [], f"Extraction failed: {error}", "{}", ""

    table_rows = [[test.get(header) for header in TABLE_HEADERS] for test in result.tests]
    structured_json = json.dumps(
        {"tests": result.tests, "notes": result.notes, "request": result.request_summary},
        indent=2,
        ensure_ascii=False,
    )

    status = f"Extracted {len(result.tests)} lab values."
    if result.notes:
        status += " Notes: " + " ".join(result.notes[:3])

    return table_rows, status, structured_json, result.raw_response


def api_status() -> str:
    configured = bool(os.getenv("OPENBMB_API_KEY"))
    api_url = os.getenv("OPENBMB_API_URL") or DEFAULT_API_URL
    model = os.getenv("OPENBMB_MODEL") or DEFAULT_MODEL

    if configured:
        return f"OpenBMB API key detected. Default model: `{model}`. Endpoint: `{api_url}`."
    return (
        "OpenBMB API key is not configured. Set `OPENBMB_API_KEY` locally or add it as a "
        "Hugging Face Space secret before running extraction."
    )


CUSTOM_CSS = """
:root {
  --bte-ink: #16202a;
  --bte-muted: #5b6673;
  --bte-line: #dce4eb;
  --bte-blue: #2364aa;
  --bte-green: #1f8a70;
  --bte-paper: #f8fbfd;
}

.gradio-container {
  max-width: 1180px !important;
  color: var(--bte-ink);
}

.bte-shell {
  border: 1px solid var(--bte-line);
  border-radius: 8px;
  padding: 18px;
  background: var(--bte-paper);
}

.bte-title h1 {
  font-size: 32px !important;
  line-height: 1.12 !important;
  margin-bottom: 8px !important;
}

.bte-title p {
  color: var(--bte-muted);
  font-size: 15px;
  max-width: 780px;
}

.bte-status {
  border-left: 4px solid var(--bte-green);
  padding: 10px 12px;
  background: #ffffff;
}

button.primary {
  background: var(--bte-blue) !important;
}
"""


with gr.Blocks(title="Blood Test Explainer") as demo:
    gr.HTML(
        """
        <section class="bte-title">
          <h1>Blood Test Explainer</h1>
          <p>Upload a lab report and extract the raw markers, values, units, ranges, and flags into a structured table. This first version performs extraction only.</p>
        </section>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["bte-shell"]):
                uploaded = gr.File(
                    label="Medical test document",
                    file_count="single",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".txt", ".csv"],
                    type="filepath",
                )
                run_button = gr.Button("Extract lab values", variant="primary")

        with gr.Column(scale=3, min_width=300):
            with gr.Group(elem_classes=["bte-shell"]):
                gr.Markdown("### Extraction backend")
                api_url = gr.Textbox(
                    label="OpenBMB chat completions URL",
                    value=os.getenv("OPENBMB_API_URL") or DEFAULT_API_URL,
                    interactive=True,
                )
                model = gr.Textbox(
                    label="Model",
                    value=os.getenv("OPENBMB_MODEL") or DEFAULT_MODEL,
                    interactive=True,
                )
                api_key_override = gr.Textbox(
                    label="OpenBMB API key override",
                    type="password",
                    placeholder="Optional. Paste exact token here for local testing.",
                    interactive=True,
                )
                max_pages = gr.Slider(1, 8, value=3, step=1, label="PDF pages to inspect")
                gr.Markdown(api_status(), elem_classes=["bte-status"])

    status = gr.Markdown("Ready.")
    results = gr.Dataframe(
        headers=TABLE_HEADERS,
        datatype=["str", "str", "str", "str", "str", "number", "str"],
        row_count=(0, "dynamic"),
        column_count=(len(TABLE_HEADERS), "fixed"),
        label="Extracted values",
        wrap=True,
    )

    with gr.Accordion("Structured JSON", open=True):
        structured_json = gr.Code(language="json", label="Normalized extraction")

    with gr.Accordion("Raw model response", open=False):
        raw_response = gr.Textbox(label="Raw response", lines=12)

    run_button.click(
        extract_lab_values,
        inputs=[uploaded, api_url, model, api_key_override, max_pages],
        outputs=[results, status, structured_json, raw_response],
    )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
