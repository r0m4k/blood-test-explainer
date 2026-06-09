from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import gradio as gr

from src.local_env import load_local_env
from src.extraction import build_extractor
from src.openbmb_client import DEFAULT_API_URL, DEFAULT_MODEL


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
) -> tuple[list[list[Any]], str, str, str, str, Any]:
    if not uploaded_file:
        return (
            [],
            _status_html("Waiting for a document", "Upload a lab report to begin extraction."),
            "{}",
            "",
            empty_report_html("No document uploaded", "Choose a file first, then run extraction again."),
            gr.update(visible=True),
        )

    # Backend chosen by EXTRACTOR_BACKEND (auto|local|api). Local = offline MiniCPM-V GGUF;
    # api = the hosted OpenBMB endpoint (dev fallback). Defaults to auto.
    extractor = build_extractor(
        api_url=api_url,
        model=model,
        api_key=(api_key_override or "").strip() or None,
    )

    try:
        result = extractor.extract(uploaded_file, max_pages=int(max_pages))
    except Exception as error:
        return (
            [],
            _status_html("Extraction failed", str(error), tone="danger"),
            "{}",
            "",
            empty_report_html("Extraction failed", "The model response could not be converted into a report."),
            gr.update(visible=True),
        )

    table_rows = [[test.get(header) for header in TABLE_HEADERS] for test in result.tests]
    structured_json = json.dumps(
        {"tests": result.tests, "notes": result.notes, "request": result.request_summary},
        indent=2,
        ensure_ascii=False,
    )

    status_text = f"Extracted {len(result.tests)} lab values."
    if result.notes:
        status_text += " Notes: " + " ".join(result.notes[:3])

    return (
        table_rows,
        _status_html("Extraction complete", status_text),
        structured_json,
        result.raw_response,
        report_html(result.tests, result.notes),
        gr.update(visible=True),
    )


def api_status() -> str:
    configured = bool(os.getenv("OPENBMB_API_KEY"))
    api_url = os.getenv("OPENBMB_API_URL") or DEFAULT_API_URL
    model = os.getenv("OPENBMB_MODEL") or DEFAULT_MODEL

    if configured:
        return f"<strong>OpenBMB key detected.</strong><span>{escape(model)} · {escape(api_url)}</span>"
    return (
        "<strong>OpenBMB key missing.</strong><span>Set <code>OPENBMB_API_KEY</code> locally or as a Space secret.</span>"
    )


def _status_html(title: str, detail: str, tone: str = "success") -> str:
    return f"""
    <div class="bte-run-status bte-run-status--{escape(tone)}">
      <strong>{escape(title)}</strong>
      <span>{escape(detail)}</span>
    </div>
    """


def show_processing() -> tuple[str, Any, str]:
    return (
        _status_html("Reading document", "Extracting markers, values, units, ranges, and confidence signals.", tone="loading"),
        gr.update(visible=True),
        loading_report_html(),
    )


def upload_state(uploaded_file: str | None) -> tuple[Any, Any]:
    if not uploaded_file:
        return (
            gr.update(visible=True),
            gr.update(visible=False, value=selected_document_html()),
        )

    filename = os.path.basename(uploaded_file)
    return (
        gr.update(visible=False),
        gr.update(visible=True, value=selected_document_html(filename)),
    )


def selected_document_html(filename: str | None = None) -> str:
    if not filename:
        filename = "Document ready"
    return f"""
    <section class="bte-selected-document">
      <div class="bte-selected-icon" aria-hidden="true">
        <span></span>
      </div>
      <div>
        <p class="bte-kicker">Document loaded</p>
        <h3>{escape(filename)}</h3>
        <p>Ready to extract markers, values, units, ranges, and confidence signals.</p>
      </div>
    </section>
    """


def empty_report_html(
    title: str = "Report preview",
    detail: str = "Extracted lab markers will render here as an interactive medical document.",
) -> str:
    return f"""
    <section class="bte-report bte-report--empty">
      <div>
        <p class="bte-kicker">Interactive document draft</p>
        <h2>{escape(title)}</h2>
        <p>{escape(detail)}</p>
      </div>
    </section>
    """


def loading_report_html() -> str:
    return """
    <section class="bte-report bte-loading-report">
      <div class="bte-loader-orbit" aria-hidden="true">
        <span></span>
        <i></i>
      </div>
      <div class="bte-loading-copy">
        <p class="bte-kicker">Extraction in progress</p>
        <h2>Reading your test results</h2>
        <p>The model is locating markers, values, units, reference ranges, and status flags. The full report will appear here when extraction is complete.</p>
      </div>
      <div class="bte-loading-stack" aria-hidden="true">
        <div><span></span><strong></strong></div>
        <div><span></span><strong></strong></div>
        <div><span></span><strong></strong></div>
      </div>
    </section>
    """


def formation_animation_html() -> str:
    return """
    <section class="bte-formation" aria-label="Responsive document formation animation">
      <div class="bte-formation-stage">
        <div class="bte-source-doc">
          <div class="bte-doc-top">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="bte-doc-line bte-doc-line--wide"></div>
          <div class="bte-doc-line"></div>
          <div class="bte-doc-line bte-doc-line--short"></div>
          <div class="bte-doc-table">
            <span></span><span></span><span></span>
            <span></span><span></span><span></span>
            <span></span><span></span><span></span>
          </div>
          <div class="bte-scan-band"></div>
        </div>

        <div class="bte-flow">
          <span></span>
          <i></i>
          <b></b>
        </div>

        <div class="bte-smart-report">
          <div class="bte-report-window">
            <div class="bte-report-header">
              <strong>12 markers</strong>
              <small>ready to review</small>
            </div>
            <div class="bte-mini-card bte-mini-card--green">
              <span>Hemoglobin</span>
              <strong>Normal</strong>
            </div>
            <div class="bte-mini-card bte-mini-card--amber">
              <span>Vitamin D</span>
              <strong>Low</strong>
            </div>
            <div class="bte-mini-chart">
              <span style="height: 34%"></span>
              <span style="height: 56%"></span>
              <span style="height: 42%"></span>
              <span style="height: 74%"></span>
              <span style="height: 61%"></span>
            </div>
          </div>
        </div>
      </div>
    </section>
    """


def report_html(tests: list[dict[str, Any]], notes: list[str]) -> str:
    total = len(tests)
    high = _count_status(tests, "high")
    low = _count_status(tests, "low")
    abnormal = _count_status(tests, "abnormal")
    normal = _count_status(tests, "normal")
    needs_review = high + low + abnormal

    cards = "\n".join(_marker_card(test) for test in tests) or _empty_marker_card()
    notes_html = "".join(f"<li>{escape(note)}</li>" for note in notes[:6])
    notes_block = f"<ul>{notes_html}</ul>" if notes_html else "<p>No extraction notes returned.</p>"

    return f"""
    <section class="bte-report">
      <header class="bte-report-hero">
        <div>
          <p class="bte-kicker">Lab extraction report</p>
          <h2>{total} markers found</h2>
          <p>Review the extracted values before using them for interpretation. This draft is an extraction view only.</p>
        </div>
        <div class="bte-score">
          <span>{needs_review}</span>
          <small>need review</small>
        </div>
      </header>

      <div class="bte-metrics">
        {_metric("Total", total, "All extracted markers")}
        {_metric("Review", needs_review, "High, low, or abnormal")}
        {_metric("Normal", normal, "Marked normal")}
        {_metric("Unknown", max(total - needs_review - normal, 0), "Needs confirmation")}
      </div>

      <div class="bte-status-strip">
        {_pill("High", high, "high")}
        {_pill("Low", low, "low")}
        {_pill("Abnormal", abnormal, "abnormal")}
        {_pill("Normal", normal, "normal")}
      </div>

      <div class="bte-report-grid">
        <section class="bte-marker-list">
          {cards}
        </section>
        <aside class="bte-report-aside">
          <h3>Extraction notes</h3>
          {notes_block}
          <div class="bte-disclaimer">
            <strong>Draft only</strong>
            <span>These are raw extracted values, not medical advice. Confirm values against the original document.</span>
          </div>
        </aside>
      </div>
    </section>
    """


def _count_status(tests: list[dict[str, Any]], status: str) -> int:
    return sum(1 for test in tests if str(test.get("status") or "").lower() == status)


def _metric(label: str, value: int, caption: str) -> str:
    return f"""
    <div class="bte-metric">
      <span>{escape(str(value))}</span>
      <strong>{escape(label)}</strong>
      <small>{escape(caption)}</small>
    </div>
    """


def _pill(label: str, value: int, tone: str) -> str:
    return f'<span class="bte-pill bte-pill--{escape(tone)}">{escape(label)} <strong>{escape(str(value))}</strong></span>'


def _marker_card(test: dict[str, Any]) -> str:
    marker = _text(test.get("marker"), "Unknown marker")
    value = _text(test.get("value"), "-")
    unit = _text(test.get("unit"), "")
    reference = _text(test.get("reference_range"), "Reference range not extracted")
    status = _text(test.get("status"), "unknown").lower()
    source = _text(test.get("source_text"), "No source snippet returned.")
    confidence = _confidence_percent(test.get("confidence"))

    return f"""
    <details class="bte-marker bte-marker--{escape(status)}" open>
      <summary>
        <span class="bte-marker-main">
          <strong>{escape(marker)}</strong>
          <small>{escape(reference)}</small>
        </span>
        <span class="bte-marker-value">
          <strong>{escape(value)}</strong>
          <small>{escape(unit)}</small>
        </span>
        <span class="bte-marker-status">{escape(status.title())}</span>
      </summary>
      <div class="bte-marker-body">
        <div>
          <span>Confidence</span>
          <div class="bte-confidence"><i style="width: {confidence}%"></i></div>
          <small>{confidence}%</small>
        </div>
        <blockquote>{escape(source)}</blockquote>
      </div>
    </details>
    """


def _empty_marker_card() -> str:
    return """
    <div class="bte-marker-empty">
      No markers extracted yet.
    </div>
    """


def _text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _confidence_percent(value: Any) -> int:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, round(score * 100)))


CUSTOM_CSS = """
:root {
  color-scheme: light;
  --bte-ink: #111827;
  --bte-muted: #65707f;
  --bte-soft: #8b97a7;
  --bte-line: #e5eaf0;
  --bte-blue: #2563eb;
  --bte-blue-soft: #eff6ff;
  --bte-green: #12805c;
  --bte-green-soft: #eaf8f2;
  --bte-red: #bf3434;
  --bte-red-soft: #fff1f1;
  --bte-amber: #9a6700;
  --bte-amber-soft: #fff7df;
  --bte-paper: #f7f9fc;
  --bte-card: rgba(255, 255, 255, 0.92);
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

body,
gradio-app,
.gradio-container,
.dark,
.dark .gradio-container {
  color-scheme: light !important;
  --body-background-fill: #f7f9fc !important;
  --body-text-color: #111827 !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f7f9fc !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: #e5eaf0 !important;
  --block-border-width: 1px !important;
  --block-info-text-color: #65707f !important;
  --block-label-background-fill: #ffffff !important;
  --block-label-text-color: #111827 !important;
  --block-title-text-color: #111827 !important;
  --border-color-primary: #e5eaf0 !important;
  --border-color-accent: #2563eb !important;
  --input-background-fill: #ffffff !important;
  --input-border-color: #d8e2ee !important;
  --input-placeholder-color: #8b97a7 !important;
  --input-shadow: none !important;
  --button-primary-background-fill: #111827 !important;
  --button-primary-background-fill-hover: #263244 !important;
  --button-primary-text-color: #ffffff !important;
  --slider-color: #2563eb !important;
  --shadow-drop: 0 18px 48px rgba(17, 24, 39, 0.06) !important;
}

.gradio-container {
  max-width: 1240px !important;
  width: 100% !important;
  margin: 0 auto !important;
  box-sizing: border-box !important;
  color: var(--bte-ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  background:
    radial-gradient(circle at 20% 0%, rgba(37, 99, 235, 0.08), transparent 34%),
    linear-gradient(180deg, #fbfdff 0%, #f4f7fb 100%);
}

.gradio-container,
.gradio-container * {
  letter-spacing: 0 !important;
}

.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
  background: transparent !important;
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container span {
  color: inherit;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container input,
.gradio-container textarea {
  color: var(--bte-ink) !important;
}

.gradio-container input,
.gradio-container textarea {
  background: #ffffff !important;
  border-color: var(--bte-line) !important;
  border-radius: 12px !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .input-container,
.gradio-container fieldset {
  background: #ffffff !important;
  border-color: var(--bte-line) !important;
}

.gradio-container .form {
  border: 0 !important;
}

.gradio-container .block {
  box-shadow: none !important;
}

.bte-shell {
  border: 1px solid var(--bte-line);
  border-radius: 22px;
  padding: 18px;
  background: var(--bte-card);
  box-shadow: 0 18px 48px rgba(17, 24, 39, 0.06);
}

.bte-engine {
  padding: 18px 18px 20px;
}

.bte-engine-compact {
  margin-top: 8px;
  padding: 16px;
  box-shadow: none;
}

.bte-shell > div,
.bte-shell > div > div {
  background: transparent !important;
}

.bte-shell h3,
.bte-shell h3 * {
  color: var(--bte-ink) !important;
  font-size: 18px !important;
  line-height: 1.15 !important;
  margin-bottom: 10px !important;
}

.bte-title {
  padding: 26px 12px 18px;
}

.bte-title h1,
.bte-report h2 {
  font-size: clamp(34px, 4vw, 42px) !important;
  line-height: 1.12 !important;
  margin-bottom: 8px !important;
  letter-spacing: 0 !important;
  color: var(--bte-ink) !important;
}

.bte-title p {
  color: var(--bte-muted);
  font-size: 16px;
  max-width: 820px;
  margin: 0;
}

.bte-hero-grid {
  align-items: stretch !important;
}

.bte-upload-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.bte-formation {
  height: 100%;
  min-height: 390px;
  border: 1px solid var(--bte-line);
  border-radius: 24px;
  padding: 22px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(247, 251, 255, 0.86)),
    #ffffff;
  box-shadow: 0 22px 70px rgba(17, 24, 39, 0.08);
  overflow: hidden;
}

.bte-formation-stage {
  height: 100%;
  min-height: 342px;
  display: grid;
  grid-template-columns: minmax(138px, 0.95fr) 74px minmax(176px, 1.12fr);
  align-items: center;
  gap: 14px;
}

.bte-source-doc,
.bte-report-window {
  position: relative;
  border: 1px solid rgba(216, 226, 238, 0.95);
  border-radius: 18px;
  background: #ffffff;
  box-shadow: 0 20px 48px rgba(17, 24, 39, 0.1);
}

.bte-source-doc {
  min-height: 280px;
  padding: 18px;
  overflow: hidden;
  animation: bte-doc-float 5.6s ease-in-out infinite;
}

.bte-doc-top {
  display: flex;
  gap: 6px;
  margin-bottom: 18px;
}

.bte-doc-top span {
  width: 10px;
  aspect-ratio: 1;
  border-radius: 999px;
  background: #d8e2ee;
}

.bte-doc-top span:nth-child(2) {
  background: #91d7c0;
}

.bte-doc-top span:nth-child(3) {
  background: #9cbcff;
}

.bte-doc-line {
  height: 11px;
  width: 76%;
  border-radius: 999px;
  margin-bottom: 10px;
  background: #e7edf5;
}

.bte-doc-line--wide {
  width: 92%;
  height: 15px;
  background: #dce8fb;
}

.bte-doc-line--short {
  width: 54%;
}

.bte-doc-table {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 22px;
}

.bte-doc-table span {
  min-height: 28px;
  border-radius: 9px;
  background: #f0f4f8;
  border: 1px solid #e2eaf3;
}

.bte-scan-band {
  position: absolute;
  left: -20%;
  right: -20%;
  top: 22%;
  height: 34px;
  transform: rotate(-6deg);
  background: linear-gradient(90deg, transparent, rgba(18, 128, 92, 0.18), rgba(37, 99, 235, 0.16), transparent);
  animation: bte-scan-doc 2.9s ease-in-out infinite;
}

.bte-flow {
  display: grid;
  justify-items: center;
  align-items: center;
  gap: 10px;
}

.bte-flow span {
  width: 100%;
  height: 2px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(37, 99, 235, 0), rgba(37, 99, 235, 0.8), rgba(18, 128, 92, 0));
  animation: bte-flow-line 1.8s ease-in-out infinite;
}

.bte-flow i,
.bte-flow b {
  display: block;
  width: 34px;
  aspect-ratio: 1;
  border-radius: 12px;
  border: 1px solid #dbeafe;
  background: #ffffff;
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.12);
  animation: bte-flow-tile 2.4s ease-in-out infinite;
}

.bte-flow b {
  width: 22px;
  border-color: #cfeee3;
  animation-delay: 0.3s;
}

.bte-smart-report {
  animation: bte-report-rise 5.6s ease-in-out infinite;
}

.bte-report-window {
  min-height: 302px;
  padding: 16px;
}

.bte-report-header {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;
}

.bte-report-header strong {
  font-size: 22px;
  color: var(--bte-ink);
}

.bte-report-header small {
  color: var(--bte-muted);
}

.bte-mini-card {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  border-radius: 14px;
  padding: 12px;
  margin-bottom: 10px;
  border: 1px solid var(--bte-line);
  background: #fbfdff;
  animation: bte-card-pop 4.4s ease-in-out infinite;
}

.bte-mini-card--green {
  border-color: rgba(18, 128, 92, 0.2);
  background: var(--bte-green-soft);
}

.bte-mini-card--amber {
  border-color: rgba(154, 103, 0, 0.2);
  background: var(--bte-amber-soft);
  animation-delay: 0.35s;
}

.bte-mini-card span {
  color: var(--bte-muted);
  font-size: 13px;
}

.bte-mini-card strong {
  color: var(--bte-ink);
  font-size: 13px;
}

.bte-mini-chart {
  height: 116px;
  display: flex;
  align-items: end;
  gap: 10px;
  padding: 14px;
  border: 1px solid var(--bte-line);
  border-radius: 16px;
  background: #ffffff;
}

.bte-mini-chart span {
  flex: 1;
  min-width: 10px;
  border-radius: 999px 999px 4px 4px;
  background: linear-gradient(180deg, #2563eb, #12805c);
  animation: bte-bar-grow 2.6s ease-in-out infinite;
}

.bte-mini-chart span:nth-child(2) { animation-delay: 0.12s; }
.bte-mini-chart span:nth-child(3) { animation-delay: 0.24s; }
.bte-mini-chart span:nth-child(4) { animation-delay: 0.36s; }
.bte-mini-chart span:nth-child(5) { animation-delay: 0.48s; }

.bte-kicker {
  color: var(--bte-blue);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0 !important;
  margin: 0 0 8px;
  text-transform: uppercase;
}

.bte-status,
.bte-run-status {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid var(--bte-line);
  border-radius: 14px;
  padding: 12px 14px;
  background: #ffffff;
  color: var(--bte-muted);
  font-size: 13px;
}

.bte-status span,
.bte-run-status span {
  color: var(--bte-muted);
}

.bte-status code {
  color: var(--bte-blue);
}

.bte-status strong,
.bte-run-status strong {
  color: var(--bte-ink);
}

.bte-run-status--success {
  border-color: rgba(18, 128, 92, 0.22);
  background: var(--bte-green-soft);
}

.bte-run-status--danger {
  border-color: rgba(191, 52, 52, 0.24);
  background: var(--bte-red-soft);
}

.bte-run-status--loading {
  border-color: rgba(37, 99, 235, 0.22);
  background: var(--bte-blue-soft);
}

button.primary {
  background: #111827 !important;
  border-radius: 14px !important;
  min-height: 44px !important;
  border: 0 !important;
  box-shadow: 0 12px 26px rgba(17, 24, 39, 0.2) !important;
  font-size: 0 !important;
}

button.primary::after {
  content: "Extract test results";
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 16px !important;
  font-weight: 720 !important;
}

.bte-action {
  margin-top: 12px !important;
}

.bte-uploader {
  border: 0 !important;
}

.bte-shell .file-preview,
.bte-shell [data-testid="file"],
.bte-shell .upload-container,
.bte-shell .file-drop,
.bte-shell .file-dropzone,
.bte-shell .file-container,
.bte-shell .dropzone,
.bte-shell [class*="drop"],
.bte-shell [class*="upload"] {
  background: linear-gradient(180deg, #fbfdff, #f4f7fb) !important;
  border-color: #d8e2ee !important;
  border-radius: 18px !important;
  color: var(--bte-ink) !important;
}

.bte-uploader [class*="drop"],
.bte-uploader [class*="upload"] {
  min-height: 220px !important;
}

.bte-selected-document {
  display: grid;
  grid-template-columns: 68px minmax(0, 1fr);
  gap: 16px;
  align-items: center;
  min-height: 220px;
  border: 1px solid #d8e2ee;
  border-radius: 18px;
  padding: 22px;
  background:
    linear-gradient(180deg, rgba(234, 248, 242, 0.78), rgba(255, 255, 255, 0.94)),
    #ffffff;
}

.bte-selected-document h3 {
  margin: 0 0 6px !important;
  color: var(--bte-ink) !important;
  font-size: 22px !important;
  line-height: 1.15 !important;
  overflow-wrap: anywhere;
}

.bte-selected-document p:last-child {
  margin: 0;
  color: var(--bte-muted);
  font-size: 14px;
}

.bte-selected-icon {
  width: 68px;
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border-radius: 18px;
  border: 1px solid rgba(18, 128, 92, 0.22);
  background: #ffffff;
  box-shadow: 0 14px 28px rgba(18, 128, 92, 0.12);
}

.bte-selected-icon span {
  width: 30px;
  aspect-ratio: 0.78;
  display: block;
  border-radius: 7px;
  border: 2px solid var(--bte-green);
  position: relative;
}

.bte-selected-icon span::after {
  content: "";
  position: absolute;
  width: 15px;
  height: 8px;
  left: 7px;
  top: 9px;
  border-left: 2px solid var(--bte-green);
  border-bottom: 2px solid var(--bte-green);
  transform: rotate(-45deg);
}

.bte-shell [class*="drop"] *,
.bte-shell [class*="upload"] * {
  color: var(--bte-ink) !important;
  -webkit-text-fill-color: var(--bte-ink) !important;
}

.bte-shell [class*="drop"] {
  border-style: dashed !important;
  border-width: 2px !important;
}

.bte-shell svg,
.bte-shell .icon-wrap {
  color: var(--bte-blue) !important;
}

button.bte-action,
button.bte-action *,
.bte-action button,
.bte-action button * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.bte-uploader button,
.bte-uploader button *,
.bte-uploader [class*="drop"] *,
.bte-uploader [class*="upload"] * {
  color: var(--bte-ink) !important;
  -webkit-text-fill-color: var(--bte-ink) !important;
}

.bte-upload-card button.bte-action,
.bte-upload-card button.bte-action *,
.bte-upload-card .bte-action,
.bte-upload-card .bte-action * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.gradio-container details {
  border-color: var(--bte-line) !important;
  border-radius: 14px !important;
  background: rgba(255, 255, 255, 0.86) !important;
  box-shadow: none !important;
}

.gradio-container details > summary {
  min-height: 42px !important;
  color: var(--bte-ink) !important;
  font-weight: 650 !important;
}

.gradio-container footer {
  display: none !important;
}

.bte-report {
  border: 1px solid var(--bte-line);
  border-radius: 24px;
  padding: 22px;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 22px 70px rgba(17, 24, 39, 0.08);
}

.bte-report--empty {
  min-height: 360px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--bte-muted);
  background:
    linear-gradient(180deg, rgba(239, 246, 255, 0.55), rgba(255, 255, 255, 0.96)),
    #ffffff;
}

.bte-report--empty h2 {
  font-size: 32px !important;
}

.bte-loading-report {
  min-height: 430px;
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  align-items: center;
  gap: 28px;
  overflow: hidden;
  position: relative;
  background:
    linear-gradient(120deg, rgba(239, 246, 255, 0.92), rgba(255, 255, 255, 0.96)),
    #ffffff;
}

.bte-loading-report::after {
  content: "";
  position: absolute;
  inset: 0;
  transform: translateX(-100%);
  background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.08), transparent);
  animation: bte-scan 2.2s ease-in-out infinite;
}

.bte-loader-orbit {
  width: 148px;
  aspect-ratio: 1;
  position: relative;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #ffffff;
  border: 1px solid #dbeafe;
  box-shadow: 0 18px 48px rgba(37, 99, 235, 0.14);
}

.bte-loader-orbit span {
  width: 86px;
  aspect-ratio: 1;
  display: block;
  border-radius: 50%;
  background:
    radial-gradient(circle at 50% 50%, #ffffff 0 42%, transparent 43%),
    conic-gradient(from 30deg, var(--bte-blue), var(--bte-green), #dbeafe, var(--bte-blue));
  animation: bte-spin 1.4s linear infinite;
}

.bte-loader-orbit i {
  position: absolute;
  width: 12px;
  aspect-ratio: 1;
  border-radius: 50%;
  background: var(--bte-green);
  box-shadow: 0 0 0 8px rgba(18, 128, 92, 0.12);
  animation: bte-pulse 1.4s ease-in-out infinite;
}

.bte-loading-copy {
  position: relative;
  z-index: 1;
}

.bte-loading-copy h2 {
  margin-top: 0 !important;
}

.bte-loading-copy p:last-child {
  max-width: 620px;
  color: var(--bte-muted);
}

.bte-loading-stack {
  grid-column: 1 / -1;
  position: relative;
  z-index: 1;
  display: grid;
  gap: 10px;
}

.bte-loading-stack div {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: 14px;
  align-items: center;
  padding: 14px;
  border: 1px solid var(--bte-line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.74);
}

.bte-loading-stack span,
.bte-loading-stack strong {
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, #edf3fb, #dce9fb, #edf3fb);
  background-size: 220% 100%;
  animation: bte-shimmer 1.35s ease-in-out infinite;
}

.bte-loading-stack strong {
  height: 16px;
}

@keyframes bte-spin {
  to { transform: rotate(360deg); }
}

@keyframes bte-pulse {
  0%, 100% { transform: scale(0.86); opacity: 0.72; }
  50% { transform: scale(1.08); opacity: 1; }
}

@keyframes bte-scan {
  0% { transform: translateX(-100%); }
  48%, 100% { transform: translateX(100%); }
}

@keyframes bte-shimmer {
  0% { background-position: 180% 0; }
  100% { background-position: -40% 0; }
}

@keyframes bte-doc-float {
  0%, 100% { transform: translateY(0) rotate(-1.5deg); }
  50% { transform: translateY(-8px) rotate(-0.5deg); }
}

@keyframes bte-scan-doc {
  0% { transform: translateY(-98px) rotate(-6deg); opacity: 0; }
  18% { opacity: 1; }
  72% { opacity: 1; }
  100% { transform: translateY(238px) rotate(-6deg); opacity: 0; }
}

@keyframes bte-flow-line {
  0%, 100% { transform: scaleX(0.42); opacity: 0.45; }
  50% { transform: scaleX(1); opacity: 1; }
}

@keyframes bte-flow-tile {
  0%, 100% { transform: translateY(0); opacity: 0.74; }
  50% { transform: translateY(-6px); opacity: 1; }
}

@keyframes bte-report-rise {
  0%, 100% { transform: translateY(5px); }
  50% { transform: translateY(-7px); }
}

@keyframes bte-card-pop {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

@keyframes bte-bar-grow {
  0%, 100% { transform: scaleY(0.86); transform-origin: bottom; }
  50% { transform: scaleY(1); transform-origin: bottom; }
}

.bte-report-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 6px 0 20px;
}

.bte-report-hero p {
  color: var(--bte-muted);
  margin: 0;
  max-width: 620px;
}

.tabs {
  border: 0 !important;
}

.tab-nav {
  border-bottom: 1px solid var(--bte-line) !important;
}

.tab-nav button {
  color: var(--bte-muted) !important;
  font-weight: 650 !important;
}

.tab-nav button.selected {
  color: var(--bte-blue) !important;
}

.bte-score {
  width: 136px;
  min-width: 136px;
  aspect-ratio: 1;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: linear-gradient(180deg, var(--bte-blue-soft), #ffffff);
  border: 1px solid #dbeafe;
}

.bte-score span {
  display: block;
  font-size: 40px;
  font-weight: 760;
}

.bte-score small {
  color: var(--bte-muted);
}

.bte-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}

.bte-metric {
  border: 1px solid var(--bte-line);
  border-radius: 16px;
  padding: 14px;
  background: #fbfdff;
}

.bte-metric span {
  display: block;
  font-size: 28px;
  font-weight: 760;
}

.bte-metric strong,
.bte-metric small {
  display: block;
}

.bte-metric small {
  color: var(--bte-muted);
}

.bte-status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 18px;
}

.bte-pill {
  border-radius: 999px;
  padding: 8px 11px;
  background: #f3f6fa;
  color: var(--bte-muted);
  font-size: 13px;
}

.bte-pill--high,
.bte-pill--abnormal {
  background: var(--bte-red-soft);
  color: var(--bte-red);
}

.bte-pill--low {
  background: var(--bte-amber-soft);
  color: var(--bte-amber);
}

.bte-pill--normal {
  background: var(--bte-green-soft);
  color: var(--bte-green);
}

.bte-report-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 16px;
}

.bte-marker-list {
  display: grid;
  gap: 10px;
}

.bte-marker {
  border: 1px solid var(--bte-line);
  border-radius: 16px;
  background: #ffffff;
  overflow: hidden;
}

.bte-marker summary {
  cursor: pointer;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 14px;
  padding: 14px;
  list-style: none;
}

.bte-marker summary::-webkit-details-marker {
  display: none;
}

.bte-marker-main strong,
.bte-marker-main small,
.bte-marker-value strong,
.bte-marker-value small {
  display: block;
}

.bte-marker-main small,
.bte-marker-value small {
  color: var(--bte-muted);
  font-size: 12px;
}

.bte-marker-value {
  text-align: right;
}

.bte-marker-value strong {
  font-size: 20px;
}

.bte-marker-status {
  border-radius: 999px;
  padding: 7px 10px;
  min-width: 82px;
  text-align: center;
  font-size: 12px;
  background: #f3f6fa;
  color: var(--bte-muted);
}

.bte-marker--high .bte-marker-status,
.bte-marker--abnormal .bte-marker-status {
  background: var(--bte-red-soft);
  color: var(--bte-red);
}

.bte-marker--low .bte-marker-status {
  background: var(--bte-amber-soft);
  color: var(--bte-amber);
}

.bte-marker--normal .bte-marker-status {
  background: var(--bte-green-soft);
  color: var(--bte-green);
}

.bte-marker-body {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 14px;
  padding: 0 14px 14px;
  color: var(--bte-muted);
}

.bte-confidence {
  height: 8px;
  border-radius: 999px;
  overflow: hidden;
  background: #edf1f6;
  margin: 8px 0 4px;
}

.bte-confidence i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--bte-blue), var(--bte-green));
}

.bte-marker blockquote {
  margin: 0;
  padding: 12px;
  border-radius: 12px;
  background: #f7f9fc;
  color: var(--bte-muted);
}

.bte-report-aside {
  border: 1px solid var(--bte-line);
  border-radius: 18px;
  padding: 16px;
  background: #fbfdff;
  align-self: start;
}

.bte-report-aside h3 {
  margin-top: 0;
}

.bte-report-aside ul {
  padding-left: 18px;
  color: var(--bte-muted);
}

.bte-disclaimer {
  display: grid;
  gap: 4px;
  margin-top: 14px;
  padding: 12px;
  border-radius: 14px;
  background: var(--bte-blue-soft);
  color: var(--bte-muted);
}

.bte-disclaimer strong {
  color: var(--bte-ink);
}

  @media (max-width: 860px) {
  body,
  html,
  gradio-app {
    overflow-x: hidden !important;
    max-width: 100vw !important;
  }

  .gradio-container {
    width: calc(100vw - 16px) !important;
    max-width: calc(100vw - 16px) !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
    overflow-x: hidden !important;
  }

  .gradio-container > *,
  .gradio-container div,
  .gradio-container .main,
  .gradio-container .contain,
  .gradio-container .wrap,
  .gradio-container .row,
  .gradio-container .column {
    max-width: 100% !important;
    min-width: 0 !important;
  }

  .gradio-container input,
  .gradio-container textarea,
  .gradio-container button {
    max-width: 100% !important;
    min-width: 0 !important;
  }

  .bte-title {
    padding: 22px 4px 16px;
  }

  .bte-title h1,
  .bte-report h2 {
    font-size: 32px !important;
    overflow-wrap: anywhere;
  }

  .bte-title p {
    font-size: 15px;
    max-width: 340px !important;
    overflow-wrap: anywhere;
  }

  .bte-shell {
    border-radius: 18px;
    padding: 12px;
  }

  .bte-engine {
    padding: 16px;
  }

  .bte-formation {
    min-height: 560px;
    padding: 14px;
    border-radius: 20px;
  }

  .bte-formation-stage {
    grid-template-columns: 1fr;
    min-height: 528px;
    gap: 12px;
  }

  .bte-source-doc {
    min-height: 220px;
    transform: none;
  }

  .bte-flow {
    grid-template-columns: 1fr auto 1fr;
    width: 100%;
  }

  .bte-flow span {
    grid-column: 1 / -1;
    grid-row: 1;
  }

  .bte-flow i,
  .bte-flow b {
    grid-row: 1;
    grid-column: 2;
  }

  .bte-report-window {
    min-height: 244px;
  }

  .bte-mini-chart {
    height: 80px;
  }

  .bte-uploader [class*="drop"],
  .bte-uploader [class*="upload"] {
    min-height: 210px !important;
  }

  .bte-selected-document {
    grid-template-columns: 1fr;
    min-height: 210px;
    text-align: center;
    justify-items: center;
  }

  .bte-status span,
  .bte-run-status span,
  .bte-report p,
  .bte-report small {
    overflow-wrap: anywhere;
  }

  .bte-report {
    border-radius: 20px;
    padding: 16px;
  }

  .bte-loading-report {
    grid-template-columns: 1fr;
    min-height: 520px;
    text-align: center;
    justify-items: center;
  }

  .bte-loading-stack {
    width: 100%;
  }

  .bte-report-hero,
  .bte-report-grid {
    grid-template-columns: 1fr;
    display: grid;
  }

  .bte-score {
    width: 110px;
    min-width: 110px;
  }

  .bte-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bte-marker summary,
  .bte-marker-body {
    grid-template-columns: 1fr;
  }

  .bte-marker-value {
    text-align: left;
  }
}

"""


with gr.Blocks(title="Blood Test Explainer") as demo:
    gr.HTML(
        """
        <section class="bte-title">
          <p class="bte-kicker">Clinical clarity from raw documents</p>
          <h1>Blood Test Explainer</h1>
          <p>Upload a lab report and turn dense medical paperwork into a polished extraction report with raw markers, values, units, reference ranges, and confidence signals.</p>
        </section>
        """
    )

    with gr.Row(equal_height=False, elem_classes=["bte-hero-grid"]):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["bte-shell", "bte-upload-card"]):
                with gr.Group() as upload_dropzone:
                    uploaded = gr.File(
                        label="Upload medical test document",
                        file_count="single",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".txt", ".csv"],
                        type="filepath",
                        elem_classes=["bte-uploader"],
                    )
                selected_document = gr.HTML(selected_document_html(), visible=False)
                run_button = gr.Button("Extract test results", variant="primary", elem_classes=["bte-action"])

        with gr.Column(scale=5, min_width=360):
            gr.HTML(formation_animation_html())

    with gr.Accordion("Advanced extraction settings", open=False):
        with gr.Group(elem_classes=["bte-shell", "bte-engine", "bte-engine-compact"]):
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
            gr.HTML(api_status(), elem_classes=["bte-status"])

    status = gr.HTML(_status_html("Ready", "Upload a lab report to create the first interactive extraction draft."))

    uploaded.change(
        upload_state,
        inputs=[uploaded],
        outputs=[upload_dropzone, selected_document],
        show_progress="hidden",
    )

    gr.HTML('<div id="bte-report-anchor" class="bte-report-anchor"></div>', visible=True)
    with gr.Group(visible=False) as report_panel:
        with gr.Tabs():
            with gr.Tab("Report"):
                report = gr.HTML(empty_report_html())
            with gr.Tab("Values"):
                results = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["str", "str", "str", "str", "str", "number", "str"],
                    row_count=(0, "dynamic"),
                    column_count=(len(TABLE_HEADERS), "fixed"),
                    label="Extracted values",
                    wrap=True,
                )
            with gr.Tab("JSON"):
                structured_json = gr.Code(language="json", label="Normalized extraction")
            with gr.Tab("Raw"):
                raw_response = gr.Textbox(label="Raw model response", lines=12)

    run_button.click(
        show_processing,
        outputs=[status, report_panel, report],
        scroll_to_output=True,
        show_progress="hidden",
        js="""
        () => {
          setTimeout(() => {
            document.getElementById("bte-report-anchor")?.scrollIntoView({
              behavior: "smooth",
              block: "start"
            });
          }, 450);
          return [];
        }
        """,
    ).then(
        extract_lab_values,
        inputs=[uploaded, api_url, model, api_key_override, max_pages],
        outputs=[results, status, structured_json, raw_response, report, report_panel],
        scroll_to_output=True,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
