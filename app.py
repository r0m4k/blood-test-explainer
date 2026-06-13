from __future__ import annotations

import os
import re
import time
import traceback
import base64
import io
from pathlib import Path
from html import escape
from typing import Any


def _patch_asyncio_event_loop_del() -> None:
    """Suppress Gradio 6 asyncio GC noise on Hugging Face Spaces (Python 3.10)."""
    try:
        import asyncio.base_events as base_events

        original_del = getattr(base_events.BaseEventLoop, "__del__", None)
        if original_del is None or getattr(original_del, "_bte_patched", False):
            return

        def patched_del(self) -> None:
            try:
                original_del(self)
            except ValueError as exc:
                if str(exc) != "Invalid file descriptor: -1":
                    raise

        patched_del._bte_patched = True  # type: ignore[attr-defined]
        base_events.BaseEventLoop.__del__ = patched_del  # type: ignore[method-assign]
    except Exception:
        pass


_patch_asyncio_event_loop_del()

import gradio as gr

from src.extraction import build_extractor
from src.interpretation_render import patterns_html
from src.local_env import load_local_env
from src.pipeline_trace import (
    build_pipeline_trace,
    empty_trace_html,
    error_trace_html,
    processing_trace_html,
    trace_hover_js,
    trace_to_html,
)
from src.report_pipeline import build_health_report


load_local_env()
_BOOT_T0 = time.perf_counter()


def _boot_log(message: str) -> None:
    elapsed = time.perf_counter() - _BOOT_T0
    print(f"[Blood Test Explainer][{elapsed:0.2f}s] {message}", flush=True)

_APP_ROOT = Path(__file__).resolve().parent
_LOGO_DIR = _APP_ROOT / "assets" / "logos"
_boot_log("environment loaded")


def extract_lab_values(
    uploaded_file: str | None,
) -> tuple[str, str, Any, str, str]:
    if not uploaded_file:
        return (
            _status_html("Waiting for a document", "Upload a lab report to begin extraction."),
            empty_report_html("No document uploaded", "Choose a file first, then run extraction again."),
            gr.update(visible=True),
            workflow_phase_html("ready"),
            empty_trace_html(),
        )

    extractor = build_extractor()

    try:
        result = extractor.extract(uploaded_file)
    except Exception as error:
        detail = _format_extraction_error(error)
        return (
            _status_html("Extraction failed", detail, tone="danger"),
            empty_report_html("Extraction failed", detail),
            gr.update(visible=True),
            workflow_phase_html("ready"),
            error_trace_html(detail),
        )

    health_report = build_health_report(result)
    summary = health_report["summary"]
    patient = health_report["patient"]
    steps = build_pipeline_trace(result, health_report, source_path=uploaded_file)

    status_text = (
        f"Extracted {summary['total_markers']} lab values and enriched "
        f"{summary['enriched_markers']} from the knowledge graph."
    )
    patient_bits = []
    if patient.get("age"):
        patient_bits.append(f"age {patient['age']}")
    if patient.get("sex") and patient["sex"] != "unknown":
        patient_bits.append(patient["sex"])
    if patient_bits:
        status_text += " Patient context: " + ", ".join(patient_bits) + "."
    if result.notes:
        status_text += " Notes: " + " ".join(result.notes[:3])

    return (
        _status_html("Extraction complete", status_text),
        report_html(health_report) + patterns_html(result.tests),
        gr.update(visible=True),
        workflow_phase_html("done"),
        trace_to_html(steps),
    )


def _status_html(title: str, detail: str, tone: str = "success") -> str:
    return f"""
    <div class="bte-run-status bte-run-status--{escape(tone)}">
      <strong>{escape(title)}</strong>
      <span>{escape(detail)}</span>
    </div>
    """


def _format_extraction_error(error: Exception) -> str:
    primary = _sanitize_error_message(error)
    lowered = primary.lower()
    if "failed to load model from file" in lowered:
        return (
            "The llama.cpp backend could not load the GGUF model. That points to a model/runtime "
            "compatibility issue, not a background worker problem."
        )
    if "hosted openbmb api backend is disabled" in lowered:
        return "Hosted API extraction is disabled. The app uses local Transformers only."
    if "401" in lowered or "unauthorized" in lowered:
        return "Authentication failed for the configured backend."
    if "could not be converted into a report" in lowered:
        return "The model produced output, but it could not be parsed into the extraction schema."
    return primary


def _sanitize_error_message(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        message = error.__class__.__name__
    if os.getenv("BTE_DEBUG_ERRORS", "0") == "1":
        tb = "".join(traceback.TracebackException.from_exception(error).format()).strip()
        return f"{message}\n\n{tb}"
    return message


def workflow_phase_html(phase: str) -> str:
    phase = phase if phase in {"ready", "processing", "done"} else "ready"
    return f"""
    <div class="bte-workflow-phase-marker" data-phase="{escape(phase)}" aria-hidden="true"></div>
    """


def _display_status_label(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized == "bad":
        return "Low"
    return normalized.title() if normalized else "Unknown"


def _logo_data_uri(filename: str) -> str | None:
    path = _LOGO_DIR / filename
    if not path.exists():
        return None
    mime_type = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _hero_badge_mark_html(slug: str, mark: str, logo_file: str) -> str:
    logo_uri = _logo_data_uri(logo_file)
    if not logo_uri:
        return escape(mark)
    return (
        f'<img class="bte-hero-badge-logo" src="{logo_uri}" '
        f'alt="{escape(slug)} logo" loading="lazy" />'
    )


def hero_hackathon_panel_html() -> str:
    hf_logo_uri = _logo_data_uri("HF.webp")
    hf_logo_inner = (
        f'<img class="bte-title-hf-logo" src="{hf_logo_uri}" alt="Hugging Face logo" loading="lazy" />'
        if hf_logo_uri
        else '<span class="bte-title-hf-logo-fallback" aria-hidden="true">HF</span>'
    )
    hf_logo_html = f'<span class="bte-title-hf-logo-wrap">{hf_logo_inner}</span>'
    badges = [
        (
            "🔌",
            "Off the Grid",
            "Extraction runs on-device through llama.cpp or ZeroGPU with no external inference API.",
        ),
        (
            "🎯",
            "Well-Tuned",
            "MiniCPM-V was fine-tuned on Modal and published on Hugging Face for lab report extraction.",
        ),
        (
            "🎨",
            "Off-Brand",
            "Custom CSS, HTML reports, and workflow panels push past the default Gradio look.",
        ),
        (
            "🦙",
            "Llama Champion",
            "GGUF models run through the llama.cpp runtime on CPU and ZeroGPU paths.",
        ),
        (
            "📡",
            "Sharing is Caring",
            "Agent traces, eval artifacts, and model cards are shared on the Hugging Face Hub.",
        ),
        (
            "📓",
            "Field Notes",
            "README, runbook, and eval docs capture how the app was built and how to run it.",
        ),
    ]
    badge_items = "\n".join(
        f"""
        <li class="bte-hack-badge" tabindex="0">
          <div class="bte-hack-badge-row">
            <span class="bte-hack-badge-icon" aria-hidden="true">{emoji}</span>
            <span class="bte-hack-badge-name">{escape(name)}</span>
          </div>
          <p class="bte-expand-detail">{escape(detail)}</p>
        </li>
        """
        for emoji, name, detail in badges
    )
    return f"""
    <div class="bte-title-hackathon-panel">
      <section class="bte-title-hf" aria-label="Hackathon project">
        {hf_logo_html}
        <p class="bte-title-hf-copy">Project for Build Small Hackathon</p>
      </section>
      <div class="bte-title-section-divider" aria-hidden="true"></div>
      <section class="bte-hack-badges">
        <p class="bte-title-side-label">Badges Collected</p>
        <ul class="bte-hack-badges-grid" aria-label="Hackathon badges collected">
          {badge_items}
        </ul>
      </section>
    </div>
    """


def hero_attribution_html() -> str:
    items = [
        (
            "Codex",
            "Build with Codex",
            "CDX",
            "codex.png",
            "Codex helped build the app UI, extraction pipeline, deployment scripts, and iteration workflow.",
        ),
        (
            "OpenBMB",
            "Enabled with OpenBMB",
            "OB",
            "openbmb.png",
            "MiniCPM-V-4.6 reads uploaded lab reports and extracts marker values, units, and status flags.",
        ),
        (
            "Modal",
            "Finetuned with Modal",
            "M",
            "modal.png",
            "Modal runs LoRA fine-tuning and evaluation jobs that produced the published extraction model.",
        ),
        (
            "ACG",
            "Created by researchers at ACG",
            "ACG",
            "acg.png",
            "Developed at The American College of Greece for the Hugging Face Build Small Hackathon.",
        ),
    ]
    badge_chunks = []
    for slug, label, mark, logo_file, detail in items:
        mark_html = _hero_badge_mark_html(slug, mark, logo_file)
        badge_chunks.append(
            f"""
        <li class="bte-hero-badge bte-hero-badge--{escape(slug.lower())}" tabindex="0">
          <div class="bte-hero-badge-row">
            <span class="bte-hero-badge-mark">{mark_html}</span>
            <span class="bte-hero-badge-text">{escape(label)}</span>
          </div>
          <p class="bte-expand-detail">{escape(detail)}</p>
        </li>
        """
        )
    badges = "\n".join(badge_chunks)
    return f"""
    <div class="bte-hero-credits">
      <ul class="bte-hero-attribution" aria-label="Project attributions">
        {badges}
      </ul>
    </div>
    """


def workflow_arrow_html(kind: str) -> str:
    kind = kind if kind in {"upload", "report"} else "upload"
    if kind == "upload":
        svg = """
        <svg viewBox="0 0 160 420" aria-hidden="true" focusable="false">
          <defs>
            <linearGradient id="bte-arrow-upload" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#67e8f9"/>
              <stop offset="48%" stop-color="#2563eb"/>
              <stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
          </defs>
          <path d="M136 22 C70 58, 34 120, 34 200 C34 276, 63 332, 108 376" fill="none" stroke="url(#bte-arrow-upload)" stroke-width="8" stroke-linecap="round"/>
          <path d="M108 376 L86 360 M108 376 L102 349" fill="none" stroke="url(#bte-arrow-upload)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """
    else:
        svg = """
        <svg viewBox="0 0 220 180" aria-hidden="true" focusable="false">
          <defs>
            <linearGradient id="bte-arrow-report" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#67e8f9"/>
              <stop offset="48%" stop-color="#2563eb"/>
              <stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
          </defs>
          <path d="M110 14 C110 52, 110 86, 110 126" fill="none" stroke="url(#bte-arrow-report)" stroke-width="8" stroke-linecap="round"/>
          <path d="M110 126 L89 106 M110 126 L132 106" fill="none" stroke="url(#bte-arrow-report)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """
    return f"""
    <div class="bte-workflow-arrow-svg bte-workflow-arrow-svg--{escape(kind)}" aria-hidden="true">
      {svg}
    </div>
    """


def show_processing() -> tuple[str, Any, str, str, str]:
    return (
        _status_html("Reading document", "Extracting patient context and markers, then matching them to the knowledge graph.", tone="loading"),
        gr.update(visible=False),
        "",
        workflow_phase_html("processing"),
        processing_trace_html(),
    )


def upload_state(uploaded_file: str | None) -> tuple[Any, Any, Any, str, str]:
    if not uploaded_file:
        return (
            gr.update(visible=True),
            gr.update(value='<p class="bte-upload-hint">Supported formats: PDF, PNG, JPEG, WebP</p>', visible=True),
            gr.update(visible=False, value=selected_document_html()),
            workflow_phase_html("ready"),
            empty_trace_html(),
        )

    preview_data_url = _uploaded_file_preview_data_url(uploaded_file)
    return (
        gr.update(visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=True, value=selected_document_html(preview_data_url=preview_data_url)),
        workflow_phase_html("processing"),
        processing_trace_html(),
    )


def selected_document_html(preview_data_url: str | None = None) -> str:
    preview_markup = (
        f'<img class="bte-upload-preview-image" src="{escape(preview_data_url)}" alt="Uploaded document preview">'
        if preview_data_url
        else """
        <div class="bte-upload-preview-placeholder">
          <div class="bte-selected-preview-header">
            <span></span><span></span><span></span>
          </div>
        </div>
        """
    )
    return f"""
    <section class="bte-selected-document">
      <div class="bte-selected-preview" aria-hidden="true">
        {preview_markup}
        <div class="bte-selected-preview-overlay"></div>
      </div>
    </section>
    """


def _uploaded_file_preview_data_url(uploaded_file: str) -> str | None:
    path = Path(uploaded_file)
    suffix = path.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=86, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"

    if suffix == ".pdf":
        import fitz

        with fitz.open(path) as document:
            if document.page_count == 0:
                return None
            page = document.load_page(0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.8, 2.8), alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            return f"data:image/png;base64,{encoded}"

    return None


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
        <p>The model is locating patient context, markers, values, units, reference ranges, and status flags. The knowledge graph report will appear when enrichment is complete.</p>
      </div>
      <div class="bte-loading-stack" aria-hidden="true">
        <div><span></span><strong></strong></div>
        <div><span></span><strong></strong></div>
        <div><span></span><strong></strong></div>
      </div>
    </section>
    """


def analysis_animation_html() -> str:
    return """
    <section class="bte-formation bte-formation--analysis" aria-label="Document analysis animation">
      <div class="bte-formation-stage bte-formation-stage--analysis">
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
      </div>
    </section>
    """


def _ideal_marker_card(test: dict[str, str]) -> str:
    status = test["status"]
    range_position_value = test.get("range_position", "50")
    range_position = escape(range_position_value)
    range_labels = test.get("range_labels", ("Low", "Normal", "Good"))
    low_label, mid_label, high_label = (escape(label) for label in range_labels)
    return f"""
    <article class="bte-ideal-marker bte-ideal-marker--{escape(status)}">
      <div class="bte-ideal-marker-head">
        <div class="bte-ideal-title-line">
          <h3>{escape(test["marker"])}</h3>
          <span class="bte-ideal-status">{escape(_display_status_label(status))}</span>
        </div>
        <div class="bte-range-scale" style="--value-position: {range_position}%; --value-position-number: {range_position}">
          <div class="bte-range-value">
            <strong>{escape(test["value"])}</strong>
            <small>{escape(test["unit"])}</small>
          </div>
          <div class="bte-range-track" aria-hidden="true">
            <span>{low_label}</span>
            <span>{mid_label}</span>
            <span>{high_label}</span>
          </div>
        </div>
      </div>
      <div class="bte-ideal-marker-body">
        <div>
          <span>Measures</span>
          <p>{escape(test["summary"])}</p>
        </div>
        <div>
          <span>Why it matters</span>
          <p>{escape(test["importance"])}</p>
        </div>
        <div>
          <span>How to improve</span>
          <p>{escape(test["improve"])}</p>
        </div>
      </div>
    </article>
    """


def report_html(report: dict[str, Any]) -> str:
    markers = list(report.get("markers") or [])
    summary = report.get("summary") or {}
    patient = report.get("patient") or {}

    total = int(summary.get("total_markers") or len(markers))
    final_statuses = [_final_status_for_marker(marker) for marker in markers]
    ideal = final_statuses.count("ideal")
    normal = final_statuses.count("normal")
    bad = final_statuses.count("bad")
    patient_context = _final_patient_context(patient)

    left_cards = "\n".join(
        _final_marker_card(marker) for index, marker in enumerate(markers) if index % 2 == 0
    )
    right_cards = "\n".join(
        _final_marker_card(marker) for index, marker in enumerate(markers) if index % 2 == 1
    )
    if not left_cards and not right_cards:
        left_cards = _empty_final_marker_card()

    return f"""
    <section class="bte-ideal-doc bte-final-report">
      <header class="bte-ideal-hero">
        <div>
          <p class="bte-kicker">Blood test report</p>
          <h2>Blood Test Report</h2>
          <p>Generated from the uploaded lab report, matched to the knowledge graph, and enriched with age and sex context. {escape(patient_context)}</p>
        </div>
      </header>

      <input class="bte-ideal-filter" type="radio" name="bte-final-filter" id="bte-final-filter-total" checked>
      <input class="bte-ideal-filter" type="radio" name="bte-final-filter" id="bte-final-filter-ideal">
      <input class="bte-ideal-filter" type="radio" name="bte-final-filter" id="bte-final-filter-normal">
      <input class="bte-ideal-filter" type="radio" name="bte-final-filter" id="bte-final-filter-bad">

      <div class="bte-ideal-stats">
        <label class="bte-ideal-stat bte-ideal-stat--total" for="bte-final-filter-total">
          <span>{total}</span>
          <strong>Total tests</strong>
        </label>
        <label class="bte-ideal-stat bte-ideal-stat--ideal" for="bte-final-filter-ideal">
          <span>{ideal}</span>
          <strong>Ideal</strong>
        </label>
        <label class="bte-ideal-stat bte-ideal-stat--normal" for="bte-final-filter-normal">
          <span>{normal}</span>
          <strong>Normal</strong>
        </label>
        <label class="bte-ideal-stat bte-ideal-stat--bad" for="bte-final-filter-bad">
          <span>{bad}</span>
          <strong>Low</strong>
        </label>
      </div>

      <div class="bte-ideal-grid">
        <div class="bte-ideal-column">
          {left_cards}
        </div>
        <div class="bte-ideal-column">
          {right_cards}
        </div>
      </div>
    </section>
    """


def _patient_context_html(patient: dict[str, Any]) -> str:
    age = _text(patient.get("age"), "Not extracted")
    age_group = _text(patient.get("age_group"), "adult")
    sex = _text(patient.get("sex"), "unknown")
    return f"""
    <dl class="bte-patient-context">
      <div><dt>Age</dt><dd>{escape(age)}</dd></div>
      <div><dt>Age group</dt><dd>{escape(age_group.title())}</dd></div>
      <div><dt>Sex</dt><dd>{escape(sex.title())}</dd></div>
    </dl>
    """


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
    marker = _preferred_marker_label(test)
    raw_name = _text(test.get("raw_name"), marker)
    value = _text(test.get("value"), "-")
    unit = _text(test.get("unit"), "")
    reference = _reference_label(test)
    status = _text(test.get("status"), "unknown").lower()
    confidence = _confidence_percent(test.get("confidence"))
    comparison = test.get("comparison") or {}
    range_position = escape(str(comparison.get("range_position", 50)))
    knowledge = test.get("knowledge") or {}
    source = _text(test.get("source_text"), "No source snippet returned.")
    description = _text(knowledge.get("description"), "No knowledge graph description available for this marker.")
    why = _text(knowledge.get("why_important"), "No knowledge graph importance note available for this marker.")
    instructions = knowledge.get("instructions_to_improve") or {}
    sex_context = knowledge.get("sex_significance") or {}
    sex_summary = _text(sex_context.get("summary"), "No major sex-specific interpretation note is stored for this marker.")
    stats_text = _statistics_text(test)
    derived = _text(test.get("derived_status"), "unknown")
    extracted = _text(test.get("extracted_status"), "unknown")

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
          <span class="bte-marker-status">{escape(_display_status_label(status))}</span>
      </summary>
      <div class="bte-marker-body">
        <div class="bte-marker-evidence">
          <span>Extraction confidence</span>
          <div class="bte-confidence"><i style="width: {confidence}%"></i></div>
          <small>{confidence}%</small>
          <span>Raw marker</span>
          <small>{escape(raw_name)}</small>
          <span>Status check</span>
          <small>Extracted: {escape(extracted.title())}. Calculated: {escape(derived.title())}.</small>
          <blockquote>{escape(source)}</blockquote>
        </div>
        <div class="bte-marker-insights">
          <div class="bte-range-scale bte-range-scale--report" style="--value-position: {range_position}%; --value-position-number: {range_position}">
            <div class="bte-range-value">
              <strong>{escape(value)}</strong>
              <small>{escape(unit)}</small>
            </div>
            <div class="bte-range-track" aria-hidden="true">
              <span>Low</span>
              <span>Reference</span>
              <span>High</span>
            </div>
          </div>
          <div class="bte-insight-grid">
            {_insight_block("Measures", description)}
            {_insight_block("Why it matters", why)}
            {_insight_block("Selected range", stats_text)}
            {_insight_block("Sex context", sex_summary)}
          </div>
          <div class="bte-guidance">
            {_guidance_column("Food", instructions.get("food"))}
            {_guidance_column("Exercise", instructions.get("exercises"))}
            {_guidance_column("Supplements", instructions.get("supplements"))}
          </div>
          <div class="bte-marker-video">
            <span>Related video</span>
            {_youtube_embed_html(knowledge.get("video_url"), title=f"{marker} overview")}
          </div>
        </div>
      </div>
    </details>
    """


def _final_marker_card(test: dict[str, Any]) -> str:
    marker = _preferred_marker_label(test)
    value = _text(test.get("value"), "-")
    unit = _text(test.get("unit"), "")
    status = _final_status_for_marker(test)
    range_position = escape(str(_final_range_position(test)))
    knowledge = test.get("knowledge") or {}
    instructions = knowledge.get("instructions_to_improve") or {}
    description = _text(knowledge.get("description"), "No knowledge graph description available for this marker.")
    why = _text(knowledge.get("why_important"), "No knowledge graph importance note available for this marker.")

    return f"""
    <article class="bte-ideal-marker bte-ideal-marker--{escape(status)}">
      <div class="bte-ideal-marker-head">
        <div class="bte-ideal-title-line">
          <h3>{escape(marker)}</h3>
          <span class="bte-ideal-status">{escape(_display_status_label(status))}</span>
        </div>
        <div class="bte-range-scale" style="--value-position: {range_position}%; --value-position-number: {range_position}">
          <div class="bte-range-value">
            <strong>{escape(value)}</strong>
            <small>{escape(unit)}</small>
          </div>
          <div class="bte-range-track" aria-hidden="true">
            <span>Low</span>
            <span>Normal</span>
            <span>Good</span>
          </div>
        </div>
      </div>
      <div class="bte-ideal-marker-body">
        <div>
          <span>Measures</span>
          <p>{escape(description)}</p>
        </div>
        <div>
          <span>Why it matters</span>
          <p>{escape(why)}</p>
        </div>
        {_improvement_block(instructions)}
        <div class="bte-marker-video-block">
          <span>Related video</span>
          {_youtube_embed_html(knowledge.get("video_url"), title=f"{marker} overview")}
        </div>
      </div>
    </article>
    """


def _final_status_for_marker(marker: dict[str, Any]) -> str:
    quality = _final_quality(marker)
    if quality is None:
        status = _text(marker.get("status"), "unknown").lower()
        if status in {"low", "high", "abnormal"}:
            return "bad"
        return "normal"
    return quality["status"]


def _preferred_marker_label(marker: dict[str, Any]) -> str:
    canonical = _text(marker.get("display_name"), "Unknown marker")
    raw_name = _text(marker.get("raw_name"), "")
    if _looks_like_marker_abbreviation(raw_name, canonical):
        return raw_name
    return canonical


def _looks_like_marker_abbreviation(raw_name: str, canonical: str) -> bool:
    raw = raw_name.strip()
    if not raw or raw.casefold() == canonical.strip().casefold():
        return False
    if len(raw) > 16:
        return False

    compact = re.sub(r"[\s._-]+", "", raw)
    letters = [char for char in compact if char.isalpha()]
    if not letters:
        return False

    if any(symbol in raw for symbol in ("%#",)):
        return True
    if "/" in raw and len(compact) <= 12:
        return True

    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    if len(compact) <= 10 and uppercase_ratio >= 0.6:
        return True

    # Common lab shorthand is often title-cased, e.g. Hct or Plt.
    canonical_is_descriptive = len(canonical) > len(raw) + 3 or " " in canonical
    return canonical_is_descriptive and len(compact) <= 5 and raw[0].isupper()


def _final_range_position(marker: dict[str, Any]) -> int:
    quality = _final_quality(marker)
    if quality is not None:
        return quality["position"]

    return int((marker.get("comparison") or {}).get("range_position") or 50)


def _final_quality(marker: dict[str, Any]) -> dict[str, Any] | None:
    values = _final_reference_values(marker)
    numeric = _final_numeric_value(marker)
    if values is None or numeric is None:
        return None

    low, normal, high = values
    if high <= low:
        return None

    if numeric < low:
        distance = (low - numeric) / max(normal - low, high - low, 1.0)
        return {"status": "bad", "position": max(6, min(30, round(26 - distance * 18)))}
    if numeric > high:
        distance = (numeric - high) / max(high - normal, high - low, 1.0)
        return {"status": "bad", "position": max(6, min(30, round(26 - distance * 18)))}

    if numeric <= normal:
        side_width = normal - low
    else:
        side_width = high - normal

    if side_width <= 0:
        closeness = 1.0
    else:
        closeness = 1 - abs(numeric - normal) / side_width
    closeness = max(0.0, min(1.0, closeness))

    if closeness >= 0.7:
        # Good/ideal zone: the closer the patient is to the KG normal value, the fuller the bar.
        position = 68 + (closeness - 0.7) / 0.3 * 26
        return {"status": "ideal", "position": max(68, min(94, round(position)))}

    # Normal zone: in range, but not close enough to the KG normal value to call it ideal.
    position = 38 + closeness / 0.7 * 26
    return {"status": "normal", "position": max(38, min(64, round(position)))}


def _final_reference_values(marker: dict[str, Any]) -> tuple[float, float, float] | None:
    selection = marker.get("reference_selection") or {}
    values = selection.get("values") or {}
    try:
        low = float(values["minimal_value"])
        normal = float(values["normal_value"])
        high = float(values["maximum_value"])
    except (KeyError, TypeError, ValueError):
        return None
    if high <= low:
        return None
    return low, normal, high


def _final_numeric_value(marker: dict[str, Any]) -> float | None:
    try:
        return float(marker.get("numeric_value"))
    except (TypeError, ValueError):
        return None


def _improvement_block(instructions: dict[str, Any]) -> str:
    return f"""
    <div class="bte-improvement-block">
      <span>How to improve</span>
      <div class="bte-guidance bte-guidance--card">
        {_guidance_column("Food", instructions.get("food"))}
        {_guidance_column("Exercise", instructions.get("exercises"))}
        {_guidance_column("Supplements", instructions.get("supplements"))}
      </div>
    </div>
    """


def _youtube_video_id(video_url: str | None) -> str | None:
    if not video_url:
        return None
    text = str(video_url).strip()
    match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        text,
    )
    return match.group(1) if match else None


def _youtube_embed_html(video_url: str | None, *, title: str = "Marker overview video") -> str:
    video_id = _youtube_video_id(video_url)
    if not video_id:
        return '<p class="bte-video-placeholder">No video is available for this marker yet.</p>'
    safe_title = escape(title)
    return f"""
    <div class="bte-video-embed">
      <iframe
        src="https://www.youtube.com/embed/{escape(video_id)}"
        title="{safe_title}"
        loading="lazy"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen
      ></iframe>
    </div>
    """


def _final_patient_context(patient: dict[str, Any]) -> str:
    age = _text(patient.get("age"), "")
    sex = _text(patient.get("sex"), "")
    age_group = _text(patient.get("age_group"), "")
    parts = []
    if age:
        parts.append(f"Age: {age}")
    if sex and sex != "unknown":
        parts.append(f"Sex: {sex.title()}")
    if age_group:
        parts.append(f"Group: {age_group.title()}")
    return " | ".join(parts)


def _empty_final_marker_card() -> str:
    return """
    <article class="bte-ideal-marker bte-ideal-marker--normal">
      <div class="bte-ideal-marker-head">
        <div class="bte-ideal-title-line">
          <h3>No markers extracted yet</h3>
          <span class="bte-ideal-status">Normal</span>
        </div>
      </div>
    </article>
    """


def _empty_marker_card() -> str:
    return """
    <div class="bte-marker-empty">
      No markers extracted yet.
    </div>
    """


def _reference_label(test: dict[str, Any]) -> str:
    lab_range = _text(test.get("lab_reference_range"), "")
    if lab_range:
        return f"Lab range: {lab_range}"
    return _statistics_text(test)


def _statistics_text(test: dict[str, Any]) -> str:
    selection = test.get("reference_selection") or {}
    values = selection.get("values") or {}
    low = values.get("minimal_value")
    normal = values.get("normal_value")
    high = values.get("maximum_value")
    if low is None and high is None:
        return "No knowledge graph range available."
    age_group = _text(selection.get("age_group"), "adult").title()
    sex = _text(selection.get("sex"), "not_applied").replace("_", " ").title()
    unit = _text(test.get("unit"), "")
    return f"{age_group}, {sex}: {low} min / {normal} typical / {high} max {unit}".strip()


def _insight_block(label: str, text: str) -> str:
    return f"""
    <div>
      <span>{escape(label)}</span>
      <p>{escape(text)}</p>
    </div>
    """


def _guidance_column(label: str, items: Any) -> str:
    if not isinstance(items, list) or not items:
        body = "<li>No guidance stored for this marker.</li>"
    else:
        body = "".join(f"<li>{escape(str(item))}</li>" for item in items[:3])
    return f"""
    <div>
      <strong>{escape(label)}</strong>
      <ul>{body}</ul>
    </div>
    """


def _source_links(markers: list[dict[str, Any]], sources: dict[str, str]) -> str:
    source_ids: list[str] = []
    for marker in markers:
        knowledge = marker.get("knowledge") or {}
        for source_id in knowledge.get("source_ids") or []:
            if source_id not in source_ids:
                source_ids.append(source_id)

    links = []
    for source_id in source_ids[:8]:
        url = sources.get(source_id)
        if not url:
            continue
        links.append(f'<li><a href="{escape(url)}" target="_blank" rel="noreferrer">{escape(source_id)}</a></li>')

    if not links:
        return "<p>No source links available for matched markers.</p>"
    return f"<ul>{''.join(links)}</ul>"


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
  --bte-page: rgb(248, 249, 252);
  --bte-paper: rgb(248, 249, 252);
  --bte-surface: #ffffff;
  --bte-card: #ffffff;
  --bte-radius: 22px;
  --bte-shadow: 0 14px 34px rgba(17, 24, 39, 0.055);
  --bte-shadow-strong: 0 18px 44px rgba(17, 24, 39, 0.07);
  --bte-active-ring: linear-gradient(120deg, var(--bte-green), var(--bte-blue), var(--bte-red));
  --bte-rail: min(94vw, 1240px);
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
  --body-background-fill: rgb(248, 249, 252) !important;
  --body-text-color: #111827 !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: rgb(248, 249, 252) !important;
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
  --shadow-drop: var(--bte-shadow) !important;
  background: var(--bte-page) !important;
}

.gradio-container {
  max-width: none !important;
  width: 100% !important;
  margin: 0 auto !important;
  box-sizing: border-box !important;
  color: var(--bte-ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  background: var(--bte-page) !important;
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
  width: 100% !important;
  max-width: 100% !important;
  border: 1px solid var(--bte-line);
  border-radius: var(--bte-radius);
  padding: 18px;
  background: var(--bte-card);
  box-shadow: var(--bte-shadow);
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

.bte-title-rail,
.bte-title-rail.block,
.bte-title-rail > div,
.bte-title-rail .form,
.bte-title-rail.gr-group,
.gradio-container .gr-group.bte-title-rail,
.gradio-container .block.bte-title-rail {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  overflow: visible !important;
}

.gradio-container .column:has(.bte-title-rail),
.gradio-container .column:has(> .row.bte-title) {
  overflow: visible !important;
}

.bte-title {
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 0 18px !important;
  padding: 28px 28px 26px;
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 0 !important;
  align-items: stretch !important;
  border: 1px solid rgba(255, 255, 255, 0.42);
  border-radius: var(--bte-radius) !important;
  overflow: hidden !important;
  background:
    linear-gradient(120deg, rgba(191, 52, 52, 0.82) 0%, rgba(37, 99, 235, 0.95) 58%, rgba(18, 128, 92, 0.98) 100%),
    #12805c;
  box-shadow: var(--bte-shadow-strong);
}

.gradio-container .row.bte-title.stretch,
.bte-title.row.stretch,
.bte-title.row {
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  flex-direction: unset !important;
  align-items: stretch !important;
  border-radius: var(--bte-radius) !important;
  overflow: hidden !important;
}

.bte-title > .column,
.bte-title > div > .column,
.bte-title .column.bte-title-copy,
.bte-title .column.bte-title-hackathon-wrap,
.bte-title .column.bte-title-credits-wrap {
  flex: none !important;
  flex-grow: 0 !important;
  flex-shrink: 0 !important;
  flex-basis: auto !important;
  width: auto !important;
  min-width: 0 !important;
  max-width: none !important;
  display: flex !important;
  flex-direction: column !important;
  align-self: stretch !important;
  height: 100% !important;
  min-height: 100% !important;
}

.bte-title .column > .block,
.bte-title .column > .form,
.bte-title .column .html-container,
.bte-title .column .prose {
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
}

.bte-title-copy .html-container > div,
.bte-title-copy .prose > div,
.bte-title-hackathon-wrap .html-container > div,
.bte-title-hackathon-wrap .prose > div,
.bte-title-credits-wrap .html-container > div,
.bte-title-credits-wrap .prose > div {
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
}

.bte-title-copy .html-container > div > p:last-child,
.bte-title-copy .prose > div > p:last-child {
  margin-top: auto !important;
}

.bte-title-hackathon-panel {
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  gap: 0;
}

.bte-title-hackathon-wrap .bte-hack-badges {
  margin-top: auto !important;
}

.bte-hero-credits {
  flex: 1 1 auto !important;
  height: 100% !important;
  min-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
}

.bte-title-credits-wrap .bte-hero-attribution {
  margin-top: auto !important;
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
  color: rgba(255, 255, 255, 0.88);
  -webkit-text-fill-color: rgba(255, 255, 255, 0.88) !important;
  font-size: 16px;
  max-width: none;
  margin: 0;
  text-align: left;
}

.bte-title-copy,
.bte-title-hackathon-wrap,
.bte-title-credits-wrap,
.bte-title .bte-title-copy,
.bte-title .bte-title-hackathon-wrap,
.bte-title .bte-title-credits-wrap {
  position: relative;
  align-self: stretch;
  min-width: 0;
}

.bte-title-copy {
  text-align: left;
  justify-self: stretch;
  padding: 0 24px 0 0;
}

.bte-title-hackathon-wrap {
  padding: 0 24px;
}

.bte-title-credits-wrap {
  padding: 0 0 0 24px;
}

.bte-title-copy::after,
.bte-title-hackathon-wrap::after,
.bte-title .bte-title-copy::after,
.bte-title .bte-title-hackathon-wrap::after {
  content: "";
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 1px;
  background: rgba(255, 255, 255, 0.42);
  pointer-events: none;
}

.bte-title-hf {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: flex-start;
  gap: 14px;
  padding-bottom: 18px;
  text-align: left;
}

.bte-title-hf-logo-wrap {
  flex: 0 0 52px;
  width: 52px;
  height: 52px;
  display: grid;
  place-items: center;
  overflow: hidden;
  border-radius: 23%;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 4px 14px rgba(17, 24, 39, 0.14);
}

.bte-title-hf-logo {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.bte-title-hf-logo-fallback {
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  color: #111827;
  font-size: 18px;
  font-weight: 800;
}

.bte-title-hf-copy {
  margin: 0;
  flex: 1;
  min-width: 0;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.07em;
  line-height: 1.35;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.88) !important;
  -webkit-text-fill-color: rgba(255, 255, 255, 0.88) !important;
}

.bte-title-section-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.34);
  margin-bottom: 18px;
}

.bte-title-side-label {
  margin: 0 0 10px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.72) !important;
  -webkit-text-fill-color: rgba(255, 255, 255, 0.72) !important;
}

.bte-hack-badges,
.bte-hack-badges-grid,
.bte-hero-credits,
.bte-hero-attribution {
  overflow: visible;
}

.bte-hack-badges-grid {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-auto-rows: minmax(36px, auto);
  align-content: start;
  gap: 8px 10px;
}

.bte-hack-badge {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0;
  width: 100%;
  max-width: 100%;
  min-height: 36px;
  max-height: 36px;
  min-width: 0;
  box-sizing: border-box;
  padding: 8px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.16);
  overflow: hidden;
  cursor: pointer;
  position: relative;
  transition:
    max-height 260ms ease,
    background 180ms ease,
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.bte-hack-badge:hover,
.bte-hack-badge:focus-within {
  max-height: 500px;
  overflow: visible;
  z-index: 3;
  background: rgba(255, 255, 255, 0.18);
  border-color: rgba(255, 255, 255, 0.32);
  box-shadow: 0 10px 24px rgba(17, 24, 39, 0.16);
  outline: none;
}

.bte-hack-badge-row {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
  min-height: 18px;
}

.bte-hack-badge-icon {
  flex: 0 0 18px;
  width: 18px;
  height: 18px;
  display: grid;
  place-items: center;
  font-size: 14px;
  line-height: 1;
}

.bte-hack-badge-name {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 10px;
  line-height: 1.2;
  font-weight: 700;
}

.bte-expand-detail {
  margin: 0;
  max-height: 0;
  opacity: 0;
  overflow: hidden;
  color: rgba(255, 255, 255, 0.76) !important;
  -webkit-text-fill-color: rgba(255, 255, 255, 0.76) !important;
  font-size: 10px !important;
  line-height: 1.4;
  font-weight: 400 !important;
  white-space: normal;
  transition:
    max-height 260ms ease,
    opacity 180ms ease,
    margin-top 180ms ease;
}

.bte-title .bte-expand-detail,
.bte-title .bte-expand-detail * {
  font-size: 10px !important;
  font-weight: 400 !important;
  line-height: 1.4 !important;
}

.bte-hack-badge:hover .bte-hack-badge-name,
.bte-hack-badge:focus-within .bte-hack-badge-name,
.bte-hero-badge:hover .bte-hero-badge-text,
.bte-hero-badge:focus-within .bte-hero-badge-text {
  white-space: normal;
  overflow: visible;
  text-overflow: clip;
}

.bte-hack-badge:hover .bte-expand-detail,
.bte-hack-badge:focus-within .bte-expand-detail,
.bte-hero-badge:hover .bte-expand-detail,
.bte-hero-badge:focus-within .bte-expand-detail {
  max-height: 400px;
  opacity: 1;
  margin-top: 6px;
  overflow: visible;
}

.bte-hero-credits {
  min-width: 0;
}

.bte-title-attribution-wrap {
  min-width: 0;
}

.bte-hero-attribution {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 10px;
}

.bte-hero-badge {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0;
  min-height: 54px;
  max-height: 54px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.18);
  backdrop-filter: blur(6px);
  overflow: hidden;
  cursor: pointer;
  position: relative;
  transition:
    max-height 260ms ease,
    background 180ms ease,
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.bte-hero-badge:hover,
.bte-hero-badge:focus-within {
  max-height: 500px;
  overflow: visible;
  z-index: 3;
  background: rgba(255, 255, 255, 0.18);
  border-color: rgba(255, 255, 255, 0.32);
  box-shadow: 0 10px 24px rgba(17, 24, 39, 0.16);
  outline: none;
}

.bte-hero-badge-row {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  min-height: 34px;
}

.bte-hero-badge-mark {
  width: 46px;
  height: 34px;
  flex: 0 0 46px;
  display: grid;
  place-items: center;
  color: #ffffff;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0;
  background: transparent;
  box-shadow: none;
  overflow: visible;
}

.bte-hero-badge-logo {
  display: block;
  width: 34px;
  max-width: 38px;
  max-height: 24px;
  object-fit: contain;
}

.bte-hero-badge-text {
  flex: 1 1 auto;
  min-width: 0;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  font-size: 13px;
  line-height: 1.3;
  font-weight: 700;
}

.bte-hero-badge--openbmb .bte-hero-badge-logo {
  width: auto;
  max-width: 44px;
  max-height: 22px;
}

.bte-hero-badge--modal .bte-hero-badge-logo {
  width: 38px;
  max-width: 40px;
}

.bte-hero-badge--acg .bte-hero-badge-logo {
  width: 32px;
  max-height: 32px;
}

.bte-title .bte-kicker,
.bte-title .bte-kicker *,
.bte-title h1,
.bte-title h1 *,
.bte-title .bte-title-copy p,
.bte-title .bte-title-copy p * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.bte-title .bte-title-copy > div > p:not(.bte-kicker) {
  color: rgba(255, 255, 255, 0.88) !important;
  -webkit-text-fill-color: rgba(255, 255, 255, 0.88) !important;
}

.bte-title h1 {
  font-size: clamp(38px, 5vw, 56px) !important;
  line-height: 1.04 !important;
  text-align: left !important;
}

.bte-title .bte-kicker {
  text-align: left !important;
}

.bte-title > div,
.bte-title > div > div,
.bte-title .column,
.bte-title .block,
.bte-title .form {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-hero-grid {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  align-items: stretch !important;
  gap: 16px !important;
}

.bte-hero-grid > div,
.bte-hero-grid .column {
  min-width: 0 !important;
  height: 100% !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-hero-grid > div > div,
.bte-hero-grid .column > div {
  height: 100% !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.bte-hero-grid .block,
.bte-hero-grid .form,
.bte-hero-grid .html-container {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card),
.bte-hero-grid .bte-panel-upload div:has(> .bte-upload-card) {
  height: 430px !important;
  min-height: 430px !important;
  border: 1px solid var(--bte-line) !important;
  border-radius: var(--bte-radius) !important;
  padding: 18px !important;
  background: var(--bte-page) !important;
  box-shadow: var(--bte-shadow) !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
}

.bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card) .bte-shell,
.bte-hero-grid .bte-panel-upload div:has(> .bte-upload-card) .bte-shell,
.bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card) .bte-upload-card,
.bte-hero-grid .bte-panel-upload div:has(> .bte-upload-card) > .bte-upload-card {
  height: 100% !important;
  min-height: 0 !important;
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  border: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  overflow: hidden !important;
}

.bte-hero-grid .bte-upload-card:not(.bte-panel-upload .block:has(.bte-upload-card) .bte-upload-card) {
  border: 1px solid var(--bte-line) !important;
  border-radius: var(--bte-radius) !important;
  padding: 18px !important;
  background: var(--bte-page) !important;
  box-shadow: var(--bte-shadow) !important;
  overflow: hidden !important;
}

.bte-workflow-panel {
  min-width: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-workflow-panel--upload {
  flex: 0.9 1 0 !important;
  min-width: 320px !important;
}

.bte-workflow-panel--analysis {
  flex: 1.1 1 0 !important;
  min-width: 360px !important;
}

.bte-workflow-panel > div,
.bte-workflow-panel > div > div {
  background: transparent !important;
}

.bte-workflow-panel,
.bte-final-row,
.bte-report-panel {
  position: relative;
}

.bte-report-panel,
.bte-report-panel > * {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.bte-workflow-phase,
.bte-workflow-phase-marker {
  display: none !important;
}

.bte-step-row-block,
.bte-step-row-block > div,
.gradio-container .block.bte-step-row-block,
.gradio-container .block.bte-step-row-block.padded,
.gradio-container .block.bte-step-row-block .form,
.bte-step-row-block .form {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

.bte-step-row-block .prose,
.bte-step-row-block .html-container,
.bte-step-row-block .block,
.prose.bte-step-row-block,
.gradio-container .block.bte-step-row-block .html-container,
.gradio-container .block.bte-step-row-block .prose,
.gradio-container .prose.bte-step-row-block {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

.bte-status-row,
.bte-status-row > div,
.bte-ideal-row,
.bte-ideal-row > div,
.bte-final-row,
.bte-final-row > div {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 0 !important;
  background: var(--bte-page) !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-final-row,
.bte-final-row > div,
.bte-final-row .prose,
.bte-final-row .html-container,
.bte-final-row .block {
  min-height: 0 !important;
  height: auto !important;
}

.bte-status-row .prose,
.bte-status-row .html-container,
.bte-status-row .block,
.bte-ideal-row .prose,
.bte-ideal-row .html-container,
.bte-ideal-row .block,
.bte-final-row .prose,
.bte-final-row .html-container,
.bte-final-row .block {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-status-row .bte-run-status {
  display: flex !important;
  flex-direction: column !important;
  gap: 4px !important;
  border: 1px solid rgba(18, 128, 92, 0.22) !important;
  border-radius: 14px !important;
  padding: 12px 14px !important;
  background: var(--bte-green-soft) !important;
  box-shadow: var(--bte-shadow) !important;
}

.bte-step-row {
  width: 100% !important;
  max-width: 100% !important;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  align-items: stretch;
  gap: 16px;
  margin: 0 0 16px;
  background: transparent !important;
  background-color: transparent !important;
}

.bte-step-block,
.bte-step-block > div {
  height: auto !important;
  min-height: 0 !important;
  flex: 0 0 auto !important;
  overflow: visible !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-step-heading {
  min-height: 112px;
  height: 100%;
  display: flex;
  align-items: start;
  gap: 12px;
  margin: 0;
  padding: 18px;
  border: 1px solid var(--bte-line);
  border-radius: var(--bte-radius);
  background: var(--bte-surface);
  box-shadow: var(--bte-shadow);
  overflow: hidden;
  opacity: 0.55;
  filter: saturate(0.6);
  transition: opacity 220ms ease, filter 220ms ease, box-shadow 220ms ease, transform 220ms ease, border-color 220ms ease, background 220ms ease;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-step-row-block .bte-step-heading--upload,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-step-row-block .bte-step-heading--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-step-row-block .bte-step-heading--report {
  opacity: 1;
  filter: saturate(1.08);
  transform: translateY(-1px);
  border: 2px solid transparent;
  background:
    linear-gradient(var(--bte-surface), var(--bte-surface)) padding-box,
    var(--bte-active-ring) border-box;
  box-shadow: var(--bte-shadow-strong);
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-step-row-block .bte-step-heading--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-step-row-block .bte-step-heading--report,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-step-row-block .bte-step-heading--upload,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-step-row-block .bte-step-heading--report,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-step-row-block .bte-step-heading--upload,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-step-row-block .bte-step-heading--analysis {
  opacity: 0.38;
  filter: saturate(0.45);
  transform: none;
  background: var(--bte-surface);
  box-shadow: var(--bte-shadow);
  border-color: rgba(216, 226, 238, 0.92);
}

.bte-step-heading span {
  width: 34px;
  min-width: 34px;
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border-radius: 50%;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: linear-gradient(135deg, var(--bte-green), var(--bte-blue));
  font-size: 15px;
  font-weight: 780;
}

.bte-step-heading span,
.bte-step-heading span * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.bte-step-heading h2 {
  margin: 0 !important;
  color: var(--bte-ink) !important;
  font-size: clamp(18px, 2.1vw, 24px) !important;
  line-height: 1.18 !important;
  letter-spacing: 0 !important;
  text-align: left !important;
}

.bte-panel-upload .bte-upload-card,
.bte-panel-analysis .bte-formation,
.bte-panel-result .bte-agent-panel,
.bte-final-row .bte-report {
  transition: opacity 220ms ease, filter 220ms ease, box-shadow 220ms ease, transform 220ms ease, border-color 220ms ease, background 220ms ease;
}

.bte-step-heading--report {
  margin-top: 0;
  min-height: 112px;
  padding: 18px;
}

.bte-upload-card {
  height: 100% !important;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  min-height: 0 !important;
  overflow: hidden !important;
  position: relative !important;
}

.bte-panel-upload .bte-upload-dropzone,
.bte-panel-upload .bte-upload-card > .block:has(.bte-upload-dropzone),
.bte-panel-upload .bte-upload-card > .form:has(.bte-upload-dropzone) {
  position: absolute !important;
  inset: 0 !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
  border: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  z-index: 1 !important;
}

.bte-upload-card:has(.bte-selected-document) .bte-upload-dropzone,
.bte-upload-card:has(.bte-selected-document) .bte-upload-hint-wrap,
.bte-upload-card:has(.bte-selected-document) > .block:has(.bte-upload-dropzone),
.bte-upload-card:has(.bte-selected-document) > .form:has(.bte-upload-dropzone) {
  display: none !important;
}

.bte-upload-card:has(.bte-selected-document) .block:has(.bte-selected-document),
.bte-upload-card:has(.bte-selected-document) .html-container:has(.bte-selected-document) {
  position: absolute !important;
  inset: 0 !important;
  z-index: 5 !important;
  width: 100% !important;
  height: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

.bte-upload-card:has(.bte-selected-document) .prose.bte-selected-document-wrap,
.bte-upload-card:has(.bte-selected-document) .html-container:has(.bte-selected-document) > *,
.bte-upload-card:has(.bte-selected-document) .bte-selected-document,
.bte-upload-card:has(.bte-selected-document) .bte-selected-preview {
  height: 100% !important;
  min-height: 100% !important;
}

.bte-upload-card:has(.bte-selected-document) .prose:has(.bte-selected-document) {
  padding: 0 !important;
  margin: 0 !important;
  max-width: none !important;
}

.bte-panel-upload .bte-upload-dropzone > .block,
.bte-panel-upload .bte-upload-dropzone > .form,
.bte-panel-upload .bte-upload-dropzone > div {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}

.bte-panel-upload .bte-upload-hint-wrap {
  flex: 0 0 auto;
  position: relative !important;
  z-index: 2 !important;
  pointer-events: none !important;
}

.bte-upload-hint {
  margin: 0 0 12px !important;
  color: var(--bte-ink) !important;
  font-size: 18px !important;
  font-weight: 700 !important;
  text-align: center !important;
}

.bte-panel-upload .bte-upload-card .block:has(.bte-uploader),
.bte-panel-upload .bte-upload-card .form:has(.bte-uploader),
.bte-panel-upload .bte-shell > .block:has(.bte-uploader),
.bte-panel-upload .bte-shell > .form:has(.bte-uploader),
.bte-panel-upload .bte-upload-card > .block:has(.bte-uploader) {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}

.bte-panel-upload .bte-uploader,
.bte-panel-upload .bte-uploader > div,
.bte-panel-upload .bte-uploader > div > div,
.bte-panel-upload .bte-uploader .wrap {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  overflow: hidden !important;
}

.bte-formation {
  width: 100% !important;
  max-width: 100% !important;
  height: 430px !important;
  min-height: 430px;
  border: 1px solid var(--bte-line);
  border-radius: var(--bte-radius);
  padding: 22px;
  background: var(--bte-surface);
  box-shadow: var(--bte-shadow);
  overflow: hidden;
}

.bte-formation-stage {
  height: 100%;
  min-height: 382px;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  justify-items: center;
  align-items: center;
  gap: 14px;
}

.bte-formation-stage--analysis .bte-source-doc,
.bte-formation-stage--result .bte-smart-report,
.bte-formation-stage--result .bte-report-window {
  width: 100%;
}

.bte-panel-analysis .bte-formation--analysis,
.bte-panel-result .bte-agent-panel {
  overflow: hidden;
}

.bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel),
.bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) {
  height: 430px !important;
  min-height: 430px !important;
  max-height: 430px !important;
  border: 1px solid var(--bte-line) !important;
  border-radius: var(--bte-radius) !important;
  padding: 16px !important;
  background: var(--bte-page) !important;
  box-shadow: var(--bte-shadow) !important;
  overflow: hidden !important;
  display: flex !important;
  flex-direction: column !important;
  box-sizing: border-box !important;
}

.bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel) .bte-shell,
.bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) .bte-shell,
.bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel) .bte-agent-panel,
.bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) > .bte-agent-panel {
  height: 100% !important;
  min-height: 0 !important;
  flex: 1 1 auto !important;
  display: flex !important;
  flex-direction: column !important;
  border: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  overflow: hidden !important;
  gap: 0 !important;
}

.bte-agent-panel,
.bte-agent-panel > div,
.bte-agent-panel > .block,
.bte-agent-panel > .form {
  display: flex !important;
  flex-direction: column !important;
  flex: 1 1 auto !important;
  min-height: 0 !important;
  width: 100% !important;
}

.bte-agent-panel .block:has(.bte-agent-trace),
.bte-agent-panel .block:has(.bte-trace-panel),
.bte-agent-panel .html-container:has(.bte-trace-panel),
.bte-panel-trace .bte-agent-panel .block,
.bte-panel-trace .bte-agent-panel .form,
.bte-panel-trace .bte-agent-panel .wrap,
.bte-panel-trace .bte-agent-panel .html-container,
.bte-panel-trace .bte-agent-panel .prose {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  height: 100% !important;
  max-height: 100% !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  box-sizing: border-box !important;
  display: flex !important;
  flex-direction: column !important;
}

.bte-panel-trace .bte-agent-panel .html-container:has(.bte-trace-panel),
.bte-panel-trace .bte-agent-panel .prose:has(.bte-trace-panel) {
  width: 100% !important;
}

.bte-trace-panel {
  flex: 1 1 auto !important;
  height: 100% !important;
  max-height: 100% !important;
  min-height: 0 !important;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 4px 10px 0;
  box-sizing: border-box;
}

.bte-trace-steps {
  flex: 1 1 auto;
  min-height: 0;
  max-height: calc(430px - 32px - 16px);
  overflow-x: hidden;
  overflow-y: auto !important;
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
  padding: 0 2px 8px 0;
  scrollbar-gutter: stable;
}

.bte-trace-steps::-webkit-scrollbar {
  width: 8px;
}

.bte-trace-steps::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 999px;
}

.bte-trace-steps::-webkit-scrollbar-track {
  background: transparent;
}

.bte-trace-empty {
  margin: 0;
  color: var(--bte-muted);
  font-size: 14px;
  line-height: 1.5;
}

.bte-trace-empty--active {
  color: var(--bte-ink);
}

.bte-trace-empty--error {
  color: #b42318;
}

.bte-trace-status {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.bte-trace-status--complete {
  color: #067647;
  background: #ecfdf3;
  border: 1px solid #abefc6;
}

.bte-trace-status--running {
  color: #175cd3;
  background: #eff8ff;
  border: 1px solid #b2ddff;
}

.bte-trace-status--failed {
  color: #b42318;
  background: #fef3f2;
  border: 1px solid #fecdca;
}

.bte-trace-status--pending {
  color: #667085;
  background: #f9fafb;
  border: 1px solid #d0d5dd;
}

.bte-trace-status--unknown {
  color: #344054;
  background: #f2f4f7;
  border: 1px solid #eaecf0;
}

.bte-trace-step-summary {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 4px;
  padding: 10px 12px;
  cursor: pointer;
  list-style: none;
  transition: background-color 0.22s ease, border-color 0.22s ease;
}

.bte-trace-panel[data-interactive="true"] .bte-trace-step-summary:hover,
.bte-trace-panel[data-interactive="true"] .bte-trace-step.is-open .bte-trace-step-summary {
  background: #f8fbff;
}

.bte-trace-panel[data-interactive="true"] .bte-trace-step.is-open .bte-trace-step-summary {
  border-bottom: 1px solid #eef2f7;
}

.bte-trace-step-collapse {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 0.34s cubic-bezier(0.4, 0, 0.2, 1);
}

.bte-trace-panel[data-interactive="true"] .bte-trace-step.is-open .bte-trace-step-collapse {
  grid-template-rows: 1fr;
}

.bte-trace-step-collapse > .bte-trace-step-body {
  overflow: hidden;
  min-height: 0;
  transform: translateY(-6px);
  opacity: 0;
  transition:
    transform 0.34s cubic-bezier(0.4, 0, 0.2, 1),
    opacity 0.28s ease;
}

.bte-trace-panel[data-interactive="true"] .bte-trace-step.is-open .bte-trace-step-collapse > .bte-trace-step-body {
  transform: translateY(0);
  opacity: 1;
}

.bte-trace-step-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  width: 100%;
}

.bte-trace-step--locked .bte-trace-step-summary {
  cursor: default;
}

.bte-trace-panel--locked .bte-trace-step--locked .bte-trace-step-summary:hover {
  background: transparent;
}

.bte-trace-step-body {
  padding: 6px 14px 14px;
}

.bte-trace-explanation,
.bte-trace-result {
  padding-left: 2px;
  padding-right: 2px;
}

.bte-trace-step-meta {
  color: #475467;
  font-size: 11px;
  font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.bte-trace-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 12px;
  margin: 0 0 10px;
  padding: 10px;
  border: 1px solid #eef2f7;
  border-radius: 10px;
  background: #f8fafc;
}

.bte-trace-meta dt {
  margin: 0;
  color: #667085;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.bte-trace-meta dd {
  margin: 2px 0 0;
  color: #101828;
  font-size: 12px;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}

.bte-trace-step-summary::-webkit-details-marker {
  display: none;
}

.bte-trace-step {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  margin-bottom: 8px;
  background: #fff;
  overflow: hidden;
}

.bte-trace-step:last-child {
  margin-bottom: 0;
}

.bte-trace-step-title {
  color: var(--bte-ink);
  font-size: 14px;
  font-weight: 700;
}

.bte-trace-step-teaser {
  color: var(--bte-muted);
  font-size: 12px;
  line-height: 1.4;
}

.bte-trace-summary {
  margin: 0 0 8px;
  color: #334155;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.bte-trace-explanation {
  margin: 0 0 8px;
  color: #1f2937;
  font-size: 14px;
  line-height: 1.55;
}

.bte-trace-result {
  margin: 0 0 8px;
  color: #475569;
  font-size: 13px;
  line-height: 1.5;
}

.bte-trace-technical {
  display: grid;
  gap: 8px;
}

.bte-trace-technical-text {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.45;
  color: #334155;
}

.bte-trace-subdetails {
  margin-top: 8px;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 8px 10px;
  background: #f9fafb;
}

.bte-trace-subdetails summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: #334155;
}

.bte-trace-subdetails pre {
  margin: 8px 0 0;
  padding: 8px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid #e5e7eb;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 220px;
  overflow: auto;
}

.bte-panel-trace {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-trace-panel[data-interactive="false"] .bte-trace-step,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-trace-panel[data-interactive="false"] .bte-trace-step {
  pointer-events: none;
}

.bte-panel-result .bte-mini-card,
.bte-panel-result .bte-mini-chart span {
  animation-play-state: paused !important;
}

.bte-panel-analysis .bte-source-doc,
.bte-panel-analysis .bte-scan-band,
.bte-panel-analysis .bte-flow span,
.bte-panel-analysis .bte-flow i,
.bte-panel-analysis .bte-flow b {
  animation-play-state: paused !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-result .bte-agent-panel,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-result .bte-agent-panel,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis {
  opacity: 0.42;
  filter: saturate(0.5);
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card),
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload div:has(> .bte-upload-card),
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel),
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) {
  opacity: 1;
  filter: saturate(1.08);
  transform: translateY(-1px);
  border: 2px solid transparent !important;
  background:
    linear-gradient(var(--bte-page), var(--bte-page)) padding-box,
    var(--bte-active-ring) border-box !important;
  box-shadow: var(--bte-shadow-strong) !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card) .bte-shell,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .block:has(.bte-upload-card) .bte-upload-card {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel),
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) {
  background:
    linear-gradient(var(--bte-surface), var(--bte-surface)) padding-box,
    var(--bte-active-ring) border-box !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-result .bte-agent-panel,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-result .bte-agent-panel,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis {
  animation-play-state: paused !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace .block:has(.bte-agent-panel),
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-trace div:has(> .bte-agent-panel) {
  animation-play-state: running !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-source-doc,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-scan-band,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow span,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow i,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow b {
  animation-play-state: running !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card {
  animation-play-state: paused !important;
}

.bte-source-doc,
.bte-report-window {
  position: relative;
  border: 1px solid rgba(216, 226, 238, 0.95);
  border-radius: 18px;
  background: var(--bte-surface);
  box-shadow: 0 14px 34px rgba(17, 24, 39, 0.075);
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
  background: var(--bte-surface);
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
  background: var(--bte-surface);
  animation: bte-card-pop 4.4s ease-in-out infinite;
}

.bte-mini-card--green {
  border-color: rgba(18, 128, 92, 0.2);
  background: var(--bte-green-soft);
}

.bte-mini-card--red {
  border-color: rgba(200, 70, 70, 0.2);
  background: var(--bte-red-soft);
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
  background: var(--bte-surface);
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
  width: 100% !important;
  max-width: 100% !important;
  margin-left: auto !important;
  margin-right: auto !important;
  display: flex;
  flex-direction: column;
  gap: 4px;
  border: 1px solid var(--bte-line);
  border-radius: 14px;
  padding: 12px 14px;
  background: var(--bte-surface);
  color: var(--bte-muted);
  font-size: 13px;
  box-shadow: var(--bte-shadow);
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

.bte-uploader {
  border: 0 !important;
}

.bte-uploader .label-wrap,
.bte-uploader > label {
  display: none !important;
}

.bte-upload-hint-wrap,
.bte-upload-hint-wrap > div {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-panel-upload .bte-uploader [class*="drop"],
.bte-panel-upload .bte-uploader [class*="upload"] {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  max-height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
}

.bte-panel-upload .bte-uploader [data-testid="block-label"],
.bte-panel-upload .bte-uploader [data-testid="status-tracker"],
.bte-panel-upload .bte-uploader .icon-button-wrapper,
.bte-panel-upload .bte-uploader .file-preview-holder {
  display: none !important;
}

.bte-panel-upload .bte-uploader > button,
.bte-panel-upload .bte-uploader button[class*="center"] {
  flex: 1 1 auto !important;
  width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
}

.bte-panel-upload .bte-uploader button .wrap:not(:has(.uploading)) {
  font-size: 0 !important;
  line-height: 0 !important;
  color: transparent !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0 !important;
}

.bte-panel-upload .bte-uploader button .or {
  display: none !important;
}

.bte-panel-upload .bte-uploader .wrap:has(.uploading),
.bte-panel-upload .bte-uploader .wrap:has(.progress-bar) {
  flex: 1 1 auto !important;
  width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  font-size: 0 !important;
  color: transparent !important;
}

.bte-panel-upload .bte-uploader .wrap:has(.uploading) > *,
.bte-panel-upload .bte-uploader .wrap:has(.progress-bar) > * {
  display: none !important;
}

.bte-panel-upload .bte-uploader .wrap:has(.uploading)::before,
.bte-panel-upload .bte-uploader .wrap:has(.progress-bar)::before {
  content: "";
  width: 72px;
  height: 72px;
  border-radius: 50%;
  flex: 0 0 auto;
  background:
    radial-gradient(circle at 50% 50%, var(--bte-page) 0 56%, transparent 57%),
    conic-gradient(from 0deg, var(--bte-green), var(--bte-blue), var(--bte-red), var(--bte-green));
  animation: bte-spin 1.05s linear infinite;
  box-shadow: 0 12px 30px rgba(17, 24, 39, 0.08);
}

.bte-uploader [class*="drop"],
.bte-uploader [class*="upload"] {
  min-height: 220px !important;
}

.bte-panel-upload .bte-uploader svg,
.bte-panel-upload .bte-shell .icon-wrap,
.bte-panel-upload .bte-shell .icon-wrap svg {
  width: 72px !important;
  height: 72px !important;
  flex: 0 0 auto !important;
}

.bte-panel-upload .bte-uploader button .icon-wrap {
  background: var(--bte-active-ring) !important;
  -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") !important;
  mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") !important;
  -webkit-mask-repeat: no-repeat !important;
  mask-repeat: no-repeat !important;
  -webkit-mask-position: center !important;
  mask-position: center !important;
  -webkit-mask-size: contain !important;
  mask-size: contain !important;
}

.bte-panel-upload .bte-uploader button .icon-wrap svg {
  opacity: 0 !important;
  visibility: hidden !important;
  pointer-events: none !important;
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
  background: var(--bte-page) !important;
  border: 0 !important;
  border-radius: 18px !important;
  color: var(--bte-ink) !important;
  box-shadow: none !important;
  outline: none !important;
}

.bte-panel-upload .bte-shell .block,
.bte-panel-upload .bte-shell .form,
.bte-panel-upload .bte-shell .html-container,
.bte-panel-upload .bte-uploader,
.bte-panel-upload .bte-uploader > div,
.bte-panel-upload .bte-uploader > div > div,
.bte-panel-upload .bte-uploader [class*="drop"],
.bte-panel-upload .bte-uploader [class*="upload"],
.bte-panel-upload .bte-shell [class*="drop"],
.bte-panel-upload .bte-shell [class*="upload"] {
  border: 0 !important;
  outline: none !important;
  box-shadow: none !important;
}

.bte-selected-document {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
  align-items: stretch;
  height: 100%;
  min-height: 100%;
  border: 0;
  border-radius: 0;
  padding: 0;
  background: transparent;
}

.bte-upload-card:has(.bte-selected-document) .bte-selected-document {
  height: 100% !important;
  min-height: 100% !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}

.bte-selected-preview {
  position: relative;
  height: 100%;
  min-height: 100%;
  border-radius: 14px;
  border: 0;
  background: var(--bte-page);
  overflow: hidden;
  display: block;
}

.bte-upload-card:has(.bte-selected-document) .bte-selected-preview {
  height: 100% !important;
  min-height: 100% !important;
  border: 0 !important;
  border-radius: 12px !important;
}

.bte-upload-preview-image,
.bte-upload-preview-placeholder {
  position: absolute;
  inset: 0;
  border-radius: 12px;
}

.bte-upload-preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center center;
  filter: saturate(0.98) contrast(1.03);
  transform: none;
  box-shadow: none;
}

.bte-upload-card:has(.bte-selected-document) .bte-upload-preview-image {
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  object-fit: contain !important;
  object-position: center center !important;
}

.bte-upload-preview-placeholder {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(245, 248, 252, 0.96));
  border: 1px solid rgba(217, 226, 238, 0.82);
  box-shadow: 0 16px 35px rgba(18, 32, 56, 0.08);
  filter: blur(0.35px);
}

.bte-selected-preview-header {
  display: flex;
  gap: 6px;
  padding: 18px 20px 0;
}

.bte-selected-preview-header span {
  width: 11px;
  height: 11px;
  border-radius: 999px;
  background: rgba(95, 126, 180, 0.18);
}

.bte-selected-preview-body {
  padding: 22px 22px 20px;
}

.bte-preview-line {
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(27, 76, 161, 0.26), rgba(23, 161, 122, 0.22));
  filter: blur(1.2px);
  margin-bottom: 14px;
}

.bte-preview-line--title {
  width: 64%;
  height: 22px;
  margin-bottom: 18px;
  background: linear-gradient(90deg, rgba(17, 24, 39, 0.18), rgba(37, 99, 235, 0.16));
}

.bte-preview-line--lg {
  width: 78%;
}

.bte-preview-line--md {
  width: 54%;
}

.bte-preview-line--sm {
  width: 39%;
}

.bte-preview-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 20px;
}

.bte-preview-grid span {
  height: 34px;
  border-radius: 10px;
  background: rgba(36, 105, 235, 0.12);
  filter: blur(1.2px);
}

.bte-selected-preview-overlay {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  pointer-events: none;
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
  background: var(--bte-surface);
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

.bte-shell svg,
.bte-shell .icon-wrap {
  color: var(--bte-blue) !important;
}

.bte-panel-upload .bte-uploader button .icon-wrap {
  color: transparent !important;
  -webkit-text-fill-color: transparent !important;
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
  background: var(--bte-surface) !important;
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
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  border: 1px solid var(--bte-line);
  border-radius: var(--bte-radius);
  padding: 22px;
  background: var(--bte-page);
  box-shadow: var(--bte-shadow);
}

.bte-report--empty {
  min-height: 360px;
  display: grid;
  place-items: center;
  text-align: center;
  color: var(--bte-muted);
  background: var(--bte-page);
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
  background: var(--bte-page);
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
  background: var(--bte-surface);
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
  background: var(--bte-surface);
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
  background: var(--bte-surface);
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
  background: var(--bte-surface);
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

.bte-marker-evidence {
  display: grid;
  align-content: start;
  gap: 8px;
}

.bte-marker-evidence > span {
  color: var(--bte-ink);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bte-marker-insights {
  min-width: 0;
  display: grid;
  gap: 14px;
}

.bte-range-scale--report {
  padding: 12px 0 4px;
}

.bte-insight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.bte-insight-grid div {
  min-width: 0;
  padding-top: 10px;
  border-top: 1px solid var(--bte-line);
}

.bte-insight-grid span,
.bte-guidance strong {
  display: block;
  margin-bottom: 6px;
  color: var(--bte-ink);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bte-insight-grid p {
  margin: 0;
  color: var(--bte-muted);
  font-size: 14px;
  line-height: 1.48;
}

.bte-guidance {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.bte-guidance div {
  min-width: 0;
  padding-top: 10px;
  border-top: 1px solid var(--bte-line);
}

.bte-guidance ul {
  margin: 0;
  padding-left: 18px;
  color: var(--bte-muted);
  font-size: 13px;
  line-height: 1.45;
}

.bte-marker-video,
.bte-marker-video-block {
  min-width: 0;
}

.bte-marker-video span,
.bte-marker-video-block span,
.bte-improvement-block > span {
  display: block;
  margin-bottom: 6px;
  color: var(--bte-ink);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bte-improvement-block .bte-guidance--card {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.bte-improvement-block .bte-guidance--card div {
  min-width: 0;
  padding-top: 10px;
  border-top: 1px solid var(--bte-line);
  background: transparent;
  border-radius: 0;
}

.bte-video-embed {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 12px;
  overflow: hidden;
  background: #0f172a;
}

.bte-video-embed iframe {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border: 0;
}

.bte-video-placeholder {
  margin: 0;
  color: var(--bte-muted);
  font-size: 14px;
  line-height: 1.5;
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
  background: var(--bte-page);
  color: var(--bte-muted);
}

.bte-report-aside {
  border: 1px solid var(--bte-line);
  border-radius: 18px;
  padding: 16px;
  background: var(--bte-surface);
  align-self: start;
}

.bte-report-aside h3 {
  margin-top: 0;
}

.bte-report-aside ul {
  padding-left: 18px;
  color: var(--bte-muted);
}

.bte-report-aside a {
  color: var(--bte-blue);
  overflow-wrap: anywhere;
}

.bte-patient-context {
  display: grid;
  gap: 8px;
  margin: 0 0 16px;
}

.bte-patient-context div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--bte-line);
}

.bte-patient-context dt {
  color: var(--bte-muted);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bte-patient-context dd {
  margin: 0;
  color: var(--bte-ink);
  font-weight: 650;
  text-align: right;
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

.bte-ideal-doc {
  width: 100% !important;
  max-width: 100% !important;
  margin-left: auto !important;
  margin-right: auto !important;
  margin-top: 0;
  display: grid;
  gap: 0;
  background: var(--bte-page);
}

.bte-final-report {
  --bte-report-stack-gap: 12px;
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin: 0 auto !important;
  background: rgb(248, 249, 252) !important;
  align-content: start;
  gap: var(--bte-report-stack-gap);
}

.bte-final-report > .bte-ideal-hero,
.bte-final-report > .bte-ideal-stats,
.bte-final-report > .bte-ideal-grid {
  margin: 0;
}

.bte-final-report .bte-ideal-marker {
  box-shadow: 0 4px 10px rgba(17, 24, 39, 0.035);
}

.bte-final-report .bte-ideal-marker:hover,
.bte-final-report .bte-ideal-marker:focus-within {
  box-shadow: 0 6px 14px rgba(17, 24, 39, 0.05);
}

.bte-ideal-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 22px;
  padding: 24px 28px;
  margin: 0;
  border: 1px solid rgba(255, 255, 255, 0.42);
  border-radius: var(--bte-radius);
  color: #ffffff;
  background:
    linear-gradient(120deg, rgba(191, 52, 52, 0.82) 0%, rgba(37, 99, 235, 0.95) 58%, rgba(18, 128, 92, 0.98) 100%),
    #12805c;
  box-shadow: 0 6px 16px rgba(17, 24, 39, 0.045);
}

.bte-ideal-hero .bte-kicker,
.bte-ideal-hero h2,
.bte-ideal-hero p {
  color: #ffffff !important;
}

.bte-ideal-hero h2 {
  font-size: clamp(30px, 4vw, 42px) !important;
  line-height: 1.06 !important;
  margin: 0 0 8px !important;
}

.bte-ideal-hero p {
  max-width: 680px;
  margin: 0;
  opacity: 0.88;
}

.bte-ideal-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: var(--bte-report-stack-gap, 12px);
  margin: 0;
}

.bte-ideal-filter {
  display: none !important;
}

.bte-ideal-stat {
  --bte-stat-bg: var(--bte-surface);
  --bte-stat-ring: linear-gradient(120deg, var(--bte-green), var(--bte-blue), var(--bte-red));
  padding: 16px;
  border: 2px solid var(--bte-line);
  border-radius: 18px;
  background: var(--bte-stat-bg);
  box-shadow: var(--bte-shadow);
  overflow: hidden;
  cursor: pointer;
  transition: background 220ms ease, box-shadow 220ms ease, border-color 220ms ease, filter 220ms ease;
}

.bte-ideal-stat span,
.bte-ideal-stat strong {
  display: block;
}

.bte-ideal-stat span {
  font-size: 34px;
  font-weight: 780;
  line-height: 1;
  margin-bottom: 8px;
}

.bte-ideal-stat--total span {
  color: var(--bte-ink);
}

.bte-ideal-doc:has(#bte-filter-total:checked) .bte-ideal-stat--total,
.bte-ideal-doc:has(#bte-filter-ideal:checked) .bte-ideal-stat--ideal,
.bte-ideal-doc:has(#bte-filter-normal:checked) .bte-ideal-stat--normal,
.bte-ideal-doc:has(#bte-filter-bad:checked) .bte-ideal-stat--bad,
.bte-ideal-doc:has(#bte-final-filter-total:checked) .bte-ideal-stat--total,
.bte-ideal-doc:has(#bte-final-filter-ideal:checked) .bte-ideal-stat--ideal,
.bte-ideal-doc:has(#bte-final-filter-normal:checked) .bte-ideal-stat--normal,
.bte-ideal-doc:has(#bte-final-filter-bad:checked) .bte-ideal-stat--bad {
  border-color: transparent;
  background:
    linear-gradient(var(--bte-stat-bg), var(--bte-stat-bg)) padding-box,
    var(--bte-stat-ring) border-box;
  box-shadow: var(--bte-shadow-strong);
  filter: saturate(1.08);
}

.bte-ideal-stat--ideal {
  --bte-stat-bg: var(--bte-green-soft);
  --bte-stat-ring: linear-gradient(120deg, var(--bte-green), #22c7a0, var(--bte-blue));
  border-color: rgba(18, 128, 92, 0.22);
}

.bte-ideal-stat--ideal span {
  color: var(--bte-green);
}

.bte-ideal-stat--normal {
  --bte-stat-bg: var(--bte-blue-soft);
  --bte-stat-ring: linear-gradient(120deg, var(--bte-blue), #52a8ff, var(--bte-green));
  border-color: rgba(37, 99, 235, 0.22);
}

.bte-ideal-stat--normal span {
  color: var(--bte-blue);
}

.bte-ideal-stat--bad {
  --bte-stat-bg: var(--bte-red-soft);
  --bte-stat-ring: linear-gradient(120deg, var(--bte-red), #ff8f8f, var(--bte-blue));
  border-color: rgba(191, 52, 52, 0.2);
}

.bte-ideal-stat--bad span {
  color: var(--bte-red);
}

.bte-ideal-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: start;
  gap: var(--bte-report-stack-gap, 12px);
  margin: 0;
}

.bte-ideal-column {
  display: grid;
  align-content: start;
  gap: var(--bte-report-stack-gap, 12px);
}

.bte-ideal-doc:has(#bte-filter-ideal:checked) .bte-ideal-marker:not(.bte-ideal-marker--ideal),
.bte-ideal-doc:has(#bte-filter-normal:checked) .bte-ideal-marker:not(.bte-ideal-marker--normal),
.bte-ideal-doc:has(#bte-filter-bad:checked) .bte-ideal-marker:not(.bte-ideal-marker--bad),
.bte-ideal-doc:has(#bte-final-filter-ideal:checked) .bte-ideal-marker:not(.bte-ideal-marker--ideal),
.bte-ideal-doc:has(#bte-final-filter-normal:checked) .bte-ideal-marker:not(.bte-ideal-marker--normal),
.bte-ideal-doc:has(#bte-final-filter-bad:checked) .bte-ideal-marker:not(.bte-ideal-marker--bad) {
  display: none;
}

.bte-ideal-marker {
  --bte-marker-color: var(--bte-green);
  --bte-marker-soft: var(--bte-green-soft);
  border: 1px solid var(--bte-line);
  border-radius: var(--bte-radius);
  padding: 18px;
  background: var(--bte-surface);
  box-shadow: var(--bte-shadow);
  overflow: hidden;
  transition: transform 700ms ease, box-shadow 700ms ease, border-color 700ms ease;
}

.bte-ideal-marker:hover,
.bte-ideal-marker:focus-within {
  transform: translateY(-2px);
  box-shadow: var(--bte-shadow-strong);
}

.bte-ideal-marker--ideal {
  --bte-marker-color: var(--bte-green);
  --bte-marker-soft: var(--bte-green-soft);
  border-color: rgba(18, 128, 92, 0.22);
}

.bte-ideal-marker--normal {
  --bte-marker-color: var(--bte-blue);
  --bte-marker-soft: var(--bte-blue-soft);
  border-color: rgba(37, 99, 235, 0.22);
}

.bte-ideal-marker--bad {
  --bte-marker-color: var(--bte-red);
  --bte-marker-soft: var(--bte-red-soft);
  border-color: rgba(191, 52, 52, 0.2);
  background: linear-gradient(180deg, #fff8f8, #ffffff);
}

.bte-ideal-marker-head {
  display: grid;
  grid-template-columns: minmax(180px, 0.8fr) minmax(230px, 1fr);
  gap: 18px;
  align-items: center;
  padding-bottom: 0;
  border-bottom: 1px solid transparent;
  transition: padding-bottom 1300ms ease, border-color 1100ms ease;
}

.bte-ideal-marker:hover .bte-ideal-marker-head,
.bte-ideal-marker:focus-within .bte-ideal-marker-head {
  padding-bottom: 12px;
  border-bottom-color: var(--bte-line);
}

.bte-ideal-marker-head h3 {
  margin: 0 !important;
  color: var(--bte-ink) !important;
  font-size: 22px !important;
  line-height: 1.1 !important;
}

.bte-ideal-title-line {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.bte-ideal-title-line h3 {
  min-width: 0;
}

.bte-ideal-marker-head small {
  color: var(--bte-muted);
  font-size: 13px;
}

.bte-ideal-status {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 7px 10px;
  font-size: 12px;
  font-weight: 760;
}

.bte-ideal-marker--ideal .bte-ideal-status {
  color: var(--bte-green);
  background: var(--bte-green-soft);
}

.bte-ideal-marker--normal .bte-ideal-status {
  color: var(--bte-blue);
  background: var(--bte-blue-soft);
}

.bte-ideal-marker--bad .bte-ideal-status {
  color: var(--bte-red);
  background: var(--bte-red-soft);
}

.bte-range-scale {
  display: grid;
  gap: 8px;
  min-width: 0;
}

.bte-range-value {
  position: relative;
  left: var(--value-position);
  display: inline-flex;
  align-items: baseline;
  justify-self: start;
  gap: 4px;
  color: var(--bte-marker-color);
  transform: translateX(-50%);
  white-space: nowrap;
}

.bte-range-value strong {
  color: inherit;
  font-size: 19px;
  line-height: 1;
}

.bte-range-value small {
  color: var(--bte-muted);
  font-size: 12px;
  font-weight: 760;
}

.bte-range-track {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  min-height: 32px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 999px;
  overflow: hidden;
  background: #ffffff;
}

.bte-range-track::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--value-position);
  border-radius: inherit;
  background:
    linear-gradient(90deg, rgba(191, 52, 52, 0.86) 0%, rgba(37, 99, 235, 0.84) 52%, rgba(18, 128, 92, 0.86) 100%);
  background-size: calc(10000% / var(--value-position-number, 100)) 100%;
  box-shadow: inset 0 0 18px rgba(255, 255, 255, 0.2);
}

.bte-range-track span {
  position: relative;
  z-index: 1;
  display: grid;
  place-items: center;
  color: var(--bte-ink);
  font-size: 10px;
  font-weight: 820;
  text-transform: uppercase;
}

.bte-ideal-marker-body {
  display: grid;
  gap: 12px;
  max-height: 0;
  overflow: hidden;
  padding-top: 0;
  opacity: 0;
  transform: translateY(-6px);
  transition: max-height 2200ms cubic-bezier(0.22, 1, 0.36, 1), padding-top 1700ms ease, opacity 1400ms ease, transform 1700ms ease;
}

.bte-ideal-marker:hover .bte-ideal-marker-body,
.bte-ideal-marker:focus-within .bte-ideal-marker-body {
  max-height: 1400px;
  padding-top: 14px;
  opacity: 1;
  transform: translateY(0);
}

.bte-ideal-marker-body .bte-improvement-block,
.bte-ideal-marker-body .bte-marker-video-block {
  padding: 12px;
  border-radius: 14px;
  background: var(--bte-page);
}

.bte-ideal-marker-body .bte-improvement-block .bte-guidance--card div {
  padding: 0;
  border-top: 0;
  background: transparent;
}

.bte-ideal-marker-body div {
  padding: 12px;
  border-radius: 14px;
  background: var(--bte-page);
}

.bte-ideal-marker-body span {
  display: block;
  margin-bottom: 5px;
  color: var(--bte-ink);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bte-ideal-marker-body p {
  margin: 0;
  color: var(--bte-muted);
  font-size: 14px;
  line-height: 1.5;
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

  .gradio-container {
    padding-inline: 12px !important;
    width: calc(100vw - 64px) !important;
    max-width: calc(100vw - 64px) !important;
    overflow-x: hidden !important;
  }

  .bte-title,
  .bte-step-row-block,
  .bte-hero-grid,
  .bte-shell,
  .bte-formation {
    width: calc(100vw - 88px) !important;
    max-width: calc(100vw - 88px) !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
  }

  .bte-title > *,
  .bte-title *,
  .bte-hero-grid > *,
  .bte-title-copy,
  .bte-title-copy * {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-wrap: anywhere;
  }

  .gradio-container input,
  .gradio-container textarea,
  .gradio-container button {
    max-width: 100% !important;
    min-width: 0 !important;
  }

  .bte-title {
    padding: 22px 16px 18px;
    grid-template-columns: 1fr;
    gap: 18px;
  }

  .bte-title-copy {
    padding-right: 0;
    padding-bottom: 18px;
  }

  .bte-title-copy::after,
  .bte-title-hackathon-wrap::after {
    display: none;
  }

  .bte-title-hackathon-wrap {
    padding-right: 0;
    padding-bottom: 18px;
  }

  .bte-title-copy,
  .bte-title-hackathon-wrap,
  .bte-title-credits-wrap {
    border-bottom: 1px solid rgba(255, 255, 255, 0.42);
  }

  .bte-title-credits-wrap {
    padding-left: 0;
    border-bottom: 0;
  }

  .bte-hack-badges-grid {
    grid-template-columns: 1fr;
  }

  .bte-title-attribution-wrap {
    justify-self: start;
    width: 100%;
  }

  .bte-hero-attribution {
    grid-template-columns: 1fr;
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

  .bte-hero-grid {
    display: grid !important;
    grid-template-columns: 1fr !important;
  }

  .bte-step-row {
    grid-template-columns: 1fr;
  }

  .bte-title,
  .bte-step-row .bte-step-heading:nth-child(2),
  .bte-step-row .bte-step-heading:nth-child(3),
  .bte-hero-grid > :nth-child(2),
  .bte-hero-grid > :nth-child(3),
  .bte-run-status,
  .bte-report-anchor,
  .bte-ideal-row {
    display: none !important;
  }

  .bte-final-report {
    display: grid !important;
    width: calc(100vw - 88px) !important;
    max-width: calc(100vw - 88px) !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
  }

  .bte-workflow-panel--upload,
  .bte-workflow-panel--analysis {
    min-width: 0 !important;
  }

  .bte-step-heading {
    min-height: auto;
    align-items: start;
    margin-bottom: 10px;
  }

  .bte-step-heading h2 {
    font-size: 19px !important;
    overflow-wrap: anywhere;
  }

  .bte-shell {
    border-radius: var(--bte-radius);
    padding: 12px;
  }

  .bte-engine {
    padding: 16px;
  }

  .bte-formation {
    min-height: 560px;
    padding: 14px;
    border-radius: var(--bte-radius);
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

  .bte-panel-upload .bte-uploader [class*="drop"],
  .bte-panel-upload .bte-uploader [class*="upload"] {
    min-height: 0 !important;
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
    border-radius: var(--bte-radius);
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

  .bte-insight-grid,
  .bte-guidance,
  .bte-improvement-block .bte-guidance--card {
    grid-template-columns: 1fr;
  }

  .bte-marker-value {
    text-align: left;
  }

  .bte-ideal-hero {
    display: grid;
    padding: 18px;
  }

  .bte-ideal-stats,
  .bte-ideal-grid {
    grid-template-columns: 1fr;
  }

  .bte-ideal-marker-head {
    display: grid;
  }

  .bte-ideal-marker-head strong {
    white-space: normal;
  }
}

"""


with gr.Blocks(title="Blood Test Explainer") as demo:
    with gr.Group(elem_classes=["bte-title-rail"]):
        with gr.Row(equal_height=True, elem_classes=["bte-title"]):
            with gr.Column(scale=1, min_width=0, elem_classes=["bte-title-copy"]):
                gr.HTML(
                    """
                    <div>
                      <p class="bte-kicker">Clinical clarity from raw documents</p>
                      <h1>Blood Test Explainer</h1>
                      <p>Upload a lab report and turn dense medical paperwork into a polished health report with extracted values, age and sex context, and knowledge graph explanations.</p>
                    </div>
                    """
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["bte-title-hackathon-wrap"]):
                gr.HTML(hero_hackathon_panel_html())
            with gr.Column(scale=1, min_width=0, elem_classes=["bte-title-credits-wrap"]):
                gr.HTML(hero_attribution_html())

    workflow_phase = gr.HTML(
        workflow_phase_html("ready"),
        elem_classes=["bte-workflow-phase"],
    )

    gr.HTML(
        """
        <div class="bte-step-row">
          <div class="bte-step-heading bte-step-heading--upload">
            <span>1</span>
            <h2>Upload your blood tests in any suitable format</h2>
          </div>
          <div class="bte-step-heading bte-step-heading--analysis">
            <span>2</span>
            <h2>Wait until it's analysed by our AI Agents</h2>
          </div>
          <div class="bte-step-heading bte-step-heading--report">
            <span>3</span>
            <h2>Review agents' steps and the blood test report</h2>
          </div>
        </div>
        """,
        elem_classes=["bte-step-row-block"],
    )

    with gr.Row(equal_height=False, elem_classes=["bte-hero-grid"]):
        with gr.Column(scale=4, min_width=320, elem_classes=["bte-workflow-panel", "bte-panel-upload"]):
            with gr.Group(elem_classes=["bte-shell", "bte-upload-card"]):
                upload_hint = gr.HTML(
                    '<p class="bte-upload-hint">Supported formats: PDF, PNG, JPEG, WebP</p>',
                    elem_classes=["bte-upload-hint-wrap"],
                )
                with gr.Group(elem_classes=["bte-upload-dropzone"]) as upload_dropzone:
                    uploaded = gr.File(
                        label="Upload medical test document",
                        file_count="single",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".txt", ".csv"],
                        type="filepath",
                        elem_classes=["bte-uploader"],
                    )
                selected_document = gr.HTML(
                    selected_document_html(),
                    visible=False,
                    elem_classes=["bte-selected-document-wrap"],
                )

        with gr.Column(scale=4, min_width=300, elem_classes=["bte-workflow-panel", "bte-panel-analysis"]):
            gr.HTML(analysis_animation_html())

        with gr.Column(scale=4, min_width=300, elem_classes=["bte-workflow-panel", "bte-panel-result", "bte-panel-trace"]):
            with gr.Group(elem_classes=["bte-shell", "bte-agent-panel"]):
                agent_trace = gr.HTML(
                    empty_trace_html(),
                    elem_classes=["bte-agent-trace"],
                )

    status = gr.HTML(
        _status_html("Ready", "Upload a lab report to create the first interactive extraction draft."),
        elem_classes=["bte-status-row"],
        visible=False,
    )

    report_panel = gr.Group(visible=False, elem_classes=["bte-report-panel", "bte-final-row"])
    with report_panel:
        report = gr.HTML(empty_report_html())

    uploaded.change(
        upload_state,
        inputs=[uploaded],
        outputs=[upload_dropzone, upload_hint, selected_document, workflow_phase, agent_trace],
        show_progress="hidden",
    ).then(
        show_processing,
        outputs=[status, report_panel, report, workflow_phase, agent_trace],
        scroll_to_output=True,
        show_progress="hidden",
    ).then(
        extract_lab_values,
        inputs=[uploaded],
        outputs=[status, report, report_panel, workflow_phase, agent_trace],
        scroll_to_output=True,
        show_progress="hidden",
    )


if __name__ == "__main__":
    _boot_log("launching Gradio demo")
    demo.launch(css=CUSTOM_CSS, js=trace_hover_js())
