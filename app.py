from __future__ import annotations

import os
import re
import time
import traceback
from html import escape
from typing import Any

import gradio as gr

from src.extraction import build_extractor
from src.interpretation_render import patterns_html
from src.local_env import load_local_env
from src.report_pipeline import build_health_report


load_local_env()
_BOOT_T0 = time.perf_counter()


def _boot_log(message: str) -> None:
    elapsed = time.perf_counter() - _BOOT_T0
    print(f"[Blood Test Explainer][{elapsed:0.2f}s] {message}", flush=True)

# The hosted API key field is only relevant when the API backend is active. The current Space
# path is ZeroGPU, so users should not see model/API configuration controls.
_API_MODE = os.getenv("EXTRACTOR_BACKEND", "auto").strip().lower() == "api"
_boot_log("environment loaded")


def extract_lab_values(
    uploaded_file: str | None,
) -> tuple[str, str, Any, str]:
    if not uploaded_file:
        return (
            _status_html("Waiting for a document", "Upload a lab report to begin extraction."),
            empty_report_html("No document uploaded", "Choose a file first, then run extraction again."),
            gr.update(visible=True),
            workflow_phase_html("ready"),
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
        )

    health_report = build_health_report(result)
    summary = health_report["summary"]
    patient = health_report["patient"]

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
        # Per-marker insight comes from the knowledge-graph report; append the cross-marker
        # patterns (anemia picture, liver cluster, lipid risk) which the per-marker report omits.
        report_html(health_report) + patterns_html(result.tests),
        gr.update(visible=True),
        workflow_phase_html("done"),
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
    if "401" in lowered or "unauthorized" in lowered:
        return (
            "The OpenBMB endpoint rejected the request. Check the API key or switch to the local "
            "ZeroGPU path."
        )
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


def show_processing() -> tuple[str, Any, str, str]:
    return (
        _status_html("Reading document", "Extracting patient context and markers, then matching them to the knowledge graph.", tone="loading"),
        gr.update(visible=False),
        "",
        workflow_phase_html("processing"),
    )


def upload_state(uploaded_file: str | None) -> tuple[Any, Any]:
    if not uploaded_file:
        return (
            gr.update(visible=True),
            gr.update(visible=False, value=selected_document_html()),
            workflow_phase_html("ready"),
        )

    filename = os.path.basename(uploaded_file)
    return (
        gr.update(visible=False),
        gr.update(visible=True, value=selected_document_html(filename)),
        workflow_phase_html("processing"),
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
        <p>Ready to extract markers, patient context, values, units, ranges, and confidence signals.</p>
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


def result_preview_html() -> str:
    return """
    <section class="bte-formation bte-formation--result" aria-label="Clear lab results preview">
      <div class="bte-formation-stage bte-formation-stage--result">
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
            <div class="bte-mini-card bte-mini-card--red">
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
    improve = _final_improvement_text(instructions)
    context = _final_context_text(test)

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
        <div>
          <span>How to improve</span>
          <p>{escape(improve)}</p>
        </div>
        <div>
          <span>Reference context</span>
          <p>{escape(context)}</p>
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


def _final_improvement_text(instructions: dict[str, Any]) -> str:
    parts = []
    for label, key in (("Food", "food"), ("Exercise", "exercises"), ("Supplements", "supplements")):
        items = instructions.get(key)
        if isinstance(items, list) and items:
            parts.append(f"{label}: {' '.join(str(item) for item in items[:2])}")
    return " ".join(parts) or "No improvement guidance is stored for this marker."


def _final_context_text(marker: dict[str, Any]) -> str:
    reference = _reference_label(marker)
    sex_context = ((marker.get("knowledge") or {}).get("sex_significance") or {}).get("summary")
    confidence = _confidence_percent(marker.get("confidence"))
    source = _text(marker.get("source_text"), "")
    parts = [reference, f"Extraction confidence: {confidence}%."]
    if sex_context:
        parts.append(str(sex_context))
    if source:
        parts.append(f"Source row: {source}")
    return " ".join(parts)


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

.bte-title {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin: 0 auto 18px !important;
  padding: 30px 28px 28px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 0.46fr);
  gap: 32px;
  align-items: center;
  border: 1px solid rgba(255, 255, 255, 0.42);
  border-radius: var(--bte-radius);
  background:
    linear-gradient(120deg, rgba(18, 128, 92, 0.98) 0%, rgba(37, 99, 235, 0.95) 58%, rgba(191, 52, 52, 0.82) 100%),
    #12805c;
  box-shadow: var(--bte-shadow-strong);
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
  max-width: 820px;
  margin: 0;
}

.bte-title-copy {
  min-width: 0;
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

.bte-hero-grid .bte-upload-card {
  border: 1px solid var(--bte-line) !important;
  border-radius: var(--bte-radius) !important;
  padding: 18px !important;
  background: var(--bte-card) !important;
  box-shadow: var(--bte-shadow) !important;
  overflow: hidden !important;
}

.bte-hero-grid .block:has(.bte-upload-card),
.bte-hero-grid div:has(> .bte-upload-card) {
  height: 430px !important;
  min-height: 430px !important;
  border: 1px solid var(--bte-line) !important;
  border-radius: var(--bte-radius) !important;
  padding: 18px !important;
  background: var(--bte-card) !important;
  box-shadow: var(--bte-shadow) !important;
  overflow: hidden !important;
}

.bte-hero-grid .block:has(.bte-upload-card) .bte-upload-card,
.bte-hero-grid div:has(> .bte-upload-card) > .bte-upload-card {
  height: 100% !important;
  min-height: 0 !important;
  border: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
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
.bte-final-row {
  position: relative;
}

.bte-workflow-phase,
.bte-workflow-phase-marker {
  display: none !important;
}

.bte-step-row-block,
.bte-step-row-block > div {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.bte-step-row-block .prose,
.bte-step-row-block .html-container,
.bte-step-row-block .block {
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
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
  filter: saturate(1);
  transform: translateY(-1px);
  border-color: rgba(37, 99, 235, 0.32);
  background: linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(255, 255, 255, 0.98));
  box-shadow: 0 16px 34px rgba(37, 99, 235, 0.1);
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
.bte-panel-result .bte-formation,
.bte-final-row .bte-report {
  transition: opacity 220ms ease, filter 220ms ease, box-shadow 220ms ease, transform 220ms ease, border-color 220ms ease, background 220ms ease;
}

.bte-step-heading--report {
  margin-top: 0;
  min-height: 112px;
  padding: 18px;
}

.bte-upload-card {
  height: 430px !important;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 430px;
  overflow: visible !important;
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
.bte-panel-result .bte-formation--result {
  overflow: visible;
}

.bte-panel-result .bte-smart-report,
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
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis {
  opacity: 0.42;
  filter: saturate(0.5);
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result {
  opacity: 1;
  filter: saturate(1);
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="ready"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-upload .bte-upload-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis {
  animation-play-state: paused !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-formation--analysis,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result {
  animation-play-state: running !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-smart-report,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-mini-card,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-mini-chart span {
  animation-play-state: running !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-source-doc,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-scan-band,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow span,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow i,
.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="processing"]) ~ .bte-hero-grid .bte-panel-analysis .bte-flow b {
  animation-play-state: running !important;
}

.bte-workflow-phase:has(.bte-workflow-phase-marker[data-phase="done"]) ~ .bte-hero-grid .bte-panel-result .bte-formation--result .bte-smart-report,
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

.bte-upload-hint {
  margin: 0 0 10px !important;
  color: var(--bte-muted) !important;
  font-size: 13px !important;
  font-weight: 650 !important;
  text-align: center !important;
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
  border-color: #d8e2ee !important;
  border-radius: 18px !important;
  color: var(--bte-ink) !important;
}

.bte-uploader [class*="drop"],
.bte-uploader [class*="upload"] {
  min-height: 250px !important;
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
  background: var(--bte-surface);
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
  margin-top: 34px;
  display: grid;
  gap: 16px;
  background: var(--bte-page);
}

.bte-final-report {
  width: var(--bte-rail) !important;
  max-width: var(--bte-rail) !important;
  margin: 0 auto !important;
  background: rgb(248, 249, 252) !important;
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
  padding: 28px;
  border: 1px solid rgba(255, 255, 255, 0.42);
  border-radius: var(--bte-radius);
  color: #ffffff;
  background:
    linear-gradient(120deg, rgba(18, 128, 92, 0.98) 0%, rgba(37, 99, 235, 0.95) 58%, rgba(191, 52, 52, 0.82) 100%),
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
  gap: 12px;
  margin: 0;
}

.bte-ideal-filter {
  position: absolute;
  inline-size: 1px;
  block-size: 1px;
  opacity: 0;
  pointer-events: none;
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
  gap: 16px;
}

.bte-ideal-column {
  display: grid;
  align-content: start;
  gap: 16px;
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
  max-height: 520px;
  padding-top: 14px;
  opacity: 1;
  transform: translateY(0);
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
  .bte-guidance {
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
    # Flatten the report container so only the inner .bte-report card shows (no double box).
    gr.HTML(
        "<style>.bte-report-panel,.bte-report-panel>*{background:transparent !important;"
        "border:0 !important;box-shadow:none !important;padding:0 !important;}</style>"
    )
    with gr.Row(equal_height=True, elem_classes=["bte-title"]):
        with gr.Column(scale=1, min_width=420, elem_classes=["bte-title-copy"]):
            gr.HTML(
                """
                <div>
                  <p class="bte-kicker">Clinical clarity from raw documents</p>
                  <h1>Blood Test Explainer</h1>
                  <p>Upload a lab report and turn dense medical paperwork into a polished health report with extracted values, age and sex context, and knowledge graph explanations.</p>
                </div>
                """
            )

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
            <h2>Get your blood test results in the clearest possible format</h2>
          </div>
        </div>
        """,
        elem_classes=["bte-step-row-block"],
    )

    with gr.Row(equal_height=False, elem_classes=["bte-hero-grid"]):
        with gr.Column(scale=4, min_width=320, elem_classes=["bte-workflow-panel", "bte-panel-upload"]):
            with gr.Group(elem_classes=["bte-shell", "bte-upload-card"]):
                gr.HTML(
                    '<p class="bte-upload-hint">Supported formats: PDF, PNG, JPEG</p>',
                    elem_classes=["bte-upload-hint-wrap"],
                )
                with gr.Group() as upload_dropzone:
                    uploaded = gr.File(
                        label="Upload medical test document",
                        file_count="single",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".txt", ".csv"],
                        type="filepath",
                        elem_classes=["bte-uploader"],
                    )
                selected_document = gr.HTML(selected_document_html(), visible=False)

        with gr.Column(scale=4, min_width=300, elem_classes=["bte-workflow-panel", "bte-panel-analysis"]):
            gr.HTML(analysis_animation_html())

        with gr.Column(scale=4, min_width=300, elem_classes=["bte-workflow-panel", "bte-panel-result"]):
            gr.HTML(result_preview_html())

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
        outputs=[upload_dropzone, selected_document, workflow_phase],
        show_progress="hidden",
    ).then(
        show_processing,
        outputs=[status, report_panel, report, workflow_phase],
        scroll_to_output=True,
        show_progress="hidden",
    ).then(
        extract_lab_values,
        inputs=[uploaded],
        outputs=[status, report, report_panel, workflow_phase],
        scroll_to_output=True,
        show_progress="hidden",
    )


if __name__ == "__main__":
    _boot_log("launching Gradio demo")
    demo.launch(css=CUSTOM_CSS)
