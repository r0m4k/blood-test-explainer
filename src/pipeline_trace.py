"""Build a step-by-step trace of the analysis pipeline for the agent trace panel."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.interpretation import Interpretation, build_interpretation
from src.openbmb_client import EXTRACTION_PROMPT, ExtractionResult

_MAX_PREVIEW = 2400

_PIPELINE_STEP_DEFS: tuple[tuple[str, str], ...] = (
    ("document_intake", "Step 1 — Document intake"),
    ("vision_extraction", "Step 2 — Vision extraction (LLM)"),
    ("schema_normalization", "Step 3 — Schema normalization"),
    ("knowledge_graph", "Step 4 — Knowledge graph enrichment"),
    ("pattern_detection", "Step 5 — Cross-marker pattern detection"),
)

_STEP_COPY: dict[str, dict[str, str]] = {
    "document_intake": {
        "explanation": (
            "This step gets your upload ready for the AI. It checks the file type, turns PDF "
            "pages into images when needed, and packages the report so the vision model can read it."
        ),
        "pending": "Waiting for you to upload a lab report.",
        "running": "Reading your file and preparing it for the vision model.",
    },
    "vision_extraction": {
        "explanation": (
            "This step is where the AI actually reads your lab report. The vision model scans "
            "tables, values, units, and labels, then turns what it sees into structured lab data."
        ),
        "pending": "The vision model has not run yet.",
        "running": "The vision model is reading your report and extracting lab values.",
    },
    "schema_normalization": {
        "explanation": (
            "Raw model output can be messy. This step cleans it up into a consistent list of "
            "markers, values, units, and patient details the rest of the app can trust."
        ),
        "pending": "Structured marker parsing has not started yet.",
        "running": "Turning the model output into clean, structured lab values.",
    },
    "knowledge_graph": {
        "explanation": (
            "Numbers alone are hard to interpret. This step matches each marker to our knowledge "
            "graph so the report can explain what a result generally means in plain language."
        ),
        "pending": "Knowledge graph enrichment has not started yet.",
        "running": "Matching extracted markers to educational explanations.",
    },
    "pattern_detection": {
        "explanation": (
            "Single markers tell part of the story. This step looks across your results for "
            "related flags and patterns that may deserve extra attention together."
        ),
        "pending": "Cross-marker pattern checks have not started yet.",
        "running": "Checking how markers relate to each other across your report.",
    },
}


@dataclass(frozen=True)
class PipelineStep:
    id: str
    title: str
    status: str
    summary: str
    return_code: int | None = 0
    technical_details: str | None = None
    prompt: str | None = None
    input_preview: str | None = None
    output_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _step_copy(step_id: str, phase: str = "complete") -> str:
    copy = _STEP_COPY[step_id]
    if phase == "pending":
        return copy["pending"]
    if phase == "running":
        return copy["running"]
    return copy["explanation"]


def _summary_with_result(explanation: str, result_note: str | None) -> str:
    if not result_note:
        return explanation
    return f"{explanation}\n\nIn this run: {result_note}"


def _truncate(text: str | None, limit: int = _MAX_PREVIEW) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _marker_preview(tests: list[dict[str, Any]], limit: int = 3) -> str:
    lines: list[str] = []
    for test in tests[:limit]:
        marker = test.get("marker", "?")
        value = test.get("value", "?")
        unit = test.get("unit") or ""
        status = test.get("status") or "unknown"
        lines.append(f"- {marker}: {value} {unit} ({status})".strip())
    if len(tests) > limit:
        lines.append(f"- … and {len(tests) - limit} more")
    return "\n".join(lines)


def build_pipeline_trace(
    extraction: ExtractionResult,
    health_report: dict[str, Any],
    *,
    source_path: str | None = None,
) -> list[PipelineStep]:
    summary = extraction.request_summary or {}
    patient = health_report.get("patient") or extraction.patient or {}
    report_summary = health_report.get("summary") or {}
    interpretation = build_interpretation(extraction.tests)

    backend = summary.get("backend") or summary.get("api_url") or "unknown"
    file_name = Path(source_path).name if source_path else None
    runtime_return_code = summary.get("return_code", 0)

    intake_lines = [
        f"Backend: {backend}",
        f"Input modality: {summary.get('input_modality', 'unknown')}",
        f"Document parts: {summary.get('document_parts', '?')}",
    ]
    if summary.get("pages_rendered") is not None:
        intake_lines.append(f"Pages rendered to images: {summary.get('pages_rendered')}")
    if summary.get("max_pages") is not None:
        intake_lines.append(f"Max pages: {summary.get('max_pages')}")
    if file_name:
        intake_lines.append(f"File: {file_name}")
    preview = summary.get("user_message_preview") or {}
    if preview:
        intake_lines.append(
            f"Payload preview: {preview.get('image_count', 0)} image(s), "
            f"{preview.get('text_characters', 0)} text character(s)"
        )

    intake_result_parts: list[str] = []
    if file_name:
        intake_result_parts.append(f"we prepared “{file_name}”")
    modality = summary.get("input_modality")
    if modality:
        intake_result_parts.append(f"the input was treated as a {modality} document")
    pages_rendered = summary.get("pages_rendered")
    if pages_rendered is not None:
        intake_result_parts.append(
            f"{pages_rendered} page(s) were sent to the model as image(s)"
        )
    elif preview.get("image_count"):
        intake_result_parts.append(
            f"{preview.get('image_count')} image(s) were included in the model payload"
        )
    intake_result = ", and ".join(intake_result_parts) + "." if intake_result_parts else None

    model_name = summary.get("model") or summary.get("repo") or backend
    extraction_result_parts = [f"the {model_name} model extracted structured lab data from your report"]
    if summary.get("duration_ms"):
        seconds = max(1, int(summary["duration_ms"]) // 1000)
        extraction_result_parts.append(f"in about {seconds} second(s)")
    extraction_result = ", ".join(extraction_result_parts) + "."

    normalization_result = (
        f"we parsed {len(extraction.tests)} marker(s) and {len(extraction.notes)} note(s), "
        f"with patient context recorded as {patient.get('sex', 'unknown')} / "
        f"{patient.get('age_group', 'unknown')} when available"
    ) + "."

    enriched = report_summary.get("enriched_markers", 0)
    total = report_summary.get("total_markers", 0)
    unmatched = len(report_summary.get("unmatched_markers") or [])
    knowledge_result = (
        f"{enriched} of {total} marker(s) were matched to knowledge-base explanations"
        + (f" and {unmatched} marker(s) had no close match" if unmatched else "")
    ) + "."

    flagged = len(interpretation.flagged)
    patterns = len(interpretation.patterns)
    pattern_result = (
        f"we flagged {flagged} marker(s) for attention and found {patterns} cross-marker pattern(s), "
        f"with {interpretation.normal_count} marker(s) recognized as in-range"
    ) + "."

    steps: list[PipelineStep] = [
        PipelineStep(
            id="document_intake",
            title="Step 1 — Document intake",
            status="complete",
            return_code=0,
            summary=_summary_with_result(_step_copy("document_intake"), intake_result),
            technical_details="\n".join(intake_lines),
            input_preview=file_name,
            metadata={
                "backend": backend,
                "document_parts": summary.get("document_parts"),
                "max_pages": summary.get("max_pages"),
                "file": file_name,
                **preview,
            },
        ),
        PipelineStep(
            id="vision_extraction",
            title="Step 2 — Vision extraction (LLM)",
            status="complete",
            return_code=runtime_return_code,
            summary=_summary_with_result(_step_copy("vision_extraction"), extraction_result),
            technical_details=(
                f"Model/backend: {model_name}\n"
                f"HTTP status: {summary.get('http_status', '—')}\n"
                f"Duration (ms): {summary.get('duration_ms', '—')}\n"
                f"Document parts: {summary.get('document_parts', '?')}"
            ),
            prompt=summary.get("extraction_prompt") or EXTRACTION_PROMPT,
            input_preview=_stringify_preview(
                summary.get("composed_prompt") or summary.get("messages_preview")
            ),
            output_preview=_truncate(extraction.raw_response),
            metadata={
                "backend": backend,
                "model": summary.get("model") or summary.get("repo"),
                "api_url": summary.get("api_url") or summary.get("url"),
                "http_status": summary.get("http_status"),
                "duration_ms": summary.get("duration_ms"),
                "return_code": runtime_return_code,
                "document_parts": summary.get("document_parts"),
            },
        ),
        PipelineStep(
            id="schema_normalization",
            title="Step 3 — Schema normalization",
            status="complete",
            return_code=0,
            summary=_summary_with_result(_step_copy("schema_normalization"), normalization_result),
            technical_details=(
                f"Markers parsed: {len(extraction.tests)}\n"
                f"Notes parsed: {len(extraction.notes)}\n"
                f"Patient sex: {patient.get('sex', 'unknown')}\n"
                f"Patient age group: {patient.get('age_group', 'unknown')}"
            ),
            output_preview=_marker_preview(extraction.tests),
            metadata={
                "markers_parsed": len(extraction.tests),
                "notes_parsed": len(extraction.notes),
                "patient_sex": patient.get("sex", "unknown"),
                "patient_age_group": patient.get("age_group", "unknown"),
                "notes": extraction.notes[:5],
            },
        ),
        PipelineStep(
            id="knowledge_graph",
            title="Step 4 — Knowledge graph enrichment",
            status="complete",
            return_code=0,
            summary=_summary_with_result(_step_copy("knowledge_graph"), knowledge_result),
            technical_details=(
                f"Enriched markers: {enriched}\n"
                f"Total markers: {total}\n"
                f"Unmatched markers: {unmatched}"
            ),
            output_preview=_truncate(json.dumps(report_summary, indent=2)),
            metadata={
                "enriched_markers": enriched,
                "total_markers": total,
                "unmatched_markers": report_summary.get("unmatched_markers") or [],
            },
        ),
        PipelineStep(
            id="pattern_detection",
            title="Step 5 — Cross-marker pattern detection",
            status="complete",
            return_code=0,
            summary=_summary_with_result(_step_copy("pattern_detection"), pattern_result),
            technical_details=_pattern_summary(interpretation),
            output_preview=_pattern_output(interpretation),
            metadata={
                "flagged_markers": flagged,
                "patterns_detected": patterns,
                "normal_count": interpretation.normal_count,
            },
        ),
    ]
    return steps


def _stringify_preview(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _truncate(value)
    return _truncate(json.dumps(value, indent=2))


def _pattern_summary(interpretation: Interpretation) -> str:
    flagged = len(interpretation.flagged)
    patterns = len(interpretation.patterns)
    return (
        f"Flagged markers: {flagged}. "
        f"Cross-marker patterns detected: {patterns}. "
        f"In-range recognized markers: {interpretation.normal_count}."
    )


def _pattern_output(interpretation: Interpretation) -> str | None:
    if not interpretation.patterns and not interpretation.flagged:
        return "No flagged markers or cross-marker patterns."
    lines: list[str] = []
    for insight in interpretation.flagged[:6]:
        note = insight.note or "(no KB note)"
        lines.append(f"- {insight.marker} ({insight.status}): {note}")
    if len(interpretation.flagged) > 6:
        lines.append(f"- … and {len(interpretation.flagged) - 6} more flagged marker(s)")
    for pattern in interpretation.patterns:
        lines.append(f"- Pattern — {pattern.name}: {pattern.note}")
    return "\n".join(lines)


def _format_return_code(code: int | None) -> str:
    if code is None:
        return "—"
    return str(code)


def _format_meta_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _metrics_table(step: PipelineStep) -> str:
    rows: list[tuple[str, str]] = [
        ("Status", step.status.replace("_", " ").title()),
        ("Return code", _format_return_code(step.return_code)),
    ]
    skip_keys = {"notes", "unmatched_markers"}
    for key, value in step.metadata.items():
        if key in skip_keys or value in (None, "", [], {}):
            continue
        label = key.replace("_", " ").title()
        rows.append((label, _format_meta_value(value)))

    cells = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in rows
    )
    return f'<dl class="bte-trace-meta">{cells}</dl>'


def trace_hover_js() -> str:
    """Frontend JS for Gradio launch(); inline scripts in gr.HTML are stripped."""
    return """
(function () {
  if (window.__bteTraceHoverInit) return;
  window.__bteTraceHoverInit = true;

  var OPEN_DELAY_MS = 180;
  var CLOSE_DELAY_MS = 140;
  var openTimers = new WeakMap();
  var closeTimers = new WeakMap();

  function isInteractiveStep(step) {
    if (!step || !step.classList.contains("bte-trace-step")) return false;
    if (step.classList.contains("bte-trace-step--locked")) return false;
    var panel = step.closest(".bte-trace-panel");
    return panel && panel.dataset.interactive === "true";
  }

  function clearOpenTimer(step) {
    var timer = openTimers.get(step);
    if (timer) {
      clearTimeout(timer);
      openTimers.delete(step);
    }
  }

  function clearCloseTimer(step) {
    var timer = closeTimers.get(step);
    if (timer) {
      clearTimeout(timer);
      closeTimers.delete(step);
    }
  }

  function openStep(step) {
    clearCloseTimer(step);
    step.classList.add("is-open");
  }

  function closeStep(step) {
    clearOpenTimer(step);
    step.classList.remove("is-open");
  }

  function scheduleOpen(step) {
    clearCloseTimer(step);
    if (step.classList.contains("is-open")) return;
    clearOpenTimer(step);
    openTimers.set(
      step,
      window.setTimeout(function () {
        openTimers.delete(step);
        openStep(step);
      }, OPEN_DELAY_MS)
    );
  }

  function scheduleClose(step) {
    if (step.dataset.pinned === "1") return;
    clearOpenTimer(step);
    clearCloseTimer(step);
    closeTimers.set(
      step,
      window.setTimeout(function () {
        closeTimers.delete(step);
        closeStep(step);
      }, CLOSE_DELAY_MS)
    );
  }

  document.addEventListener("mouseover", function (event) {
    var step = event.target.closest(".bte-trace-step");
    if (!isInteractiveStep(step)) return;
    scheduleOpen(step);
  });

  document.addEventListener("mouseout", function (event) {
    var step = event.target.closest(".bte-trace-step");
    if (!isInteractiveStep(step)) return;
    if (step.contains(event.relatedTarget)) return;
    scheduleClose(step);
  });

  document.addEventListener("click", function (event) {
    var summary = event.target.closest(".bte-trace-step-summary");
    if (!summary) return;
    var step = summary.closest(".bte-trace-step");
    if (!isInteractiveStep(step)) return;
    event.preventDefault();
    if (step.dataset.pinned === "1") {
      step.dataset.pinned = "0";
      closeStep(step);
      return;
    }
    step.dataset.pinned = "1";
    openStep(step);
  });
})();
"""


def _trace_block(body: str, *, interactive: bool = True) -> str:
    panel_class = "bte-trace-panel" if interactive else "bte-trace-panel bte-trace-panel--locked"
    interactive_attr = "true" if interactive else "false"
    return f"""
    <section class="{panel_class}" aria-label="Agent actions" data-interactive="{interactive_attr}">
      <div class="bte-trace-steps">
        {body}
      </div>
    </section>
    """


def _technical_details_block(step: PipelineStep) -> str | None:
    sections: list[str] = []
    metrics = _metrics_table(step)
    if metrics:
        sections.append(metrics)
    if step.technical_details:
        sections.append(
            f'<pre class="bte-trace-technical-text">{html.escape(step.technical_details)}</pre>'
        )
    if not sections:
        return None
    return (
        '<details class="bte-trace-subdetails">'
        "<summary>Technical details</summary>"
        f'<div class="bte-trace-technical">{"".join(sections)}</div>'
        "</details>"
    )


def _render_summary_html(summary: str) -> str:
    parts = summary.split("\n\n", 1)
    chunks = [f'<p class="bte-trace-explanation">{html.escape(parts[0])}</p>']
    if len(parts) > 1:
        chunks.append(f'<p class="bte-trace-result">{html.escape(parts[1])}</p>')
    return "".join(chunks)


def _step_body_sections(step: PipelineStep) -> str:
    sections: list[str] = [_render_summary_html(step.summary)]
    technical = _technical_details_block(step)
    if technical:
        sections.append(technical)
    if step.prompt:
        sections.append(
            '<details class="bte-trace-subdetails">'
            "<summary>Full prompt</summary>"
            f"<pre>{html.escape(step.prompt)}</pre>"
            "</details>"
        )
    if step.input_preview:
        sections.append(
            '<details class="bte-trace-subdetails">'
            "<summary>Input preview</summary>"
            f"<pre>{html.escape(step.input_preview)}</pre>"
            "</details>"
        )
    if step.output_preview:
        sections.append(
            '<details class="bte-trace-subdetails">'
            "<summary>Output preview</summary>"
            f"<pre>{html.escape(step.output_preview)}</pre>"
            "</details>"
        )
    return "".join(sections)


def step_to_html(step: PipelineStep, *, interactive: bool = True) -> str:
    title_html = f'<span class="bte-trace-step-title">{html.escape(step.title)}</span>'
    if not interactive:
        return f"""
    <div class="bte-trace-step bte-trace-step--locked">
      <div class="bte-trace-step-summary">
        <div class="bte-trace-step-heading">{title_html}</div>
      </div>
    </div>
    """
    return f"""
    <div class="bte-trace-step" data-pinned="0">
      <div class="bte-trace-step-summary" role="button" tabindex="0">
        <div class="bte-trace-step-heading">{title_html}</div>
      </div>
      <div class="bte-trace-step-collapse">
        <div class="bte-trace-step-body">
          {_step_body_sections(step)}
        </div>
      </div>
    </div>
    """


def _placeholder_pipeline_steps(
    *,
    status: str,
    return_code: int | None = None,
    pipeline_phase: str,
) -> list[PipelineStep]:
    phase = "pending" if status == "pending" else "running" if status == "running" else "complete"
    return [
        PipelineStep(
            id=step_id,
            title=title,
            status=status,
            return_code=return_code,
            summary=_step_copy(step_id, phase),
            metadata={"pipeline_phase": pipeline_phase},
        )
        for step_id, title in _PIPELINE_STEP_DEFS
    ]


def trace_to_html(steps: list[PipelineStep], *, interactive: bool = True) -> str:
    body = "".join(step_to_html(step, interactive=interactive) for step in steps)
    return _trace_block(body, interactive=interactive)


def empty_trace_html() -> str:
    steps = _placeholder_pipeline_steps(
        status="pending",
        return_code=None,
        pipeline_phase="ready",
    )
    return trace_to_html(steps, interactive=False)


def processing_trace_html() -> str:
    steps = _placeholder_pipeline_steps(
        status="running",
        return_code=None,
        pipeline_phase="processing",
    )
    return trace_to_html(steps, interactive=False)


def error_trace_html(message: str) -> str:
    failed_step = PipelineStep(
        id="vision_extraction",
        title="Step 2 — Vision extraction (LLM)",
        status="failed",
        return_code=1,
        summary=(
            f"{_step_copy('vision_extraction')}\n\n"
            f"In this run: the vision model could not finish reading your report. {message}"
        ),
        technical_details=message,
        metadata={"pipeline_phase": "failed"},
    )
    body = step_to_html(failed_step, interactive=False)
    return _trace_block(body, interactive=False)

