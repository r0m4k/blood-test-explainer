"""Build a step-by-step trace of the analysis pipeline for the agent trace panel."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class PipelineStep:
    id: str
    title: str
    status: str
    summary: str
    return_code: int | None = 0
    prompt: str | None = None
    input_preview: str | None = None
    output_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    steps: list[PipelineStep] = [
        PipelineStep(
            id="document_intake",
            title="Step 1 — Document intake",
            status="complete",
            return_code=0,
            summary="\n".join(intake_lines),
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
            summary=(
                f"Model/backend: {summary.get('model') or summary.get('repo') or backend}. "
                f"Extracted structured JSON from the document."
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
            summary=(
                f"Parsed {len(extraction.tests)} marker(s), "
                f"{len(extraction.notes)} note(s). "
                f"Patient sex: {patient.get('sex', 'unknown')}; "
                f"age group: {patient.get('age_group', 'unknown')}."
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
            summary=(
                f"Enriched {report_summary.get('enriched_markers', 0)} of "
                f"{report_summary.get('total_markers', 0)} marker(s). "
                f"Unmatched: {len(report_summary.get('unmatched_markers') or [])}."
            ),
            output_preview=_truncate(json.dumps(report_summary, indent=2)),
            metadata={
                "enriched_markers": report_summary.get("enriched_markers", 0),
                "total_markers": report_summary.get("total_markers", 0),
                "unmatched_markers": report_summary.get("unmatched_markers") or [],
            },
        ),
        PipelineStep(
            id="pattern_detection",
            title="Step 5 — Cross-marker pattern detection",
            status="complete",
            return_code=0,
            summary=_pattern_summary(interpretation),
            output_preview=_pattern_output(interpretation),
            metadata={
                "flagged_markers": len(interpretation.flagged),
                "patterns_detected": len(interpretation.patterns),
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


def _step_teaser(step: PipelineStep) -> str:
    first_line = step.summary.strip().split("\n", 1)[0]
    if len(first_line) > 96:
        return first_line[:93].rstrip() + "..."
    return first_line


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


def _status_badge(status: str) -> str:
    css = {
        "complete": "bte-trace-status--complete",
        "running": "bte-trace-status--running",
        "failed": "bte-trace-status--failed",
        "pending": "bte-trace-status--pending",
    }.get(status, "bte-trace-status--unknown")
    label = status.replace("_", " ").title()
    return f'<span class="bte-trace-status {css}">{html.escape(label)}</span>'


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


def _trace_block(body: str) -> str:
    return f"""
    <section class="bte-trace-panel" aria-label="Agent actions">
      <div class="bte-trace-steps">
        {body}
      </div>
    </section>
    """


def step_to_html(step: PipelineStep) -> str:
    sections: list[str] = [
        _metrics_table(step),
        f'<p class="bte-trace-summary">{html.escape(step.summary)}</p>',
    ]
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
    return f"""
    <details class="bte-trace-step">
      <summary class="bte-trace-step-summary">
        <span class="bte-trace-step-heading">
          <span class="bte-trace-step-title">{html.escape(step.title)}</span>
          {_status_badge(step.status)}
        </span>
        <span class="bte-trace-step-meta">
          Return code: {html.escape(_format_return_code(step.return_code))}
        </span>
        <span class="bte-trace-step-teaser">{html.escape(_step_teaser(step))}</span>
      </summary>
      <div class="bte-trace-step-body">
        {"".join(sections)}
      </div>
    </details>
    """


def _placeholder_pipeline_steps(
    *,
    status: str,
    summary: str,
    return_code: int | None = None,
    pipeline_phase: str,
) -> list[PipelineStep]:
    return [
        PipelineStep(
            id=step_id,
            title=title,
            status=status,
            return_code=return_code,
            summary=summary,
            metadata={"pipeline_phase": pipeline_phase},
        )
        for step_id, title in _PIPELINE_STEP_DEFS
    ]


def trace_to_html(steps: list[PipelineStep]) -> str:
    body = "".join(step_to_html(step) for step in steps)
    return _trace_block(body)


def empty_trace_html() -> str:
    steps = _placeholder_pipeline_steps(
        status="pending",
        summary="Waiting for a lab report upload to start analysis.",
        return_code=None,
        pipeline_phase="ready",
    )
    return trace_to_html(steps)


def processing_trace_html() -> str:
    steps = _placeholder_pipeline_steps(
        status="running",
        summary="Waiting for upstream steps to finish…",
        return_code=None,
        pipeline_phase="processing",
    )
    return trace_to_html(steps)


def error_trace_html(message: str) -> str:
    failed_step = PipelineStep(
        id="vision_extraction",
        title="Step 2 — Vision extraction (LLM)",
        status="failed",
        return_code=1,
        summary=message,
        metadata={"pipeline_phase": "failed"},
    )
    body = step_to_html(failed_step)
    return _trace_block(body)


def step_to_markdown(step: PipelineStep) -> str:
    parts = [f"**{step.title}**", step.summary]
    if step.prompt:
        parts.append(
            f"<details><summary>Full prompt</summary>\n\n```\n{step.prompt}\n```\n</details>"
        )
    if step.input_preview:
        parts.append(
            f"<details><summary>Input preview</summary>\n\n```\n{step.input_preview}\n```\n</details>"
        )
    if step.output_preview:
        parts.append(
            f"<details><summary>Output preview</summary>\n\n```\n{step.output_preview}\n```\n</details>"
        )
    return "\n\n".join(parts)


def trace_to_chat_messages(steps: list[PipelineStep]) -> list[dict[str, str]]:
    intro = (
        "**Analysis pipeline complete.** Below are the agent steps that processed your document."
    )
    messages = [{"role": "assistant", "content": intro}]
    for step in steps:
        messages.append({"role": "assistant", "content": step_to_markdown(step)})
    return messages


def serialize_steps(steps: list[PipelineStep]) -> list[dict[str, Any]]:
    return [asdict(step) for step in steps]


def interpretation_to_dict(interpretation: Interpretation) -> dict[str, Any]:
    return {
        "flagged": [
            {
                "marker": item.marker,
                "value": item.value,
                "unit": item.unit,
                "status": item.status,
                "reference_range": item.reference_range,
                "note": item.note,
                "questions": list(item.questions),
            }
            for item in interpretation.flagged
        ],
        "normal_count": interpretation.normal_count,
        "patterns": [{"name": p.name, "note": p.note} for p in interpretation.patterns],
        "disclaimer": interpretation.disclaimer,
    }


def extraction_to_dict(extraction: ExtractionResult) -> dict[str, Any]:
    return {
        "patient": extraction.patient,
        "tests": extraction.tests,
        "notes": extraction.notes,
        "raw_response": extraction.raw_response,
        "request_summary": extraction.request_summary,
    }


def build_session_state(
    extraction: ExtractionResult,
    health_report: dict[str, Any],
    steps: list[PipelineStep],
) -> dict[str, Any]:
    interpretation = build_interpretation(extraction.tests)
    return {
        "extraction": extraction_to_dict(extraction),
        "health_report": health_report,
        "interpretation": interpretation_to_dict(interpretation),
        "trace_steps": serialize_steps(steps),
    }
