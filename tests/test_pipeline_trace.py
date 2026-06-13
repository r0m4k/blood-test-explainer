import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openbmb_client import EXTRACTION_PROMPT, ExtractionResult
from src.pipeline_trace import build_pipeline_trace, empty_trace_html, trace_to_html, trace_to_chat_messages
from src.report_pipeline import build_health_report


def _sample_extraction() -> ExtractionResult:
    return ExtractionResult(
        patient={"age": "42y", "age_years": 42.0, "sex": "female", "age_group": "adult"},
        tests=[
            {
                "marker": "Hemoglobin",
                "value": "11.2",
                "unit": "g/dL",
                "reference_range": "12.0-16.0",
                "status": "low",
                "source_text": "Hgb 11.2",
                "confidence": 0.95,
            },
            {
                "marker": "WBC",
                "value": "6.5",
                "unit": "10^3/uL",
                "reference_range": "4.5-11.0",
                "status": "normal",
                "source_text": "WBC 6.5",
                "confidence": 0.9,
            },
        ],
        notes=["Sample note"],
        raw_response='{"tests":[{"marker":"Hemoglobin","value":"11.2"}]}',
        request_summary={
            "backend": "test",
            "extraction_prompt": EXTRACTION_PROMPT,
            "document_parts": 2,
            "http_status": 200,
            "return_code": 0,
            "duration_ms": 842,
            "user_message_preview": {"image_count": 1, "text_characters": 120},
        },
    )


def test_build_pipeline_trace_has_five_steps():
    extraction = _sample_extraction()
    report = build_health_report(extraction)
    steps = build_pipeline_trace(extraction, report, source_path="/tmp/report.pdf")
    assert len(steps) == 5
    assert [step.id for step in steps] == [
        "document_intake",
        "vision_extraction",
        "schema_normalization",
        "knowledge_graph",
        "pattern_detection",
    ]


def test_extraction_step_includes_full_prompt():
    extraction = _sample_extraction()
    report = build_health_report(extraction)
    steps = build_pipeline_trace(extraction, report)
    extraction_step = steps[1]
    assert extraction_step.prompt == EXTRACTION_PROMPT
    assert extraction_step.return_code == 0
    assert extraction_step.metadata["http_status"] == 200
    assert extraction_step.metadata["duration_ms"] == 842


def test_trace_to_html_collapsible_steps():
    extraction = _sample_extraction()
    report = build_health_report(extraction)
    steps = build_pipeline_trace(extraction, report)
    html = trace_to_html(steps)
    assert "bte-trace-step" in html
    assert html.count('<details class="bte-trace-step">') == len(steps)
    assert "Return code" in html
    assert "bte-trace-status--complete" in html
    assert "Full prompt" in html


def test_empty_trace_html_shows_pending_steps():
    html = empty_trace_html()
    assert html.count('<details class="bte-trace-step">') == 5
    assert html.count("bte-trace-status--pending") == 5
    assert "Pending" in html
    assert "Step 1 — Document intake" in html
    assert "Step 5 — Cross-marker pattern detection" in html


def test_trace_to_chat_messages_shape():
    extraction = _sample_extraction()
    report = build_health_report(extraction)
    steps = build_pipeline_trace(extraction, report)
    messages = trace_to_chat_messages(steps)
    assert messages[0]["role"] == "assistant"
    assert all(msg["role"] == "assistant" for msg in messages)
    assert len(messages) == len(steps) + 1


if __name__ == "__main__":
    test_build_pipeline_trace_has_five_steps()
    test_extraction_step_includes_full_prompt()
    test_trace_to_html_collapsible_steps()
    test_trace_to_chat_messages_shape()
    print("test_pipeline_trace: ok")
