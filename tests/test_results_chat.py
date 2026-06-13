import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openbmb_client import ExtractionResult
from src.pipeline_trace import build_pipeline_trace, build_session_state
from src.report_pipeline import build_health_report
from src.results_chat import ResultsChatAssistant, build_chat_context


def _session() -> dict:
    extraction = ExtractionResult(
        patient={"age_years": 42, "sex": "female"},
        tests=[
            {
                "marker": "Hemoglobin",
                "value": "11.2",
                "unit": "g/dL",
                "reference_range": "12.0-16.0",
                "status": "low",
                "confidence": 0.9,
            }
        ],
        notes=[],
        raw_response="{}",
        request_summary={"backend": "test", "document_parts": 1},
    )
    report = build_health_report(extraction)
    steps = build_pipeline_trace(extraction, report)
    return build_session_state(extraction, report, steps)


def test_build_chat_context_includes_patient_and_markers():
    context = build_chat_context(_session())
    assert "female" in context
    assert "Hemoglobin" in context
    assert "Report summary" in context


def test_reply_requires_session():
    assistant = ResultsChatAssistant()
    reply = assistant.reply("What is low hemoglobin?", [], {})
    assert "Upload and analyze" in reply


def test_reply_uses_llm_when_session_present():
    assistant = ResultsChatAssistant()
    session = _session()
    with patch("src.results_chat.generate_text_chat", return_value="Educational reply."):
        reply = assistant.reply("Explain my hemoglobin.", [], session)
    assert reply == "Educational reply."


if __name__ == "__main__":
    test_build_chat_context_includes_patient_and_markers()
    test_reply_requires_session()
    test_reply_uses_llm_when_session_present()
    print("test_results_chat: ok")
