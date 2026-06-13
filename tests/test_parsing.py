"""Parser must survive MiniCPM-V's thinking mode + bare-array output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openbmb_client import _normalize_tests, _parse_json_response  # noqa: E402


def test_think_block_then_bare_array():
    raw = (
        "<think>\nWe need to extract the tests...\n</think>\n\n"
        '[{"marker": "Glucose", "value": "95", "unit": "mg/dL", "reference_range": "70-99"}]'
    )
    parsed = _parse_json_response(raw)
    assert set(parsed) == {"tests", "notes"}
    tests = _normalize_tests(parsed["tests"])
    assert len(tests) == 1 and tests[0]["marker"] == "Glucose"


def test_plain_object_still_works():
    raw = '{"tests": [{"marker": "ALT", "value": "30"}], "notes": ["ok"]}'
    parsed = _parse_json_response(raw)
    assert parsed["notes"] == ["ok"]
    assert _normalize_tests(parsed["tests"])[0]["marker"] == "ALT"


def test_code_fenced_array():
    raw = '```json\n[{"marker": "TSH", "value": "2.1", "unit": "mIU/L"}]\n```'
    tests = _normalize_tests(_parse_json_response(raw)["tests"])
    assert tests[0]["marker"] == "TSH"


def test_prose_then_object_is_recovered():
    raw = 'Here are the results:\n{"tests": [{"marker": "HDL", "value": "55"}], "notes": []}'
    assert _normalize_tests(_parse_json_response(raw)["tests"])[0]["marker"] == "HDL"


def test_extraction_prompt_includes_field_guide_and_few_shot_examples():
    from src.openbmb_client import EXTRACTION_PROMPT

    assert "Field guide" in EXTRACTION_PROMPT
    assert "Few-shot examples" in EXTRACTION_PROMPT
    assert "Hemoglobin (Hb)" in EXTRACTION_PROMPT
    assert "Neutrophil, Absolute" in EXTRACTION_PROMPT
    assert "Remove thousands separators" in EXTRACTION_PROMPT
