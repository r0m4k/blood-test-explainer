from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from json_repair import loads as repair_json_loads

from src.local_env import load_local_env


load_local_env()

EXTRACTION_PROMPT = """
You are extracting laboratory test results from a medical document.

Return only valid JSON with this exact shape:
{
  "patient": {
    "age": "string or null",
    "age_years": 0.0,
    "sex": "male | female | unknown"
  },
  "tests": [
    {
      "marker": "string",
      "value": "string",
      "unit": "string or null",
      "reference_range": "string or null",
      "status": "low | normal | high | abnormal | unknown",
      "source_text": "short source snippet",
      "confidence": 0.0
    }
  ],
  "notes": ["string"]
}

Field guide (how each test row should look):
- marker: The test name exactly as printed on the report. Keep abbreviations in parentheses when shown (e.g. "Hemoglobin (Hb)", "Mean Cell Volume (MCV)"). One JSON object per result row — do not merge percent differentials with absolute counts.
- value: The measured result only, as a string. Use digits for numeric results (no unit). Remove thousands separators ("150,000" → "150000"). Qualitative results stay as printed ("Negative", "Positive", "<5").
- unit: The measurement unit as printed (e.g. "g/dL", "%", "K/mcL", "mg/dL", "U/L", "cumm"). null if absent.
- reference_range: The lab's printed normal interval or limit, copied verbatim when possible (e.g. "13.0 - 17.0", "70-99", "<200"). null if missing.
- status: Compare value to the printed reference range or report flag. Map H/L, LOW/HIGH, *, or out-of-range arrows to "low" or "high". Use "abnormal" for qualitative out-of-range results. Use "unknown" when the range or flag is missing.
- source_text: A short verbatim snippet from the document for that row (test name + value + unit/flag), under ~120 characters.
- confidence: 0.0–1.0 — how certain you are that marker, value, and unit are correct.

Rules:
- Extract pure lab values only.
- Do not diagnose, interpret, recommend food, supplements, or exercise.
- Extract patient age and sex only when visibly present in the document.
- Normalize sex to "male", "female", or "unknown"; do not infer sex from the patient's name.
- Use null for age and age_years when age is missing.
- Do not invent missing values.
- Preserve units and reference ranges exactly as shown when possible.
- If a marker is unreadable, omit it or add a short note in "notes".
- Use null for missing units or reference ranges.
- Confidence must be a number from 0 to 1.

Few-shot examples (format only — extract from the uploaded document, not these samples):

Example 1 — CBC with flags:
{
  "patient": {"age": "58 years", "age_years": 58.0, "sex": "female"},
  "tests": [
    {
      "marker": "Hemoglobin (Hb)",
      "value": "12.5",
      "unit": "g/dL",
      "reference_range": "13.0 - 17.0",
      "status": "low",
      "source_text": "Hemoglobin (Hb) 12.5 g/dL L",
      "confidence": 0.96
    },
    {
      "marker": "Packed Cell Volume (PCV)",
      "value": "57.5",
      "unit": "%",
      "reference_range": "40 - 50",
      "status": "high",
      "source_text": "Packed Cell Volume (PCV) 57.5 % H",
      "confidence": 0.94
    },
    {
      "marker": "Platelet Count",
      "value": "150000",
      "unit": "cumm",
      "reference_range": "150000 - 410000",
      "status": "normal",
      "source_text": "Platelet Count 150000 cumm",
      "confidence": 0.93
    }
  ],
  "notes": []
}

Example 2 — chemistry panel:
{
  "patient": {"age": null, "age_years": null, "sex": "unknown"},
  "tests": [
    {
      "marker": "Glucose",
      "value": "95",
      "unit": "mg/dL",
      "reference_range": "70-99",
      "status": "normal",
      "source_text": "Glucose 95 mg/dL",
      "confidence": 0.95
    },
    {
      "marker": "ALT",
      "value": "42",
      "unit": "U/L",
      "reference_range": "7-56",
      "status": "normal",
      "source_text": "ALT 42 U/L",
      "confidence": 0.92
    },
    {
      "marker": "Creatinine",
      "value": "1.8",
      "unit": "mg/dL",
      "reference_range": "0.6-1.2",
      "status": "high",
      "source_text": "Creatinine 1.8 mg/dL H",
      "confidence": 0.94
    }
  ],
  "notes": []
}

Example 3 — absolute counts (separate rows from % differentials):
{
  "patient": {"age": "34", "age_years": 34.0, "sex": "male"},
  "tests": [
    {
      "marker": "Neutrophils",
      "value": "60",
      "unit": "%",
      "reference_range": "50 - 62",
      "status": "normal",
      "source_text": "Neutrophils 60 %",
      "confidence": 0.91
    },
    {
      "marker": "Neutrophil, Absolute",
      "value": "3.5",
      "unit": "K/mcL",
      "reference_range": "1.8-7.8",
      "status": "normal",
      "source_text": "Neutrophil, Absolute 3.5 K/mcL",
      "confidence": 0.90
    },
    {
      "marker": "White Blood Cell (WBC)",
      "value": "6.9",
      "unit": "K/mcL",
      "reference_range": "4.8-10.8",
      "status": "normal",
      "source_text": "White Blood Cell (WBC) 6.9 K/mcL",
      "confidence": 0.93
    }
  ],
  "notes": []
}
""".strip()


@dataclass(frozen=True)
class ExtractionResult:
    tests: list[dict[str, Any]]
    notes: list[str]
    raw_response: str
    request_summary: dict[str, Any]
    patient: dict[str, Any] = field(default_factory=dict)


def summarize_document_parts(parts: list[dict[str, Any]]) -> dict[str, int]:
    """Lightweight payload stats for pipeline traces (no base64 blobs)."""
    image_count = 0
    text_characters = 0
    for part in parts:
        if part.get("type") == "image_url":
            image_count += 1
        elif part.get("type") == "text":
            text_characters += len(str(part.get("text") or ""))
    return {"image_count": image_count, "text_characters": text_characters}


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _strip_think(_strip_code_fence(text))
    parsed = _loads_model_json(cleaned)

    # Some models (e.g. MiniCPM-V in "thinking" mode) return a bare array of tests
    # instead of the {tests, notes} object. Wrap it so the rest of the app is unchanged.
    if isinstance(parsed, list):
        return {"tests": parsed, "notes": []}
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object or array.")
    return parsed


def _loads_model_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]|\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Model response did not contain JSON.")
            snippet = match.group(0)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                try:
                    return json.loads(snippet, strict=False)
                except json.JSONDecodeError:
                    return repair_json_loads(snippet)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop <think>...</think> reasoning blocks some models emit before the JSON."""
    return _THINK_RE.sub("", text).strip()


def _normalize_tests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    tests: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        tests.append(
            {
                "marker": str(item.get("marker") or "").strip(),
                "value": str(item.get("value") or "").strip(),
                "unit": _optional_string(item.get("unit")),
                "reference_range": _optional_string(item.get("reference_range")),
                "status": str(item.get("status") or "unknown").strip().lower(),
                "source_text": _optional_string(item.get("source_text")),
                "confidence": _confidence(item.get("confidence")),
            }
        )

    return [test for test in tests if test["marker"] and test["value"]]


def _normalize_notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(note).strip() for note in value if str(note).strip()]


def _normalize_patient(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"age": None, "age_years": None, "sex": "unknown"}

    age = _optional_string(value.get("age") or value.get("age_text") or value.get("patient_age"))
    age_years = _optional_float(value.get("age_years"))
    sex = _normalize_sex(value.get("sex") or value.get("patient_sex") or value.get("gender"))
    return {
        "age": age,
        "age_years": age_years,
        "sex": sex,
    }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sex(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    return "unknown"


def _confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
