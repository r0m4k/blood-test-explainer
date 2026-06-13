from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from json_repair import loads as repair_json_loads
from requests import HTTPError

from src.document_processing import document_intake_metadata, document_to_payload_parts
from src.local_env import load_local_env


load_local_env()

DEFAULT_API_URL = "http://35.203.155.71:8003/v1/chat/completions"
DEFAULT_MODEL = "MiniCPM-V-4.6"


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

Rules:
- Extract pure lab values only.
- Do not diagnose, interpret, recommend food, supplements, or exercise.
- Extract patient age and sex only when visibly present in the document.
- Normalize sex to "male", "female", or "unknown"; do not infer sex from the patient's name.
- Use null for age and age_years when age is missing.
- Do not invent missing values.
- Preserve the units and reference ranges exactly as shown when possible.
- If a marker is unreadable, omit it or add a short note.
- Use null for missing units or reference ranges.
- Confidence must be a number from 0 to 1.
""".strip()


@dataclass(frozen=True)
class ExtractionResult:
    tests: list[dict[str, Any]]
    notes: list[str]
    raw_response: str
    request_summary: dict[str, Any]
    patient: dict[str, Any] = field(default_factory=dict)


class OpenBMBExtractor:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        self.api_url = (api_url or os.getenv("OPENBMB_API_URL") or DEFAULT_API_URL).strip()
        self.model = (model or os.getenv("OPENBMB_MODEL") or DEFAULT_MODEL).strip()
        self.api_key = _normalize_api_key(api_key or os.getenv("OPENBMB_API_KEY"))
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def extract(self, file_path: str, max_pages: int | None = None) -> ExtractionResult:
        if not self.api_key:
            raise RuntimeError(
                "OpenBMB API key is not configured. Set OPENBMB_API_KEY locally or add it as a Hugging Face Space secret."
            )

        document_parts = document_to_payload_parts(file_path, max_pages=max_pages)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        *document_parts,
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }

        started = time.perf_counter()
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        try:
            response.raise_for_status()
        except HTTPError as error:
            if response.status_code == 401:
                raise RuntimeError(
                    "OpenBMB rejected the API key with 401 Unauthorized. Check that the token is exact, active, and belongs to this endpoint."
                ) from error
            raise

        raw_response = _extract_message_content(response.json())
        parsed = _parse_json_response(raw_response)

        return ExtractionResult(
            patient=_normalize_patient(parsed.get("patient", {})),
            tests=_normalize_tests(parsed.get("tests", [])),
            notes=_normalize_notes(parsed.get("notes", [])),
            raw_response=raw_response,
            request_summary={
                "backend": "api",
                "api_url": self.api_url,
                "model": self.model,
                "document_parts": len(document_parts),
                "pages": max_pages or "auto",
                "extraction_prompt": EXTRACTION_PROMPT,
                "user_message_preview": summarize_document_parts(document_parts),
                **document_intake_metadata(file_path, document_parts),
                "http_status": response.status_code,
                "return_code": 0,
                "duration_ms": duration_ms,
            },
        )


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


def _extract_message_content(payload: dict[str, Any]) -> str:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("OpenBMB response did not include choices[0].message.") from error

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(text_parts).strip()

    raise ValueError("OpenBMB response message content was not text.")


def _normalize_api_key(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = value.strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()

    return cleaned or None


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
