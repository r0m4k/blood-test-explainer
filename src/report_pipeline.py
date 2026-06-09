"""Extraction-to-health-report pipeline.

This module keeps the agentic part focused on reading the document. Everything after that is
deterministic: marker resolution, age/sex reference selection, status comparison, and shaping the
object consumed by the UI.
"""

from __future__ import annotations

import re
from typing import Any

from src.knowledge_graph import LabKnowledgeGraph, default_knowledge_graph
from src.openbmb_client import ExtractionResult


AGE_GROUPS = ("child", "teenager", "adult", "elder")
KNOWN_STATUSES = {"low", "normal", "high", "abnormal", "unknown"}


def build_health_report(
    extraction: ExtractionResult,
    knowledge_graph: LabKnowledgeGraph | None = None,
) -> dict[str, Any]:
    """Merge extracted lab values with knowledge-graph context for rendering."""
    graph = knowledge_graph or default_knowledge_graph()
    patient = normalize_patient(getattr(extraction, "patient", {}))
    markers = [
        enrich_marker(test, patient=patient, knowledge_graph=graph)
        for test in extraction.tests
    ]

    status_counts = _status_counts(markers)
    enriched_count = sum(1 for marker in markers if marker.get("knowledge") is not None)
    unmatched = [
        marker["raw_name"]
        for marker in markers
        if marker.get("knowledge") is None
    ]

    return {
        "patient": patient,
        "markers": markers,
        "notes": list(extraction.notes),
        "summary": {
            "total_markers": len(markers),
            "enriched_markers": enriched_count,
            "unmatched_markers": unmatched,
            "status_counts": status_counts,
            "needs_review": (
                status_counts.get("high", 0)
                + status_counts.get("low", 0)
                + status_counts.get("abnormal", 0)
            ),
        },
        "knowledge_graph": {
            "schema_version": graph.payload.get("schema_version"),
            "title": graph.payload.get("title"),
            "medical_disclaimer": graph.payload.get("medical_disclaimer"),
            "sex_significance_policy": graph.payload.get("sex_significance_policy"),
            "sources": graph.payload.get("sources", {}),
        },
        "request_summary": extraction.request_summary,
        "raw_response": extraction.raw_response,
    }


def enrich_marker(
    extracted: dict[str, Any],
    patient: dict[str, Any],
    knowledge_graph: LabKnowledgeGraph,
) -> dict[str, Any]:
    raw_name = _text(extracted.get("marker"), "Unknown marker")
    node = knowledge_graph.resolve(raw_name)
    numeric_value = parse_numeric_value(extracted.get("value"))
    extracted_status = normalize_status(extracted.get("status"))
    lab_interval = parse_reference_interval(extracted.get("reference_range"))
    kg_selection = (
        knowledge_graph.select_statistics(node, patient["age_group"], patient["sex"])
        if node is not None
        else None
    )
    kg_interval = _interval_from_statistics(kg_selection)

    comparison_interval = lab_interval or kg_interval
    reference_basis = "lab_reference_range" if lab_interval else "knowledge_graph"
    derived_status = status_from_interval(numeric_value, comparison_interval)
    final_status = extracted_status if extracted_status != "unknown" else (derived_status or "unknown")

    return {
        "raw_name": raw_name,
        "canonical_id": node.get("id") if node else None,
        "display_name": node.get("display_name") if node else raw_name,
        "value": _text(extracted.get("value"), "-"),
        "numeric_value": numeric_value,
        "unit": _text(extracted.get("unit"), node.get("unit", "") if node else ""),
        "lab_reference_range": _optional_text(extracted.get("reference_range")),
        "status": final_status,
        "extracted_status": extracted_status,
        "derived_status": derived_status or "unknown",
        "confidence": _confidence(extracted.get("confidence")),
        "source_text": _optional_text(extracted.get("source_text")),
        "comparison": {
            "basis": reference_basis,
            "interval": comparison_interval,
            "range_position": range_position(numeric_value, comparison_interval),
        },
        "reference_selection": kg_selection,
        "knowledge": _knowledge_payload(node),
    }


def normalize_patient(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    raw_age = (
        source.get("age")
        or source.get("age_text")
        or source.get("age_years")
        or source.get("patient_age")
    )
    age_years = parse_age_years(source.get("age_years"))
    if age_years is None:
        age_years = parse_age_years(raw_age)

    sex = normalize_sex(source.get("sex") or source.get("patient_sex") or source.get("gender"))
    return {
        "age": _optional_text(raw_age),
        "age_years": age_years,
        "age_group": age_group_for(age_years),
        "sex": sex,
        "raw": source,
    }


def parse_age_years(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None

    text = str(value).strip().casefold()
    if not text:
        return None

    # Common report format: "25y 10m 26d".
    years = _first_number_before(text, ("y", "yr", "yrs", "year", "years"))
    months = _first_number_before(text, ("mo", "mos", "month", "months", "m"))
    days = _first_number_before(text, ("d", "day", "days"))
    if years is not None or months is not None or days is not None:
        return round((years or 0.0) + (months or 0.0) / 12 + (days or 0.0) / 365.25, 2)

    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        parsed = float(match.group(0))
        return parsed if parsed >= 0 else None
    return None


def normalize_sex(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().casefold()
    if text in {"m", "male", "man", "boy"}:
        return "male"
    if text in {"f", "female", "woman", "girl"}:
        return "female"
    return "unknown"


def age_group_for(age_years: float | None) -> str:
    if age_years is None:
        return "adult"
    if age_years < 13:
        return "child"
    if age_years < 18:
        return "teenager"
    if age_years < 65:
        return "adult"
    return "elder"


def parse_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_reference_interval(value: Any) -> dict[str, float | None] | None:
    text = _optional_text(value)
    if not text:
        return None

    cleaned = text.casefold().replace("–", "-").replace("—", "-")
    numbers = [float(match.replace(",", "")) for match in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", cleaned)]
    if len(numbers) >= 2 and re.search(r"\d\s*-\s*\d", cleaned):
        low, high = numbers[0], numbers[1]
        return {"low": min(low, high), "high": max(low, high)}

    if numbers and re.search(r"(up to|less than|<|<=|≤|below)", cleaned):
        return {"low": None, "high": numbers[0]}

    if numbers and re.search(r"(greater than|>|>=|≥|above|at least)", cleaned):
        return {"low": numbers[0], "high": None}

    return None


def status_from_interval(
    value: float | None,
    interval: dict[str, float | None] | None,
) -> str | None:
    if value is None or not interval:
        return None
    low = interval.get("low")
    high = interval.get("high")
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    return "normal"


def range_position(
    value: float | None,
    interval: dict[str, float | None] | None,
) -> int:
    if value is None or not interval:
        return 50
    low = interval.get("low")
    high = interval.get("high")
    if low is not None and high is not None and high > low:
        return _clamp_percent((value - low) / (high - low) * 100)
    if high is not None and high > 0:
        return _clamp_percent(value / high * 100)
    if low is not None and low > 0:
        return _clamp_percent(value / low * 100)
    return 50


def normalize_status(value: Any) -> str:
    status = str(value or "unknown").strip().casefold()
    if status in {"l", "lo"}:
        return "low"
    if status in {"h", "hi"}:
        return "high"
    if status in {"ok", "within range", "in range"}:
        return "normal"
    return status if status in KNOWN_STATUSES else "unknown"


def _knowledge_payload(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "description": node.get("description"),
        "why_important": node.get("why_important"),
        "instructions_to_improve": node.get("instructions_to_improve") or {},
        "sex_significance": node.get("sex_significance") or {},
        "related_tests": node.get("related_tests") or [],
        "source_ids": node.get("source_ids") or [],
        "category": node.get("category"),
        "unit": node.get("unit"),
    }


def _interval_from_statistics(selection: dict[str, Any] | None) -> dict[str, float | None] | None:
    if not selection:
        return None
    values = selection.get("values") or {}
    low = values.get("minimal_value")
    high = values.get("maximum_value")
    if low is None and high is None:
        return None
    return {"low": float(low) if low is not None else None, "high": float(high) if high is not None else None}


def _first_number_before(text: str, suffixes: tuple[str, ...]) -> float | None:
    suffix_pattern = "|".join(re.escape(suffix) for suffix in suffixes)
    match = re.search(rf"(\d+(?:\.\d+)?)\s*(?:{suffix_pattern})\b", text)
    return float(match.group(1)) if match else None


def _status_counts(markers: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(KNOWN_STATUSES)}
    for marker in markers:
        status = normalize_status(marker.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _text(value: Any, fallback: str) -> str:
    text = _optional_text(value)
    return text if text is not None else fallback


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _clamp_percent(value: float) -> int:
    return max(0, min(100, round(value)))
