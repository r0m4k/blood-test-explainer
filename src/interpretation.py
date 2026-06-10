"""Interpretation engine (Phase 3.2–3.4): turn extracted values into grounded insight.

The model extracts the numbers; THIS module decides what is worth saying, and every word of
medical content comes from the knowledge base (kb/knowledge_base.py), never invented. Output:
  - per-marker insight cards for flagged (high/low) markers, with doctor-questions  [3.2, 3.4]
  - cross-marker patterns that mean more together than alone                         [3.3]
  - an educational disclaimer

Pure functions + plain dataclasses so the app can render them and the eval can test them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kb.knowledge_base import DISCLAIMER, PATTERNS, interpret, questions_for
from src.markers import resolve

_PATTERN_NOTE = {p.name: p.note for p in PATTERNS}


@dataclass(frozen=True)
class MarkerInsight:
    marker: str            # canonical name
    value: str
    unit: str | None
    status: str            # "low" | "high"
    reference_range: str
    note: str | None       # grounded educational note from the KB
    questions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PatternInsight:
    name: str
    note: str


@dataclass(frozen=True)
class Interpretation:
    flagged: tuple[MarkerInsight, ...]   # abnormal markers only
    normal_count: int                    # how many recognized markers were in range
    patterns: tuple[PatternInsight, ...]
    disclaimer: str = DISCLAIMER

    @property
    def has_findings(self) -> bool:
        return bool(self.flagged or self.patterns)


def _num(value: object) -> float | None:
    """Pull the first numeric token out of a value like '12.3', '12.3 *', or '< 0.5'."""
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group(0)) if m else None


def _status_of(test: dict, marker) -> str:
    """Trust the model's status when valid, else compute it from value vs the marker's range."""
    status = str(test.get("status") or "").strip().lower()
    if status in {"low", "normal", "high"}:
        return status
    value = _num(test.get("value"))
    if value is not None:
        return marker.status_for(value)
    return "unknown"


def build_interpretation(tests: list[dict]) -> Interpretation:
    """Build the grounded interpretation from a list of extracted test dicts."""
    flagged: list[MarkerInsight] = []
    normal = 0
    status_by_marker: dict[str, str] = {}

    for test in tests:
        marker = resolve(str(test.get("marker") or ""))
        if marker is None:
            continue  # unknown marker -> we don't invent meaning for it
        status = _status_of(test, marker)
        status_by_marker[marker.name] = status
        if status in {"low", "high"}:
            flagged.append(
                MarkerInsight(
                    marker=marker.name,
                    value=str(test.get("value") or "").strip(),
                    unit=(test.get("unit") or marker.unit),
                    status=status,
                    reference_range=marker.ref_range_text(),
                    note=interpret(marker.name, status),
                    questions=questions_for(marker.name),
                )
            )
        elif status == "normal":
            normal += 1

    return Interpretation(
        flagged=tuple(flagged),
        normal_count=normal,
        patterns=tuple(_detect_patterns(status_by_marker)),
    )


def _detect_patterns(s: dict[str, str]) -> list[PatternInsight]:
    """Apply the cross-marker pattern logic (the human-readable triggers live in the KB)."""
    out: list[PatternInsight] = []

    def at(name: str, *statuses: str) -> bool:
        return s.get(name) in statuses

    liver_high = sum(at(m, "high") for m in ("ALT", "AST", "ALP", "GGT"))

    if at("Hemoglobin", "low") and at("Hematocrit", "low"):
        out.append(_pattern("Anemia picture"))
    if at("Ferritin", "low") and at("MCV", "low"):
        out.append(_pattern("Iron-deficiency pattern"))
    if at("MCV", "high") and at("Vitamin B12", "low"):
        out.append(_pattern("B12/folate pattern"))
    if liver_high >= 2:
        out.append(_pattern("Liver cluster"))
    if (at("LDL Cholesterol", "high") or at("Triglycerides", "high")) and at("HDL Cholesterol", "low"):
        out.append(_pattern("Lipid / cardiovascular risk"))
    if at("Creatinine", "high") and at("eGFR", "low"):
        out.append(_pattern("Kidney-function pattern"))
    if (at("TSH", "high") and at("Free T4", "low")) or (at("TSH", "low") and at("Free T4", "high")):
        out.append(_pattern("Thyroid pattern"))
    if at("Glucose", "high") and at("HbA1c", "high"):
        out.append(_pattern("Glycemic pattern"))

    return out


def _pattern(name: str) -> PatternInsight:
    return PatternInsight(name=name, note=_PATTERN_NOTE[name])
