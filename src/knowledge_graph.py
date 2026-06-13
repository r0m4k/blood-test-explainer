"""Knowledge-graph lookup and reference selection for lab markers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_GRAPH_PATH = ROOT / "kb" / "cbc_knowledge_graph.json"  # lab-wide marker graph (107 tests)


class LabKnowledgeGraph:
    """Small deterministic lookup layer over the JSON knowledge graph."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.tests: list[dict[str, Any]] = list(payload.get("tests", []))
        self._by_id = {str(test.get("id", "")).casefold(): test for test in self.tests}
        self._alias_index = self._build_alias_index()

    @classmethod
    def load(cls, path: str | Path = DEFAULT_KNOWLEDGE_GRAPH_PATH) -> "LabKnowledgeGraph":
        graph_path = Path(path)
        return cls(json.loads(graph_path.read_text(encoding="utf-8")))

    def resolve(self, marker_name: str | None) -> dict[str, Any] | None:
        """Return the graph node matching a raw marker name or alias."""
        for key in _candidate_keys(marker_name):
            match = self._alias_index.get(key)
            if match is not None:
                return match
        return None

    def get(self, marker_id: str | None) -> dict[str, Any] | None:
        if not marker_id:
            return None
        return self._by_id.get(str(marker_id).casefold())

    def select_statistics(
        self,
        node: dict[str, Any],
        age_group: str,
        sex: str,
    ) -> dict[str, Any] | None:
        """Select the best statistics block for age/sex context.

        Sex-specific ranges are preferred when available. The JSON keeps an
        `unknown` sex bucket for high-impact markers, and age-only statistics
        remain the compatibility fallback for every marker.
        """
        normalized_sex = sex if sex in {"male", "female"} else "unknown"
        sex_stats = node.get("sex_specific_statistics_per_group_age")
        if isinstance(sex_stats, dict):
            group_stats = sex_stats.get(age_group)
            if isinstance(group_stats, dict):
                values = group_stats.get(normalized_sex) or group_stats.get("unknown")
                if isinstance(values, dict):
                    return {
                        "basis": "sex_specific_statistics_per_group_age",
                        "age_group": age_group,
                        "sex": normalized_sex,
                        "values": values,
                    }

        age_stats = node.get("statistics_per_group_age", {})
        values = age_stats.get(age_group)
        if isinstance(values, dict):
            return {
                "basis": "statistics_per_group_age",
                "age_group": age_group,
                "sex": "not_applied",
                "values": values,
            }
        return None

    def _build_alias_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for test in self.tests:
            names = [
                test.get("id"),
                test.get("display_name"),
                *(test.get("aliases") or []),
            ]
            for name in names:
                for key in _candidate_keys(name):
                    index.setdefault(key, test)
        return index


@lru_cache(maxsize=1)
def default_knowledge_graph() -> LabKnowledgeGraph:
    return LabKnowledgeGraph.load()


def _candidate_keys(value: str | None) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    pieces = {text}
    pieces.add(re.sub(r"\([^)]*\)", "", text).strip())

    for inner in re.findall(r"\(([^)]*)\)", text):
        pieces.add(inner)
        pieces.update(part.strip() for part in re.split(r"[/,;]", inner))

    keys: list[str] = []
    for piece in pieces:
        key = _marker_key(piece)
        if key and key not in keys:
            keys.append(key)
    return keys


def _marker_key(value: str) -> str:
    normalized = value.casefold()
    normalized = normalized.replace("µ", "u").replace("μ", "u")
    normalized = normalized.replace("percent", "%").replace("number", "#")
    return re.sub(r"[^a-z0-9%#]+", "", normalized)
