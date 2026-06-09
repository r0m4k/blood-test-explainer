"""Field-level scoring for extraction quality.

Compares predicted lab values against gold labels and reports the metrics that matter for the
OpenBMB before/after story:
  - **marker P / R / F1** — did we find the right markers (matched by canonical name/alias)?
  - **value / unit / status accuracy** — for matched markers, are the fields right?

Pure functions, no model or I/O, so they are unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.markers import resolve


def _canon(name: str) -> str:
    m = resolve(name)
    return m.name.casefold() if m else (name or "").strip().casefold()


def _num(s) -> float | None:
    try:
        return float(str(s).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _value_match(a, b, rel_tol: float = 0.001) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= rel_tol * max(1.0, abs(nb))
    return str(a).strip().casefold() == str(b).strip().casefold()


def _unit_match(a, b) -> bool:
    norm = lambda s: (str(s or "").strip().casefold().replace(" ", ""))
    return norm(a) == norm(b)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    value_ok: int = 0
    unit_ok: int = 0
    status_ok: int = 0
    matched: int = 0
    by_marker_fn: dict[str, int] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def value_acc(self) -> float:
        return self.value_ok / self.matched if self.matched else 0.0

    @property
    def unit_acc(self) -> float:
        return self.unit_ok / self.matched if self.matched else 0.0

    @property
    def status_acc(self) -> float:
        return self.status_ok / self.matched if self.matched else 0.0


def score_report(gold_tests: list[dict], pred_tests: list[dict], m: Metrics) -> None:
    """Accumulate one report's gold-vs-pred comparison into `m`."""
    gold_by = {_canon(t.get("marker", "")): t for t in gold_tests}
    pred_by = {_canon(t.get("marker", "")): t for t in pred_tests}

    for key, g in gold_by.items():
        p = pred_by.get(key)
        if p is None:
            m.fn += 1
            m.by_marker_fn[key] = m.by_marker_fn.get(key, 0) + 1
            continue
        m.tp += 1
        m.matched += 1
        m.value_ok += _value_match(p.get("value"), g.get("value"))
        m.unit_ok += _unit_match(p.get("unit"), g.get("unit"))
        m.status_ok += str(p.get("status", "")).strip().casefold() == str(g.get("status", "")).strip().casefold()

    for key in pred_by:
        if key not in gold_by:
            m.fp += 1


def score(gold_rows: list[dict], pred_rows: list[dict]) -> Metrics:
    """Score aligned lists of {tests:[...]} rows (same order/length)."""
    m = Metrics()
    for g, p in zip(gold_rows, pred_rows):
        score_report(g.get("tests", []), p.get("tests", []), m)
    return m


def format_metrics(m: Metrics) -> str:
    return (
        f"  markers   P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1:.3f}  "
        f"(tp={m.tp} fp={m.fp} fn={m.fn})\n"
        f"  fields    value={m.value_acc:.3f}  unit={m.unit_acc:.3f}  status={m.status_acc:.3f}  "
        f"(matched={m.matched})"
    )
