import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_scoring import score  # noqa: E402


def _row(tests):
    return {"tests": tests}


def _t(marker, value, unit="mg/dL", status="normal"):
    return {"marker": marker, "value": value, "unit": unit, "status": status}


def test_perfect_match():
    gold = [_row([_t("Glucose", "95"), _t("ALT", "30", "U/L")])]
    m = score(gold, gold)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert m.value_acc == 1.0 and m.unit_acc == 1.0 and m.status_acc == 1.0


def test_alias_is_matched_to_canonical():
    gold = [_row([_t("Creatinine", "1.0")])]
    pred = [_row([_t("Cr", "1.0")])]  # alias
    m = score(gold, pred)
    assert m.tp == 1 and m.fp == 0 and m.fn == 0


def test_missing_and_extra_markers():
    gold = [_row([_t("Glucose", "95"), _t("ALT", "30")])]
    pred = [_row([_t("Glucose", "95"), _t("HDL", "55")])]  # missed ALT, hallucinated HDL
    m = score(gold, pred)
    assert m.tp == 1 and m.fn == 1 and m.fp == 1
    assert "alt" in m.by_marker_fn


def test_value_numeric_tolerance_and_status():
    gold = [_row([_t("Glucose", "95", status="normal")])]
    pred = [_row([_t("Glucose", "95.0", status="high")])]  # value ok (numeric), status wrong
    m = score(gold, pred)
    assert m.value_ok == 1
    assert m.status_ok == 0


def test_unit_mismatch():
    gold = [_row([_t("ALT", "30", "U/L")])]
    pred = [_row([_t("ALT", "30", "IU/L")])]
    m = score(gold, pred)
    assert m.unit_ok == 0 and m.value_ok == 1
