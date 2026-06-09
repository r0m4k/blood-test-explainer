"""End-to-end check of the data pipeline: generate → SFT convert → self-score."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_scoring import score  # noqa: E402
from train.synth_reports import generate  # noqa: E402
from train.to_sft_dataset import convert  # noqa: E402


def test_generate_produces_valid_labels(tmp_path):
    labels = generate(5, tmp_path, seed=1)
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    assert len(rows) == 5
    for r in rows:
        assert (tmp_path / r["image"]).exists()
        assert r["tests"], "every report should have at least one marker"
        for t in r["tests"]:
            assert t["status"] in {"low", "normal", "high"}
            assert set(t) >= {"marker", "value", "unit", "reference_range", "status"}


def test_gold_scores_perfectly_against_itself(tmp_path):
    labels = generate(8, tmp_path, seed=2)
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    m = score(rows, rows)
    assert m.recall == 1.0 and m.precision == 1.0
    assert m.value_acc == 1.0 and m.status_acc == 1.0


def test_sft_conversion_targets_are_valid_json(tmp_path):
    labels = generate(4, tmp_path, seed=3)
    out = tmp_path / "sft.jsonl"
    n = convert(labels, out)
    assert n == 4
    for line in out.read_text().splitlines():
        rec = json.loads(line)
        assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
        json.loads(rec["messages"][1]["content"])  # assistant target parses as JSON
        assert rec["images"] and Path(rec["images"][0]).exists()
