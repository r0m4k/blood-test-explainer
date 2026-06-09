#!/usr/bin/env python3
"""Run extraction evaluation: gold labels vs predictions.

Two modes:
  1. Score precomputed predictions:
        python eval/run_eval.py --labels eval/data/synth_eval/labels.jsonl \
                                --predictions runs/pred.jsonl
  2. Run the configured extractor over the images and score live (needs the model):
        EXTRACTOR_BACKEND=local LOCAL_MODEL_PATH=... LOCAL_MMPROJ_PATH=... \
        python eval/run_eval.py --labels eval/data/synth_eval/labels.jsonl --run

Use mode 2 twice (base vs fine-tuned GGUF) to produce the OpenBMB before/after numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_scoring import format_metrics, score  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predict_live(labels: list[dict], labels_path: Path) -> list[dict]:
    from src.extraction import build_extractor

    extractor = build_extractor()
    base = labels_path.parent
    preds = []
    for i, row in enumerate(labels):
        image_path = str((base / row["image"]).resolve())
        try:
            result = extractor.extract(image_path, max_pages=3)
            preds.append({"tests": result.tests})
        except Exception as error:  # keep going; a failed page is a miss
            print(f"  [{i}] extraction failed: {error}", file=sys.stderr)
            preds.append({"tests": []})
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="precomputed predictions JSONL")
    ap.add_argument("--run", action="store_true", help="run the configured extractor live")
    args = ap.parse_args()

    gold = _load_jsonl(args.labels)
    if args.predictions:
        pred = _load_jsonl(args.predictions)
    elif args.run:
        pred = _predict_live(gold, args.labels)
    else:
        ap.error("provide --predictions or --run")

    if len(pred) != len(gold):
        ap.error(f"predictions ({len(pred)}) and labels ({len(gold)}) length mismatch")

    m = score(gold, pred)
    print(f"\n  Extraction eval — {args.labels.name} ({len(gold)} reports)\n")
    print(format_metrics(m))
    worst = sorted(m.by_marker_fn.items(), key=lambda kv: -kv[1])[:5]
    if worst:
        print("\n  most-missed markers:", ", ".join(f"{k}×{n}" for k, n in worst))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
