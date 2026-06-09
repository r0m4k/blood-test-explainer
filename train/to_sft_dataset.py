#!/usr/bin/env python3
"""Convert synthetic (image, gold-JSON) pairs → a vision SFT dataset for MiniCPM-V LoRA.

Output is one JSON object per line in the messages+images format consumed by ms-swift /
common MiniCPM-V finetune recipes:

    {"messages": [
        {"role": "user", "content": "<image>" + EXTRACTION_PROMPT},
        {"role": "assistant", "content": "<the gold {tests,notes} JSON>"}],
     "images": ["/abs/path/to/report.png"]}

The assistant turn is exactly the schema the GBNF grammar enforces at inference, so training
target and serving format are identical. Image paths are absolute so the trainer can find them
regardless of working directory.

Usage:
    python train/to_sft_dataset.py --labels train/data/synth/labels.jsonl --out train/data/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.openbmb_client import EXTRACTION_PROMPT  # noqa: E402

IMAGE_TAG = "<image>"


def to_example(row: dict, labels_dir: Path) -> dict:
    image_abs = str((labels_dir / row["image"]).resolve())
    target = json.dumps({"tests": row.get("tests", []), "notes": row.get("notes", [])},
                        ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": IMAGE_TAG + "\n" + EXTRACTION_PROMPT},
            {"role": "assistant", "content": target},
        ],
        "images": [image_abs],
    }


def convert(labels_path: Path, out_path: Path) -> int:
    labels_dir = labels_path.parent
    rows = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_example(row, labels_dir), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("train/data/sft.jsonl"))
    args = ap.parse_args()
    n = convert(args.labels, args.out)
    print(f"Wrote {n} SFT examples → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
