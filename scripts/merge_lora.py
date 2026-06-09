#!/usr/bin/env python3
"""Merge the LoRA adapters into the MiniCPM-V base → a standalone HF model for GGUF conversion.

    python scripts/merge_lora.py \
        --base openbmb/MiniCPM-V-4_6 \
        --adapters ./adapters/minicpmv-lab-lora \
        --out ./merged-minicpmv-lab
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base model HF id or path")
    ap.add_argument("--adapters", required=True, help="LoRA adapter dir (from Modal volume)")
    ap.add_argument("--out", required=True, help="output dir for the merged model")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    print(f"Loading base {args.base} ...")
    model = AutoModel.from_pretrained(
        args.base, trust_remote_code=True, torch_dtype=torch.float16
    )
    print(f"Applying adapters {args.adapters} ...")
    model = PeftModel.from_pretrained(model, args.adapters)
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
    except Exception:
        pass  # some MiniCPM-V revisions bundle the processor in the tokenizer

    print(f"Merged model written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
