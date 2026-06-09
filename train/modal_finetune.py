#!/usr/bin/env python3
"""LoRA fine-tune of MiniCPM-V for lab extraction, on Modal.

Strategy: generate the synthetic dataset **on the GPU box** (it's pure Python + PIL, fully
reproducible from a seed), convert to the vision-SFT format, then LoRA fine-tune MiniCPM-V with
ms-swift. No image upload, no dataset drift. Adapters are saved to a Modal Volume; pull them
down and convert to GGUF (see scripts/convert_to_gguf.sh).

    modal run train/modal_finetune.py --n 4000

Running the fine-tune on Modal also satisfies the Modal prize.

⚠️ VERIFY-ON-FIRST-RUN: confirm the exact ms-swift/LLaMA-Factory `--model_type` for MiniCPM-V
4.6 against the finetune guide at github.com/OpenBMB/MiniCPM-V before running. Trainer CLIs
evolve; pin the value in MODEL_TYPE below after checking the current guide.
"""

from __future__ import annotations

import modal

# TODO(verify): confirm the exact ms-swift/LLaMA-Factory model_type for MiniCPM-V 4.6
# against the finetune guide at github.com/OpenBMB/MiniCPM-V before running.
MODEL_ID = "openbmb/MiniCPM-V-4.6"          # HF id of the base vision model
MODEL_TYPE = "minicpm-v-v2_6-chat"          # placeholder until confirmed from the guide

app = modal.App("blood-test-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.44",
        "peft>=0.12",
        "accelerate>=0.33",
        "pillow>=10",
        "ms-swift>=2.5",
        "timm",
        "sentencepiece",
    )
    # Mount our generator + converter + marker reference so the box builds its own data.
    .add_local_dir("src", "/root/app/src")
    .add_local_dir("train", "/root/app/train")
)

adapters = modal.Volume.from_name("blood-test-adapters", create_if_missing=True)
hf_cache = modal.Volume.from_name("blood-test-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3 * 60 * 60,
    volumes={"/adapters": adapters, "/root/.cache/huggingface": hf_cache},
)
def train(n: int = 4000, epochs: int = 2, seed: int = 13) -> str:
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/app")
    from train.synth_reports import generate
    from train.to_sft_dataset import convert

    # 1) build the dataset on the box
    data_dir = Path("/root/app/train/data/synth")
    labels = generate(n, data_dir, seed=seed)
    sft_path = Path("/root/app/train/data/sft.jsonl")
    n_examples = convert(labels, sft_path)
    print(f"Generated {n_examples} SFT examples at {sft_path}")

    # 2) LoRA fine-tune with ms-swift
    out_dir = "/adapters/minicpmv-lab-lora"
    cmd = [
        "swift", "sft",
        "--model_type", MODEL_TYPE,
        "--model_id_or_path", MODEL_ID,
        "--sft_type", "lora",
        "--dataset", str(sft_path),
        "--num_train_epochs", str(epochs),
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--learning_rate", "1e-4",
        "--batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--max_length", "2048",
        "--output_dir", out_dir,
        "--save_total_limit", "1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    adapters.commit()
    return out_dir


@app.local_entrypoint()
def main(n: int = 4000, epochs: int = 2) -> None:
    path = train.remote(n=n, epochs=epochs)
    print(f"\nLoRA adapters saved to Modal volume 'blood-test-adapters' at {path}")
    print("Next: download adapters, merge into the base model, convert to GGUF + mmproj,")
    print("quantize Q4_K_M, and bundle into the Space (see scripts/convert_to_gguf.sh).")
