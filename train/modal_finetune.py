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

MODEL_ID = "openbmb/MiniCPM-V-4.6"          # HF id of the base vision model
MODEL_TYPE = "minicpm-v-4_6"                # ms-swift model_type for MiniCPM-V 4.6

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
        # Pulled in by the data-prep import chain (to_sft_dataset -> openbmb_client ->
        # document_processing). Needed so the box can build its own dataset.
        "pymupdf",
        "requests",
        "json-repair",
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
    timeout=6 * 60 * 60,
    volumes={"/adapters": adapters, "/root/.cache/huggingface": hf_cache},
)
def train(n: int = 4000, epochs: int = 2, seed: int = 13) -> str:
    import os
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
    # ms-swift 4.x CLI: --model (was --model_id_or_path), --per_device_train_batch_size (was
    # --batch_size). model_type is inferred from --model. train_type defaults to "lora" and is
    # configured by --lora_rank/--lora_alpha, so we don't pass --train_type explicitly (this
    # build rejected it as an unknown arg).
    cmd = [
        "swift", "sft",
        "--model", MODEL_ID,
        "--dataset", str(sft_path),
        "--num_train_epochs", str(epochs),
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--learning_rate", "1e-4",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--max_length", "2048",
        "--output_dir", out_dir,
        "--save_total_limit", "1",
    ]
    # Pull weights from HuggingFace (fast from Modal + cached in the hf_cache volume) instead of
    # ModelScope, whose ~600kB/s crawl from Modal's US network blew past the client heartbeat.
    env = {**os.environ, "USE_HF": "1"}
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    adapters.commit()
    return out_dir


@app.local_entrypoint()
def main(n: int = 4000, epochs: int = 2) -> None:
    path = train.remote(n=n, epochs=epochs)
    print(f"\nLoRA adapters saved to Modal volume 'blood-test-adapters' at {path}")
    print("Next: merge the adapter into the base model and push it to the Hub:")
    print("  modal run train/modal_finetune.py::merge --repo-id <owner>/<model-name>")
    print("then set ZEROGPU_MODEL_ID=<owner>/<model-name> on the Space (Transformers path; no GGUF).")


@app.function(
    image=image,
    gpu="A100",
    timeout=2 * 60 * 60,
    volumes={"/adapters": adapters, "/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_and_push(repo_id: str, adapter_dir: str = "/adapters/minicpmv-lab-lora") -> str:
    """Merge the newest LoRA checkpoint into MiniCPM-V 4.6 and push the merged model to the Hub.

    Deploy by setting ZEROGPU_MODEL_ID=<repo_id> on the Space (Transformers path; no GGUF needed).
    Requires a Modal secret named 'huggingface-secret' exposing HF_TOKEN with write access.
    """
    import glob
    import os
    import subprocess

    checkpoints = [
        c for c in sorted(glob.glob(f"{adapter_dir}/v*/checkpoint-*"), key=os.path.getmtime)
        if not c.endswith("-merged")
    ]
    if not checkpoints:
        raise RuntimeError(f"No LoRA checkpoints found under {adapter_dir}")
    latest = checkpoints[-1]
    print(f"Merging LoRA checkpoint: {latest}")

    env = {**os.environ, "USE_HF": "1"}
    # ms-swift merges the adapter into the base model; the merged weights land in <ckpt>-merged.
    subprocess.run(["swift", "export", "--adapters", latest, "--merge_lora", "true"], check=True, env=env)

    merged = sorted(glob.glob(f"{adapter_dir}/v*/checkpoint-*-merged"), key=os.path.getmtime)
    if not merged:
        raise RuntimeError("swift export did not produce a *-merged directory")
    merged_dir = merged[-1]
    print(f"Merged model directory: {merged_dir}")

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — add it to the 'huggingface-secret' Modal secret.")
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=merged_dir, repo_id=repo_id, repo_type="model")
    adapters.commit()
    print(f"Pushed merged model to https://huggingface.co/{repo_id}")
    return repo_id


@app.local_entrypoint()
def merge(repo_id: str, adapter_dir: str = "/adapters/minicpmv-lab-lora") -> None:
    pushed = merge_and_push.remote(repo_id=repo_id, adapter_dir=adapter_dir)
    print(f"\nDone. On the Space set:  ZEROGPU_MODEL_ID={pushed}")
    print("Rebuild the Space -> the fine-tuned model is live (Well-Tuned badge).")
