"""Medical-reasoning LoRA fine-tune (Track 1) — earns Well-Tuned WITHOUT touching extraction.

Roman's idea, done as LoRA (not full FT, which would catastrophically forget the vision/extraction
ability). We freeze the vision encoder and LoRA the LLM only, on the general medical-reasoning
dataset FreedomIntelligence/medical-o1-reasoning-SFT (TEXT, no images). The result is used as the
*interpretation phraser* that speaks the KB-grounded facts fluently — extraction stays on base.

A held-out slice is used for eval (reasoning loss). Gate A also re-runs the extraction eval on the
merged model to confirm extraction did not regress.

    modal run train/modal_medreason.py::main --n 100 --epochs 1     # cheap smoke test first
    modal run --detach train/modal_medreason.py::main               # full run (n=4000)
    modal run train/modal_finetune.py::merge --adapter-dir /adapters/medreason-lora --repo-id <owner>/<name>-medreason
    modal run train/modal_eval.py::compare --finetuned-id <owner>/<name>-medreason   # Gate A: extraction unharmed?
"""

from __future__ import annotations

import modal

MODEL_ID = "openbmb/MiniCPM-V-4.6"

app = modal.App("blood-test-medreason")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Pin the exact ms-swift that recognizes MiniCPM-V 4.6 (the extraction run used this);
        # an unpinned `datasets` previously dragged ms-swift down to a version that didn't. ms-swift
        # brings a compatible `datasets`, so we don't add it ourselves.
        "torch",
        "transformers>=5.7.0",
        "peft>=0.12",
        "accelerate>=0.33",
        "ms-swift==4.3.0",
        "sentencepiece",
        "timm",
    )
)

adapters = modal.Volume.from_name("blood-test-adapters", create_if_missing=True)
hf_cache = modal.Volume.from_name("blood-test-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={"/adapters": adapters, "/root/.cache/huggingface": hf_cache},
)
def train_medreason(n: int = 4000, epochs: int = 1, lr: float = 1e-4, n_eval: int = 500, seed: int = 13) -> str:
    import json
    import os
    import subprocess
    from pathlib import Path

    from datasets import load_dataset

    os.environ["USE_HF"] = "1"  # pull dataset + weights from HF (fast on Modal), not ModelScope

    # 1) medical-o1 reasoning data (English) -> text chat messages (Question -> CoT + Response)
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n + n_eval, len(ds))))

    def to_messages(ex: dict) -> dict:
        q = (ex.get("Question") or "").strip()
        cot = (ex.get("Complex_CoT") or "").strip()
        resp = (ex.get("Response") or "").strip()
        answer = f"{cot}\n\n{resp}".strip() if cot else resp
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]}

    rows = [to_messages(ex) for ex in ds]
    val_rows, train_rows = rows[:n_eval], rows[n_eval:]

    data_dir = Path("/root/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "medreason_train.jsonl"
    val_path = data_dir / "medreason_val.jsonl"
    train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows), encoding="utf-8")
    val_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows), encoding="utf-8")
    print(f"medical-o1: {len(train_rows)} train, {len(val_rows)} held-out eval examples")

    # 2) LoRA the LLM only (freeze vision) on the reasoning text — keeps extraction untouched.
    out_dir = "/adapters/medreason-lora"
    cmd = [
        "swift", "sft",
        "--model", MODEL_ID,
        "--dataset", str(train_path),
        "--val_dataset", str(val_path),
        "--num_train_epochs", str(epochs),
        "--lora_rank", "16",
        "--lora_alpha", "32",
        "--learning_rate", str(lr),
        "--warmup_ratio", "0.05",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--max_length", "4096",       # medical CoT answers are long
        "--freeze_vit", "true",       # do not touch the vision encoder (extraction lives there)
        "--eval_steps", "50",         # report held-out reasoning loss during training
        "--output_dir", out_dir,
        "--save_total_limit", "1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "USE_HF": "1"})

    adapters.commit()
    return out_dir


@app.local_entrypoint()
def main(n: int = 4000, epochs: int = 1, lr: float = 1e-4) -> None:
    path = train_medreason.remote(n=n, epochs=epochs, lr=lr)
    print(f"\nMedical-reasoning LoRA saved to volume 'blood-test-adapters' at {path}")
    print("Next:")
    print("  modal run train/modal_finetune.py::merge --adapter-dir /adapters/medreason-lora \\")
    print("      --repo-id dimitriskl/blood-test-minicpmv-4_6-medreason")
    print("  modal run train/modal_eval.py::compare --finetuned-id dimitriskl/blood-test-minicpmv-4_6-medreason")
    print("  ^ Gate A: confirms extraction did NOT regress on the reasoning-tuned model.")
