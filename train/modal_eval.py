"""Before/after extraction eval on Modal — the OpenBMB proof.

Runs the labeled reports through the BASE and the FINE-TUNED model on a GPU and reports the
field-level accuracy jump. The model runs through the same ZeroGPU/Transformers backend the Space
uses (here `@spaces.GPU` is a no-op because the `spaces` package isn't installed, so generation
runs directly on the Modal GPU).

    modal run train/modal_eval.py::compare --finetuned-id dimitriskalligaridis/blood-test-minicpmv-4_6

Writes eval/before_after.json locally; render the chart with:  python eval/make_chart.py
"""

from __future__ import annotations

import modal

app = modal.App("blood-test-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "torchvision",
        "transformers[torch]>=5.7.0",
        "accelerate",
        "pillow",
        "pymupdf",
        "av",
        "requests",
        "json-repair",
    )
    # NOTE: deliberately no `spaces` package -> @spaces.GPU is a no-op -> runs on the Modal GPU.
    .add_local_dir("src", "/root/app/src")
    .add_local_dir("kb", "/root/app/kb")
    .add_local_dir("eval", "/root/app/eval")
)

hf_cache = modal.Volume.from_name("blood-test-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval_model(model_id: str, labels_rel: str = "eval/data/real/labels.jsonl") -> dict:
    """Run the configured model over the labeled reports and return field-level metrics."""
    import json
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/app")
    os.environ["ZEROGPU_QUANTIZE"] = "0"  # bf16 — a clean, representative eval

    from src.eval_scoring import format_metrics, score
    # Import the ZeroGPU backend directly (not via the factory) so we don't pull in the llama.cpp
    # backend, which isn't installed in this eval image.
    from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor

    labels_path = Path("/root/app") / labels_rel
    gold = [json.loads(ln) for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    extractor = ZeroGPUTransformersExtractor(model_id=model_id)
    base_dir = labels_path.parent

    preds: list[dict] = []
    for i, row in enumerate(gold):
        image_path = str((base_dir / row["image"]).resolve())
        try:
            result = extractor.extract(image_path, max_pages=3)
            preds.append({"tests": result.tests})
            print(f"[{i}] {row['image']}: {len(result.tests)} markers")
        except Exception as error:  # a failed report is a miss, keep going
            print(f"[{i}] {row['image']}: FAILED — {error}")
            preds.append({"tests": []})

    m = score(gold, preds)
    print(f"\n=== {model_id} ===\n{format_metrics(m)}\n")
    return {
        "model": model_id,
        "n": len(gold),
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "value_acc": m.value_acc,
        "unit_acc": m.unit_acc,
        "status_acc": m.status_acc,
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "matched": m.matched,
    }


@app.local_entrypoint()
def compare(
    finetuned_id: str,
    base_id: str = "openbmb/MiniCPM-V-4.6",
    labels_rel: str = "eval/data/real/labels.jsonl",
) -> None:
    """Eval base vs fine-tuned and write the before/after numbers for the chart."""
    import json
    from pathlib import Path

    base = eval_model.remote(base_id, labels_rel)
    fine = eval_model.remote(finetuned_id, labels_rel)

    metrics = ("f1", "recall", "precision", "value_acc", "unit_acc", "status_acc")
    print(f"\n  Extraction before/after — {base['n']} labeled reports\n")
    print(f"  {'metric':<12}{'base':>9}{'fine-tuned':>14}{'delta':>10}")
    for key in metrics:
        b, f = base[key], fine[key]
        print(f"  {key:<12}{b:>9.3f}{f:>14.3f}{('+' if f >= b else '') + f'{f - b:.3f}':>10}")

    out = Path("eval/before_after.json")
    out.write_text(json.dumps({"base": base, "finetuned": fine}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}  ->  render the chart with:  python eval/make_chart.py\n")


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def draft_labels(
    pdf_dir_rel: str = "eval/data/real",
    model_id: str = "openbmb/MiniCPM-V-4.6",
    exclude: tuple[str, ...] = ("06_drlogy_cbc.pdf", "02_cbc_umc_johndoe.pdf"),
) -> list[dict]:
    """Run the BASE model over the real PDFs to produce DRAFT labels you then correct by hand.

    Excludes the held-out eval reports so train/eval stay separate (no leakage).
    """
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/app")
    os.environ["ZEROGPU_QUANTIZE"] = "0"
    from src.extraction.zerogpu_transformers import ZeroGPUTransformersExtractor

    pdf_dir = Path("/root/app") / pdf_dir_rel
    extractor = ZeroGPUTransformersExtractor(model_id=model_id)
    drafts: list[dict] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        if pdf.name in exclude:
            continue
        try:
            tests = extractor.extract(str(pdf), max_pages=3).tests
            print(f"{pdf.name}: {len(tests)} draft markers")
        except Exception as error:
            print(f"{pdf.name}: FAILED — {error}")
            tests = []
        drafts.append({"image": pdf.name, "tests": tests, "notes": []})
    return drafts


@app.local_entrypoint()
def label(pdf_dir: str = "eval/data/real", out: str = "eval/data/real/labels_train_draft.jsonl") -> None:
    """Generate draft labels (base model) for you to correct, then mix into training."""
    import json
    from pathlib import Path

    drafts = draft_labels.remote(pdf_dir_rel=pdf_dir)
    out_path = Path(out)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in drafts:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {len(drafts)} DRAFT labels -> {out_path}")
    print("  Correct each line (fix marker/value/unit/status, delete junk rows), save it as")
    print("  eval/data/real/labels_train.jsonl, then retrain with the real mix-in:")
    print("    modal run train/modal_finetune.py::main --real-labels eval/data/real/labels_train.jsonl\n")
