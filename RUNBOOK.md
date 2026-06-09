# Runbook — Offline extraction + MiniCPM-V fine-tune (Items 1 & 2)

This branch makes extraction run **fully offline on a fine-tuned MiniCPM-V** (off-grid +
fine-tune + quantization badges), while keeping the hosted API as a dev fallback so nothing
breaks before the GGUF exists.

## What changed (and why)
| Area | What | Earns |
|---|---|---|
| `src/extraction/` | Backend abstraction: **local MiniCPM-V (llama.cpp)** + API fallback + `build_extractor()` factory | off-grid |
| `src/grammar.py` | GBNF grammar → output is always valid `{tests,notes}` JSON | reliability |
| `src/markers.py` | 31-marker canonical reference (names, aliases, units, ranges) — also the KB seed | extraction + eval + KB |
| `train/synth_reports.py` | Renders format-diverse lab-report **images + gold JSON** (perfect labels) | fine-tune data |
| `train/to_sft_dataset.py` | → vision-SFT (`messages`+`images`) format | fine-tune data |
| `train/modal_finetune.py` | LoRA fine-tune MiniCPM-V **on Modal** (generates data on the box) | fine-tune + Modal prize |
| `scripts/merge_lora.py`, `scripts/convert_to_gguf.sh` | merge → GGUF + mmproj → **Q4_K_M** | quantization |
| `eval/run_eval.py`, `src/eval_scoring.py` | field-level P/R/F1 + value/unit/status accuracy | OpenBMB before/after |
| `app.py` | routes through `build_extractor()` (env `EXTRACTOR_BACKEND`) | wiring |

Backend selection — `EXTRACTOR_BACKEND`:
- `auto` (default): local if a GGUF is configured + llama.cpp importable, else API. **Stays working today, flips to offline the moment the GGUF lands — no code change.**
- `local`: force offline MiniCPM-V. `api`: force the hosted endpoint.

## Local dev / tests (no model needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install pillow pytest requests pymupdf json-repair
pytest -q                                # data pipeline + eval scoring
python train/synth_reports.py --n 12 --out train/data/preview   # eyeball the images
```

## The fine-tune → offline pipeline (run these in order)
```bash
# 1) Fine-tune MiniCPM-V on Modal (data generated on the box; needs Modal credits).
#    First confirm MODEL_TYPE/MODEL_ID in train/modal_finetune.py against `swift sft --help`.
modal run train/modal_finetune.py --n 4000 --epochs 2

# 2) Pull adapters from the Modal volume 'blood-test-adapters', then merge.
python scripts/merge_lora.py --base openbmb/MiniCPM-V-4_6 \
    --adapters ./adapters/minicpmv-lab-lora --out ./merged-minicpmv-lab

# 3) Convert to GGUF + mmproj and quantize Q4_K_M (needs a llama.cpp checkout).
LLAMA_CPP=./llama.cpp bash scripts/convert_to_gguf.sh ./merged-minicpmv-lab

# 4) Point the app at the local model and go offline.
export EXTRACTOR_BACKEND=local
export LOCAL_MODEL_PATH=./models/minicpmv-lab.Q4_K_M.gguf
export LOCAL_MMPROJ_PATH=./models/minicpmv-lab.mmproj.gguf
python app.py
```

## Before/after (the OpenBMB proof)
```bash
# held-out eval set
python train/synth_reports.py --n 60 --out eval/data/synth_eval --seed 99

# base model (point LOCAL_MODEL_PATH at the un-fine-tuned GGUF), then the fine-tuned one:
EXTRACTOR_BACKEND=local LOCAL_MODEL_PATH=... LOCAL_MMPROJ_PATH=... \
  python eval/run_eval.py --labels eval/data/synth_eval/labels.jsonl --run
# also drop in the ~15-30 real synthetic reports you collect → the credibility number
```

## Deploy (off-grid) checklist
- [ ] Bundle the two `.gguf` files via **git-lfs** in the Space repo (zero network at runtime).
- [ ] Space secrets/vars: `EXTRACTOR_BACKEND=local`, `LOCAL_MODEL_PATH`, `LOCAL_MMPROJ_PATH`.
- [ ] `LOCAL_N_GPU_LAYERS>0` if on ZeroGPU; `0` for pure CPU.
- [ ] Verify zero egress (no `requests` call on the local path).

## Verify-on-hardware (can't be tested in CI)
1. `train/modal_finetune.py`: confirm the ms-swift `MODEL_TYPE` for **MiniCPM-V 4.6**.
2. `scripts/convert_to_gguf.sh`: llama.cpp multimodal script paths/`MINICPMV_VERSION` for 4.6.
3. `src/extraction/local_minicpmv.py`: the `LOCAL_CHAT_HANDLER` class for your llama-cpp-python build.

## Still TODO (next PRs, not this one)
- Knowledge base + **cross-marker reasoning** (makes MiniCPM the star, the OpenBMB clincher).
- **Agent trace** (Item 3): turn the single call into a streamed Ingest→…→Render pipeline.
- Real sample reports bundled + downloadable `.html` document + video/social/model-card/dataset.
