#!/usr/bin/env bash
# Convert the merged, fine-tuned MiniCPM-V LLM into a quantized GGUF for llama.cpp, and
# download the official MiniCPM-V 4.6 mmproj. Produces the two files the offline backend loads:
#   models/minicpmv-lab.Q4_K_M.gguf   (LOCAL_MODEL_PATH)
#   models/minicpmv-lab.mmproj.gguf   (LOCAL_MMPROJ_PATH)
#
# LoRA touches the LLM, not the vision encoder, so the official mmproj remains valid.
#
# Prereqs: a merged HF model (scripts/merge_lora.py), a local llama.cpp checkout, and
# huggingface-cli (`pip install huggingface_hub`).
set -euo pipefail

MERGED="${1:-./merged-minicpmv-lab}"          # merged HF model dir
LLAMA="${LLAMA_CPP:-./llama.cpp}"             # path to a llama.cpp checkout
OUT="${OUT_DIR:-./models}"
mkdir -p "$OUT"

echo "==> 1/3  Download official MiniCPM-V 4.6 GGUF assets (includes mmproj)"
huggingface-cli download openbmb/MiniCPM-V-4.6-gguf --local-dir "$OUT"
MMPROJ="$(find "$OUT" -maxdepth 1 -type f -iname '*mmproj*.gguf' | head -n 1)"
if [[ -z "${MMPROJ:-}" ]]; then
  echo "No mmproj GGUF found in $OUT after download" >&2
  exit 1
fi
if [[ "$MMPROJ" != "$OUT/minicpmv-lab.mmproj.gguf" ]]; then
  cp "$MMPROJ" "$OUT/minicpmv-lab.mmproj.gguf"
fi

echo "==> 2/3  Convert the merged LLM to GGUF (f16)"
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "$OUT/minicpmv-lab.f16.gguf"

echo "==> 3/3  Quantize to Q4_K_M"
"$LLAMA/llama-quantize" "$OUT/minicpmv-lab.f16.gguf" "$OUT/minicpmv-lab.Q4_K_M.gguf" Q4_K_M

echo
echo "Done. Set in the Space:"
echo "  LOCAL_MODEL_PATH=$OUT/minicpmv-lab.Q4_K_M.gguf"
echo "  LOCAL_MMPROJ_PATH=$OUT/minicpmv-lab.mmproj.gguf"
echo "  EXTRACTOR_BACKEND=local"
echo "Track both .gguf files with git-lfs and commit them into the Space repo."
