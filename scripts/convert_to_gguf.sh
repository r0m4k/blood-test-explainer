#!/usr/bin/env bash
# Convert the merged, fine-tuned MiniCPM-V into a quantized GGUF + vision projector (mmproj)
# for llama.cpp, then quantize to Q4_K_M. Produces the two files the offline backend loads:
#   models/minicpmv-lab.Q4_K_M.gguf   (LOCAL_MODEL_PATH)
#   models/minicpmv-lab.mmproj.gguf   (LOCAL_MMPROJ_PATH)
#
# Prereqs: a merged HF model (scripts/merge_lora.py) and a local llama.cpp checkout.
#
# ⚠️ MiniCPM-V GGUF conversion lives under llama.cpp's multimodal tooling and the exact script
# names/paths move between releases (older: examples/llava/*, newer: tools/mtmd/*). Check your
# llama.cpp version and adjust the three SCRIPT paths below. The flow is stable; the paths drift.
set -euo pipefail

MERGED="${1:-./merged-minicpmv-lab}"          # merged HF model dir
LLAMA="${LLAMA_CPP:-./llama.cpp}"             # path to a llama.cpp checkout
OUT="${OUT_DIR:-./models}"
VER="${MINICPMV_VERSION:-3}"                  # MiniCPM-V arch version flag; confirm for 4.6
mkdir -p "$OUT"

echo "==> 1/4  Split vision encoder + LLM (surgery)"
python "$LLAMA/examples/llava/minicpmv-surgery.py" -m "$MERGED"

echo "==> 2/4  Build the vision projector (mmproj) GGUF"
python "$LLAMA/examples/llava/minicpmv-convert-image-encoder-to-gguf.py" \
  -m "$MERGED" \
  --minicpmv-projector "$MERGED/minicpmv.projector" \
  --output-dir "$OUT" \
  --minicpmv_version "$VER"
mv "$OUT"/*mmproj*.gguf "$OUT/minicpmv-lab.mmproj.gguf" 2>/dev/null || true

echo "==> 3/4  Convert the LLM to GGUF (f16)"
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED/model" --outfile "$OUT/minicpmv-lab.f16.gguf"

echo "==> 4/4  Quantize to Q4_K_M"
"$LLAMA/llama-quantize" "$OUT/minicpmv-lab.f16.gguf" "$OUT/minicpmv-lab.Q4_K_M.gguf" Q4_K_M

echo
echo "Done. Set in the Space:"
echo "  LOCAL_MODEL_PATH=$OUT/minicpmv-lab.Q4_K_M.gguf"
echo "  LOCAL_MMPROJ_PATH=$OUT/minicpmv-lab.mmproj.gguf"
echo "  EXTRACTOR_BACKEND=local"
echo "Track both .gguf files with git-lfs and commit them into the Space repo."
