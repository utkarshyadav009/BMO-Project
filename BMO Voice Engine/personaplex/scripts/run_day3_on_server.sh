#!/usr/bin/env bash
# Day 3 server commands (run from personaplex repo root with BMO-Project conda env).
# Adjust MIMI_CKPT and paths to match the H100 host layout.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/moshi:${PYTHONPATH:-}"
export BMO_SO_PATH="${BMO_SO_PATH:-${ROOT}/build/libbmo.so}"

GGUF_V5="${GGUF_V5:-${ROOT}/bmo_septq_v5.gguf}"
PT_CKPT="${PT_CKPT:-${ROOT}/bmo_temporal_half_cushion_max.pt}"
GOLDEN="${GOLDEN:-${ROOT}/build/v5_matvec_golden.bin}"
FP16_ST="${FP16_ST:-${ROOT}/v5_step1500_split.safetensors}"
MIMI_CKPT="${MIMI_CKPT:?Set MIMI_CKPT to tokenizer-e351c8d8-checkpoint125.safetensors (or your Mimi weights)}"
TOKENIZER="${TOKENIZER:-${ROOT}/tokenizer_spm_32k_3.model}"

echo "=== D2: generate golden + build + run bmo_v5_runtime_test ==="
python scripts/gen_v5_matvec_golden.py --pt "$PT_CKPT" --out "$GOLDEN"
cmake --build build --target bmo_shared bmo_v5_runtime_test -j"$(nproc)"
./build/bmo_v5_runtime_test "$GOLDEN" "$GGUF_V5"

echo "=== D3: end2end v5 GGUF vs FP16 safetensors ==="
HARNESS="${HARNESS:-${ROOT}/pt_dump_final/harness_input.json}"
python scripts/end2end_v5_vs_fp16.py \
  --gguf "$GGUF_V5" \
  --fp16 "$FP16_ST" \
  --so-path "$BMO_SO_PATH" \
  --harness-input "$HARNESS"

echo "=== D4: short stream A/B (requires Mimi + tokenizer; ~63 frames @ Mimi frame rate) ==="
# bmo_inference loads GGUF only. For FP16 A/B, set GGUF_FP16 to a dense/FP16-exported GGUF
# (safetensors is used in D3 via PyTorch, not here).
GGUF_FP16="${GGUF_FP16:-}"
OUT_V5="${ROOT}/path_b_day3_audio_test.wav"
OUT_FP16="${ROOT}/path_b_day3_audio_test_fp16.wav"
python bmo_inference.py stream \
  --so-path "$BMO_SO_PATH" \
  --gguf "$GGUF_V5" \
  --mimi "$MIMI_CKPT" \
  --tokenizer "$TOKENIZER" \
  --depth-mode cpp \
  --force-text-pad \
  --n-frames 63 \
  --n-ctx 256 \
  --output-wav "$OUT_V5" \
  --output-text "${OUT_V5%.wav}.json"

if [[ -n "${GGUF_FP16}" && -f "${GGUF_FP16}" ]]; then
  python bmo_inference.py stream \
    --so-path "$BMO_SO_PATH" \
    --gguf "$GGUF_FP16" \
    --mimi "$MIMI_CKPT" \
    --tokenizer "$TOKENIZER" \
    --depth-mode cpp \
    --force-text-pad \
    --n-frames 63 \
    --n-ctx 256 \
    --output-wav "$OUT_FP16" \
    --output-text "${OUT_FP16%.wav}.json"
  echo "D4 outputs: $OUT_V5  $OUT_FP16"
else
  echo "Skip FP16 stream: set GGUF_FP16 to a FP16/dense GGUF path for path_b_day3_audio_test_fp16.wav"
  echo "D4 v5 output: $OUT_V5"
fi
