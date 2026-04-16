#!/usr/bin/env bash
set -euo pipefail

# Run from personaplex root on the server.
# You can override any variable at invocation time, e.g.:
# BF16_CKPT=/path/to/model.safetensors OUT_CKPT=my_fp32.pt bash run_fp32_identity_diagnostic.sh

export PYTHONPATH="${PYTHONPATH:-/home/jovyan/work/BMO-Project/personaplex_repo/moshi}"

BF16_CKPT="${BF16_CKPT:-v5_step1500.safetensors}"
EIGENVECTORS="${EIGENVECTORS:-bmo_slicegpt_eigenvectors.pt}"
CONFIG="${CONFIG:-bmo_config.json}"
OUT_CKPT="${OUT_CKPT:-bmo_slicegpt_4096_identity_fp32.pt}"

DEVICE_EXPORT="${DEVICE_EXPORT:-cuda:0}"
DEVICE_EVAL="${DEVICE_EVAL:-cuda}"

INPUT_WAV="${INPUT_WAV:-tellmeajoke_padded.wav}"
VOICE_PROMPT_WAV="${VOICE_PROMPT_WAV:-bmo_621.wav}"
TEXT_PROMPT="${TEXT_PROMPT:-Tell me a joke.}"
MIMI_WEIGHT="${MIMI_WEIGHT:-tokenizer-e351c8d8-checkpoint125.safetensors}"
TOKENIZER="${TOKENIZER:-tokenizer_spm_32k_3.model}"
VOICE_RATIO="${VOICE_RATIO:-0.25}"

RUN_FULL_ROLLOUT="${RUN_FULL_ROLLOUT:-true}"
RUNTIME_PATCH="${RUNTIME_PATCH:-true}"

EXPORT_LOG="fp32_identity_export.log"
STEP0_LOG="fp32_stage_step0.log"
TF64_LOG="fp32_teacher_forced.log"

echo "[STARTING FP32 IDENTITY DIAGNOSTIC PACK]"
echo "[INFO] PYTHONPATH=${PYTHONPATH}"
echo "[INFO] runtime_patch=${RUNTIME_PATCH}"

echo "[RUNNING: FP32 IDENTITY EXPORT]"
python apply_slicegpt.py \
  --bf16 "${BF16_CKPT}" \
  --eigenvectors "${EIGENVECTORS}" \
  --config "${CONFIG}" \
  --out "${OUT_CKPT}" \
  --d-new 4096 \
  --dtype float32 \
  --device "${DEVICE_EXPORT}" \
  --attn-rope-mode quarot \
  | tee "${EXPORT_LOG}"

echo "[RUNNING: FP32 STEP-0 STAGE PROBE]"
python verify_int4_rollout_drift.py \
  --bf16 "${BF16_CKPT}" \
  --int4 "${OUT_CKPT}" \
  --device "${DEVICE_EVAL}" \
  --teacher-forced true \
  --runtime-patch "${RUNTIME_PATCH}" \
  --teacher-dtype float32 \
  --student-dtype float32 \
  --steps 1 \
  --report-step 0 \
  --input-wav "${INPUT_WAV}" \
  --voice-prompt-wav "${VOICE_PROMPT_WAV}" \
  --text-prompt "${TEXT_PROMPT}" \
  --mimi-weight "${MIMI_WEIGHT}" \
  --tokenizer "${TOKENIZER}" \
  --voice-ratio "${VOICE_RATIO}" \
  --layer-ladder true \
  --layer-ladder-step 0 \
  --layer-ladder-unrotate-student true \
  --layer-stage-probe true \
  | tee "${STEP0_LOG}"

if [[ "${RUN_FULL_ROLLOUT}" == "true" ]]; then
  echo "[RUNNING: FP32 FULL TEACHER-FORCED ROLLOUT]"
  python verify_int4_rollout_drift.py \
    --bf16 "${BF16_CKPT}" \
    --int4 "${OUT_CKPT}" \
    --device "${DEVICE_EVAL}" \
    --teacher-forced true \
    --runtime-patch "${RUNTIME_PATCH}" \
    --teacher-dtype float32 \
    --student-dtype float32 \
    --steps 64 \
    --report-step 63 \
    --input-wav "${INPUT_WAV}" \
    --voice-prompt-wav "${VOICE_PROMPT_WAV}" \
    --text-prompt "${TEXT_PROMPT}" \
    --mimi-weight "${MIMI_WEIGHT}" \
    --tokenizer "${TOKENIZER}" \
    --voice-ratio "${VOICE_RATIO}" \
    | tee "${TF64_LOG}"
fi

echo "===================================================="
echo "          FP32 IDENTITY DIAGNOSTIC SUMMARY          "
echo "===================================================="

grep -E "RoPE-safe attention shape sanity|Saved SliceGPT checkpoint" "${EXPORT_LOG}" || true

grep -E "=== REPORT STEP ===|Mean cosine|Mean MSE|Top-1 agreement|Worst cosine|Worst MSE" "${STEP0_LOG}" || true

grep -E "step=00 layer=(00|04|05|15|31) cos=" "${STEP0_LOG}" || true
grep -E "step=00 layer=00 stage=(norm1|self_attn|norm2|gating) cos=" "${STEP0_LOG}" || true

if [[ "${RUN_FULL_ROLLOUT}" == "true" ]]; then
  grep -E "=== REPORT STEP ===|Mean cosine|Mean MSE|Top-1 agreement|Worst cosine|Worst MSE" "${TF64_LOG}" || true
  echo "[DONE] Logs: ${EXPORT_LOG}, ${STEP0_LOG}, ${TF64_LOG}"
else
  echo "[DONE] Logs: ${EXPORT_LOG}, ${STEP0_LOG}"
fi
