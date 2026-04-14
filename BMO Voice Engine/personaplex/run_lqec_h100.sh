#!/usr/bin/env bash
set -euo pipefail

# Run from personaplex root on H100 host.
python train_lqec.py \
  --teacher v5_step1500.safetensors \
  --student bmo_temporal_int4_base.pt \
  --mimi-weight tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer tokenizer_spm_32k_3.model \
  --input-wav tellmeajoke_padded.wav \
  --text-prompt "Tell me a joke." \
  --steps 50 \
  --rank 64 \
  --alpha 16 \
  --lr 1e-4 \
  --device cuda:0 \
  --out lqec_overfit_step50.pt \
  --log-json lqec_overfit_log.json
