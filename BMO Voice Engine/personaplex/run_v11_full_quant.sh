#!/bin/bash
set -e

# ============================================================
# BMO v11 Full Quantization Pipeline
# ============================================================
# Changes vs Half Cushion Max (v10):
#   1. REMOVED --skip-modules "self_attn.out_proj"
#      → out_proj is now quantized (saves ~1.8 GB in GGUF)
#   2. CHANGED --quantize-layers 0-31 (was 0-30)
#      → L31 is now quantized (saves ~0.7 GB in GGUF)
#   3. Same proven 5.72 BPW ratios (2/12/36/50)
#   4. Same max calibration data (857 clips, 16384 samples)
# ============================================================

export PYTHONPATH="/home/jovyan/work/BMO-Project/personaplex_repo/moshi"
export CUDA_VISIBLE_DEVICES=1

# --- Output directory ---
RUN_DIR="v11_full_quant_run"
mkdir -p "$RUN_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=== BMO v11 Full Quant Pipeline — $TIMESTAMP ===" | tee "$RUN_DIR/run.log"
echo "Output directory: $RUN_DIR" | tee -a "$RUN_DIR/run.log"

# ============================================================
# STEP 1: Multi-tier SEPTQ quantization
# ============================================================
echo ""
echo "=== STEP 1: SEPTQ Quantization (all 32 layers, all 4 modules) ===" | tee -a "$RUN_DIR/run.log"
echo "  Layers: 0-31 (all 32)"
echo "  Modules: in_proj + gating_in + gating_out + out_proj (128 total)"
echo "  BPW: 5.72 (2% FP16 / 12% INT8 / 36% INT4 / 50% INT2)"
echo "  Calibration: 857 clips, 16384 samples"
echo ""

python -u apply_septq_multitier.py \
  --device cuda:0 \
  --bf16 v5_step1500_split.safetensors \
  --calibration-clips bmo_dataset_clean \
  --bits 2 \
  --ratio-fp16 0.02 \
  --ratio-int8 0.12 \
  --ratio-int4 0.36 \
  --block-size 128 \
  --max-calibration-samples 16384 \
  --max-steps-per-clip 750 \
  --max-clips 857 \
  --quantize-layers 0-31 \
  --skip-modules "none" \
  --out "$RUN_DIR/bmo_temporal_v11_full.pt" 2>&1 | tee "$RUN_DIR/septq_quant.log"

echo ""
echo "=== STEP 1 COMPLETE ===" | tee -a "$RUN_DIR/run.log"
echo ""

# ============================================================
# STEP 2: Zero-shot drift verification
# ============================================================
echo "=== STEP 2: Z_S Drift Verification ===" | tee -a "$RUN_DIR/run.log"

python -u verify_septq_zs_drift.py \
  --device cuda:0 \
  --teacher v5_step1500_split.safetensors \
  --student "$RUN_DIR/bmo_temporal_v11_full.pt" \
  --steps 125 \
  --min-median-cos 0.997 \
  --save-json "$RUN_DIR/zs_v11_full.json" 2>&1 | tee "$RUN_DIR/zs_drift.log"

echo ""
echo "=== STEP 2 COMPLETE ===" | tee -a "$RUN_DIR/run.log"
echo ""

# ============================================================
# STEP 3: QAT fine-tuning
# ============================================================
echo "=== STEP 3: QAT Fine-Tuning ===" | tee -a "$RUN_DIR/run.log"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u qat_septq.py \
  --teacher v5_step1500_split.safetensors \
  --student-quant-meta "$RUN_DIR/bmo_temporal_v11_full.pt" \
  --calibration-clips bmo_dataset_clean \
  --mimi-weight tokenizer-e351c8d8-checkpoint125.safetensors \
  --device cuda \
  --train-layers 0-31 \
  --max-clips 32 \
  --max-steps-per-clip 125 \
  --train-max-steps-per-clip 64 \
  --eval-clips 16 \
  --eval-max-steps-per-clip 64 \
  --max-train-steps 1200 \
  --min-train-steps 0 \
  --checkpoint-every 50 \
  --lr 3e-6 \
  --warmup-steps 100 \
  --backward-mode per-token \
  --target-median-cos 0.999 \
  --flatline-median-cos 0.0 \
  --out-dir "$RUN_DIR/qat_output" \
  --seed 1234 \
  --log-every 10 2>&1 | tee "$RUN_DIR/qat.log"

echo ""
echo "=== STEP 3 COMPLETE ===" | tee -a "$RUN_DIR/run.log"
echo ""

# ============================================================
# DONE
# ============================================================
echo "========================================" | tee -a "$RUN_DIR/run.log"
echo "=== ALL STEPS COMPLETE ===" | tee -a "$RUN_DIR/run.log"
echo "  Quant output: $RUN_DIR/bmo_temporal_v11_full.pt" | tee -a "$RUN_DIR/run.log"
echo "  ZS drift:     $RUN_DIR/zs_v11_full.json" | tee -a "$RUN_DIR/run.log"
echo "  QAT output:   $RUN_DIR/qat_output/" | tee -a "$RUN_DIR/run.log"
echo "  Logs:         $RUN_DIR/*.log" | tee -a "$RUN_DIR/run.log"
echo "========================================" | tee -a "$RUN_DIR/run.log"
