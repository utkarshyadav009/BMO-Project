# Server-Side Run Commands

Run these on LineBreaker (the H100 server), in order. Stop at each gate and decide before proceeding.

## Step 0 — Sync edited files from local to server

From your local machine (NOT inside the agent):

```bash
# Adjust the SSH alias/path to your actual server config
LOCAL_REPO="/path/to/local/BMO-Project/personaplex_repo"
SERVER_REPO="jovyan@linebreaker:/home/jovyan/work/BMO-Project/personaplex_repo"

rsync -avz \
  "$LOCAL_REPO/apply_septq_multitier.py" \
  "$LOCAL_REPO/qat_septq.py" \
  "$SERVER_REPO/"
```

## Step 1 — Sanity check on server

SSH into the server, then:

```bash
cd /home/jovyan/work/BMO-Project/personaplex_repo/
export PYTHONPATH="/home/jovyan/work/BMO-Project/personaplex_repo/moshi"

# Confirm inputs exist
ls -lh v5_step1500_split.safetensors qat_septq_final_run/qat_best.pt
ls bmo_dataset_clean | head -3

# Confirm Python imports succeed (this DOES execute imports, unlike the local syntax check)
python -c "import apply_septq_multitier; import qat_septq; print('IMPORT OK')"

# Create experiment directory
mkdir -p tile_region_experiment
```

Stop and confirm everything passed before continuing.

## Step 2 — PTQ at validated 2/12/36/50 ratios, tile-region mode (~2-4 hours)

This is the critical experiment. Uses the validated qat_best.pt ratios but with tile-region allocation instead of per-element. The question is how much pre-QAT quality tile-region costs versus per-element.

```bash
export CUDA_VISIBLE_DEVICES=0

python -u apply_septq_multitier.py \
  --device cuda:0 \
  --bf16 v5_step1500_split.safetensors \
  --calibration-clips bmo_dataset_clean \
  --bits 2 \
  --ratio-fp16 0.02 \
  --ratio-int8 0.12 \
  --ratio-int4 0.36 \
  --allocation-mode tile-region \
  --tile-size 128 \
  --tile-aggregate p95 \
  --block-size 128 \
  --max-calibration-samples 16384 \
  --max-steps-per-clip 750 \
  --max-clips 857 \
  --quantize-layers 0-30 \
  --skip-modules "self_attn.out_proj" \
  --quantize-depth-int8 \
  --out tile_region_experiment/bmo_tile_region_p95.pt \
  2>&1 | tee tile_region_experiment/ptq_p95.log
```

Sanity-check the log after completion:
- Look for `[DEPTH-INT8] quantized=N skipped=M` with N > 0
- Look for tile_region_metadata being saved
- Check `.pt` size: `ls -lh tile_region_experiment/bmo_tile_region_p95.pt`

## Step 3 — Verify z_s drift pre-QAT (~10 min)

```bash
python -u verify_septq_zs_drift.py \
  --device cuda:0 \
  --teacher v5_step1500_split.safetensors \
  --student tile_region_experiment/bmo_tile_region_p95.pt \
  --steps 125 \
  --min-median-cos 0.85 \
  --save-json tile_region_experiment/zs_p95.json \
  2>&1 | tee tile_region_experiment/zs_p95.log
```

Note: `--min-median-cos 0.85` is the pre-QAT gate, not the ship gate. The Heavy Cushion config hit 0.893 pre-QAT and recovered to 0.973 post-QAT — that's the playbook.

## Step 4 — Decision gate

Read the cos_median from the verify log and apply this table:

| Pre-QAT cos_median | Decision |
|---|---|
| ≥ 0.90 | Tile-region p95 is competitive with per-element. Proceed to Step 5 (QAT). |
| 0.85 - 0.90 | Borderline. Proceed to Step 5, expect ship-borderline post-QAT. |
| 0.70 - 0.85 | Try `--tile-aggregate max` instead (re-run Step 2 with that flag, new output path) before QAT. |
| < 0.70 | **STOP.** Tile-region at these ratios is structurally inadequate. Do not run QAT speculatively. Report back. The validated `qat_septq_final_run/qat_best.pt` remains the ship fallback. |

If you re-run Step 2 with `max` aggregation:

```bash
python -u apply_septq_multitier.py \
  --device cuda:0 \
  --bf16 v5_step1500_split.safetensors \
  --calibration-clips bmo_dataset_clean \
  --bits 2 \
  --ratio-fp16 0.02 --ratio-int8 0.12 --ratio-int4 0.36 \
  --allocation-mode tile-region \
  --tile-size 128 \
  --tile-aggregate max \
  --block-size 128 \
  --max-calibration-samples 16384 --max-steps-per-clip 750 --max-clips 857 \
  --quantize-layers 0-30 \
  --skip-modules "self_attn.out_proj" \
  --quantize-depth-int8 \
  --out tile_region_experiment/bmo_tile_region_max.pt \
  2>&1 | tee tile_region_experiment/ptq_max.log

python -u verify_septq_zs_drift.py \
  --device cuda:0 \
  --teacher v5_step1500_split.safetensors \
  --student tile_region_experiment/bmo_tile_region_max.pt \
  --steps 125 \
  --min-median-cos 0.85 \
  --save-json tile_region_experiment/zs_max.json \
  2>&1 | tee tile_region_experiment/zs_max.log
```

## Step 5 — QAT (gated on Step 4; ~7 hours)

Only if Step 4 returned ≥ 0.85 pre-QAT. Use whichever .pt won (p95 or max) as the student. The example below uses p95.

**Before running:** confirm exact flag names with `python qat_septq.py --help`. The flags below mirror the script that produced qat_best.pt; adjust if the actual script differs.

```bash
python -u qat_septq.py \
  --device cuda:0 \
  --teacher v5_step1500_split.safetensors \
  --student tile_region_experiment/bmo_tile_region_p95.pt \
  --calibration-clips bmo_dataset_clean \
  --max-train-steps 600 \
  --warmup-steps 100 \
  --train-layers 0-30 \
  --skip-modules "self_attn.out_proj" \
  --out tile_region_experiment/qat_tile_region \
  --teacher-dtype bf16 \
  --student-dtype bf16 \
  2>&1 | tee tile_region_experiment/qat_p95.log
```

Re-verify after QAT:

```bash
python -u verify_septq_zs_drift.py \
  --device cuda:0 \
  --teacher v5_step1500_split.safetensors \
  --student tile_region_experiment/qat_tile_region/qat_best.pt \
  --steps 125 \
  --min-median-cos 0.97 \
  --save-json tile_region_experiment/zs_qat_p95.json \
  2>&1 | tee tile_region_experiment/zs_qat_p95.log
```

## Step 6 — Pull final .pt back to local for testing

From local machine:

```bash
SERVER_PT="jovyan@linebreaker:/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_tile_region/qat_best.pt"
LOCAL_DEST="/path/to/local/BMO-Project/personaplex_repo/tile_region_experiment/qat_best_tile_region.pt"
mkdir -p "$(dirname $LOCAL_DEST)"
rsync -avz --progress "$SERVER_PT" "$LOCAL_DEST"
```

Then run the 5x joke loop test locally using the existing test script. Adjust the MODEL path in that script to point at the synced file.
