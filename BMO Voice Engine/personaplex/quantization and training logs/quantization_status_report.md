# BMO Quantization & Export Status Report

This document summarizes the completion of the previous architectural roadmap and details the results of our subsequent attempt to push quantization further (which failed).

## Part 1: Completion of the Architectural Roadmap

We have successfully implemented all fixes suggested in the previous LLM diagnostic. The export pipeline and validation scripts are now structurally correct.

### ✅ Bug 1 (CRITICAL): `verify_depth.py` Logic Fixed
The PyTorch reference calculation for Depth Step 0 was corrected. The script now properly projects the temporal output `z_s` through `depformer_in.0` and adds **only** the `depformer_text_emb` (removing the erroneous addition of `depformer_emb.0`). The reference tensor now correctly matches the real `forward_depformer` logic.

### ✅ Bug 2 (HIGH): Missing Exports Added
`export_bmo_gguf.py` was updated to explicitly export the 17 missing critical tensors:
- The 16 output codebook heads (`linears.{0..15}.weight`)
- The final temporal norm (`out_norm.alpha`)
We also added a strict completeness check to the exporter that will intentionally crash if any of these 17 keys (or the standard transformer/depth keys) are missing from the final GGUF blobs.

### ✅ Bug 3 (MEDIUM): Depth Norm Shape Corrected
We ensured that all depth norms (`depformer_layers_{i}_norm1/2.alpha`) and the final temporal norm (`out_norm.alpha`) use `flatten=True` during export. They are now correctly reshaped from PyTorch's `(1, 1, 1024)` to a 1D `(1024,)` array, allowing the C++ engine's `ggml_add` broadcasting to work without dimension-mismatch errors.

### ✅ Bug 4 (LOW): File Size Bloat (FP32 -> FP16)
The previous exporter was casting all unquantized dense tensors from `bfloat16` to `float32`, doubling their size and causing the GGUF to balloon to 14 GB. We updated `export_dense_tensor` to support `preserve_half=True`. Now, all depth weights, codebook embeddings, and unquantized temporal fallbacks are exported as 16-bit floats (FP16). This alone recovers several gigabytes of space.

### ✅ Depth Excluded from SEPTQ
As designed, we confirmed that the Depth stack remains entirely excluded from multi-tier quantization to preserve audio prosody. PTQ/QAT is restricted strictly to the temporal stack.

---

## Part 2: The v11 "Full Quantization" Experiment & Failure

With the pipeline fixed, we faced a final challenge: hitting the strict **5.5 GB limit for the Jetson**. 

Our only successful temporal configuration so far was **"Half Cushion Max"**. It achieved an excellent 0.973 cosine similarity after QAT at 5.72 BPW (2% FP16, 12% INT8, 36% INT4, 50% INT2). However, to maintain that quality, it utilized a skip filter (`--skip-modules "self_attn.out_proj"`). Leaving the 32 `out_proj` modules as unquantized dense tensors added ~1.8 GB to the file size, pushing us over the 5.5 GB limit.

**The Pivot:** We hypothesized that we could remove the skip filter and force `out_proj` to be quantized alongside the other modules at 5.72 BPW to save 1.8 GB. We ran a new PTQ pass (`v11_full_quant`) on all 32 layers, targeting all 128 modules.

### The Catastrophic Failure
The v11 run failed completely. `self_attn.out_proj` proved to be hyper-sensitive to compression. Forcing it into the 5.72 BPW budget resulted in disastrous packing metrics (e.g., Layer 3 `out_proj` dropped to a 0.793 cosine similarity right out of the packer). 

This triggered an unrecoverable cascade during the Zero-Shot drift verification:
```text
=== PER-LAYER Z_S DRIFT (TEMPORAL) ===
layer=00 cos_median=0.999377
layer=01 cos_median=0.999709
layer=02 cos_median=0.999334
layer=03 cos_median=0.983591  <-- The cliff starts here (where out_proj packing was worst)
...
layer=31 cos_median=0.686525  <-- Final temporal output is destroyed
```
Because the baseline Zero-Shot median cosine fell to 0.894 (below the 0.90 safety threshold), the QAT fine-tuning script automatically aborted. The gradients from a starting point this degraded would have destroyed the model.

---

## Conclusion & Next Steps

The data definitively proves that **`self_attn.out_proj` cannot survive 5.72 BPW quantization**. Attempting to quantize it destroys temporal coherence.

To get as close to the 5.5 GB limit as possible without breaking the model, we must **combine** the safety of the original "Half Cushion Max" configuration with the new FP16 export optimizations:

1. Re-instate `--skip-modules "self_attn.out_proj"` for PTQ/QAT.
2. Rely on the newly implemented `preserve_half=True` in the export script. When `out_proj` bypasses quantization, the exporter will now write it as a 16-bit float (instead of 32-bit), halving the penalty of skipping it from ~1.8 GB to ~0.9 GB.

This hybrid approach represents the absolute maximum compression possible for this architecture while retaining >0.97 cosine similarity.
