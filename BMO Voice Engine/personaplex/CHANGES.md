# CHANGES.md — Tile-Region Activation Mismatch Fix

**Date:** 2026-06-12
**File:** `apply_septq_multitier.py`

## Problem

In `--allocation-mode tile-region`, tile tiers were computed in a **pre-pass** over all
layers on the pristine BF16 model, then applied during a separate sequential quantization
loop that modifies the model in place. By layer 3+ the solver ran on progressively-quantized
activations while the tiers were optimized for clean BF16 ones. This mismatch caused an
early quality cliff (z_s median 0.29). The per-element path was immune because it computes
its mask from the same activations it solves against (validated: 0.893, graceful decay).

## Fix Summary

### 1. Pre-pass deleted

The entire block that began with:
```
if str(args.allocation_mode) == "tile-region":
    print("[INFO] tile-region allocation enabled; collecting global tile scores...")
```
…including the `collect_inputs_for_entries` loop over all layers, `compute_tile_region_scores`
per entry, `assign_global_tile_region_tiers` consuming the collected scores, and the
pre-flight sanity check — has been **deleted**.

### 2. Scoring inlined into the quantization loop

For each tensor `name`, **after** `x = inputs.get(name)` is captured (the same `x` passed to
`septq_quantize_weight`) and **before** calling `septq_quantize_weight`, when
`allocation_mode == "tile-region"`:

1. `compute_tile_region_scores(weight=weight, activations=x, ...)` — scores this tensor
   against the **identical** `x` the solver will use.
2. `assign_global_tile_region_tiers({name: scores_dict}, ...)` — per-tensor tier assignment
   via single-tensor dict; produces the correct 2/12/36/50 split for this tensor alone.
3. `expand_2d_tile_tiers(...)` — expands tile tiers to per-element `forced_tier_mask`.
4. `tile_region_metadata["tiles"][name] = tile_info` — accumulates into running metadata.

### 3. Activation identity guaranteed

The `x` tensor object used for `compute_tile_region_scores` is the **exact same Python
object** passed to `septq_quantize_weight`. No re-capture, no re-collection between scoring
and quantizing. This identity is the entire fix.

### 4. `assign_global_tile_region_tiers` confirmed per-tensor

The function already implements per-tensor ranking (iterates `for name in ordered_names` and
computes budgets `n_fp16 = int(ratio * n)` within each tensor). Calling it with a
single-tensor dict `{name: scores_dict}` correctly produces that tensor's own ratio split.
No global-pool logic exists that would misbehave with one entry.

### 5. Metadata completeness

`tile_region_metadata` is initialized before the loop with all static fields:
- `version`, `layout_contract_version`, `layout_contract`
- `tile_size`, `tile_shape`, `tile_shape_per_tier`
- `layout_target`, `aggregate`, `int2_storage`
- `"tiles": {}` — populated per-tensor during the loop

After the loop, `tile_region_metadata["tiles"]` contains an entry for every quantized tensor
with identical structure to what the old pre-pass produced:
`n_tiles_total`, `tile_dim`, `tile_shape`, `tile_grid`, `pad_rows`, `pad_cols`, `tile_tiers`.

All downstream consumers (`tile_packed_streams`, projected sizes, reports, payload build)
read from `tile_region_metadata` unchanged.

### 6. Dangling references

- `tile_region_scores` — only exists as function parameter names and the inline call; no
  stale variable references remain in `main()`.
- `tile_assignments` — completely removed (zero hits in file).
- `"collecting global tile scores"` print — removed.

### 7. What was NOT touched

- `compute_tile_region_scores`, `aggregate_2d_tile_scores`, `expand_2d_tile_tiers`,
  `septq_quantize_weight` internals — unchanged.
- The per-element / block-aligned / block-max-tier paths — unchanged.
- Depth-INT8, QAT script, CUDA, GGUF export — unchanged.
- Dense weights still stored for eval.

**Per-element mode dequantized values are unaffected** — that code path was not modified.

## Verification

```
python -c "import ast; ast.parse(open('apply_septq_multitier.py').read()); print('SYNTAX OK')"
# Output: SYNTAX OK

findstr "tile_assignments" apply_septq_multitier.py
# Output: (empty — no hits)

findstr "collecting global tile scores" apply_septq_multitier.py
# Output: (empty — no hits)
```
