# CHANGES.md — Tile-Region Activation Mismatch Fix

**Date:** 2026-06-12
**File:** `apply_septq_multitier.py`

<<<<<<< Updated upstream
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
=======
### apply_septq_multitier.py

#### Previous changes (preserved)
- Added `--tile-shape`, `--tile-layout-target`, and `--int2-storage` flags for hardware-aware tile-region export preparation.
- Replaced 128x1 row-stripe tile aggregation/expansion with padded 2D tile geometry; default `--tile-shape auto --tile-layout-target ampere` resolves to 64x64 tiles.
- Stored authoritative per-tile tiers in `tile_region_metadata.tiles[name].tile_tiers`, plus `mask_representation="per-tile"` and `tier_masks_authoritative=False` for tile-region mode.
- Preserved dense dequantized weights and retained `tier_masks_uint2` as a non-authoritative debug/QAT compatibility artifact in tile-region mode.
- Added tile geometry metadata: `tile_shape_per_tier`, `layout_target`, per-tensor `tile_grid`, `pad_rows`, and `pad_cols`.
- Added `int2_storage` metadata and projected packed-size/effective-bpw reports for both raw packed INT2 and INT4-container INT2.
- Reworked `tile_packed_streams` into nested per-tier descriptors with tile indices, tile geometry, region labels, per-tile FP16 scale/zero-point arrays, and simple tile-major packed values.
- Added `layout_contract_version=1` and a payload contract string documenting that future exporters should implement warp-coalesced/swizzled byte order later.

#### New: Outlier extraction for tile-region mode (2026-06-13)

**Problem:** In tile-region mode, precision tier is assigned per tile. Single highest-magnitude weights (outliers) inside low-bit tiles get crushed to the low-bit grid because one outlier doesn't lift its tile's aggregate score. This flattens z_s while keeping bulk weight-cosine deceptively high.

**Solution:** Magnitude-based per-tensor outlier extraction (paper 2311.16442 two-region split). Before bulk quantization, the top-k outliers by |w| are pulled into a separate FP16 sidecar, zeroed in the bulk, then written back exactly into the dense `state_dict` weights that PyTorch eval loads.

**New CLI flags:**
- `--outlier-extraction {none,magnitude}` (default: `none`). Only applies when `--allocation-mode tile-region`.
- `--outlier-threshold-pct FLOAT` (default: `0.005`). Fraction of each tensor's weights to extract (0.005 = top 0.5%).

**New helper functions added:**
- `extract_outliers_by_magnitude(weight, threshold_pct)` — identifies top-k elements by |w|, returns flat indices + exact BF16 values.
- `apply_outlier_zeroing(weight, outlier_info)` — returns `W` with outlier positions set to 0.0.
- `restore_outlier_values(q_weight, outlier_info, original_weight)` — overwrites outlier positions in the dequantized weight with exact original values.

**Acceptance signal — per-tensor log line:**
```
[OUTLIER] {name}: extracted={N} ({pct:.3f}%) threshold_mag={mag:.6f}
[OUTLIER] {name}: orig_absmax={a:.4f} stored_absmax={b:.4f}
```
`orig_absmax` and `stored_absmax` must match closely when extraction is on. If they don't match, extraction didn't work.

**Effective-BPW accounting update:**
- Outlier FP16 values (16 bits) + int32 flat indices (32 bits) per outlier are included in a new `outlier_bpw_overhead` metric.
- New RESULT lines: `outlier_total_count`, `outlier_total_bytes`, `outlier_bpw_overhead`, `effective_bpw_with_outliers`, `estimated_weight_gib_with_outliers`.

**Checkpoint payload additions:**
- `payload["outlier_metadata"]` — per-tensor dict of `{indices, values, count, threshold_pct, threshold_value, orig_absmax, stored_absmax}`.
- `payload["outlier_extraction"]` — the extraction mode string.

#### Confirmed: 32x32 tile shape already fully functional

`parse_tile_shape("32x32")` correctly parses explicit `RxC` (line 797). `resolve_tile_shape_per_tier` applies it to all tiers (line 821). `aggregate_2d_tile_scores` uses generic padding math with no hardcoded 64 — validated for arbitrary tile shapes including 32x32. `expand_2d_tile_tiers` uses `repeat_interleave` + crop, also fully generic. No code changes needed.

#### Confirmed: `--int2-storage packed` is the default and fully working

Default is `"packed"` (line 2018). The packed path uses `_pack_uint2()` for raw 2-bit packing (16 weights per uint32). The int4-container path uses `_pack_uint4()`. Dense `state_dict` weights (what eval loads) are identical regardless of `int2-storage` choice — the choice only affects `tile_packed_streams`. Both paths in `build_tile_packed_streams` are complete.

#### Confirmed: `--tile-aggregate max` handles padding correctly

`aggregate_2d_tile_scores` with `agg == "max"` pads with `float("-inf")` (line 873) and masks via `masked_fill(~valid_tiles, float("-inf")).amax(dim=-1)` (line 905). Padding elements are excluded from the max. No code changes needed.

## Files NOT modified
- Per-element / block-aligned / block-max-tier quantization paths — completely untouched.
- `verify_septq_zs_drift.py` — the eval path loads dense weights which now contain corrected outliers via the state_dict. No changes needed.
- CUDA/C++ runtime (`bmo.cpp`, `bmo_compute.cpp`, `bmo_cuda_kernels*.cu`, `bmo.h`, etc.)
- `export_bmo_gguf.py` (later exporter/kernel session)
- `qat_septq.py`
- Any test scripts
>>>>>>> Stashed changes
