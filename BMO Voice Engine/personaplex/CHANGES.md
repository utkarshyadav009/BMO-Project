# Local Edits Summary

## Files Modified

### qat_septq.py

#### New: Outlier-Aware QAT Support (2026-06-13)

**Problem:** QAT fake-quantization re-quantizes weights every forward pass based on a per-element tier mask. Magnitude outliers (which PTQ protected at FP16 precision) would normally be crushed back to low-bit grids in QAT if their mask positions carry low-bit tiers. This destroys the absmax preservation.

**Solution:** Force outlier positions to tier 0 (FP16) in the per-element mask before QAT fake-quantization registration. This ensures the fake-quant leaves them at full precision.

**New CLI flags:**
- `--outlier-meta PATH` (default: `"auto"`): Path to student checkpoint or `'auto'` to read it from `--student-quant-meta`. Set to `'none'` to disable outlier remapping.
- `--freeze-outliers` / `--no-freeze-outliers` (default: `True`): Force outlier positions to tier 0 in QAT.

**New helper function added:**
- `force_outlier_tier0_in_masks(tier_masks_uint2, tier_masks_meta, outlier_meta)`: Unpacks the uint2 masks, sets outlier indices to tier 0 (FP16), repacks, and prints the total number of forced positions.

**Verification log output:**
- Prints at initialization: `[INFO] forced X outlier positions to tier0(FP16) across Y tensors`.

**Sanity check reference (PTQ statistics):**
- Total model-wide outliers: `29,255,258`.
- Model-wide forced count expectation: `~24M - 25M` remapped positions (approx. 84% of outliers).
- Representative single tensor expectation: `~250k` total outliers, of which `~210k` are forced to tier 0.
- A forced count of `0` or an implausibly small value will trigger a QAT validation failure.

**Metadata Pass-Through on Checkpoint Save:**
- Inside `save_qat_checkpoint()`, all core PTQ layout/outlier metadata keys (`depth_int8_meta`, `outlier_metadata`, `tile_region_metadata`, `allocation_mode`, `mask_representation`, `outlier_extraction`) are copied from the original student quantized checkpoint and preserved in the saved QAT checkpoint payload.

---

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

#### Outlier extraction for tile-region mode (2026-06-13)
- Magnitude-based per-tensor outlier extraction (paper 2311.16442 two-region split). Before bulk quantization, the top-k outliers by |w| are pulled into a separate FP16 sidecar, zeroed in the bulk, then written back exactly into the dense `state_dict` weights that PyTorch eval loads.
- Added CLI flags: `--outlier-extraction {none,magnitude}` (default: `none`) and `--outlier-threshold-pct FLOAT` (default: `0.005`).
- Added helper functions: `extract_outliers_by_magnitude()`, `apply_outlier_zeroing()`, and `restore_outlier_values()`.
- Updated effective-bpw accounting and checkpoint payload saving.

---

## Mask Packing Convention Verification

As part of the correctness requirements, the uint2 packed mask layout was explicitly cross-checked and verified between the packing, unpacking, and remapping code paths:

1. **Packing File & Implementation**:
   - Verified in [apply_septq_multitier.py](file:///c:/Users/raouy/OneDrive/Documents/GitHub/BMO%20Project/BMO-Project/BMO%20Voice%20Engine/personaplex/apply_septq_multitier.py) under `pack_tier_mask_uint2` (lines 1259-1283):
     ```python
     packed = (
         flat[0::4]
         | (flat[1::4] << 2)
         | (flat[2::4] << 4)
         | (flat[3::4] << 6)
     ).to(dtype=torch.uint8)
     ```

2. **Unpacking Files & Implementations**:
   - Verified in [compare_fakequant_vs_gguf_weights.py](file:///c:/Users/raouy/OneDrive/Documents/GitHub/BMO%20Project/BMO-Project/BMO%20Voice%20Engine/personaplex/compare_fakequant_vs_gguf_weights.py) under `_tier_per_block_numpy` (lines 156-162):
     ```python
     shifts = np.tile(np.array([0, 2, 4, 6], dtype=np.uint8), n_mask_bytes)[:n_blocks]
     return ((pm_rep >> shifts) & np.uint8(0x3)).astype(np.int64)
     ```
   - Verified in [qat_septq.py](file:///c:/Users/raouy/OneDrive/Documents/GitHub/BMO%20Project/BMO-Project/BMO%20Voice%20Engine/personaplex/qat_septq.py) under `unpack_tier_mask_uint2` (lines 183-190):
     ```python
     for i in range(4):
         expanded[i::4] = (packed >> (i * 2)) & 0b11
     ```

**Confirmed Convention:**
The mask uses a **little-endian (LE)** bitwise packing ordering where 4 tiers (values 0-3) are packed per byte:
- `tier = (byte >> (2 * lane)) & 0x3`
- lane 0 = lowest 2 bits (bits 0-1)
- lane 1 = bits 2-3
- lane 2 = bits 4-5
- lane 3 = highest 2 bits (bits 6-7)
- byte index = block_idx // 4
- lane index = block_idx % 4

The newly implemented helper `force_outlier_tier0_in_masks()` uses this exact convention to unpack, zero-out outlier positions, and repack correctly.
