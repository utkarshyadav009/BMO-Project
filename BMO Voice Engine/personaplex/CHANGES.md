# Local Edits Summary

## Files Modified

### apply_septq_multitier.py
- Added `--tile-shape`, `--tile-layout-target`, and `--int2-storage` flags for hardware-aware tile-region export preparation.
- Replaced 128x1 row-stripe tile aggregation/expansion with padded 2D tile geometry; default `--tile-shape auto --tile-layout-target ampere` resolves to 64x64 tiles.
- Stored authoritative per-tile tiers in `tile_region_metadata.tiles[name].tile_tiers`, plus `mask_representation="per-tile"` and `tier_masks_authoritative=False` for tile-region mode.
- Preserved dense dequantized weights and retained `tier_masks_uint2` as a non-authoritative debug/QAT compatibility artifact in tile-region mode.
- Added tile geometry metadata: `tile_shape_per_tier`, `layout_target`, per-tensor `tile_grid`, `pad_rows`, and `pad_cols`.
- Added `int2_storage` metadata and projected packed-size/effective-bpw reports for both raw packed INT2 and INT4-container INT2.
- Reworked `tile_packed_streams` into nested per-tier descriptors with tile indices, tile geometry, region labels, per-tile FP16 scale/zero-point arrays, and simple tile-major packed values.
- Added `layout_contract_version=1` and a payload contract string documenting that future exporters should implement warp-coalesced/swizzled byte order later.

## Files NOT modified
- CUDA/C++ runtime (`bmo.cpp`, `bmo_compute.cpp`, `bmo_cuda_kernels*.cu`, `bmo.h`, etc.)
- `export_bmo_gguf.py` (later exporter/kernel session)
- `qat_septq.py`
- Any test scripts
