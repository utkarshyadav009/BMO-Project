# Local Edits Summary

## Files Modified

### apply_septq_multitier.py
- Added `--quantize-depth-int8` and `--depth-int8-skip-modules` flags (Part 1)
- Added `quantize_depth_int8()` helper function (Part 1)
- Wired depth-INT8 into `main()` after temporal SEPTQ loop (Part 1)
- Added `--allocation-mode`, `--tile-size`, `--tile-aggregate`, `--emit-tile-packed` flags (Part 2)
- Added tile-region global tile-score collection and tile-mask branch in the SEPTQ quantization flow (Part 2)
- Added `tile_region_metadata` plus optional `tile_packed_streams` to saved payload (Part 2)

### qat_septq.py
- Added `freeze_all_params(student)` log before fake-quant registration (Part 1)
- Added `depth_int8_meta` carry-through in QAT save block (Part 1)

## Files NOT modified
- CUDA/C++ runtime (`bmo.cpp`, `bmo_compute.cpp`, `bmo_cuda_kernels*.cu`, `bmo.h`, etc.)
- `export_bmo_gguf.py` (touched in a later session)
- Any test scripts
