# Task 0 Report: Candidate C GGUF Integration

## 1. Stock moshi.cpp Reference Q4_K Baseline
- **Execution**: Coherent conversational generation output ("Hello, this is a live call from Switchboard...") under seed 1782579450.
- **Generation Speed**: **19.44 FPS** (exceeds real-time target of 12.5 FPS).
- **Runtime VRAM**: **6331 MiB** exact tensor allocation (11,094 MiB global delta due to CUDA context).

## 2. Load failure of qat_heavy_int2.gguf
- **Failure Symptom**: Segmentation fault (exit code 139) during system prompt execution (inside `moshi_lm_start`).
- **Root Cause**: The GGUF file has underscore-separated names (e.g., `transformer_layers_0_norm1_weight`), but `moshi.cpp` model code queries dot-separated names (e.g., `transformer.layers.0.norm1.weight`). As a result, `WeightLoader::get_tensor` returns `NULL`, and `fetch` returns `false`. Because assertions are disabled in Release builds (`-DNDEBUG`), the `NULL` tensor pointer is silently accepted, leading to a segfault during the first forward step execution.

## 3. Format Types and Native Mapping Confirmation
- **Format 3 (Depth Stack Q4_0)**: Maps directly to standard GGUF `GGML_TYPE_Q4_0`. Can be loaded directly with no custom dequant kernels.
- **Format 4 (Embeddings Q4_0)**: Maps directly to standard GGUF `GGML_TYPE_Q4_0`. Can be loaded directly.
- **Format 2 (Attention INT4)**: Stored as custom split tensors (`packed_weights`, `scales`, `zeros`, etc.) for asymmetric group-INT4 (group size 128). Does not map directly to a native GGUF layout on disk, but can be dequantized on the CPU to FP32 and then re-quantized/loaded as a native `Q4_K` or `Q4_1` tensor.
- **Format 1 (FFN gating)**: Custom tile-region multi-tier (64x64 tiles, 70% INT2). Requires a genuinely custom `GGML_TYPE_*` and corresponding dequantization kernels.

## 4. Reuse from WHT-NF2 Work
- Interception of `fetch` for custom type allocation.
- Type registration and name-mapping patterns.
- CUDA dequantization skeleton structure in `convert.cu`.
