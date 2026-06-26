# Progress Report: KV-Cache Quantization & WHT Rotation in moshi.cpp

This document outlines the current progress, completed tasks, verification details, and design decisions for KV cache quantization and the TurboQuant Walsh-Hadamard Transform (WHT) trick.

---

## 1. Completed Tasks

*   **KV-Cache Configuration & Propagation**:
    *   Added `kv_cache_type` to `moshi_config_t` (in `moshi.h`).
    *   Initialized default type to `GGML_TYPE_BF16` (in `config.h`).
    *   Added command line argument `-k` / `--kv-type` supporting values `bf16`, `f16`, `q8_0`, `q4_0`, `q2_k` (in `personaplex.cpp`).
    *   Added generic `fill_quant` allocator to `StateContext` (in `context.h`) to allocate quantized state buffers.
    *   Propagated the KV cache type through `moshi_lm_t`, `moshi_lm_start` down to the `StateContext`.
*   **TurboQuant Walsh-Hadamard Transform (WHT) Trick**:
    *   Added normalized Sylvester Hadamard matrices of sizes 64 and 128 to `StateContext` (in `context.h`).
    *   Implemented head dimension double-pointer mapping (`hadamard64` and `hadamard128`) in `moshi_smha_state_t` (in `transformer.h`) to support dynamic resolution at runtime and prevent matrix dimension collision between Depformer (head dim 64) and LM (head dim 128).
    *   Applied WHT rotation to Query ($Q$), Key ($K$), and Value ($V$) tensors before insertion into the KV cache, and applied the corresponding inverse WHT rotation to the final attention output ($x$) inside `moshi_streaming_multihead_attention` (in `transformer.h`).
*   **GPU Read-Path Dequantization & Copy Support**:
    *   Implemented GPU-side dequantization block copy kernels in `cpy.cu` for `GGML_TYPE_Q8_0`, `GGML_TYPE_Q4_0`, and `GGML_TYPE_Q2_K` to map float/query activation formats directly and avoid CPU fallbacks.
*   **Block-Size Compatibility & 2-bit Quantification Fallback**:
    *   Added block size check: since standard K-quants in GGML (like `q2_k`) have a block size of 256, but Moshi head dimensions are 64/128, native `GGML_TYPE_Q2_K` cannot be allocated directly.
    *   Designed the kv-type CLI flag and the cast path in `transformer.h` to accept future 2-bit / custom-2bit formats. If the block size is incompatible with the head dimension, the system prints a warning and falls back to a `q4_0` placeholder, avoiding crashes while preparing all cast logic to be fully localized for future custom formats.

---

## 2. Verification & Benchmarking

Benchmarks were executed using `personaplex` in benchmark-only mode (`-b`) over 125 frames (equivalent to 10 seconds of interaction) on an **NVIDIA H100 PCIe (80GB, CC 9.0)**:

| Cache Type | Device Memory Delta | Exact Tensor VRAM | Tensor VRAM Savings | Generation Speed | Output Coherence | Output Transcription |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **BF16** (Baseline) | 11,094 MiB | **6,331 MiB** | *Reference* | **19.78 FPS** | Coherent | "Hello, this is Alexander." |
| **Q8_0** (with WHT) | 8,896 MiB | **5,630 MiB** | **701 MiB (11.1%)** | **17.09 FPS** | Coherent | "Hello, this is Kisha." |
| **Q4_0** (with WHT) | 8,522 MiB | **5,255 MiB** | **1,076 MiB (17.0%)** | **17.59 FPS** | Coherent | "Hello, this is Maya." |
| **Q2_K** (Fallback) | 8,522 MiB | **5,255 MiB** | **1,076 MiB (17.0%)** | **17.53 FPS** | Coherent | "Hello, welcome to the podcast! This is Liza..." |

*   **Safety Fallback**: Passing `-k q2_k` outputs a warning: `warning: KV cache type q2_K has block size 256, which is incompatible with head dimension 128. Falling back to q4_0 placeholder.` and completes generation successfully with fully coherent speech.
