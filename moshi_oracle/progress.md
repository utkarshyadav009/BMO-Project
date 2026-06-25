# Progress Report: KV-Cache Quantization in moshi.cpp

This document outlines the current progress, bug fixes, diagnostic details, and next steps to resume this task on a different machine.

---

## 1. Completed Tasks

*   **KV-Cache Configuration & Propagation**:
    *   Added `kv_cache_type` to `moshi_config_t` (in `moshi.h`).
    *   Initialized default type to `GGML_TYPE_BF16` (in `config.h`).
    *   Added command line argument `-k` / `--kv-type` supporting values `bf16`, `f16`, `q8_0`, `q4_0` (in `personaplex.cpp`).
    *   Added generic `fill_quant` allocator to `StateContext` (in `context.h`) to allocate quantized state buffers.
    *   Propagated the KV cache type through `moshi_lm_t`, `moshi_lm_start` down to the `StateContext`.
*   **Attention Read/Write Changes**:
    *   Updated `moshi_kv_cache_state` to allocate `keys` and `values` using the configured `kv_cache_type` (in `transformer.h`).
    *   Updated `moshi_kv_cache_insert_kv` to cast key and value input tensors to `GGML_TYPE_F32` before insertion via `ggml_set_rows`, enabling on-the-fly quantization in CUDA kernels.
    *   Updated `moshi_streaming_multihead_attention` to cast/dequantize cached `k` and `v` tensors dynamically back to the query activation type (`q->type`) before attention dot product computation.
*   **Windows Path Resolution Bugfix**:
    *   Discovered a Windows limitation in `check_arg_path` (in `common_utils.h`) where folders ending with trailing slashes `/` or `\` caused `_stat64` to fail (reporting that the directory does not exist).
    *   Fixed this by stripping trailing slashes in `check_arg_path` before querying the filesystem.

---

## 2. Diagnostic Summary & Crash Site Discovery

To debug why the executable exited immediately with code `1` (and empty `stderr` in `conda run`), we instrumented `init_ggml` and the model loading/initialization pipeline with detailed `DEBUG` prints and `fflush(stdout)`.

### Successful Initialization Path:
*   CUDA initializes successfully on device 0: `NVIDIA GeForce RTX 4070 Ti SUPER`.
*   Model config `personaplex-config.json` is located and read successfully.
*   All file paths (moshi model `.gguf`, Mimi codec `.gguf`, tokenizer `.model`) are verified and exist.
*   `moshi_alloc` successfully sets up context and CPU/GPU scratch spaces.
*   `moshi_lm_from_files` loads model weights successfully.
*   Generator, tokenizer, and Mimi codec are allocated successfully.
*   `moshi_lm_load` successfully loads the weights.
*   Encoder and decoder contexts are allocated successfully.

### The Crash Site:
The execution enters `moshi_lm_start` and prints:
```
DEBUG: calling moshi_lm_start
DEBUG: moshi_lm_start starting
DEBUG: creating StateContext, backend=000001F46B44E040
DEBUG: state_ctx created
DEBUG: kv_cache_type set to 30
DEBUG: calling moshi_lmmodel_states (default path)
DEBUG: moshi_lmmodel_states returned successfully
DEBUG: calling state_ctx->alloc()
DEBUG: calling state_ctx->init()
DEBUG: calling init (scratch)
DEBUG: creating gen->ctx ScratchContext
DEBUG: ScratchContext created
DEBUG: processing system prompts
```
It immediately crashes or exits within **`moshi_lmgen_step_system_prompts`** before printing the success message.
Specifically:
1. `moshi_lmgen_step_system_prompts` (in `lm.h`) calls `moshi_lmgen_step_audio_silence`.
2. `moshi_lmgen_step_audio_silence` calls `moshi_lmgen_step`.
3. `moshi_lmgen_step` executes the first model forward pass.
4. During this forward pass, our modified self-attention / KV cache quantization logic in `transformer.h` is executed for the first time.
5. The crash happens here, pointing to a memory stride, shape mismatch, or invalid cast during dynamic dequantization on CUDA.

---

## 3. Next Steps to Resume & Resolve

1.  **Revert Debug Instrumentation** (Optional):
    *   Remove/clean up the `printf("DEBUG: ...")` statements in `common_ggml.h`, `moshi.cpp`, and `personaplex.cpp` if desired, or keep them for further tracing.
2.  **Debug Attention/KV Cache logic** in `transformer.h`:
    *   Examine `moshi_kv_cache_insert_kv` and `moshi_streaming_multihead_attention` (around lines 247 and 570 in `src/moshi/modules/transformer.h`).
    *   Check if casting the quantized cache tensors `k` and `v` back to `q->type` via `ggml_cast` causes an Access Violation on CUDA.
    *   Check if `ggml_set_rows` destination tensor size/shape is fully compatible with on-the-fly quantization of float inputs to `Q8_0` or `Q4_0`.
    *   Run the command with a C++ debugger (`gdb`, `lldb`, or VS Debugger) to catch the crash location and view the stack trace.
3.  **Benchmark and Verify**:
    *   Once the crash is fixed, run the benchmark:
        ```cmd
        conda run --prefix C:\Users\u521785\AppData\Local\miniconda3\envs\cuda_env moshi.cpp\build\bin\personaplex.exe -m models/personaplex/ -k q8_0 -b
        ```
    *   Ensure speech generation is coherent and doesn't degrade into gibberish.
    *   Record VRAM usage, inference FPS, and verify coherence for:
        *   `bf16` (baseline)
        *   `q8_0`
        *   `q4_0`
4.  **Optional Task 2 (WHT Rotation)**:
    *   If `q4_0` suffers from coherence issues due to channel outliers, implement Walsh-Hadamard Transform (WHT) rotation on Q and K before insertion/attention as described in `implementation_plan.md`.
