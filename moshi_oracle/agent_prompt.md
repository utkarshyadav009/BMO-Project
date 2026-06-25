# Instructions for the next Agent

You are resuming a pair-programming task to implement KV-Cache Quantization (TurboQuant-style) in `moshi.cpp`.

## Context & Current Goal
The goal is to quantize the temporal KV cache of the PersonaPlex model to `Q8_0` or `Q4_0` to reduce runtime VRAM from the 7.26 GB baseline on RTX 4070 Ti SUPER, while keeping speech generation coherent (avoiding gibberish).

We are working on the git branch: `experiment/kv-quant`.

We have already completed Phase 1 implementation (allocation, argument parsing, write path F32 cast, and read path query type dynamic cast). However, the program crashes during the very first forward step (when processing system prompts).

## Your Tasks

1.  **Read the Progress Report**:
    Read the file [progress.md](file:///d:/LocalWorkDir/u521785/BMO-Project/moshi_oracle/progress.md) to understand the changes made and the details of the crash site.
2.  **Debug the Crash in KV Cache Attention**:
    The crash happens inside `moshi_lmgen_step_system_prompts` -> `moshi_lmgen_step_audio_silence` -> `moshi_lmgen_step` (which executes the forward pass).
    Specifically, investigate `moshi_kv_cache_insert_kv` and `moshi_streaming_multihead_attention` in [transformer.h](file:///d:/LocalWorkDir/u521785/BMO-Project/moshi_oracle/moshi.cpp/src/moshi/modules/transformer.h).
    *   Check if casting the quantized cache tensors `k` and `v` back to `q->type` via `ggml_cast` causes an Access Violation on CUDA.
    *   Verify if `ggml_set_rows` destination tensor size/shape is fully compatible with on-the-fly quantization of float inputs to `Q8_0` or `Q4_0`.
    *   Build the project using the MSVC environment:
        ```cmd
        cmd /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" && cd moshi.cpp\build && ninja'
        ```
    *   Use print statements or run with a debugger to trace the exact line.
3.  **Run Benchmarks & Verify Coherence**:
    *   Once the crash is fixed, run the benchmark for the quantized types:
        ```cmd
        conda run --prefix C:\Users\u521785\AppData\Local\miniconda3\envs\cuda_env moshi.cpp\build\bin\personaplex.exe -m models/personaplex/ -k q8_0 -b
        ```
    *   Verify that it runs without errors and produces coherent output (no gibberish).
    *   Fill out the benchmarking table (VRAM, FPS, Coherence) for `bf16`, `q8_0`, `q4_0`.
4.  **Implement Walsh-Hadamard Transform (WHT) Rotation (Task 2)**:
    If `q4_0` suffers from coherence loss (gibberish/static), implement a Sylvester construction Hadamard matrix $H_{128}$ in `StateContext` and rotate Q and K tensors before insertion/attention as detailed in `implementation_plan.md`.
