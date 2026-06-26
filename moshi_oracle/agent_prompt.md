# Instructions for the Windows Laptop Agent (RTX 2060 Optimization)

You are resuming a pair-programming task to run/verify KV-Cache Quantization in `moshi.cpp` on a laptop with an **NVIDIA RTX 2060 (6GB VRAM, Turing architecture, CUDA compute capability 7.5)**.

---

## 1. Goal & VRAM Constraint
The target machine has a strict VRAM limit of **6.0 GB (approx. 5.5 GB usable)**. 
- The baseline FP16 model uses ~7.2 GB of VRAM, which causes Out-Of-Memory (OOM) or massive paging slowdowns on the RTX 2060.
- By utilizing the **`Q4_0` KV cache quantization** and tuning the **context length** via `-c`, we can fit the model entirely in VRAM and run it at full speed (15+ FPS).

---

## 2. Your Tasks

### Step 1: Pull the latest changes
Ensure you are on the `experiment/kv-quant` branch and pull the latest commits which contain:
1. The CUDA copy kernel serialization fix in `cpy.cu`.
2. The dynamic two-step cast fix in `transformer.h`.
3. The exact tensor VRAM allocation tracking API in `personaplex.cpp`/`moshi-sts.cpp`.

```bash
git checkout experiment/kv-quant
git pull origin experiment/kv-quant
```

### Step 2: Clean & Recompile for Turing (RTX 2060)
Clean any previous CMakeCache files and compile for the Turing architecture (`-DCMAKE_CUDA_ARCHITECTURES=75`).

You can use the following Windows batch script:
```batch
@echo off
set BASE_DIR=%CD%

echo --- PURGING OLD CACHES ---
if exist "%BASE_DIR%\ggml\build\CMakeCache.txt" del /f /q "%BASE_DIR%\ggml\build\CMakeCache.txt"
if exist "%BASE_DIR%\moshi.cpp\build\CMakeCache.txt" del /f /q "%BASE_DIR%\moshi.cpp\build\CMakeCache.txt"

echo --- PHASE 1: GGML (Turing arch 75) ---
cd /d "%BASE_DIR%\ggml\build"
cmake .. -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DCMAKE_C_FLAGS="/Zc:preprocessor" ^
    -DCMAKE_CXX_FLAGS="/Zc:preprocessor" ^
    -DCMAKE_CUDA_FLAGS="-Xcompiler=\"/Zc:preprocessor\""
cmake --build . --config Release -j 6

echo --- PHASE 2: Moshi.cpp ---
cd /d "%BASE_DIR%\moshi.cpp\build"
set GGML_BUILD=%BASE_DIR%\ggml\build
set SPM_BUILD=%BASE_DIR%\sentencepiece\build
set SDL2_DIR=%BASE_DIR%\tools\SDL2-2.30.11
set FFMPEG_DIR=%BASE_DIR%\tools\ffmpeg-master-latest-win64-lgpl-shared
set CUDA_LIB_DIR=C:\CUDA_LIBS

set LINKER_FLAGS=/WHOLEARCHIVE:"%GGML_BUILD%\src\ggml-cpu.lib" /WHOLEARCHIVE:"%GGML_BUILD%\src\ggml-cuda\ggml-cuda.lib" %CUDA_LIB_DIR%\cudart_static.lib %CUDA_LIB_DIR%\cublas.lib %CUDA_LIB_DIR%\cublasLt.lib %CUDA_LIB_DIR%\cuda.lib

cmake .. -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY ^
    -DGGML_LIBRARY="%GGML_BUILD%\src\ggml.lib" ^
    -DGGML_BASE_LIBRARY="%GGML_BUILD%\src\ggml-base.lib" ^
    -DGGML_CPU_LIBRARY="%GGML_BUILD%\src\ggml-cpu.lib" ^
    -DGGML_CUDA_LIBRARY="%GGML_BUILD%\src\ggml-cuda\ggml-cuda.lib" ^
    -DSentencePiece_LIBRARY="%SPM_BUILD%\src\sentencepiece.lib" ^
    -DGGML_INCLUDE_DIR="%BASE_DIR%\ggml\include" ^
    -DSentencePiece_INCLUDE_DIR="%BASE_DIR%\sentencepiece\src" ^
    -DCMAKE_PREFIX_PATH="%SDL2_DIR%" ^
    -DFFmpeg_DIR="%FFMPEG_DIR%" ^
    -DCMAKE_EXE_LINKER_FLAGS="%LINKER_FLAGS%"
cmake --build . --config Release -j 6
echo --- BUILD COMPLETE ---
```

### Step 3: Run with Q4_0 and Context Tuning
On the H100 with context length `3000`, the exact VRAM footprint for `q4_0` is **5,253 MiB** (5.13 GB).
To ensure it stays safely under the usable 5.5 GB VRAM limit of the RTX 2060 on Windows (accounting for the Windows display driver base allocation), run the tool with:
- **`-k q4_0`**: Enable 4-bit KV cache.
- **`-c <length>`**: Constrain the context length (e.g., `-c 1500` or `-c 2000`). Lowering the context length linearly drops the KV-cache VRAM footprint.

Run Command (interactive mode with SDL audio on Windows):
```cmd
moshi.cpp\build\bin\personaplex.exe -m models/personaplex/ -k q4_0 -c 1500 -v NATF0
```

Verify that the console prints:
1. `exact tensor VRAM allocation` is below 5.5 GB.
2. Generating speed (FPS) is smooth and speech remains coherent.
