# moshi_oracle

This repository contains a C++ and GGML port of Kyutai's Moshi speech-to-speech model, with additional support for NVIDIA's PersonaPlex voice-cloning model. 

This document serves as a guide for other developers and agents to understand the repository structure, build dependencies, run demos, and inspect the quantization details of the models.

---

## Directory Structure

*   **`moshi.cpp/`**: The core C++ project containing the inference engine, library API, and utility binaries.
    *   `src/`: Core implementation files (`moshi.cpp`, `loader.h`, `safetensor.cpp`, etc.).
    *   `src/moshi/`: Model-specific code (gating, convolutions, transformer, seanet, quantizer, etc.).
    *   `include/`: API headers.
    *   `CMakeLists.txt`: CMake build configuration.
*   **`ggml/`**: A git submodule/fork of the GGML library containing backend enhancements for CPU, CUDA, and Vulkan acceleration.
*   **`sentencepiece/`**: A copy of Google's SentencePiece library used for text tokenization.
*   **`models/`**: Holds model files, configs, and cached voices.
    *   `models/personaplex/`: Configuration for PersonaPlex (`personaplex-config.json`), the quantized language model (`model-q4_k.gguf`), and the audio codec (`mimi-e351c8d8-125.gguf`).
*   **`tools/`**: External tools and dependencies such as prebuilt binaries for `SDL2`, `aria2c.exe`, and `ffmpeg`.

---

## Probed Quantization Analysis

We analyzed the quantization of the PersonaPlex models in the `models/personaplex` folder:

### 1. Language Model (`model-q4_k.gguf`)
*   **Total Tensors**: 655
*   **Quantization Scheme**: Mixed precision using `Q4_K` and `Q4_0` for weights, and `F32` for normalization scaling.
*   **Quantized Layers**:
    *   **`Q4_K`** (545 tensors): All main transformer projection weights, gating linear layers, self-attention query/key/value projections, and feedforward output projection layers.
    *   **`Q4_0`** (33 tensors): All embedding layers (text embeddings `emb` and `text_emb`, as well as `depformer_emb` and `depformer_text_emb`).
*   **Untouched (Non-Quantized) Layers**:
    *   **`F32`** (77 tensors): The scaling/multiplier coefficients (`alpha` parameters) of the layer normalizations. Specifically:
        *   `lm.transformer.layers.[0-31].norm1.alpha` and `norm2.alpha` (64 tensors)
        *   `lm.depformer.layers.[0-5].norm1.alpha` and `norm2.alpha` (12 tensors)
        *   `lm.out_norm.alpha` (1 tensor)
    *   *Why?* Layer norm parameters require very high precision for numerical stability and are extremely small (vector size 1024 or 4096), making quantization ineffective for memory savings while introducing high performance degradation.

### 2. Mimi Audio Codec Model (`mimi-e351c8d8-125.gguf`)
*   **Total Tensors**: 254
*   **Quantization Scheme**: **NOT quantized** (0 quantized tensors).
*   **Precision Breakdown**:
    *   **`F16`** (29 tensors): Weight tensors for the 1D convolution layers (`mimi.encoder.model.X.block.Y.conv.conv.weight`, `mimi.decoder.model.X.block.Y.conv.conv.weight`, and projection layers).
    *   **`F32`** (225 tensors): Bias tensors, normalization weights/biases, linear transformer weights, and VQ (vector quantization) codebook embeddings.

---

## How to Build the Repository

### Prerequisites
*   Latest **MSVC Runtimes** (for Windows)
*   **CMake** (version 3.20+ recommended)
*   CUDA Toolkit (if building with CUDA acceleration)

### Build Command (Windows example using CMake and NMake)
```cmd
cd moshi.cpp
mkdir build
cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DGGML_INCLUDE_DIR=../../ggml/include ^
  -DGGML_LIBRARY_DIR=../../ggml/build/src ^
  -DSentencePiece_INCLUDE_DIR=../../sentencepiece/src ^
  -DSentencePiece_LIBRARY_DIR=../../sentencepiece/build/src ^
  -DCMAKE_PREFIX_PATH=../../tools/SDL2-2.30.11 ^
  -DFFmpeg_DIR=../../tools/ffmpeg-master-latest-win64-lgpl-shared
cmake --build .
```

---

## Running PersonaPlex

1.  Navigate to the `build/bin` directory (or wherever binary outputs are generated).
2.  Ensure you have downloaded the default models (placed in `models/personaplex/`).
3.  Run the PersonaPlex CLI tool:
    *   **Real-time voice cloning and custom prompt**:
        ```cmd
        .\personaplex -v voice_sample.wav -p "You are a helpful assistant."
        ```
    *   **Using default preset voice**:
        ```cmd
        .\personaplex -v NATF0
        ```
