#!/bin/bash
# Build the standalone BMO kernel microbench (no CMake — fast nvcc iteration).
# Links only libggml-base.a (gguf parsing); the kernels live in the bench itself.
set -e
cd "$(dirname "$0")/.."
mkdir -p build/bin
nvcc -O3 -std=c++17 -arch=sm_87 -lineinfo \
  tools/bmo_kernel_bench.cu \
  -I ../ggml/include \
  -o build/bin/bmo_kernel_bench \
  ../ggml/build/src/libggml-base.a \
  -lpthread -lm -ldl
echo "built build/bin/bmo_kernel_bench"
