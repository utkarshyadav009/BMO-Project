#!/bin/bash
# Build the standalone mimi conv_transpose_1d microbench (no CMake — fast nvcc
# iteration, same pattern as build_bmo_kernel_bench.sh). Fully self-contained:
# shapes are hardcoded (nsys-certified), no ggml linkage needed.
set -e
cd "$(dirname "$0")/.."
mkdir -p build/bin
nvcc -O3 -std=c++17 -arch=sm_87 -lineinfo \
  tools/mimi_conv_bench.cu \
  -o build/bin/mimi_conv_bench \
  -lm
echo "built build/bin/mimi_conv_bench"
