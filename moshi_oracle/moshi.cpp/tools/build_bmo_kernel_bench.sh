#!/bin/bash
# Build the standalone BMO kernel microbench (no CMake — fast nvcc iteration).
# Links only libggml-base.a (gguf parsing); the kernels live in the bench itself.
#
# Single nvcc invocation, two -gencode targets: sm_90 (H100, this dev box —
# runnable here) and sm_87 (Jetson Orin Nano target — cross-compile sanity
# only, cannot be run on this machine). Compiling both from one pass means
# any device-code issue specific to either arch surfaces here. The resulting
# binary is a single fat executable; on H100 it runs the native sm_90 cubin.
set -e
cd "$(dirname "$0")/.."
mkdir -p build/bin
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_87,code=sm_87 \
  -gencode arch=compute_90,code=sm_90 \
  tools/bmo_kernel_bench.cu \
  -I ../ggml/include \
  -o build/bin/bmo_kernel_bench \
  ../ggml/build/src/libggml-base.a \
  -lpthread -lm -ldl
echo "built build/bin/bmo_kernel_bench (sm_87 + sm_90)"
