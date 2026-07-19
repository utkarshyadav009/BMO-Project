#!/bin/bash
# Build the Phase-1 attention-numerics harness (no CMake — links the ggml
# static libs directly, same fast-iteration pattern as the other benches).
set -e
cd "$(dirname "$0")/.."
mkdir -p build/bin
g++ -O2 -std=c++17 \
  tools/attn_fattn_check.cpp \
  -I ../ggml/include \
  -o build/bin/attn_fattn_check \
  ../ggml/build/src/libggml.a \
  ../ggml/build/src/ggml-cuda/libggml-cuda.a \
  ../ggml/build/src/libggml-cpu.a \
  ../ggml/build/src/libggml-base.a \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lcuda \
  -fopenmp -lpthread -lm -ldl
echo "built build/bin/attn_fattn_check"
