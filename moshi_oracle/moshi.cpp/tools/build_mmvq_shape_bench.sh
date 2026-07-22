#!/bin/bash
# Build the mmvq shape bench (ggml static libs, same pattern as
# build_attn_fattn_check.sh).
set -e
cd "$(dirname "$0")/.."
mkdir -p build/bin
g++ -O2 -std=c++17 \
  tools/mmvq_shape_bench.cpp \
  -I ../ggml/include \
  -o build/bin/mmvq_shape_bench \
  ../ggml/build/src/libggml.a \
  ../ggml/build/src/ggml-cuda/libggml-cuda.a \
  ../ggml/build/src/libggml-cpu.a \
  ../ggml/build/src/libggml-base.a \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lcuda \
  -fopenmp -lpthread -lm -ldl
echo "built build/bin/mmvq_shape_bench"
