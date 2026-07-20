#!/bin/bash
# Informal (non-official, current boot, no preflight/clock-lock) smoke run
# for the mimi encode/decode pipelining change. Runs pipelined mode then
# serial mode back-to-back (sequentially, same GPU, to avoid clock/thermal
# cross-contamination) and prints both TOKEN_HASH lines for comparison.
# This is NOT the official validation run — that requires fresh boot +
# preflight + nvpmodel -m 2 + jetson_clocks per protocol.
set -e
cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

echo "=== PIPELINED (BMO_PIPELINE=1, default) ==="
GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/mimi_smoke_pipelined_stdout.txt 2>/tmp/mimi_smoke_pipelined_stderr.txt
echo "PIPELINED_EXIT=$?"

echo "=== SERIAL (BMO_PIPELINE=0) ==="
BMO_PIPELINE=0 GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/mimi_smoke_serial_stdout.txt 2>/tmp/mimi_smoke_serial_stderr.txt
echo "SERIAL_EXIT=$?"

echo "--- hashes ---"
grep TOKEN_HASH /tmp/mimi_smoke_pipelined_stdout.txt
grep TOKEN_HASH /tmp/mimi_smoke_serial_stdout.txt
echo "--- underrun report (pipelined only) ---"
grep -E "PIPELINE_STATS|QUEUE_UNDERRUNS" /tmp/mimi_smoke_pipelined_stdout.txt
echo "--- cuda errors (either file) ---"
grep -iE "cuda error|misaligned|trap|abort|segfault" /tmp/mimi_smoke_pipelined_stdout.txt /tmp/mimi_smoke_pipelined_stderr.txt /tmp/mimi_smoke_serial_stdout.txt /tmp/mimi_smoke_serial_stderr.txt || echo "none found"
echo SMOKE_DONE
