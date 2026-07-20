#!/bin/bash
# fattn consumer-1 OFFICIAL gates (Phase 2). Usage: step8_official_run.sh <ctx> <tag>
#   run 1: 256 c256   (phase table vs 200.5-saturated baseline)
#   run 2: 512 c512   (slope pair point — both fattn-native; 384 % 256 != 0
#                      would silently fall back to the old chain)
# Preconditions done by owner: fresh boot, hardened preflight PASS,
# nvpmodel -m 2, jetson_clocks. Standard command/seed; BENCH_DUMP_PCM unset.
CTX=$1
TAG=$2
rm -f /tmp/step8_${TAG}_stdout.txt /tmp/step8_${TAG}_stderr.txt /tmp/step8_${TAG}_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step8_${TAG}_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c $CTX \
  > /tmp/step8_${TAG}_stdout.txt 2>/tmp/step8_${TAG}_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step8_${TAG}_* 2>/dev/null
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
