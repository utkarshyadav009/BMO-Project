#!/bin/bash
# Deliberately no `set -e` — personaplex is expected to possibly exit nonzero
# (OOM/crash is one of the outcomes this run is testing for), and the
# tegrastats kill + chmod + DONE marker must still happen in that case.
rm -f /tmp/run2_stdout.txt /tmp/run2_stderr.txt /tmp/run2_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/run2_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/run2_stdout.txt 2>/tmp/run2_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/run2_stdout.txt /tmp/run2_stderr.txt /tmp/run2_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
