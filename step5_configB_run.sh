#!/bin/bash
# CONFIG WAVE — config B: config A (reuse_buffer default-on) PLUS
# GGML_CUDA_GRAPHS=ON (ggml rebuilt and relinked; binary saved as
# bin/personaplex_graphs_on before ggml was toggled back to OFF for config
# A). Same invocation/seed as the 321.0ms baseline. Report reads the first
# 12 PHASE_TIMING windows (25 frames/window = 300 frames) per task item 3.
# No set -e on purpose: on a crash we still want tegrastats killed, files
# readable, and the exit code printed.
rm -f /tmp/step5b_stdout.txt /tmp/step5b_stderr.txt /tmp/step5b_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step5b_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex_graphs_on -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step5b_stdout.txt 2>/tmp/step5b_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step5b_stdout.txt /tmp/step5b_stderr.txt /tmp/step5b_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
