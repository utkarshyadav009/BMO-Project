#!/bin/bash
# OFFICIAL mimi-pipelining validation run. Requires: fresh boot, preflight
# PASS already logged, nvpmodel -m 2 && jetson_clocks already run (in that
# order) by the user in their own terminal. Same invocation/seed as the
# STEP 3 baseline (step3_integration_run.sh) and the 648.7ms/321.0ms prior
# runs, so phase numbers are directly comparable. BMO_PIPELINE defaults to 1
# (pipelined) in the binary — no env var needed for the official run.
# No set -e on purpose: on a crash we still want tegrastats killed, files
# readable, and the exit code printed.
rm -f /tmp/step4p_stdout.txt /tmp/step4p_stderr.txt /tmp/step4p_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step4p_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step4p_stdout.txt 2>/tmp/step4p_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step4p_stdout.txt /tmp/step4p_stderr.txt /tmp/step4p_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
