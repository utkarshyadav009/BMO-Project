#!/bin/bash
# STEP 4: phase-timing run with the new PHASE_TIMING instrumentation.
# Power mode already confirmed MAXN_SUPER/max-clocks in this session — no
# second config needed. No frame-count CLI flag exists (bench mode always
# runs to 1250); we just read the first 300 frames' worth of PHASE_TIMING
# windows from the log rather than truncating the run.
rm -f /tmp/step4_stdout.txt /tmp/step4_stderr.txt /tmp/step4_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step4_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step4_stdout.txt 2>/tmp/step4_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step4_stdout.txt /tmp/step4_stderr.txt /tmp/step4_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
