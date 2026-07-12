#!/bin/bash
# STEP 3 official integration run: rewritten BMO GEMV kernels (band-major
# payload, fused outliers, shape dispatch). Same invocation as the 648.7ms
# baseline run (step4_phase_timing.sh). Bench mode runs 1250 frames; the
# 300-frame report reads the first 12 PHASE_TIMING windows from the log.
# No set -e on purpose: on a crash we still want tegrastats killed, files
# readable, and the exit code printed.
rm -f /tmp/step3i_stdout.txt /tmp/step3i_stderr.txt /tmp/step3i_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step3i_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step3i_stdout.txt 2>/tmp/step3i_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step3i_stdout.txt /tmp/step3i_stderr.txt /tmp/step3i_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
