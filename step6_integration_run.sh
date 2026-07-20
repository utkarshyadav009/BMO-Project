#!/bin/bash
# conv_transpose_1d replacement — OFFICIAL integration run (STEP 3).
# Preconditions (user-side, this script does not check them): fresh boot,
# hardened preflight PASS (stops competing services), nvpmodel -m 2,
# jetson_clocks. Standard command/seed; BENCH_DUMP_PCM deliberately UNSET —
# the official phase table must come from the unmodified measurement config.
# Phase table read from the first 12 PHASE_TIMING windows (300 frames) vs the
# 320.3 ms config-B baseline; full 1250-frame run executes to completion per
# convention. Token hash must equal the serial reference 0x7b2f7f1c39d47848.
rm -f /tmp/step6_stdout.txt /tmp/step6_stderr.txt /tmp/step6_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step6_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step6_stdout.txt 2>/tmp/step6_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step6_stdout.txt /tmp/step6_stderr.txt /tmp/step6_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
