#!/bin/bash
# CONFIG WAVE — config A: GraphContext::reuse_buffer default-on for all three
# per-frame scratch contexts (mimi encode, mimi decode, LM gen->ctx), serial
# mode, GGML_CUDA_GRAPHS still OFF. This is just the current HEAD binary
# (commit 87d2396) — reuse_buffer has been the unconditional default since
# commit 52b6de9, it was just never benchmarked in isolation from the
# (reverted) pipelining path. Same invocation/seed as the 321.0ms baseline
# (step3_integration_run.sh) so phase numbers are directly comparable.
# Report reads the first 4 PHASE_TIMING windows (25 frames/window = 100
# frames) per the task's "(1) first, 100-frame check" instruction — full
# bench run still executes to completion, matching established convention
# of never truncating the standard command mid-run.
# No set -e on purpose: on a crash we still want tegrastats killed, files
# readable, and the exit code printed.
rm -f /tmp/step5a_stdout.txt /tmp/step5a_stderr.txt /tmp/step5a_tegrastats.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step5a_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step5a_stdout.txt 2>/tmp/step5a_stderr.txt
PERSONAPLEX_EXIT=$?

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step5a_stdout.txt /tmp/step5a_stderr.txt /tmp/step5a_tegrastats.txt
echo "PERSONAPLEX_EXIT_CODE=$PERSONAPLEX_EXIT"
echo DONE
