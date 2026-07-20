#!/bin/bash
# conv_transpose_1d replacement — waveform-parity run (NOT an official timing
# run: current boot session, BENCH_DUMP_PCM set). Standard command/seed so
# tokens are the deterministic serial reference stream; the dumped PCM is the
# decoded waveform for the old-vs-new rel_l2 < 1e-4 gate.
# Usage: step6_wave_run.sh <binary-name> <tag>
BIN=$1
TAG=$2
rm -f /tmp/wave_${TAG}.pcm /tmp/wave_${TAG}_stdout.txt /tmp/wave_${TAG}_stderr.txt

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  BENCH_DUMP_PCM=/tmp/wave_${TAG}.pcm \
  ./bin/$BIN -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/wave_${TAG}_stdout.txt 2>/tmp/wave_${TAG}_stderr.txt
echo "PERSONAPLEX_EXIT_CODE=$?"
chmod a+r /tmp/wave_${TAG}.pcm /tmp/wave_${TAG}_stdout.txt /tmp/wave_${TAG}_stderr.txt 2>/dev/null
echo DONE
