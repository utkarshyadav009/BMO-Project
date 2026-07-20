#!/bin/bash
# STEP 7 — nsys re-attribution at the 199.2 ms baseline (bit-exact conv
# kernel, GRAPHS=ON build, commit 7ce78cf). Usage: step7_nsys.sh <ctx> <tag>
#   run A: step7_nsys.sh 256 c256
#   run B: step7_nsys.sh 384 c384
#
# DELAY=380 (vs 285 in the config-wave capture) for two reasons:
#   1. tracing must not attach during the memory-tight load window (~275-280s,
#      established OOM mode);
#   2. the -c 384 run's KV window only saturates at frame 384 (~361s wall);
#      capturing earlier mixes context-GROWTH frames into the sample and
#      contaminates the c256-vs-c384 scaling comparison. Same delay for both
#      runs so the windows are like-for-like (both fully saturated).
# nsys terminates the app when the collection window ends (delay+duration).
CTX=$1
TAG=$2
DELAY=${DELAY:-380}
DURATION=${DURATION:-90}
rm -f /tmp/step7_${TAG}_stdout.txt /tmp/step7_${TAG}_stderr.txt \
      /tmp/step7_${TAG}_tegrastats.txt /tmp/step7_${TAG}_profile.*

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

tegrastats --interval 1000 > /tmp/step7_${TAG}_tegrastats.txt 2>&1 &
TEGRA_PID=$!

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  nsys profile -t cuda,osrt -o /tmp/step7_${TAG}_profile -y $DELAY --duration=$DURATION --force-overwrite=true \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c $CTX \
  > /tmp/step7_${TAG}_stdout.txt 2>/tmp/step7_${TAG}_stderr.txt
echo "NSYS_EXIT=$?"

kill "$TEGRA_PID" 2>/dev/null
chmod a+r /tmp/step7_${TAG}_* 2>/dev/null
echo "=== stats export ==="
# Fresh sqlite per tag (stale-cache pitfall: nsys stats silently reuses an old
# export at the same path); force-export on the first call regenerates it.
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output /tmp/step7_${TAG}_kernsum /tmp/step7_${TAG}_profile.nsys-rep
nsys stats --report cuda_api_sum --format csv --output /tmp/step7_${TAG}_apisum /tmp/step7_${TAG}_profile.nsys-rep
nsys stats --report cuda_gpu_trace --format csv --output /tmp/step7_${TAG}_gputrace /tmp/step7_${TAG}_profile.nsys-rep
chmod a+r /tmp/step7_${TAG}_* 2>/dev/null
echo STATS_DONE
