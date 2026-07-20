#!/bin/bash
# CONFIG WAVE — nsys capture on config B (reuse_buffer + GGML_CUDA_GRAPHS=ON),
# ~50 frames of decode, for t_temporal attribution beyond the gating GEMVs
# (top 10 kernels + gap/launch share). Same duration-budget approach as
# step3_nsys.sh: load takes ~270s wall time before decode starts on this
# hardware; budget load + enough decode time to comfortably clear 50 frames,
# rounded up for nsys tracing overhead. If config B's fps differs materially
# from STEP 3's ~1.54fps, adjust DURATION before running.
# Two prior attempts OOM-crashed during model load (layer 25-28, error 12)
# even with the extra background services stopped -- nsys's own collection
# overhead was enough to tip an already razor-thin load window (~200-400MiB
# free at that point) over the edge, on BOTH attempts, while the un-traced
# config A/B runs completed cleanly. Fix: delay collection start until well
# after load finishes (~275-280s observed) so nsys isn't buffering/tracing
# during the tight window at all -- only attach once decode is underway and
# memory has stabilized (~5200+ MiB driver delta by frame 25 in both prior
# runs).
DELAY=${DELAY:-285}
DURATION=${DURATION:-90}
rm -f /tmp/step5_stdout.txt /tmp/step5_stderr.txt /tmp/step5_profile.nsys-rep

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  nsys profile -t cuda,osrt -o /tmp/step5_profile -y $DELAY --duration=$DURATION --force-overwrite=true \
  ./bin/personaplex_graphs_on -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step5_stdout.txt 2>/tmp/step5_stderr.txt

chmod a+r /tmp/step5_stdout.txt /tmp/step5_stderr.txt /tmp/step5_profile.nsys-rep 2>/dev/null
echo DONE

echo "=== generating kernel summary stats ==="
# cuda_gpu_kern_sum: per-kernel-name totals for top-10 attribution.
# cuda_api_sum: host-side launch/sync API time (proxy for launch overhead).
# cuda_gpu_trace: full per-call timeline, used to compute gap = wall time
# spanned by the capture minus GPU-busy time (no built-in "gaps" report on
# this nsys version, 2024.5.4).
nsys stats --report cuda_gpu_kern_sum --format csv --output /tmp/step5_kernsum /tmp/step5_profile.nsys-rep
nsys stats --report cuda_api_sum --format csv --output /tmp/step5_apisum /tmp/step5_profile.nsys-rep
nsys stats --report cuda_gpu_trace --format csv --output /tmp/step5_gputrace /tmp/step5_profile.nsys-rep
chmod a+r /tmp/step5_kernsum* /tmp/step5_apisum* /tmp/step5_gputrace* 2>/dev/null
echo STATS_DONE
