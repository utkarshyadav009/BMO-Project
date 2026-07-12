#!/bin/bash
# STEP 3: nsys profile capturing CUDA + OS runtime API for load + ~50 frames
# of decode. Corrected duration: STEP 4 (no nsys) showed load alone takes
# ~273s wall time before decode starts. Budget 273s load + ~110s decode
# (~70 frames at 1.54fps, comfortably over the 50-frame target) = 400s,
# rounded up for nsys tracing overhead.
rm -f /tmp/step3_stdout.txt /tmp/step3_stderr.txt /tmp/step3_profile.nsys-rep

cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build

GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  nsys profile -t cuda,osrt -o /tmp/step3_profile --duration=500 --force-overwrite=true \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 \
  > /tmp/step3_stdout.txt 2>/tmp/step3_stderr.txt

chmod a+r /tmp/step3_stdout.txt /tmp/step3_stderr.txt /tmp/step3_profile.nsys-rep 2>/dev/null
echo DONE

echo "=== generating kernel summary stats ==="
nsys stats --report cuda_gpu_kern_sum --format csv --output /tmp/step3_kernsum /tmp/step3_profile.nsys-rep
nsys stats --report cuda_api_sum --format csv --output /tmp/step3_apisum /tmp/step3_profile.nsys-rep
chmod a+r /tmp/step3_kernsum* /tmp/step3_apisum* 2>/dev/null
echo STATS_DONE
