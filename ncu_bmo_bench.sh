#!/bin/bash
# Round 2: profile v10 (final candidate) with Nsight Compute.
# Profiles the FULL v10 kernel on linear_in. Needs root.
# Output: /tmp/ncu_bmo_v10.txt (world-readable).
cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp

SECTIONS="--section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis --section Occupancy"

{
  echo "================ PROFILE: v10 full (linear_in) ================"
  BMO_BENCH_ONLY=v10_tilepar \
  /usr/local/cuda/bin/ncu -k v10_gemv_kernel --launch-skip 5 --launch-count 2 $SECTIONS \
    ./build/bin/bmo_kernel_bench ../models/qat_heavy_int2_dir/qat_heavy_int2.gguf 2>&1
} > /tmp/ncu_bmo_v10.txt
chmod a+r /tmp/ncu_bmo_v10.txt
echo NCU_DONE
grep -c "Section:" /tmp/ncu_bmo_v10.txt
