#!/bin/bash
rm -f /tmp/step3_profile.sqlite /tmp/step3_kernsum_cuda_gpu_kern_sum.csv /tmp/step3_apisum_cuda_api_sum.csv
nsys stats --report cuda_gpu_kern_sum --format csv --output /tmp/step3_kernsum /tmp/step3_profile.nsys-rep
nsys stats --report cuda_api_sum --format csv --output /tmp/step3_apisum /tmp/step3_profile.nsys-rep
chmod a+r /tmp/step3_profile.sqlite /tmp/step3_kernsum_cuda_gpu_kern_sum.csv /tmp/step3_apisum_cuda_api_sum.csv 2>/dev/null
echo STATS_DONE
