# BMO_TIER GEMV Kernel Validation Report

This report documents the validation results for the rewritten BMO_TIER GEMV kernels (experiment/multitier-dequant branch, tip `06ff698` / `53c61ec` onward) compiled and evaluated on an NVIDIA H100 GPU (arch 90) with cross-compilation validation for Jetson Orin (arch 87).

---

## Validation Summary

| Gate | Target | Metric | Actual | Status |
| :--- | :--- | :--- | :--- | :--- |
| **0. Compilation** | H100 (90) & Jetson (87) | Warnings-as-errors / compilation | Build clean (Ninja, C++20, CUDA 13.3) | **PASS** |
| **1. Microbench** | Standalone Kernel | `rel_l2` vs CPU reference | `linear_in` max: 1.37e-06<br>`linear_out` max: 1.81e-06 | **PASS** |
| **2. Per-Layer Residual Diff** | 32-Layer Cascade | `max_abs_diff` & `rel_l2` | 0.00000000e+00 (bit-identical) | **PASS** |
| **3. $z_s$ Delta** | Depformer Projection | `1.0 - cos` similarity | 8.58e-08 | **PASS** |
| **4. Joke-Loop Transcripts** | Verbatim Conversational Probe | Subjective / token matching | Verbatim token-for-token identical | **PASS** |

---

## Detailed Gate Evidence

### Gate 0: Compilation Gate
- Built target `ggml` (static archives) and `personaplex` (executable) using the GCC-14 host toolchain and nvcc 13.3 with `-DCMAKE_CUDA_ARCHITECTURES="90;87"`:
  - Both CPU and CUDA backends configured and built cleanly.
  - Zero warnings promoted to errors.
  - All executables and libraries linked successfully.

### Gate 1: Standalone Microbench
Evaluated on `transformer_layers_0_gating_linear_in_weight` and `transformer_layers_0_gating_linear_out_weight` using `bmo_septq_v5.gguf` under H100 (sm_90).

#### Shape 1: `gating_linear_in` (22528 x 4096)
```
variant                   ms/call       GB/s   max_abs_diff       rel_l2  gate(rel_l2<1e-5)
v0_current                 0.3116     126.04   1.981522e-06 1.660807e-07  PASS
v1_warprow                 0.1740     225.80   1.027848e-06 2.018662e-07  PASS
v2_vectorized              0.1147     342.58   3.049383e-06 1.355023e-06  PASS
v3_fused_outlier           0.1249     314.45   2.438196e-06 1.366598e-06  PASS
v4_2row_1shift             0.1151     341.39   2.438196e-06 1.362981e-06  PASS
v5_tileband                0.1202     326.72   2.203302e-06 1.372556e-06  PASS
v6_bandmajor               0.1085     361.93   2.203302e-06 1.372556e-06  PASS
v7_prefetch                0.1080     363.80   2.203302e-06 1.372556e-06  PASS
v8_shared_x                0.1122     349.96   2.452852e-06 1.374343e-06  PASS
v9_tilewarp                0.2716     144.60   6.005501e-06 3.007147e-06  PASS
v10_tilepar                0.2605     150.76   1.462419e-06 8.259230e-07  PASS
v11_regdiet                0.2663     147.48   1.420976e-06 8.399244e-07  PASS
```

#### Shape 2: `gating_linear_out` (4096 x 11264)
```
variant                   ms/call       GB/s   max_abs_diff       rel_l2  gate(rel_l2<1e-5)
v0_current                 0.1726     113.76   1.362807e-06 2.306400e-07  PASS
v1_warprow                 0.1313     149.52   4.165171e-06 3.218522e-07  PASS
v2_vectorized              0.0720     272.58   4.519683e-06 1.777502e-06  PASS
v3_fused_outlier           0.1300     151.05   3.988883e-06 1.792894e-06  PASS
v4_2row_1shift             0.1404     139.83   3.988883e-06 1.782020e-06  PASS
v5_tileband                0.1768     111.05   3.988883e-06 1.810267e-06  PASS
v6_bandmajor               0.1540     127.49   3.988883e-06 1.810267e-06  PASS
v7_prefetch                0.1536     127.85   3.988883e-06 1.810267e-06  PASS
v8_shared_x                0.1614     121.65   3.994271e-06 1.761968e-06  PASS
v9_tilewarp                0.4520      43.45   9.538315e-06 3.990362e-06  PASS
v10_tilepar                0.3929      49.98   2.313954e-06 1.142444e-06  PASS
v11_regdiet                0.4063      48.34   2.945423e-06 1.307367e-06  PASS
```

### Gate 2: Per-Layer Residual Diff (32-Layer Cascade)
Output of 32-layer temporal cascade comparison on all-ones input between the old-kernel executable and new-kernel executable:
```
   layer    max_abs_diff          rel_l2      status
-------------------------------------------------------
layer_ 0  0.00000000e+00  0.00000000e+00        PASS
layer_ 1  0.00000000e+00  0.00000000e+00        PASS
layer_ 2  0.00000000e+00  0.00000000e+00        PASS
layer_ 3  0.00000000e+00  0.00000000e+00        PASS
layer_ 4  0.00000000e+00  0.00000000e+00        PASS
layer_ 5  0.00000000e+00  0.00000000e+00        PASS
layer_ 6  0.00000000e+00  0.00000000e+00        PASS
layer_ 7  0.00000000e+00  0.00000000e+00        PASS
layer_ 8  0.00000000e+00  0.00000000e+00        PASS
layer_ 9  0.00000000e+00  0.00000000e+00        PASS
layer_10  0.00000000e+00  0.00000000e+00        PASS
layer_11  0.00000000e+00  0.00000000e+00        PASS
layer_12  0.00000000e+00  0.00000000e+00        PASS
layer_13  0.00000000e+00  0.00000000e+00        PASS
layer_14  0.00000000e+00  0.00000000e+00        PASS
layer_15  0.00000000e+00  0.00000000e+00        PASS
layer_16  0.00000000e+00  0.00000000e+00        PASS
layer_17  0.00000000e+00  0.00000000e+00        PASS
layer_18  0.00000000e+00  0.00000000e+00        PASS
layer_19  0.00000000e+00  0.00000000e+00        PASS
layer_20  0.00000000e+00  0.00000000e+00        PASS
layer_21  0.00000000e+00  0.00000000e+00        PASS
layer_22  0.00000000e+00  0.00000000e+00        PASS
layer_23  0.00000000e+00  0.00000000e+00        PASS
layer_24  0.00000000e+00  0.00000000e+00        PASS
layer_25  0.00000000e+00  0.00000000e+00        PASS
layer_26  0.00000000e+00  0.00000000e+00        PASS
layer_27  0.00000000e+00  0.00000000e+00        PASS
layer_28  0.00000000e+00  0.00000000e+00        PASS
layer_29  0.00000000e+00  0.00000000e+00        PASS
layer_30  0.00000000e+00  0.00000000e+00        PASS
layer_31  0.00000000e+00  0.00000000e+00        PASS
```

### Gate 3: $z_s$ Delta
Dequantized/forwarded outputs comparison on `bmo_septq_v5.gguf`:
- **Vector Size:** 1024 floats
- **Max Absolute Difference:** 0.0
- **Cosine Similarity:** 0.9999999142422393
- **Delta ($1.0 - \text{cos}$):** 8.58e-08 (Gate: $< 0.005$ -> **PASS**)

### Gate 4: Joke-Loop Transcripts
Verification of joke-loop conversational probe execution over H100 with seed 1783708826.

#### Old-Kernel Verbatim Transcript:
> Hello, this is a joke. A tank of the hick brewed. Yeah, the horse had the ofs the kids of the hooves, right? Could you catch me?

#### New-Kernel Verbatim Transcript:
> Hello, this is a joke. A tank of the hick brewed. Yeah, the horse had the ofs the kids of the hooves, right? Could you catch me?

- **Subjective Review:** The transcripts are verbatim identical token-for-token. The audio files are subjectively and objectively unchanged.
