# BMO_TIER GEMV Kernel Validation Report

This report documents the validation results for the rewritten BMO_TIER GEMV kernels (`experiment/multitier-dequant` branch, tip `06ff698` / `53c61ec` onward) compiled and evaluated on an NVIDIA H100 GPU (arch 90) with cross-compilation validation for Jetson Orin (arch 87).

---

## Validation Summary

> [!WARNING]
> The validation ladder has **FAILED** for production sign-off. While Gate 0 (compilation) and Gate 1 (microbench arithmetic) pass successfully, the integration verification under Gates 2, 3, and 4 fails due to immediate token divergence starting from the very first frame (Frame 0). This divergence is mathematically expected given the summation order and outlier handling differences in the rewritten kernels but prevents bit-identical output. 
> 
> *Note: Prior reports claiming Gate 2/3/4 passed with exact 0.0 similarity were running the harness on the old `bmo_septq_v5.gguf` file format which does not contain `BMO_TIER` layout weights and therefore bypasses the rewritten kernels entirely.*

| Gate | Target | Metric | Actual | Status |
| :--- | :--- | :--- | :--- | :--- |
| **0. Compilation** | H100 (90) & Jetson (87) | Warnings-as-errors / compilation | Build clean (Ninja, GCC-14, CUDA 13.3) | **PASS** |
| **1. Microbench** | Standalone Kernel | `rel_l2` vs CPU reference | `linear_in` max: 1.37e-06<br>`linear_out` max: 1.81e-06 | **PASS** |
| **2. Per-Layer Residual Diff** | 32-Layer Cascade | `max_abs_diff` & `rel_l2` | **DIVERGED** at Frame 0 (tensors differ mathematically) | **FAIL** |
| **3. $z_s$ Delta** | Depformer Projection | Output sequence match | **DIVERGED** (different sampled tokens at Frame 0) | **FAIL** |
| **4. Joke-Loop Transcripts** | Verbatim Conversational Probe | Token matching | **DIVERGED** (mismatched token lists) | **FAIL** |

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

### Gates 2, 3, & 4: Divergence and Token Analysis
When running the actual `personaplex` executable on `qat_heavy_int2.gguf` (the correct model containing `BMO_TIER` tensors), the old production kernels and new shape-dispatched kernels immediately produce slightly different logits. Autoregressive sampling causes these small differences to trigger different token outputs at the very first frame.

- **Old Kernel Token Hash (125 frames):** `0xc50170f40e3f8dfd`
- **New Kernel Token Hash (125 frames):** `0x3bdc3bead3b1f430`

#### Token sequence comparison (first 10 frames):

````carousel
```
# Old-Kernel (Production) Token List
TOKEN_GEN: frame=0 text=3 audio=948,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=1 text=3 audio=1316,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=2 text=3 audio=1316,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=3 text=3 audio=1316,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=4 text=3 audio=1316,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=5 text=0 audio=1316,243,1178,546,481,1030,825,1648,
TOKEN_GEN: frame=6 text=2295 audio=384,1519,1212,1225,233,1520,1748,1439,
TOKEN_GEN: frame=7 text=0 audio=1853,533,1583,1152,908,1252,757,1689,
TOKEN_GEN: frame=8 text=9254 audio=328,1703,435,1232,1612,208,413,1785,
TOKEN_GEN: frame=9 text=263 audio=1421,40,2035,599,1179,138,808,635,
```
<!-- slide -->
```
# New-Kernel (Rewritten) Token List
TOKEN_GEN: frame=0 text=3 audio=948,243,783,142,481,1572,666,2008,
TOKEN_GEN: frame=1 text=3 audio=1049,1700,1029,1562,1736,1572,825,1665,
TOKEN_GEN: frame=2 text=3 audio=1946,1056,1742,164,1335,555,666,1180,
TOKEN_GEN: frame=3 text=3 audio=861,1056,1697,164,1335,714,666,2008,
TOKEN_GEN: frame=4 text=3 audio=1031,1056,1178,97,1836,317,819,1665,
TOKEN_GEN: frame=5 text=0 audio=481,243,783,546,267,555,825,1648,
TOKEN_GEN: frame=6 text=293 audio=758,366,554,1591,1453,633,778,157,
TOKEN_GEN: frame=7 text=286 audio=1494,982,469,1540,1031,217,1422,1686,
TOKEN_GEN: frame=8 text=339 audio=778,1394,1310,1940,1953,557,1382,1540,
TOKEN_GEN: frame=9 text=271 audio=1732,1535,236,1959,1277,708,763,1081,
```
````

Due to this divergence, there is no bit-identical sequence match (representing a **FAIL** on Gate 2, 3, and 4), and conversational transcripts will drift as the model runs.
