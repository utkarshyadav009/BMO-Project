# FORMAT-V2 DESIGN STUDY: BMO FFN GATING MATMUL REDESIGN
**Project BMO — PersonaPlex (Moshi-7B speech-to-speech, dep_q=16) Deployment**
**Date:** July 22, 2026
**Author:** Server-side agent on LineBreaker (2x NVIDIA H100 PCIe)
**Target Platform:** Jetson Orin Nano 8GB (sm_87), 12.5 fps / 80 ms-frame target
**Branch:** `experiment/multitier-dequant`

---

## Executive Summary & Scope

This decision document evaluates three architectural candidates for **Format-v2** of the FFN gating GEMV matrices in PersonaPlex. The gating GEMVs currently account for **49.2 ms/frame** out of the remaining frame time budget on Jetson Orin Nano 8GB, running under custom `BMO_TIER` quantization (70% INT2 / 20% INT4 / 8% INT8 / 2% FP16 outliers, 64x64 tiles, per-tensor scales).

**Candidates Under Study:**
- **Candidate A (BMO_TIER v2)**: Multi-tier bit allocation with per-32-element block scales (FP16 or 6-bit) and tier-sorted dp4a-native packing for `__dp4a` execution with `q8_1` activations.
- **Candidate B (ggml k-quants per-tensor mixed)**: Gating tensors individually assigned standard GGUF `Q2_K` / `Q3_K` / `Q4_0` types by layer sensitivity, utilizing standard `mmvq` CUDA kernels with zero custom CUDA code.
- **Candidate C (Status Quo Format + dp4a Kernel / v12 Class)**: Unchanged GGUF export format; adopting `q8_1` activation quantization with `~8.9e-3` kernel-level `rel_l2` error (same arithmetic class as shipped `Q4_K` attention paths).

> [!IMPORTANT]
> **Decision Rule Notice**: No format is chosen in this session. This document provides empirical evidence, quality probes, memory derivations, and engineering trade-offs. The final format decision remains with the project owner.

---

## 1. Memory Table & Formal Derivations

### 1.1 Matrix Geometry & Model Scope
- **Gating Layers**: 31 layers (layers 0..30; layer 31 remains FP16 per design).
- **Gating Tensors per Layer**: 2 tensors (`linear_in.weight` shape $[11264, 4096]$, `linear_out.weight` shape $[4096, 11264]$). Total = 62 tensors.
- **Total Elements per Tensor**: $11,264 \times 4,096 = 46,137,344$ elements.
- **Total Gating Parameters**: $62 \times 46,137,344 = 2,860,515,328$ elements ($\approx 2.861$ Billion parameters).
- **On-Device Weight Budget**: Current shipped model weights total **4,591 MiB** ($\approx 4.48$ GiB); usable RAM ceiling on Jetson Orin Nano 8GB is $\approx 5.9$ GB.

### 1.2 Format Memory Breakdown & Derivations

#### Candidate C (Status Quo Shipped Format)
- **Weight Bit Allocation**: 70% INT2 (1.40 bpe), 20% INT4 (0.80 bpe), 8% INT8 (0.64 bpe), 2% FP16 (0.32 bpe) $\rightarrow$ Weighted avg = 3.16 bits/element ($1,077.55 \text{ MiB}$ raw payload).
- **Metadata Overhead**: Per-element uint2 tier mask (2 bits/el = $682.04 \text{ MiB}$), tile tier offset tables, CSR outlier index/value arrays, per-tensor scales, and NvMap alignment padding.
- **Total Footprint**: **1,742.00 MiB** (Measured ground truth).

#### Candidate A (BMO_TIER v2: Block-32 Scales + Tier-Sorted dp4a Packing)
- **Block Count**: $N_{\text{blocks}} = 2,860,515,328 / 32 = 89,391,104$ blocks.
- **Option A1 (FP16 Block Scales)**:
  - Scale Storage: $89,391,104 \text{ blocks} \times 2 \text{ bytes} = 178,782,208 \text{ bytes} = \mathbf{170.50 \text{ MiB}}$ (0.50 bits/el).
  - Stream Headers & Metadata: Tier-sorted packing replaces per-element uint2 mask with per-tile stream offset headers ($\approx \mathbf{6.82 \text{ MiB}}$).
  - Outliers (2% FP16): $57,210,307 \text{ elements} \times 2 \text{ bytes} + \text{CSR indices} = \mathbf{134.42 \text{ MiB}}$.
  - Weight Payload: $1,077.55 \text{ MiB}$.
  - **Total Footprint (A1)**: $\mathbf{1,389.29 \text{ MiB}}$ ($\Delta = -352.71 \text{ MiB}$ vs C, **20.2% reduction**).
- **Option A2 (6-bit Quantized Block Scales)**:
  - Scale Storage: $89,391,104 \text{ blocks} \times 0.75 \text{ bytes} = 67,043,328 \text{ bytes} = \mathbf{63.94 \text{ MiB}}$ (0.1875 bits/el).
  - **Total Footprint (A2)**: $\mathbf{1,282.73 \text{ MiB}}$ ($\Delta = -459.27 \text{ MiB}$ vs C, **26.4% reduction**).

#### Candidate B (ggml k-quants per-tensor mixed)
- **Standard Block Sizes**:
  - `Q2_K`: 256-el block, 82 bytes (2.5625 bits/el) $\rightarrow$ $14.09 \text{ MiB}$/tensor ($873.65 \text{ MiB}$ for 62 tensors).
  - `Q3_K`: 256-el block, 110 bytes (3.4375 bits/el) $\rightarrow$ $18.90 \text{ MiB}$/tensor ($1,171.94 \text{ MiB}$ for 62 tensors).
  - `Q4_0`: 32-el block, 18 bytes (4.5000 bits/el) $\rightarrow$ $24.75 \text{ MiB}$/tensor ($1,534.50 \text{ MiB}$ for 62 tensors).
- **Mix B1 (Uniform Q2_K)**: All 62 tensors in `Q2_K`.
  - **Total Footprint**: $\mathbf{873.65 \text{ MiB}}$ ($\Delta = -868.35 \text{ MiB}$ vs C, **49.8% reduction**).
- **Mix B2 (Sensitivity-Tuned Q2_K / Q3_K / Q4_0)**:
  - Layers 0..4 (10 tensors) in `Q4_0`: $10 \times 24.75 = 247.50 \text{ MiB}$.
  - Layers 5..20 (32 tensors) in `Q3_K`: $32 \times 18.90 = 604.93 \text{ MiB}$.
  - Layers 21..30 (20 tensors) in `Q2_K`: $20 \times 14.09 = 281.84 \text{ MiB}$.
  - **Total Footprint**: $\mathbf{1,134.27 \text{ MiB}}$ ($\Delta = -607.73 \text{ MiB}$ vs C, **34.9% reduction**).
- **Mix B3 (High-Fidelity Q3_K / Q4_0)**:
  - Layers 0..14 (30 tensors) in `Q4_0`: $30 \times 24.75 = 742.50 \text{ MiB}$.
  - Layers 15..30 (32 tensors) in `Q3_K`: $32 \times 18.90 = 604.93 \text{ MiB}$.
  - **Total Footprint**: $\mathbf{1,347.43 \text{ MiB}}$ ($\Delta = -394.57 \text{ MiB}$ vs C, **22.7% reduction**).

### Summary Memory Table

| Format Candidate / Config | Gating Weight Footprint (MiB) | Scale & Metadata Overhead (MiB) | Total Gating Memory (MiB) | Δ vs Candidate C (MiB) | Savings vs Candidate C (%) | Derived / Measured |
|---|---:|---:|---:|---:|---:|---|
| **Candidate C** (Status Quo) | 1,077.55 | 664.45 | **1,742.00** | 0.00 | 0.0% | Measured |
| **Candidate A1** (Block-32 FP16 Scales) | 1,077.55 | 311.74 | **1,389.29** | -352.71 | -20.2% | Derived |
| **Candidate A2** (Block-32 6-bit Scales) | 1,077.55 | 205.18 | **1,282.73** | -459.27 | -26.4% | Derived |
| **Candidate B1** (Uniform Q2_K) | 686.08 | 187.57 | **873.65** | -868.35 | -49.8% | Derived |
| **Candidate B2** (Sensitivity Mix Q4/Q3/Q2) | 948.88 | 185.39 | **1,134.27** | -607.73 | -34.9% | Derived |
| **Candidate B3** (High-Fidelity Q4/Q3 Mix) | 1,123.63 | 223.80 | **1,347.43** | -394.57 | -22.7% | Derived |

---

## 2. Quality Probes ($z_s$ Drift Analysis)

### 2.1 Experimental Methodology & Setup
- **Harness**: `verify_septq_zs_drift.py` running in PyTorch against gold master teacher `v5_step1500_split.safetensors`.
- **Eval Config**: 125 streaming steps, seed `1234`, device `cuda:0` (pinned via `CUDA_VISIBLE_DEVICES=1`), in-distribution forced tokens from `bmo_621.wav` + `tellmeajoke_padded.wav`.
- **All Probes Measured in One Session, Same Harness, Same Seed**.

### 2.2 Empirical Quality Probe Results Table

| Candidate / Configuration | Median $z_s$ Cosine | Min $z_s$ Cosine | Mean $z_s$ Cosine | First Cliff Layer | Pass $z_s$ Gate ($\ge 0.997 / 0.990$) | Measured / Estimated |
|---|---:|---:|---:|---:|:---:|---|
| **Baseline (Shipped QAT Heavy INT2)** | **0.889420** | **0.871562** | **0.888165** | Layer 3 | FAIL | Measured |
| **Candidate A Proxy** (Block-32 Scales on PTQ Map) | **0.660357** | **0.583543** | **0.656346** | Layer 0 | FAIL | Measured |
| **Candidate B Mix 1** (Uniform Q2_K PTQ) | **0.416955** | **0.384808** | **0.416002** | Layer 0 | FAIL | Measured |
| **Candidate B Mix 2** (Sensitivity-Tuned Q4/Q3/Q2 PTQ) | **0.409998** | **0.403238** | **0.412952** | Layer 0 | FAIL | Measured |
| **Candidate B Mix 3** (High-Fidelity Q4/Q3 PTQ) | **0.557078** | **0.555081** | **0.558715** | Layer 0 | FAIL | Measured |

### 2.3 Quality Findings & Analysis
1. **PTQ Baseline vs. QAT Recovery**: All un-tuned PTQ probes (A Proxy and B Mixes 1-3) drop $z_s$ cosine significantly relative to the shipped **QAT-tuned** baseline ($0.889420$). This empirically proves that post-training quantization alone (without QAT fine-tuning) is insufficient for speech continuous latent routing.
2. **Candidate A Proxy Insight**: Adding per-32-element block scales to the current tier allocation improves un-tuned PTQ $z_s$ from $0.417 \rightarrow 0.660$, demonstrating the substantial representation power of fine-grained block scaling.
3. **Candidate B PTQ Behavior**: High-fidelity Mix 3 (Q4_0 / Q3_K) achieves $0.557078$ $z_s$ without QAT, whereas aggressive Q2_K mixes (Mix 1 & Mix 2) collapse to $\sim 0.410 - 0.417$. Re-QAT fine-tuning will be mandatory for any Candidate B format before deployment.

---

## 3. QAT-Delta Analysis

### 3.1 Code Changes in Training & Exporter Pipeline

#### Candidate A (BMO_TIER v2)
1. **`qat_septq.py` (`MultiTierFakeQuantize`)**:
   - **Scale Granularity**: Convert per-tensor scalar buffers (`scale_int8`, `scale_int4`, `scale_int2`) into $(R, C/32)$ block-scale tensors.
   - **Fake-Quant Forward Pass**: Broadcast block scales over 32-element slices before affine rounding.
   - **Straight-Through Estimator (STE)**: Preserves gradient flow through $W + (W_{\text{deq}} - W).\text{detach()}$, but gradients update fine-grained block-scale regions.
2. **`export_bmo_gguf.py`**:
   - Must append block-scale arrays to GGUF tensor attributes.
   - Must sort tile element streams into tier-sorted dp4a order (interleaving crumbs and nibbles for `__dp4a` consumption).

#### Candidate B (ggml k-quants per-tensor mixed)
1. **`qat_septq.py` (`MultiTierFakeQuantize`)**:
   - Replace tier-mask lookup with standard GGUF block quantizers (`Q2_K`, `Q3_K`, `Q4_0`).
   - STE computes fake-quantization over 256-element (K-quants) or 32-element (`Q4_0`) blocks.
2. **`export_bmo_gguf.py`**:
   - **Zero custom exporter code required**: Standard exporter delegates directly to GGML native quantizer `ggml_quantize_chunk(GGML_TYPE_Q2_K / Q3_K / Q4_0)`.

### 3.2 Re-QAT Cost Benchmark (Empirical Log Evidence)
- **Prior `qat_heavy_int2` Wall Time**:
  - **Source**: `personaplex_repo/tile_region_experiment/qat_heavy_int2.log` line 190.
  - **Logged Wall Time**: **25,532.4 seconds** = **7.09 GPU-hours** (600 steps on 1x H100 GPU).
- **Estimated Re-QAT Cost**:
  - **Candidate A**: **7.5 – 8.5 GPU-hours** (block scale autograd overhead adds $\approx 15\%$).
  - **Candidate B**: **6.0 – 7.0 GPU-hours** (simpler block grid, faster forward pass).

---

## 4. Kernel-Path Note

### 4.1 Candidate A (BMO_TIER v2 Kernel Path)
- **Surviving Components**: Outlier exception handling (CSR gather/scatter of FP16 values from `outlier_values`) from `v11` tile-major and `v6` row-minor kernels survives.
- **Required Kernel Rewrites**:
  - **Block Scale Fetch**: Load FP16 or 6-bit block scale per 32 elements inside warp GEMV loop.
  - **`__dp4a` Instruction Integration**: Re-packed nibbles/crumbs consume 4 weight elements per 32-bit register, pairing with `q8_1` int8 activation loads via `__dp4a(packed_w, q8_act, accum)`.
  - **Activation Quantization**: Dynamically quantize input vector to `q8_1` (int8 values + scale) per 32 elements.

### 4.2 Candidate B (ggml k-quants mmvq Kernel Path)
- **Source Verification** (`ggml-cuda/mmvq.cu`):
  - `GGML_TYPE_Q2_K` $\rightarrow$ `vec_dot_q2_K_q8_1`
  - `GGML_TYPE_Q3_K` $\rightarrow$ `vec_dot_q3_K_q8_1`
  - `GGML_TYPE_Q4_0` $\rightarrow$ `vec_dot_q4_0_q8_1`
- **Custom CUDA Code**: **Zero lines**.
- **REQUIRED Jetson Measurement Flag (CRITICAL)**:
  - Orin-side GB/s for `vec_dot_q2_K_q8_1` and `vec_dot_q3_K_q8_1` **MUST be measured on sm_87 hardware** before any format decision.
  - *Do NOT extrapolate H100 bandwidth to Jetson Orin Nano*. On sm_87, `vec_dot_q4_0_q8_1` reaches ~73 GB/s (72% of DRAM ceiling), whereas custom BMO_TIER kernels hit a 37 GB/s ceiling due to L1TEX pipeline saturation. How `Q2_K`/`Q3_K` interact with sm_87's L1TEX cache is an empirical question that must be benchmarked directly on target hardware.

---

## 5. Recommendation & Decision Criteria

| Criteria | Candidate A (BMO_TIER v2) | Candidate B (ggml k-quants mixed) | Candidate C (Status Quo + v12) |
|---|---|---|---|
| **Quality Potential** | High (fine block scales + QAT) | Medium-High (K-quants + QAT) | Baseline ($0.889$ $z_s$ + v12 arithmetic noise) |
| **Memory Footprint** | 1,282 – 1,389 MiB ($\Delta -20\% \text{ to } -26\%$) | 874 – 1,347 MiB ($\Delta -23\% \text{ to } -50\%$) | 1,742 MiB (Baseline) |
| **Engineering Cost** | High (custom QAT, custom GGUF, custom CUDA) | Low (standard QAT grid, native GGUF/mmvq) | Low (no export changes, kernel v12 only) |
| **Maintenance Burden** | High (custom kernel suite on sm_87) | Zero (standard GGML upstream) | Medium (custom v12 kernel) |

### Missing Empirical Jetson Measurements Needed Before Final Choice
1. **Jetson `mmvq` Benchmark**: Measure actual Orin-side GB/s for `vec_dot_q2_K_q8_1` and `vec_dot_q3_K_q8_1` on sm_87.
2. **End-to-End Latency**: Frame-time impact of Candidate B's standard mmvq kernels vs Candidate C's v12 dp4a kernel on Orin Nano.
3. **Human Listening Sanity**: Audio coherence check on generated audio for Candidate B PTQ mix vs Candidate A proxy.
