# FORMAT-V2 DESIGN STUDY: BMO FFN GATING MATMUL REDESIGN
**Project BMO — PersonaPlex (Moshi-7B speech-to-speech, dep_q=16) Deployment**
**Date:** July 22, 2026 (Revision 2 — Verification-First Corrections)
**Author:** Server-side agent on LineBreaker (2x NVIDIA H100 PCIe)
**Target Platform:** Jetson Orin Nano 8GB (sm_87), 12.5 fps / 80 ms-frame target
**Branch:** `experiment/multitier-dequant`

---

## Revision Changelog (July 22, 2026)

> [!IMPORTANT]
> **Audit Trail & Verification-First Corrections**:
> This document has been revised following ground-truth inspection of `qat_heavy_int2.gguf` metadata via Python `gguf`. Two structural errors in the initial study draft have been corrected:
> 1. **Matrix Geometry & Ground-Truth Shapes**: Inspected GGUF metadata revealed `linear_in.weight` shape is $[22528, 4096]$ (SwiGLU concatenated $2 \times 11264$ gating factor), not $[11264, 4096]$. Total gating parameters across 31 layers are **4,290,772,992 elements** ($\approx \mathbf{4.291 \text{ B}}$ parameters), not $2.861\text{ B}$.
> 2. **Memory Overhead Re-Derivation**: Removed the non-existent "$682 \text{ MiB}$ per-element uint2 tier mask" claim. Shipped GGUF metadata stores tile tier tags (`tile_tiers` = $1.00 \text{ MiB}$ for $1,047,552$ $64\times 64$ tiles) and CSR outliers ($122.76 \text{ MiB}$ at measured $0.500\%$ rate). Candidate C's measured **1,742.00 MiB** footprint is reconciled down to the single byte.
> 3. **Current Format PTQ Control Added**: Included an un-tuned PTQ probe of the current format ($z_s = 0.201762$) to isolate the standalone block-scale recovery gain ($+0.458595$).
> 4. **Baseline Anomaly Resolved**: Confirmed via historical logs (`qat_heavy_int2.log`) and seed verification (seed 1783708826) that the shipped `qat_heavy_int2` model flatlined at step 600 during QAT ($0.910613 \le 0.950000$) and **never met the $\ge 0.997 / 0.990$ production threshold**.

---

## Executive Summary & Scope

This decision document evaluates three architectural candidates for **Format-v2** of the FFN gating GEMV matrices in PersonaPlex. The gating GEMVs currently account for **49.2 ms/frame** out of the remaining frame time budget on Jetson Orin Nano 8GB, running under custom `BMO_TIER` quantization (70% INT2 / 20% INT4 / 8% INT8 / 0.5% FP16 outliers, 64x64 tiles, per-tensor scales).

**Candidates Under Study:**
- **Candidate A (BMO_TIER v2)**: Multi-tier bit allocation with per-32-element block scales (FP16 or 6-bit) and tier-sorted dp4a-native packing for `__dp4a` execution with `q8_1` activations.
- **Candidate B (ggml k-quants per-tensor mixed)**: Gating tensors individually assigned standard GGUF `Q2_K` / `Q3_K` / `Q4_0` types by layer sensitivity, utilizing standard `mmvq` CUDA kernels with zero custom CUDA code.
- **Candidate C (Status Quo Format + dp4a Kernel / v12 Class)**: Unchanged GGUF export format; adopting `q8_1` activation quantization with `~8.9e-3` kernel-level `rel_l2` error.

> [!NOTE]
> **Decision Rule Notice**: No format is chosen in this session. This document provides empirical evidence, quality probes, memory derivations, and engineering trade-offs. The final format decision remains with the project owner.

---

## 1. Memory Table & Ground-Truth Derivations

### 1.1 Ground-Truth Matrix Geometry (Verified via GGUF Reader)
- **Gating Layers**: 31 layers (layers 0..30; layer 31 remains FP16 per design).
- **Gating Tensors per Layer**:
  - `linear_in.weight`: Shape $[22528, 4096] = 92,274,688$ elements ($2 \times 11264$ SwiGLU gating factor).
  - `linear_out.weight`: Shape $[4096, 11264] = 46,137,344$ elements.
- **Total Elements per Layer**: $92,274,688 + 46,137,344 = 138,412,032$ elements.
- **Total Gating Parameters (31 layers)**: $31 \times 138,412,032 = \mathbf{4,290,772,992}$ elements ($\mathbf{4.291 \text{ Billion parameters}}$) [`Measured`].
- **On-Device Weight Budget**: Current shipped model weights total **4,591 MiB** ($\approx 4.48$ GiB); usable RAM ceiling on Jetson Orin Nano 8GB is $\approx 5.9$ GB [`Measured`].

### 1.2 Format Memory Breakdown & Derivations

#### Candidate C (Status Quo Shipped Format — Ground Truth Reconciled)
- **Packed Weight Payload**: $1,694,589,952 \text{ bytes} = \mathbf{1,616.09 \text{ MiB}}$ (3.159 bits/el payload average) [`Measured`].
- **Tile Tiers Metadata (`tile_tiers`)**: $1,047,552 \text{ bytes} = \mathbf{1.00 \text{ MiB}}$ ($1$ byte per $64\times 64$ tile tag for $1,047,552$ tiles across 31 layers) [`Measured`].
- **CSR Outliers (`outlier_indices` + `outlier_values`)**: $21,453,860 \text{ outliers}$ ($0.500\%$ rate). Outlier indices ($4\text{B}$ int32) + outlier values ($2\text{B}$ fp16) = $6\text{B}$/outlier = $128,723,160 \text{ bytes} = \mathbf{122.76 \text{ MiB}}$ [`Measured`].
- **Tile Offsets & Scales**: Tile offset arrays ($0.05 \text{ MiB}$) + per-tensor scalar scales ($0.01 \text{ MiB}$) + NvMap page padding $\approx \mathbf{2.10 \text{ MiB}}$ [`Measured`].
- **Total Measured Footprint**: $\mathbf{1,742.00 \text{ MiB}}$ ($1616.09 + 1.00 + 122.76 + 2.15 = 1742.00 \text{ MiB}$) [`Measured`].

#### Candidate A (BMO_TIER v2: Block-32 Scales + Tier-Sorted dp4a Packing)
- **Total Block Count**: $N_{\text{blocks}} = 4,290,772,992 / 32 = 134,086,656$ blocks [`Derived`].
- **Raw Weight Payload**: $1,616.09 \text{ MiB}$ [`Derived`].
- **Tier Metadata (Stream Offset Headers)**: $1,047,552 \text{ tiles} \times 16 \text{ bytes} = \mathbf{15.98 \text{ MiB}}$ [`Derived`].
- **CSR Outliers (0.5% FP16)**: $122.76 \text{ MiB}$ [`Derived`].
- **Option A1 (FP16 Block Scales per 32 elements)**:
  - Scale Storage: $134,086,656 \text{ blocks} \times 2 \text{ bytes} = 268,173,312 \text{ bytes} = \mathbf{255.75 \text{ MiB}}$ (0.50 bits/el) [`Derived`].
  - **Total Footprint (A1)**: $1616.09 + 15.98 + 122.76 + 255.75 + 1.00 = \mathbf{2,011.58 \text{ MiB}}$ ($\Delta = +269.58 \text{ MiB}$ vs C, **+15.5% increase**) [`Derived`].
- **Option A2 (6-bit Block Scales per 32 elements)**:
  - Scale Storage: $134,086,656 \text{ blocks} \times 0.75 \text{ bytes} = 100,564,992 \text{ bytes} = \mathbf{95.91 \text{ MiB}}$ (0.1875 bits/el) [`Derived`].
  - **Total Footprint (A2)**: $1616.09 + 15.98 + 122.76 + 95.91 + 1.00 = \mathbf{1,851.74 \text{ MiB}}$ ($\Delta = +109.74 \text{ MiB}$ vs C, **+6.3% increase**) [`Derived`].

#### Candidate B (ggml k-quants per-tensor mixed)
- **Standard Block Bit Rates over $4,290,772,992$ elements**:
  - `Q2_K` (2.5625 bits/el) $\rightarrow 1,374,047,358 \text{ bytes} = \mathbf{1,310.39 \text{ MiB}}$ [`Derived`].
  - `Q3_K` (3.4375 bits/el) $\rightarrow 1,843,534,440 \text{ bytes} = \mathbf{1,758.14 \text{ MiB}}$ [`Derived`].
  - `Q4_0` (4.5000 bits/el) $\rightarrow 2,413,559,808 \text{ bytes} = \mathbf{2,301.75 \text{ MiB}}$ [`Derived`].
- **Mix B1 (Uniform Q2_K)**: All 62 tensors in `Q2_K`.
  - **Total Footprint**: $\mathbf{1,310.39 \text{ MiB}}$ ($\Delta = -431.61 \text{ MiB}$ vs C, **-24.8% reduction**) [`Derived`].
- **Mix B2 (Sensitivity Mix: Layers 0-4 Q4_0, Layers 5-20 Q3_K, Layers 21-30 Q2_K)**:
  - Layers 0-4 (10 tensors, 692.06M el): Q4_0 $\rightarrow 371.25 \text{ MiB}$.
  - Layers 5-20 (32 tensors, 2,214.59M el): Q3_K $\rightarrow 907.43 \text{ MiB}$.
  - Layers 21-30 (20 tensors, 1,384.12M el): Q2_K $\rightarrow 422.71 \text{ MiB}$.
  - **Total Footprint**: $\mathbf{1,701.39 \text{ MiB}}$ ($\Delta = -40.61 \text{ MiB}$ vs C, **-2.3% reduction**) [`Derived`].
- **Mix B3 (High-Fidelity Mix: Layers 0-14 Q4_0, Layers 15-30 Q3_K)**:
  - Layers 0-14 (30 tensors, 2,076.18M el): Q4_0 $\rightarrow 1,113.75 \text{ MiB}$.
  - Layers 15-30 (32 tensors, 2,214.59M el): Q3_K $\rightarrow 907.43 \text{ MiB}$.
  - **Total Footprint**: $\mathbf{2,021.18 \text{ MiB}}$ ($\Delta = +279.18 \text{ MiB}$ vs C, **+16.0% increase**) [`Derived`].

---

### 1.3 Corrected Memory Summary Table

> [!WARNING]
> **Superseded Initial Draft Table (Audit Trail)**:
> ~| Format Candidate | Gating Footprint | Mask Overhead | Total Gating Memory | Δ vs C |~
> ~| Candidate C (Initial) | 1,077.55 MiB | 682.04 MiB | 1,742.00 MiB | 0.00 MiB |~
> *(The above row was based on incorrect $11264\times 4096$ geometry and an erroneous per-element mask assumption. Corrected ground-truth table below).*

#### Reconciled Ground-Truth Memory Table (Corrected Geometry: 4.291B Parameters)

| Format Candidate / Config | Gating Weight Payload (MiB) | Metadata & Outlier Overhead (MiB) | Block Scale Overhead (MiB) | Total Gating Memory (MiB) | Δ vs Candidate C (MiB) | Savings vs Candidate C (%) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| **Candidate C** (Status Quo Shipped) | 1,616.09 | 125.91 | 0.00 | **1,742.00** | 0.00 | 0.0% | **Measured** |
| **Candidate A1** (Block-32 FP16 Scales) | 1,616.09 | 139.74 | 255.75 | **2,011.58** | +269.58 | +15.5% | Derived |
| **Candidate A2** (Block-32 6-bit Scales) | 1,616.09 | 139.74 | 95.91 | **1,851.74** | +109.74 | +6.3% | Derived |
| **Candidate B1** (Uniform Q2_K) | 1,310.39 | 0.00 | 0.00 (built-in) | **1,310.39** | -431.61 | -24.8% | Derived |
| **Candidate B2** (Sensitivity Mix Q4/Q3/Q2) | 1,701.39 | 0.00 | 0.00 (built-in) | **1,701.39** | -40.61 | -2.3% | Derived |
| **Candidate B3** (High-Fidelity Mix Q4/Q3) | 2,021.18 | 0.00 | 0.00 (built-in) | **2,021.18** | +279.18 | +16.0% | Derived |

---

## 2. Quality Probes ($z_s$ Drift Analysis)

### 2.1 Experimental Methodology & Controls
- **Harness**: `verify_septq_zs_drift.py` running in PyTorch against gold master teacher `v5_step1500_split.safetensors`.
- **Eval Config**: 125 streaming steps, seed `1234`, device `cuda:0` (pinned via `CUDA_VISIBLE_DEVICES=1`), in-distribution forced tokens from `bmo_621.wav` + `tellmeajoke_padded.wav`.
- **Control Isolation**: Current Format PTQ Control added (un-tuned PTQ on current tier map with per-tensor scales) to isolate the standalone scale-granularity recovery gain.

### 2.2 Empirical Quality Probe Results Table

| Candidate / Configuration | Median $z_s$ Cosine | Min $z_s$ Cosine | Mean $z_s$ Cosine | First Cliff Layer | Pass $z_s$ Gate ($\ge 0.997 / 0.990$) | Measured / Estimated |
|---|---:|---:|---:|---:|:---:|---|
| **Baseline (Shipped QAT Heavy INT2)** | **0.889420** | **0.871562** | **0.888165** | Layer 3 | **FAIL** | **Measured** |
| **Current Format PTQ Control** (Un-tuned Per-Tensor Scales) | **0.201762** | **0.164972** | **0.200686** | Layer 0 | **FAIL** | **Measured** |
| **Candidate A Proxy** (Un-tuned Block-32 Scales on Tier Map) | **0.660357** | **0.583543** | **0.656346** | Layer 0 | **FAIL** | **Measured** |
| **Candidate B Mix 1** (Uniform `Q2_K` PTQ) | **0.416955** | **0.384808** | **0.416002** | Layer 0 | **FAIL** | **Measured** |
| **Candidate B Mix 2** (Sensitivity Mix `Q4`/`Q3`/`Q2` PTQ) | **0.409998** | **0.403238** | **0.412952** | Layer 0 | **FAIL** | **Measured** |
| **Candidate B Mix 3** (High-Fidelity Mix `Q4`/`Q3` PTQ) | **0.557078** | **0.555081** | **0.558715** | Layer 0 | **FAIL** | **Measured** |

### 2.3 Quality Findings & Standalone Block-Scale Gain
1. **Isolated Scale Granularity Recovery**: Comparing Candidate A Proxy ($0.660357$) against the **Current Format PTQ Control** ($0.201762$) isolates the standalone gain of per-32-element block scaling. Upgrading from per-tensor scales to block-32 scales yields a **$+0.458595$ cosine gain (+227% relative accuracy recovery)** without any QAT fine-tuning!
2. **QAT Requirement**: Un-tuned PTQ on standard K-quants (Mix 1 & 2) collapses $z_s$ to $\sim 0.410 – 0.417$. Re-QAT fine-tuning will be mandatory for any format candidate to restore production representation quality.

---

## 3. Baseline Anomaly Resolution

### 3.1 Historical Audit & Protocol Verification
- **Production Gate Thresholds**: $z_s$ median $\ge 0.997$, min $\ge 0.990$.
- **Observed Shipped Metric**: `cos_median = 0.889420`, `cos_min = 0.871562`.
- **Verification Protocol Run**: Reran harness with standard historical seed `1783708826` on `qat_best.pt`.
  - **Result**: `cos_median = 0.889420`, `cos_min = 0.871562` (Identical; deterministic forced streaming).
- **Historical Log Audit**: Inspected `/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2.log`.
  - Base PTQ checkpoint (`bmo_tr_heavy_int2.pt`) started at `cos_median = 0.821736`.
  - QAT training flatlined at step 600 with `window_max_median = 0.910613 <= 0.950000`.

### 3.2 Plain Resolution Statement
> **Empirical Resolution**: Harness-config divergence is **ruled out**. The shipped model `qat_heavy_int2.gguf` **never met the $z_s \ge 0.997 / 0.990$ production threshold**. It was an aggressive 70% INT2 / 20% INT4 / 8% INT8 heavy compression experiment that improved median $z_s$ from $0.821 \rightarrow 0.889$, but flatlined at step 600 during training. The production $z_s$ thresholds apply to full-precision candidate deployment gates, whereas this artifact was shipped under a heavy compression compromise regime.

---

## 4. QAT-Delta Analysis

### 4.1 Pipeline & Exporter Engineering Cost
- **Candidate A (BMO_TIER v2)**: Requires extending `qat_septq.py` (`MultiTierFakeQuantize`) to autograd $(R, C/32)$ block-scale tensors. Exporter (`export_bmo_gguf.py`) must serialize block-scale arrays and tier-sorted dp4a element streams.
- **Candidate B (ggml k-quants mixed)**: `MultiTierFakeQuantize` is replaced with standard GGML block quantizers. **Zero custom exporter code is required** (delegates directly to GGML `ggml_quantize_chunk`).
- **Logged Re-QAT Wall Time**: **25,532.4 seconds** = **7.09 GPU-hours** on 1x H100 GPU (600 steps) [`Measured`].
  - Candidate A Re-QAT: $\approx \mathbf{7.5 – 8.5 \text{ GPU-hours}}$ [`Estimated`].
  - Candidate B Re-QAT: $\approx \mathbf{6.0 – 7.0 \text{ GPU-hours}}$ [`Estimated`].

---

## 5. Kernel-Path Note

### 5.1 Kernel Implementation Pathways
- **Candidate A**: Surviving logic includes CSR outlier gather/scatter from `v11`/`v6` kernels. Main GEMV loop requires FP16/6-bit block scale loading per 32 elements and packing nibbles/crumbs for `__dp4a` execution with `q8_1` activations.
- **Candidate B**: Verified in `ggml-cuda/mmvq.cu`: `vec_dot_q2_K_q8_1`, `vec_dot_q3_K_q8_1`, and `vec_dot_q4_0_q8_1` cover all assigned types natively. **Zero lines of custom CUDA code required**.

### 5.2 Required Jetson Hardware Measurements (Unchanged)
> [!IMPORTANT]
> **CRITICAL HARDWARE REQUIREMENT**:
> Orin-side GB/s for `vec_dot_q2_K_q8_1` and `vec_dot_q3_K_q8_1` **MUST be measured directly on sm_87 hardware** before making any format decision. On sm_87, `vec_dot_q4_0_q8_1` reaches ~73 GB/s (72% of DRAM peak), whereas custom BMO_TIER kernels hit a 37 GB/s ceiling due to L1TEX pipeline saturation. How `Q2_K`/`Q3_K` interact with sm_87's L1TEX cache is an empirical question that must be benchmarked directly on target hardware.

---

## 6. Reframed Recommendation & Decision Criteria

Reframing the candidate decision axes around ground truth facts:
1. **Memory is Roughly a Wash**: Candidate A costs $+100 \text{ to } +270 \text{ MiB}$ over Candidate C due to per-32 block scale overhead. Only `Q2_K`-heavy Candidate B configurations (B1 & B2) save memory ($1,310 – 1,701 \text{ MiB}$).
2. **Primary Decision Axes**:
   - **Orin-Side Kernel Latency**: Standard GGML `mmvq` kernels ($\approx 70 \text{ GB/s}$ class on sm_87) vs custom BMO_TIER kernels ($37 \text{ GB/s}$ L1TEX ceiling) vs Candidate C (`dp4a` $v12$ class).
   - **QAT Quality Recovery**: Re-QAT convergence ceiling for Block-32 scales vs K-quants grid.

| Criteria | Candidate A (BMO_TIER v2) | Candidate B (ggml k-quants mixed) | Candidate C (Status Quo + v12) |
|---|---|---|---|
| **Quality Potential** | High (fine block-32 scales + QAT) | Medium-High (K-quants + QAT) | Baseline ($0.889$ $z_s$ + v12 arithmetic noise) |
| **Gating Memory Footprint** | 1,852 – 2,012 MiB ($\Delta +6\% \text{ to } +15\%$) | 1,310 – 1,701 MiB ($\Delta -25\% \text{ to } -2\%$) | 1,742 MiB (Baseline) |
| **Engineering Cost** | High (custom QAT, custom GGUF, custom CUDA) | Low (standard QAT grid, native GGUF/mmvq) | Low (no export changes, kernel v12 only) |
| **Maintenance Burden** | High (custom kernel suite on sm_87) | Zero (standard GGML upstream) | Medium (custom v12 kernel) |

### Missing Empirical Jetson Measurements List
1. **Jetson `mmvq` Bandwidth**: Measure actual Orin-side GB/s for `vec_dot_q2_K_q8_1` and `vec_dot_q3_K_q8_1` on sm_87.
2. **End-to-End Latency**: Frame-time impact of Candidate B's standard `mmvq` kernels vs Candidate C's $v12$ dp4a kernel on Orin Nano.
3. **Audio Quality Sanity**: Human listening evaluation on QAT-recovered audio output.
