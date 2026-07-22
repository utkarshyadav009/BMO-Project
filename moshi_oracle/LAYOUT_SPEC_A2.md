# LAYOUT SPECIFICATION: CANDIDATE-A2 TRIAL PAYLOAD (LAYER 0)
**Project BMO — FFN Gating Matmul Candidate-A2 Trial Layout**
**Date:** July 22, 2026
**Author:** Server-side agent on LineBreaker (2x NVIDIA H100 PCIe)
**Target Platform:** Jetson Orin Nano 8GB (sm_87)
**Emitted Files:**
- [moshi_oracle/models/layer0_a2_in.bin](file:///home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models/layer0_a2_in.bin) (79.72 MiB)
- [moshi_oracle/models/layer0_a2_out.bin](file:///home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models/layer0_a2_out.bin) (39.83 MiB)
- [moshi_oracle/models/layer0_a2_ref.json](file:///home/jovyan/work/BMO-Project-Repo/BMO-Project/moshi_oracle/models/layer0_a2_ref.json)

---

## 1. Executive Summary & Purpose

This layout specification describes the binary layout format for **Candidate-A2** (BMO_TIER v2: 6-bit Block-32 Scales + Tier-Sorted dp4a-Native Packing) of Layer 0's two gating tensors (`linear_in.weight` $[22528, 4096]$ and `linear_out.weight` $[4096, 11264]$).

These binary payloads are generated for the Jetson Orin Nano 8GB (`sm_87`) benchmark agent to consume in CUDA GEMV microbenchmarking.

---

## 2. Binary Layout Specification (`.bin` File Structure)

Each `.bin` file consists of 5 sequential sections packed in little-endian binary format:

```
+-------------------------------------------------------------------+
| Section 1: Header (64 bytes)                                      |
+-------------------------------------------------------------------+
| Section 2: Column Permutations [n_tile_rows, n_tile_cols] uint16  |
+-------------------------------------------------------------------+
| Section 3: Band Stream Offsets [n_tile_rows, 4] uint32             |
+-------------------------------------------------------------------+
| Section 4: Packed Weight & Block Scale Streams (Variable bytes)   |
+-------------------------------------------------------------------+
| Section 5: CSR Outlier Arrays (Offsets + Cols + Values)            |
+-------------------------------------------------------------------+
```

### 2.1 Section 1: Header (64 Bytes)
| Field | C Type | Offset (Bytes) | Description |
|---|---|---:|---|
| `magic` | `uint32_t` | 0 | Magic identifier: `0x41324d4f` ("OMA2") |
| `rows` | `uint32_t` | 4 | Matrix rows $R$ (22528 for `in`, 4096 for `out`) |
| `cols` | `uint32_t` | 8 | Matrix cols $C$ (4096 for `in`, 11264 for `out`) |
| `tile_size` | `uint32_t` | 12 | Fixed tile dimension (64 elements) |
| `n_tile_rows` | `uint32_t` | 16 | Number of 64-row bands ($R / 64$) |
| `n_tile_cols` | `uint32_t` | 20 | Number of 64-column tile columns ($C / 64$) |
| `n_outliers` | `uint32_t` | 24 | CSR Outlier count |
| `reserved` | `uint8_t[36]` | 28 | Padding bytes to 64-byte total header size |

### 2.2 Section 2: Per-Band Column Permutations (`col_perms`)
- **Shape**: `[n_tile_rows, n_tile_cols]` of `uint16_t` values.
- **Purpose**: For each 64-row band $b$, columns are permuted so that tiles group contiguously by tier ($T_1 \rightarrow T_2 \rightarrow T_3 \rightarrow T_0$).
- **x-gather Lookup**: The Jetson CUDA GEMV kernel uses `col_perms[b][perm_idx]` to determine which 64-element block of activation vector $x$ feeds tile column `perm_idx`.

### 2.3 Section 3: Band Stream Offsets (`band_stream_offsets`)
- **Shape**: `[n_tile_rows, 4]` of `uint32_t` values.
- **Offsets**: Relative byte offsets within band $b$'s payload stream for `[off_t1, off_t2, off_t3, off_t0]`.

### 2.4 Section 4: Packed Weight & 6-Bit Block Scale Streams
Within each band, tiles are written contiguously in permuted tier order. Each $64 \times 64$ tile consists of **100 bytes of Block Scale Metadata** followed by the **dp4a-interleaved quantized weight payload**:

1. **Tile Block Scale Metadata (100 Bytes per Tile)**:
   - `s_min`: `float16` (2 bytes) — Minimum block scale value in tile.
   - `s_range`: `float16` (2 bytes) — Scale dynamic range ($\max(s) - \min(s)$).
   - `scale_indices_6bit`: 96 bytes (128 6-bit quantized scale indices packed 4 indices per 3 bytes).
     - *Dequantization formula*: $s_{\text{block\_k}} = s_{\min} + (idx_6bit / 63.0) \times s_{\text{range}}$.
2. **dp4a Weight Payload**:
   - **Tier 1 (INT2 - 2 bit)**: 1,024 bytes (4 contiguous $K$-elements per byte pairing with `__dp4a`).
   - **Tier 2 (INT4 - 4 bit)**: 2,048 bytes (2 contiguous $K$-elements per byte pairing with `__dp4a`).
   - **Tier 3 (INT8 - 8 bit)**: 4,096 bytes (1 byte per element).
   - **Tier 0 (FP16 - 16 bit)**: 8,192 bytes (2 bytes per element).

### 2.5 Section 5: CSR Outlier Arrays
- `csr_offsets`: `uint32_t[rows + 1]` array of row start indices.
- `outlier_cols`: `uint16_t[n_outliers]` array of row-sorted column indices.
- `outlier_vals`: `float16[n_outliers]` array of FP16 outlier values.

---

## 3. CPU Double-Accumulator Reference Verification

A fixed seed `1783708826` activation vector $x \sim \mathcal{N}(0, 1)$ was evaluated against the Candidate-A2 trial layout using CPU 64-bit float (`double`) accumulation:

$$y_{\text{ref\_double}} = \text{dequantized\_A2}(W_{\text{repacked}}) \cdot x + y_{\text{CSR\_outliers}}$$

### Reference Vector Checksums (`layer0_a2_ref.json`)
- **`linear_in` ($22528 \times 4096$)**:
  - $||x_{\text{in}}||_2 = 63.784534$
  - $||y_{\text{in\_ref}}||_2 = \mathbf{109.677736}$
  - $y_{\text{min}} = -15.938830, y_{\text{max}} = 16.326359$
- **`linear_out` ($4096 \times 11264$)**:
  - $||x_{\text{out}}||_2 = 105.908051$
  - $||y_{\text{out\_ref}}||_2 = \mathbf{70.880304}$
  - $y_{\text{min}} = -15.185293, y_{\text{max}} = 12.756281$

---

## 4. Orin sm_87 Benchmark Instructions for Jetson Agent
1. Load `layer0_a2_in.bin` or `layer0_a2_out.bin` directly into CUDA device memory.
2. Read header and per-band column permutations `col_perms`.
3. Launch Candidate-A2 dp4a GEMV CUDA kernel feeding `col_perms` to warp x-gather.
4. Verify kernel output $y_{\text{gpu}}$ against $y_{\text{ref\_double}}$ in `layer0_a2_ref.json` (`rel_l2 < 1e-4` gate).
