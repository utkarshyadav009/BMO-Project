# BMO Personaplex — Session Handoff

**Last updated:** 2026-05-17  
**Pickup transcript ID:** `c9c6ec77-3d61-4307-961a-d2ba4cdffadd`

When you resume on another system, paste this whole file into the new chat so the agent has full context. To continue the original chat verbatim use `[BMO RoPE Bug Investigation](c9c6ec77-3d61-4307-961a-d2ba4cdffadd)`.

**Contents**

1. [Systems-level architectural audit (mixed-precision runtime, Jetson)](#systems-level-architectural-audit-mixed-precision-runtime-jetson)
2. [Jetson offline / memory / immediate blockers](#jetson-offline--memory--immediate-blockers)
3. [Earlier session: gibberish audio / RoPE / SINE_TOKENS fix](#bmo-c-runtime-gibberish--session-handoff)

---

# Systems-Level Architectural Audit (Mixed-Precision Runtime, Jetson)

Expert-facing audit of the custom SEPTQ mixed-precision temporal transformer runtime on **NVIDIA Jetson Orin Nano 8GB** (Ampere, unified memory, `libbmo.so`). Grounded in current tree: `bmo_septq_v3.gguf`, `BMO_JETSON`, `packing_version=5`.

**Quantization policy (target):** ~2% FP16, ~12% INT8, ~32% INT4, ~50% INT2 — **per-weight (v5 per-element mask)**, not block-wise uniform.

---

## SECTION 1 — MODEL ARCHITECTURE

### Topology (Moshi / PersonaPlex 7B duplex)

| Subsystem | Role | Key dims (from runtime + export) |
|-----------|------|----------------------------------|
| **Mimi** | Neural codec: 24 kHz → ~12.5 Hz frames, 32 RVQ codebooks | SEANet `dimension=512`, quantizer `n_q=32`, `bins=2048` |
| **Temporal transformer** | Main duplex LM over text + multi-codebook audio tokens | `n_layers=32`, `n_embd=4096`, `n_heads=32`, `head_dim=128`, RoPE `theta=10000` |
| **Depth transformer (depformer)** | Per-frame autoregression over codebooks | `depth_n_layers=6`, `depth_hidden=1024`, `depth_n_heads=16`, `depth_head_dim=64`, `dep_q=16`, `num_codebooks=17` (text + 16 audio CBs) |
| **Heads** | Text + per-CB audio logits | `text_vocab=32000`, `audio_vocab=2048` (canonical output); input table 2049 rows (EPAD) |

**Temporal block (per layer `transformer_layers_{i}`):**

1. Pre-norm RMSNorm (`norm1_weight`, F32 gamma `[4096]`).
2. **Fused QKV linear** — `self_attn_in_proj_weight`: output `3×4096` (Q+K+V), input `4096`. Packed SEPTQ on Jetson.
3. RoPE on Q/K — **interleaved adjacent pairs**, `GGML_ROPE_TYPE_NORMAL` (not NeoX).
4. Self-attention — decode: **CPU eager** `apply_attention_eager_decode`; prefill: lazy `ggml_flash_attn_ext` path.
5. **Out proj** — `self_attn_out_proj_weight`: `[4096, 4096]` (packed or dense per export).
6. Residual add (GPU `apply_residual_add_gpu` on Jetson).
7. Pre-norm RMSNorm (`norm2_weight`).
8. **SwiGLU FFN** — `gating_linear_in_weight`: `[2×d_ff, 4096]` with `d_ff=11264` → `22528` rows (matches `gpu_staging_pool::SLOT_BYTES / sizeof(float)`).
9. `gating_linear_out_weight`: `[4096, d_ff]`.
10. Residual add.

**Final temporal:** `out_norm_weight` (RMSNorm γ, F32), then either `text_linear` `[32000×4096]` F16 or conditioning for depth.

**Depth block (per `depformer_layers_{i}`, dense F16):** same pattern at 1024-dim; **no** `packed_weights` in export — `ggml_mul_mat` on F16 weights.

**KV caches:**

- Temporal: FP16, shape `[head_dim, n_ctx, n_heads, n_layers]` → **512 MB** at `n_ctx=1024` (logged).
- Depth: FP16, `dep_ctx=16` (codebook steps), **384 KB** total.
- Reset: temporal grows with `n_past`; depth reset each temporal frame, grows `cb_index=0..dep_q-1`.

**Streaming / scheduling:**

- Frame rate **12.5 Hz** (80 ms/frame).
- Per frame: Mimi encode → temporal step(s) → for each codebook step `k`, depth forward → sample audio token → Mimi decode agent channels.
- `LMGen` + `streaming_forever`; C++ path: `bmo_forward` / patched `LMModel.forward_codes` with `n_token==1` decode hot path.
- `NO_CUDA_GRAPH=1` on Jetson offline — **no** CUDA graphs for temporal; eager kernels + deferred `cudaStreamSynchronize(0)` at layer boundaries.

**What is quantized (temporal only):**

Per layer, **four** matrices (when exported): `self_attn_in_proj_weight`, `self_attn_out_proj_weight`, `gating_linear_in_weight`, `gating_linear_out_weight` → **128 packed tensors** for 32 layers.

**What stays dense:**

- All **depth** weights (6 layers × attn + FFN).
- `depformer_in.{k}`, `depformer_emb.{k}`, `linears.{k}`, `text_emb`, `temporal_text_emb`, `out_norm`, norm gammas, biases.
- `text_linear` exported F16 dense (~250 MB class).

**Why FP16 pockets (~2% in assignment policy):**

Sensitivity-driven **per-element** (v5) or per-block (v4) tier `0` stores raw F16 in `fp16_values[]`; high-magnitude or outlier weights from PTQ/QAT `tier_masks_uint2` / block tier maps.

**Parameter / memory budget (typical Jetson run):**

| Pool | Size |
|------|------|
| GGUF weight bytes (accounted) | ~1798 MB |
| Host VmRSS after prepare | ~6.5 GB (weights + Python moshi shell + KV + 1 GiB work arena + staging) |
| KV temporal | 512 MB |
| Work arena | 1 GiB (`kDefaultWorkMem` in `bmo_api.cpp`) |
| GPU VRAM (Orin) | 7619 MiB reported; most weights accessed via **mapped host**, not separate `cudaMalloc` weight copies |

---

## SECTION 2 — QUANTIZATION PIPELINE

### Assignment (export: `export_bmo_gguf.py` → `create_packed_layer`)

**Static at export time** — not runtime-dynamic.

**Sources (priority):**

1. **v5 per-element:** `tier_masks_uint2` from PTQ/QAT checkpoint, byte length `(rows × padded_cols + 3) / 4`. Serialized **verbatim** — no re-tiering.
2. **v4 per-block:** `block_tier_map` or `max(abs)` thresholds / ratio ranking inside 32-wide blocks.
3. **Tier encoding in `packed_mask`:** 2 bits per slot, little-endian nibble order:  
   `0=FP16`, `1=INT8`, `2=INT4`, `3=INT2` (mask value `m` → storage tier `3-m` when deriving from external maps).

**Sensitivity:**

- v5: per-weight mask from training/PTQ (`tier_masks_uint2`).
- v4 fallback: `block_max = max(abs(block))` vs `threshold_8bit/4bit/2bit` or top‑k block ratios (`fp16_ratio_real`, etc.).

**Affine dequant (per layer, global — not per-block scale in kernel):**

```
w = (q - zp_low)   * scale_low    # INT2, q ∈ [0,3]
w = (q - zp_int4)  * scale_int4   # INT4, q ∈ [0,15]
w = (q - zp_int8)  * scale_int8   # INT8
w = fp16_values[i]                 # tier 0
```

Defaults if meta missing: `zp_low=1.5`, `zp_int4=7.5`, `zp_int8=127.5`.

### Physical layout (NOT interleaved per weight)

For each matrix `base` (e.g. `transformer_layers_0_self_attn_in_proj_weight`):

```
GGUF tensors:
  base.packed_weights   uint8[ n_2bit + n_4bit + n_8bit ]
  base.packed_mask      uint8[ mask_bytes ]
  base.fp16_values      f16[ count_tier0 ]
  base.scale_* / base.zp_*   f32 scalars
  base.rows, base.cols, base.block_size (=32), base.packing_version (=5)
  base.n_2bit_bytes, n_4bit_bytes, n_8bit_bytes
```

**`packed_weights` byte layout (contiguous tiers):**

```
[ stream_INT2 (n_2bit_bytes) | stream_INT4 (n_4bit_bytes) | stream_INT8 (n_8bit_bytes) ]
```

- **INT2 pack:** 4 values/byte, 2 bits each, LE lane `value[i] → bits (2*(i%4))`.
- **INT4 pack:** 2 values/byte, low/high nibble.
- **INT8:** raw bytes.

**`packed_mask` v5 (per-element):**

- One uint2 per **matrix element** `(row, col)`.
- Packed **4 tiers per byte**: element index `ei` → byte `ei/4`, shift `(ei%4)*2`.

**NOT** a separate sidecar bitfield per weight inside `packed_weights`; tier lives only in `packed_mask`.

**`fp16_values`:** compact array of **only** tier-0 elements, row-major walk order matching mask scan (export verifies `sum(tier==0) == len(fp16_values)`).

**Row prefix tables (runtime, device `int32*`):**

Built in `bmo_prepare_device_packed_tensors` (`bmo.cpp`):

- `row_c2[r]`, `row_c4[r]`, `row_c8[r]`, `row_c16[r]` = cumulative **stream indices** at start of row `r` for each tier’s packed stream (v5: count elements per tier while scanning mask).

**Tile alignment:** `block_size=32` for v4 block metadata and for v2 kernel’s `cols/32` iterations; v5 mask is **per-element** — `block_size` still drives loop count `n_iters = cols >> 5` (assumes `cols % 32 == 0`).

**Metadata adjacency:** mask and weights are **separate arrays**; no struct-of-array fusion. GPU reads both streams → **cache pollution** on large matrices.

---

## SECTION 3 — RUNTIME INFERENCE PIPELINE

### End-to-end (Jetson, `BMO_USE_CPP=1`)

```
Audio PCM → Mimi.encode (CPU) → codes [B, 32, T]
  → embed (Python/C++): sum temporal_audio_embs[cb] + text_emb
  → bmo_build_temporal_graph (per token or prefill chunk)
```

**Per temporal layer, per token (`n_token==1`):**

1. **RMSNorm** — `rmsnorm_kernel`: 256 threads, warp shuffle reduce, staging slot.
2. **Linear QKV** — `apply_linear_with_transient_unpack`:
   - Map `x` via `cudaHostGetDevicePointer` or memcpy to `cuda_fused_input_buffer` (pinned).
   - `launch_fused_dequant_matvec_jetson` → proto (v5) or v2 kernel.
   - Output → **staging slot** host alias (`out_lm->data = out_slot.host`).
   - Optional `ggml_add` bias on CPU graph buffer.
3. **`cudaStreamSynchronize(0)`** before attention.
4. **RoPE** — GPU `launch_rope_interleaved`.
5. **Attention** — **`apply_attention_eager_decode` on CPU**: read/write FP16 KV in **host** memory; nested loops over heads, `kv_len`, `head_dim`.
6. **Out proj** — another fused matvec (same as step 2).
7. **Residual add** — GPU.
8. **FFN** — norm → fused `gating_linear_in` → GPU SwiGLU → fused `gating_linear_out` → residual.
9. **`release_all_staging()`** at layer end.

**Weights:** not fully unpacked to dense GEMM. **Fused dequant+dot** per output row. No full `W'[rows,cols]` materialization on Jetson (non-Jetson can use `unpack_kernel` + `ggml_mul_mat`).

**Dequant location:** inside matvec kernel registers (per lane), not Tensor Cores.

**Temp buffers:**

- 32× pinned staging slots (88 KB each).
- Fused in/out buffers (~44 KB / ~86 KB pinned).
- 1 GiB `work_mem` for ggml tensor metadata + activations.
- Multi-token prefill: per-token sync + memcpy to `wctx` (`apply_linear` slow path).

**Depth (same frame, 16 steps):** dense `ggml_mul_mat` + lazy flash attention on depth KV; smaller tensors.

**Streams:** default stream `0`; kernels launched with `stream=nullptr` → stream 0. Layer-end sync drains GPU before CPU attention.

---

## SECTION 4 — CUDA KERNEL ARCHITECTURE

### 4.1 `fused_dequant_matvec_kernel_proto` (v5, production on `bmo_septq_v3.gguf`)

**Purpose:** `y = W @ x` with `W` stored SEPTQ; one output row per warp.

| Parameter | Value |
|-----------|--------|
| `ROWS_PER_BLOCK` | 8 (launch) or 4 |
| `threads/block` | `ROWS_PER_BLOCK × 32` = **256** |
| `blocks/grid` | `(rows + 7) / 8` |
| Warp mapping | `row_in_block = tid >> 5`, `lane = tid & 31` → **1 warp = 1 row** |
| Shared memory | **0** dynamic smem |
| Tensor Cores / WMMA / ldmatrix | **None** |
| cp.async | **None** |

**Hot loop (conceptual — from `bmo_cuda_kernels_proto.cu`):**

```cuda
for (k = 0; k < cols/32; ++k) {
  c = k*32 + lane;
  elem_i = row*cols + c;
  tier = (pm_elem[elem_i>>2] >> ((elem_i&3)*2)) & 3;

  // ballot: rank within warp for this tier's stream
  bm[t] = __ballot_sync(0xFFFFFFFF, tier == t);
  rank = popc(bm[tier] & ((1u<<lane)-1));
  base = (tier==0)?o16 : (tier==1)?o8 : (tier==2)?o4 : o2;
  off = base + rank;

  // scalar decode
  if (tier==0) w = half2float(fp16_vals[off]);
  else if (tier==1) w = (stream8[off]-zp8)*s8;
  else if (tier==2) { nibble extract from stream4[off>>1]; ... }
  else { 2-bit extract from stream2[off>>2]; ... }

  acc += w * x[c];

  // advance row-global stream cursors by warp-wide tier counts
  o2 += popc(bm[3]); o4 += popc(bm[2]); o8 += popc(bm[1]); o16 += popc(bm[0]);
}
// warp shuffle reduce acc → y[row]
```

**Divergence:** **4-way tier branch every iteration per lane**; lanes in a warp often differ (v5 per-element) → **warp divergence + 4× `__ballot_sync` + `__popc` per 32 elements**.

### 4.2 `fused_dequant_matvec_kernel_v2` (v4 block mask)

| Parameter | Value |
|-----------|--------|
| threads/block | 256 (8 rows × 32 lanes) |
| Dynamic smem | `8 * blocks_per_row` tier bytes + `8 * blocks_per_row` int32 offsets |
| Tier load | Lane `b` loads `s_tier[b]` from **block-granular** mask (`pm[block_global>>2]`) |
| Inner loop | `tier = s_tier[k]` where `k = col/32` → **all 32 lanes share tier** for that column block |

**Divergence:** tier branch still 4-way, but **uniform within warp** per `k` → much better than v5.

### 4.3 `unpack_kernel` (full dense expand)

- 1D grid, 256 threads, one thread per `(row,col)`.
- Used off Jetson path / debug; **not** hot path on Orin production decode.

### 4.4 `rmsnorm_kernel`, `rope` kernels, `swiglu_split_kernel`, `residual_add`

- RMSNorm: multi-warp block, shuffle reduction.
- RoPE: interleaved pairs on GPU staging buffers.
- SwiGLU: 256-thread 1D, `silu(gate)*up`.
- All **scalar FP32** math; no WMMA.

### 4.5 Attention

**No custom CUDA attention kernel on Jetson decode.** `apply_attention_eager_decode` is **host C++** triple nested loops + `exp` softmax.

---

## SECTION 5 — MEMORY LAYOUT + DATAFLOW

### Host weight residency (`bmo_load_model`)

```
pread(GGUF) → posix_memalign big_pool + scalar_pool
  → cudaHostRegister(big_pool, MAPPED|PORTABLE)  [Jetson]
  → tensor->data points into pool slices
```

**Not** lazy mmap page-in for weights (despite `gguf_mmap` field — load path uses `pread`).

**Device view:** `cudaHostGetDevicePointer` → `canonical_pw_dev`, `canonical_pm_dev`, `canonical_fv_dev` — GPU loads via **PCIe/SoC mapped pinned host** (unified memory on Orin, but still bandwidth-limited).

**Row tables:** `cudaMalloc` + `cudaMemcpy` for `row_c2/c4/c8/c16` (device-only, small).

### Activation dataflow (single token)

```
wctx F32 x[4096] (host, may be staging alias)
  → x_dev (mapped or fused_input_buffer)
  → fused matvec reads W from mapped host + mask from mapped host
  → y_dev in staging slot (pinned)
  → attn reads q,k in staging (CPU)
  → KV cache FP16 in kv_mem (host)
```

### Coalescing / alignment

- Matvec: lane `c` strides 32 — **coalesced** `x[c]` if `x` device-contiguous.
- Weight streams: `off` computed per-lane (v5) — **scattered** reads into `stream2/4/8` and `fp16_vals` → poor coalescing.
- Mask: per-lane byte index `elem_i/4` — scattered uint8 loads.

### Cache / UMA

- L2 helps repeated layers; **cold streams** per matrix still large.
- Mask + weights + fp16_values **three streams** → D-cache / L2 thrash on big layers (e.g. `12288×4096` in_proj).
- **No** `cudaMallocManaged` for weights on Jetson — explicit pinned register.

### Diagram (one matvec row)

```
Host (pinned big_pool)
  packed_weights [c2|c4|c8]
  packed_mask    [uint2 × elems]
  fp16_values    [tier0 only]
        │ cudaHostGetDevicePointer
        ▼
GPU warp (row r)
  lane 0..31: tier mask → ballot → off → load q → FMA with x[c]
        │
        ▼
staging slot y[r] (pinned)
```

---

## SECTION 6 — INT2 EXECUTION DETAILS

### Representation

- Quantized value `q3 ∈ {0,1,2,3}` stored in stream2.
- **4 weights per byte:** index `idx2 = off` (stream index, not column index), byte `stream2[idx2/4]`, shift `(idx2%4)*2`, mask `& 0x3`.

### Unpack in hot loop (proto, tier==3)

```cuda
const int idx = off;  // stream index after ballot rank
const uint8_t bb = stream2[idx >> 2];
const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
w = ((float)q - zp_low) * scale_low;
```

**Instruction count (order of magnitude per weight):** load byte → shift/mask → int→float → FMA + (v5) ballot overhead amortized over warp.

**Widening:** widened to **FP32** in registers for `acc += w * x[c]`. **Not** fed to INT8 Tensor Core paths.

**Tensor Cores:** **cannot** consume INT2 layout directly; no `mma.sync`, no `ldmatrix`.

### v2 block path INT2 (uniform tier per 32 cols)

```cuda
const int idx = off + lane;
const uint8_t bb = stream2[idx >> 2];
const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
```

Here `off` is block-base stream offset; `lane` selects element within block.

### Warp divergence (INT2-specific)

- **v5:** INT2 lanes mixed with FP16/INT8/INT4 in same warp → **severe** (ballot + 4-way branch).
- **v4:** entire warp INT2 for that 32-column block → branch uniform; only stream gather differs by `lane`.

### Scalar vs vector unpack

**Scalar per lane** — no warp-wide `uint4` vectorized decode of packed bits.

---

## SECTION 7 — PERFORMANCE BOTTLENECK ANALYSIS

### Where time goes (honest, Jetson decode `n_token=1`, `n_past` growing)

| Component | Bound | Severity |
|-----------|--------|----------|
| **Fused SEPTQ matvec (×4/layer ×32)** | Decode + memory (mask + scattered stream idx) | **High** — dominates when layers are wide |
| **CPU eager attention** | Compute on CPU, KV FP16 host | **High** — `O(n_heads × kv_len × head_dim)`; at `kv_len≈1024` this hurts |
| **Per-layer `cudaStreamSynchronize(0)`** | Latency | **Medium** |
| **`[h2_diag_matvec]` logging** | I/O | **High** if enabled (unconditional in `launch_fused_dequant_matvec_jetson`) |
| RoPE / RMSNorm / SwiGLU GPU | Moderate | Lower than above |
| Depth dense `ggml_mul_mat` ×6 ×16 steps | F16 GEMM (ggml CUDA?) | **Medium** per frame |
| Mimi CPU | Separate | Competes for **~200 MB** free RAM |

**Tensor Core utilization:** ~**0%** on temporal quantized matmul (scalar FMA). Depth may use ggml CUDA MM occasionally — not the custom SEPTQ path.

**Dominant architectural issue:** **Fused scalar dequant matvec + per-element v5 ballot** instead of block-uniform tiers or prepacked INT8/FP16 tiles for TC.

**Second:** **CPU attention** while GPU idle after sync.

**Third:** **Unified memory pressure** — 6.5 GB RSS + 512 MB KV + 1 GB arena leaves no room for Mimi on GPU (warmup device mismatch).

**Decode-bound vs compute-bound:** matvec is **decode-bound** (bit extract + branch); attention is **compute-bound on CPU**.

**Worst mistakes for this silicon:**

1. Per-element v5 mask on Orin without TC-backed GEMM.
2. Full `[h2_diag_*]` on every matvec in production builds.
3. CPU attention on every layer every token.
4. Expecting INT2→TC without layout rewrite.

**Depth INT8-only idea:** depth is **dense F16 today** (~hundreds of MB–1 GB+ in GGUF). Uniform INT8 depth cuts depth weight bytes ~2× and might free GPU for Mimi, but needs **re-export + kernel path** (currently `ggml_mul_mat` F16). It does **not** fix temporal matvec cost.

### Redesign directions (TC-focused rewrite)

1. **Drop per-element v5 in inference** — use block-uniform v4 in export for runtime, or **pre-dequant INT8 tile** per 128×128 block for `mma.sync`.
2. **GPU flash-attn** for temporal decode — stop CPU `apply_attention_eager_decode` at `kv_len>64`.
3. **Separate mask from weight streams** — or pack tier+payload in 32-byte blocks aligned for `ld.global.v4`.
4. **Depth INT8** — separate project: re-QAT depth only, `ggml_mul_mat_q8_0` or custom — saves RAM, does not fix temporal matvec.

---

## SECTION 8 — CURRENT OPTIMIZATION ATTEMPTS

| Attempt | Outcome |
|---------|---------|
| **SEPTQ multi-tier + GGUF export** | Works; ~1.8 GB temporal weights |
| **Fused dequant matvec v2** (block mask, smem tier table) | Works; v4 GGUF |
| **Proto v5** (per-element + ballot) | **Deployed** (`packing_version=5`); higher divergence, matches PTQ masks |
| **Pinned `cudaHostRegister` big pool** | Works; avoids duplicate weight VRAM copy |
| **GPU staging pool (32 slots)** | Works; avoids post-matvec memcpy on decode |
| **Eager GPU RMSNorm / RoPE / SwiGLU / residual** | Works; replaced lazy ggml race on Jetson |
| **Eager CPU attention** | Correctness win; **perf loss** vs flash |
| **Deferred stream sync** | Latency win vs sync-per-kernel |
| **CUDA graphs (`NO_CUDA_GRAPH=1`)** | Disabled on Jetson offline — graphs hung with meta-weights |
| **TensorRT** | Not integrated for SEPTQ custom layout |
| **`unpack_kernel` + dense GEMM** | Non-Jetson fallback; more memory |
| **Empty `accelerate` LM shell** | Fixes OOM loading full 7B dummy weights |
| **Mimi on CPU after GGUF** | Fits memory; needs **`.to(mimi.device)` on tokens** in warmup/decode |
| **RoPE NeoX flip experiment** | **Rejected** — wrong for Moshi layout |

---

## SECTION 9 — CODEBASE STRUCTURE

```
personaplex/
  bmo.h                    device_packed_t, pools, model/context
  bmo.cpp                  GGUF pread load, prepare packed, KV init
  bmo_compute.cpp          temporal/depth graphs, apply_linear, eager attn
  bmo_api.cpp              C ABI, work_mem 1GiB, mutex
  bmo_cuda_kernels.cu      v2 fused, unpack, rmsnorm, rope, swiglu
  bmo_cuda_kernels_proto.cu v5 proto fused matvec
  bmo_proto_kernels.h      proto launch decl
  export_bmo_gguf.py       SEPTQ pack, tier masks, GGUF writer
  bmo_inference.py         Python driver + patch_lm_for_bmo
  moshi/moshi/offline.py   offline test harness
  moshi/moshi/models/lm.py patch hooks
  moshi/moshi/models/loaders.py  empty_weights shell when BMO_USE_CPP=1
  CMakeLists.txt           BMO_JETSON, BMO_ENABLE_CUDA, bmo_shared
  scripts/generate_report_figures.py  report figs (tier heatmaps, etc.)
  REPORT_C_CPP_JETSON_CURRENT_STATE.md  report narrative (pread + cudaHostRegister)
```

**Launch sites:**

- `launch_fused_dequant_matvec_jetson` ← `apply_linear_with_transient_unpack` (`bmo_compute.cpp` ~1404).
- Metadata build: `bmo_prepare_device_packed_tensors` (`bmo.cpp` ~580+).
- Graph entry: `bmo_build_temporal_graph` / `bmo_api` forward.

**Unpack:**

- Production: inside `fused_dequant_matvec_kernel_{v2,proto}`.
- Debug/reference: `unpack_layer_to_f32_blockwise` (host, `bmo_compute.cpp`), `unpack_kernel` (CUDA).

---

## SECTION 10 — MOST IMPORTANT RAW CODE

### Tier mask extract (v5, device)

```cuda
// bmo_cuda_kernels_proto.cu
const uint8_t mbyte = pm_elem[elem_i >> 2];
const uint8_t tier = (mbyte >> ((elem_i & 3) * 2)) & 0x3;
```

### INT2 decode + ballot rank (v5 hot path)

```cuda
uint32_t bm[4];
for (int t = 0; t < 4; ++t) {
    bm[t] = __ballot_sync(0xffffffffu, (int) (tier == t));
}
const uint32_t lane_mask = (1u << lane) - 1u;
const int rank = __popc(bm[tier] & lane_mask);
// base from o2/o4/o8/o16; off = base + rank;
// INT2:
const uint8_t bb = stream2[idx >> 2];
const uint8_t q = (bb >> ((idx & 3) * 2)) & 0x3;
w = ((float) q - zp_low) * scale_low;
acc += w * x[c];
```

### Dispatch (v5 vs v2)

```cpp
// bmo_compute.cpp — launch_fused_dequant_matvec_jetson
if (dp.packing_version >= 5) {
    launch_fused_dequant_matvec_proto(..., dp.row_c16, ..., 8, stream);
} else {
    launch_fused_dequant_matvec(...);  // v2 + shared-memory block prefix
}
```

### Packed weight stream layout (export)

```python
# export_bmo_gguf.py
packed_2 = pack_2bit_values_le(q3_vals)
packed_4 = pack_4bit_values_le(q2_vals)
packed_8 = q1_vals.view(np.uint8)
packed_weights = np.concatenate([packed_2, packed_4, packed_8])
```

### Jetson mapped weights (no device copy)

```cpp
// bmo.cpp — bmo_prepare_device_packed_tensors
cudaHostGetDevicePointer(&pw_dev, dp.host_packed_weights, 0);
cudaHostGetDevicePointer(&pm_dev, dp.host_packed_mask, 0);
dp.canonical_pw_dev = pw_dev;
dp.canonical_pm_dev = pm_dev;
dp.preloaded = true;
```

### v2 block-uniform tier (lower divergence)

```cuda
// bmo_cuda_kernels.cu — inner loop
const uint8_t tier = s_tier[k];  // k = col/32, uniform in warp
const int off = s_off[k];
// INT2: idx = off + lane; extract 2 bits ...
acc += w * x[c];
```

### `device_packed_t` (Jetson)

```cpp
// bmo.h
struct device_packed_t {
    void * host_packed_weights;
    void * host_packed_mask;
    void * host_fp16_values;
    void * canonical_pw_dev;
    void * canonical_pm_dev;
    void * canonical_fv_dev;
    int32_t * row_c2, * row_c4, * row_c8, * row_c16;
    int32_t packing_version;  // 5 = per-element mask
    // scale_low, scale_int4, scale_int8, zp_* ...
};
```

---

# Jetson Offline / Memory / Immediate Blockers

## What works on Jetson (verified)

- `libbmo.so` builds: `-DBMO_ENABLE_CUDA=ON -DBMO_TARGET_JETSON=ON`, target `bmo_shared`.
- Standalone `BMOEngine(gguf, n_ctx=1024)`: load + `bmo_prepare_device_packed_tensors` + 512 MB KV completes (~6.5 GB VmRSS).
- Isolation: `get_moshi_lm(None)` → `BMOEngine` → `patch_lm_for_bmo` → `LMGen` → `streaming_forever` → **ok** (no Mimi).
- `moshi.offline` with `BMO_USE_CPP=1` reaches: GGUF engaged, Mimi on CPU, LMGen warmup — then crashes (below).

## Env for Jetson offline test

```bash
export PYTHONPATH=./moshi BMO_USE_CPP=1
export BMO_SO_PATH=./build_jetson/libbmo.so
export BMO_GGUF=./models/bmo_septq_v3.gguf
export BMO_N_CTX=1024
# Do NOT set BMO_MIMI_DEVICE=cuda without freeing BMO memory
export NO_CUDA_GRAPH=1
```

## Load order fixes already applied (sync these files)

| File | Change |
|------|--------|
| `moshi/moshi/offline.py` | **BMO_USE_CPP load order:** tokenizer → LM shell → **GGUF + patch** → **Mimi on CPU**. `BMO_MIMI_DEVICE` default `cpu`, `BMO_SINGLE_MIMI=1`. |
| `moshi/moshi/models/loaders.py` | `BMO_USE_CPP=1` → `accelerate.init_empty_weights()` LM shell (no ~7B dummy GPU weights). |
| `moshi/moshi/models/lm.py` | `device` property uses `_bmo_activation_device`. |
| `bmo_inference.py` | `patch_lm_for_bmo(..., activation_device=...)`. |
| `CMakeLists.txt` | `BMO_BUILD_TESTS=OFF` default (missing test `.cu` files). |

Build:

```bash
cmake -B build_jetson -S . -DBMO_ENABLE_CUDA=ON -DBMO_TARGET_JETSON=ON
cmake --build build_jetson -j$(nproc) --target bmo_shared
```

## Current blocker: warmup device mismatch

After GGUF + CPU Mimi load, **warmup crashes**:

```
RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0,
different from other tensors on cpu
```

**Stack:** `warmup` → `mimi.decode(tokens[:, 1:9])` — `LMGen` / patched forward returns tokens on **CUDA**; Mimi weights on **CPU**.

**Fix direction:** `tokens = tokens.to(mimi_device)` before `mimi.decode` in `warmup` and `decode_tokens_to_pcm` (`moshi/moshi/offline.py`).

**Diag spam:** `[h2_diag_matvec]` / `[h2_diag_register]` from `launch_fused_dequant_matvec_jetson` — gate behind env (e.g. `BMO_H2_DIAG=1`) for production tests.

## Why offline hung before fixes

1. `get_moshi_lm(None, device=cuda)` materialized full LM on GPU → OOM.
2. Mimi ×2 loaded before GGUF → unified memory exhausted.
3. Old order: `LMGen` before `BMOEngine` + patch → CUDA graph / meta-weight hang.
4. Ctrl+C / SSH drop under memory pressure (OOM killer / GPU reset).

## Report / docs corrections (from review)

- Loader narrative: **pread + `cudaHostRegister`**, not lazy mmap zero-copy.
- Fused kernel **does** use shared memory for tier staging on **v2**; v5 proto uses **0** dynamic smem.
- Figure 4: per-tier fraction panels, not dominant-tier heatmap.
- Jetson memory: 7.69 GB GGUF + KV + 1 GiB arena exceeds nominal RAM; explain swap/zram honestly in report.

## User ideas not yet implemented

1. **Depth transformer uniform INT8** (no mixed tier) — re-export + C++ path; may shrink GGUF and allow Mimi on GPU; does not fix temporal matvec.
2. **KV compression paper** (e.g. TurboQuant) — not integrated.
3. Gate **`BMO_H2_DIAG`** for matvec register logs.

---

# BMO C++ Runtime Gibberish — Session Handoff

**Last updated (gibberish track):** 2026-05-12 10:23 UTC+1 (SINE_TOKENS fix applied, ready to test)

---

## ✅ FIX APPLIED — TEST FIRST THING (user-channel prefill)

The user-channel prefill bug has been patched in `bmo_inference.py`. **No C++ rebuild needed.**

### What changed in `bmo_inference.py`

1. **Top-of-module comment** (lines ~145-158): replaced the "BMO empirically requires SILENCE_TOKENS on user" narrative with the actual semantics (SINE for prefill user channel, SILENCE for generation-time mimi-zero-pad fallback).
2. **`mode_stream` docstring** (around line 800): updated the phase table to show `user channels = SINE_TOKENS` for phases 1-4.
3. **Phase-prep block** (~line 941-952): now defines BOTH `user_sine_tokens` (SINE) for prefill and `user_silence_tokens` (SILENCE) for the generation-loop fallback.
4. **Prefill call sites** (phase 1/2/3/4, lines ~974/982/992/999): all four `_prefill_one(...)` calls now pass `user_sine_tokens` instead of `user_silence_tokens`. This matches `moshi/models/lm.py:_encode_sine_frame()` called from `_step_voice_prompt_frame` (lm.py:1097-1106) and `_step_audio_silence_core` (lm.py:1161-1170) and `_step_text_prompt_core` (lm.py:1183-1191).
5. **Generation-loop fallback** (line ~1021) still uses `user_silence_tokens` — this is correct because at generation time when `--input-wav` is absent/exhausted, the user channel should look like mimi-encoded silence (matches what `lm_iterate_audio(..., pad=True)` produces in moshi.offline when the wav runs out).

File passes `python -m py_compile bmo_inference.py` cleanly.

### How to test on the other system

Just run the same audio-generation command that produced gibberish yesterday. No rebuild, no resync of `libbmo.so`. Pull `bmo_inference.py` and go.

```bash
# from /home/jovyan/work/BMO-Project/personaplex_repo (or your equivalent)
git pull
export PYTHONPATH=$PWD:$PWD/moshi BMO_SO_PATH=$PWD/build/libbmo.so
# then the exact `python bmo_inference.py ...` command you ran yesterday
```

Listen to the output WAV.

### Decision rules after listening

- **Coherent audio** → runtime was fine all along, Python driver was wrong. Diff `bmo_inference.py` and consider also auditing (a) `PERSONAPLEX_DELAYS` at lines ~167-168 against `_lm_kwargs["delays"]` in `moshi/models/loaders.py`, and (b) `--force-text-pad` clamp around line ~1050 — both might be similar layered workarounds, though they may not bite if the fix above is the whole story.
- **Still gibberish but different in character** → partial fix; combine with the RoPE/K-cache investigation below.
- **Still identical gibberish** → user-channel wasn't the bug (unlikely given the evidence) or there's an additional bug. Move to the worker plan below.

---

## The one-line status

GGUF + C++ runtime produces gibberish audio. PT (.pt / .safetensors) produces coherent audio. Text logits in C++ output are **static across frames** — canonical symptom of attention being unable to see context, OR of the model being fed OOD inputs during prefill (which the user-channel token bug above would cause).

## The latest plot twist (READ THIS FIRST)

Gemini gave a diagnosis that the bug is "split-brain RoPE": multi-token prefill uses `ggml_rope(..., GGML_ROPE_TYPE_NORMAL)` (claimed half-split layout) while single-token decode uses `launch_rope_interleaved` (interleaved layout). Gemini said: flip every `GGML_ROPE_TYPE_NORMAL` to `GGML_ROPE_TYPE_NEOX` via `ggml_rope_custom(..., ctx.rope_theta, ...)`.

**Gemini is wrong. Do not apply that fix.** Proof, from the source:

```
llama.cpp/ggml/include/ggml.h:1796-1801
  GGML_ROPE_TYPE_NORMAL  n_dims = 8 --> [cscscscs]   (adjacent-pair interleaved)
  GGML_ROPE_TYPE_NEOX    n_dims = 8 --> [ccccssss]   (split-half)
```

Cross-checked:

- `moshi/moshi/modules/rope.py:67-72` uses `q.view(D//2, 2)` then `[..., 0]` and `[..., 1]` → **adjacent-pair**.
- `bmo_cuda_kernels.cu:336-343` (`rope_interleaved_kernel`) reads `x_head[2*i]` and `x_head[2*i+1]` → **adjacent-pair**.
- `llama.cpp/ggml/src/ggml-cuda/rope.cu:109-112` (`rope_norm`) reads `x[ix+0]` and `x[ix+1]` → **adjacent-pair**.
- `rope_neox` at lines 175-176 reads `x[ix+0]` and `x[ix+n_dims/2]` → split-half (LLaMA/Falcon/NeoX style).

Conclusion: `NORMAL` is the correct constant for moshi's layout. Flipping to NEOX would break the prefill K cache, not fix it. Theta is also fine — model loads `rope_theta=10000.0`, which is `ggml_rope()`'s default. So both layers Gemini flagged are non-issues.

**Bug is real, but Gemini's localization is wrong.** Still hunting.

## Background worker status

The previous worker (RoPE consistency test + K-cache dump+compare script) was **closed by the user before completing** and produced no deliverables. Do not re-launch unless the SINE_TOKENS fix above fails to produce coherent audio. If it fails, the original worker prompt is preserved in the transcript and can be re-issued; the plan is unchanged:

- **Deliverable 1:** standalone CUDA/C++ test `tests/rope_consistency_test.cpp` + CMake target `bmo_rope_consistency_test`. Compares `ggml_rope(NORMAL)` output to `launch_rope_interleaved` output for identical synthetic input. Exit 0 = math matches.
- **Deliverable 2:** `scripts/dump_kcache_after_prefill.py` + a new debug C-API entry `bmo_get_kcache_layer(int layer, float* out, int max_elements)`. Runs identical small prefill in PT and C++, dumps K-cache layer 0 and 16, reports per-head cosine.

## Decision tree (apply once worker returns results)

| rope_consistency_test | K-cache cosine | Conclusion |
|---|---|---|
| FAIL (exit 1) | n/a | RoPE math diverges between multi-token ggml fallback and single-token CUDA kernel. Fix is NOT NORMAL→NEOX. Look for stride/position mismatch in `apply_rope_gpu_interleaved` (`bmo_compute.cpp:542-599`). |
| PASS | < 0.99 | Prefill produces a bad K cache. Inspect: positions passed during prefill, KV cache write path, attention mask. Start at `bmo_build_temporal_graph` around `bmo_compute.cpp:2100-2140`. |
| PASS | ≥ 0.99 | Prefill is fine. Bug is downstream: decode-side Q computation, attention compute, out_proj, depth transformer, or sampler. |

## Hard rules for the next session

1. **Try the 🚨 lead first** (SILENCE_TOKENS → SINE_TOKENS on user channel during prefill). Five minutes, no rebuild, decisive.
2. **Do not** apply Gemini's `GGML_ROPE_TYPE_NORMAL` → `GGML_ROPE_TYPE_NEOX` flip.
3. **Do not** change `rope_theta` handling. The model uses 10000 and `ggml_rope`'s default is 10000.
4. **Do not** restart end-to-end debugging from scratch. Use the worker's two scripts to localize first if the user-channel fix doesn't work.
5. **Do not** chase Q4_K_M / generic llama.cpp quant fallback paths — user has explicitly ruled this out repeatedly. SEPTQ multi-tier is the only quantization path.
6. **Be suspicious of every "empirical" workaround in `bmo_inference.py`** — they were derived while the runtime was broken, and any one of them could be a layered bug now. Specifically audit (a) the SILENCE_TOKENS-on-user override (lead above), (b) the `PERSONAPLEX_DELAYS` array at line 167-168 vs `_lm_kwargs["delays"]` in `moshi/models/loaders.py`, and (c) the `--force-text-pad` clamp at line 1041-1042.

## Files that matter (workspace-relative)

### C++ runtime (server + Jetson)

- `bmo_compute.cpp:540-599` — `apply_rope_gpu_interleaved` (the function Gemini flagged; multi-token fallback at line 562, single-token CUDA at line 586).
- `bmo_compute.cpp:2100-2140` — temporal graph QKV split + RoPE call site.
- `bmo_compute.cpp:2675-2685` — depth graph RoPE call site (also uses NORMAL; same situation — keep it).
- `bmo_cuda_kernels.cu:310-358` — `rope_interleaved_kernel` (adjacent-pair, correct).
- `bmo.cpp:290-295` — rope_theta load (defaults 10000, model has 10000).
- `bmo.h:222` — `rope_theta` default 10000.
- `bmo.h:357` — `launch_rope_interleaved` declaration.
- `bmo_api.cpp` / `bmo_api.h` — C-API bridge for Python. Worker will add a kcache dump entry here.

### PyTorch reference

- `moshi/moshi/modules/rope.py:33-89` — `apply_rope` (adjacent-pair, max_period=10000 default).
- `moshi/moshi/modules/transformer.py` — has `StreamingMultiheadAttention` with `k_cache` / `v_cache` attributes.
- `moshi/moshi/modules/streaming.py` — streaming state machinery.
- `moshi/moshi/offline.py:131-138` — `wrap_with_system_tags` (canonical prefill flow).

### Diagnostic infrastructure

- `pt_fakequant_vs_fp16.py` — PT FP16 vs PT fake-quant reference. Already wired with H3 hooks for T1–T8 taps.
- `scripts/end2end_v5_vs_fp16.py` — orchestrates PT vs C++ end-to-end comparisons.
- `bmo_inference.py` — Python driver for hybrid inference (the one currently producing gibberish audio).

## Server context (jovyan account, H100 box)

```
$PWD = /home/jovyan/work/BMO-Project/personaplex_repo
build dir = build/  (sm_87;90 cubin both present)
env: export PYTHONPATH=$PWD:$PWD/moshi BMO_SO_PATH=$PWD/build/libbmo.so
GPU select: CUDA_VISIBLE_DEVICES=1
```

Standard rebuild after C++ changes:

```bash
cmake -B build -S . -DBMO_CUDA_ARCHS="87;90"
cmake --build build --target bmo_shared -j"$(nproc)"
```

## What's been ruled out already (don't re-litigate)

- ✅ Weights bit-exact: GGUF vs PTQ checkpoint v5 schema verified element-wise.
- ✅ Scales bit-exact: PTQ stored scales used in GGUF (Bug B fixed).
- ✅ Per-element tier mask preserved across export (Bug A fixed, packing_version=5).
- ✅ Path B kernel (`fused_dequant_matvec_kernel_proto`) bit-exact to dense unpack for v5.
- ✅ Path B kernel registers fit (REG:56 STACK:16 measured via cuobjdump).
- ✅ T4 V-slice → T5 attention: cos=0.99999998 for single-token (attention math is correct).
- ✅ T4 in_proj V output matches PTQ dense unpack bit-exactly (apparent earlier divergence was a weight-source mismatch in the reference script, not a runtime bug).
- ✅ `apply_rmsnorm_gpu` PTX toolchain issue (fixed by building sm_87+sm_90 cubins).
- ✅ `kv_len exceeds n_ctx 256` runtime crash (fixed by raising `--n-ctx` to 384).
- ❌ Static text logits across frames — still open, this is the actual gibberish symptom.

## How to resume on the other system

Re-prompt the new chat with:

> Read `.cursor/SESSION_HANDOFF.md`. Start with the **Systems-Level Architectural Audit** and **Jetson offline** sections. For gibberish audio, the SINE_TOKENS fix is already in `bmo_inference.py`. For Jetson E2E, fix warmup `tokens.to(mimi.device)` before `mimi.decode`, then run `moshi.offline` with `BMO_USE_CPP=1`.

Good luck with the test.
