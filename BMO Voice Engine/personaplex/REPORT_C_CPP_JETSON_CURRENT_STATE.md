# Report text — C++, CUDA, and Jetson deployment (current codebase state)

Use the **“Revised (use in report)”** blocks below to replace the older deployment / runtime paragraphs. The first analytical block is **unchanged** (still consistent with your empirical story). **Section 2** documents the **C++/CUDA architecture** (block-wise SEPTQ, registry, fused kernels, `ggml` hybrid). Sections **3–6** align memory, engine narrative, Jetson trade-offs, and conclusion with the repository as of this update.

**Code references (authoritative):** `bmo.cpp` (GGUF load, `bmo_prepare_device_packed_tensors`, KV, Jetson paths), `bmo_compute.cpp` (temporal/depth graphs, `apply_linear`, RoPE/RMSNorm/residual/SwiGLU hooks, Jetson fused matvec dispatch), `bmo_api.cpp` (C ABI, work-memory sizing), `bmo.h` (host/device packed metadata, staging pool), `bmo_cuda_kernels.cu` / `bmo_cuda_kernels_proto.cu` (fused dequant+matvec v2 and Path-B v5), `bmo_inference.py` + `handoff.md` (Python driver aligned with `moshi.offline` + `LMGen`).

---

## 1. Activation outliers and `self_attn.out_proj` (unchanged)

*No revision required for factual alignment with the C++ tree; keep your original wording if it still reflects your measurements.*

A critical analysis of the empirical data reveals deep architectural characteristics inherent to continuous speech foundation models. The identification of 40x to 66x activation outliers explicitly demonstrates that Moshi-family architectures distribute semantic and acoustic routing entirely differently than standard transformer quantization literature. These immense magnitude spikes suggest that the network encodes critical prosodic switches in highly localized, high-amplitude bursts that are instantly destroyed by uniform bit-width reduction.

Furthermore, the data conclusively proves that the `self_attn.out_proj` layer acts as an absolute precision floor. The unique sensitivity of the output projection mechanism likely stems from its dual responsibility: it must simultaneously synthesize the dense temporal context from the attention heads while perfectly mapping that context back into the strict geometrical boundaries required by the 12.5 Hz Mimi continuous latent space. Modifying the precision of this exact matrix mathematically fractures the latent projection.

---

## 2. C++ runtime architecture, block-wise SEPTQ, and custom CUDA kernels — **for the report**

The following is **not** a second “deployment disclaimer”: it documents the **actual engineering stack** that sits under the C ABI—custom kernels, **block-wise multi-tier packed weights**, and how they connect to `ggml` graphs.

### 2.1 Layered design

1. **Model container (`bmo_model` + `bmo_context`, `bmo.h`)** — GGUF metadata lives in a `ggml` metadata context; weights are materialized into host **scalar** and **big** pools (`bmo_load_model` in `bmo.cpp`). Tensors are still **logically block-structured**: each quantized linear is described by `packed_weights` (interleaved 2/4/8-bit streams), `packed_mask` (per-block or, from **packing_version ≥ 5**, per-element tier selection), FP16 exception values, `block_size`, `n_blocks`, per-tier byte counts, and global dequantization scales / zero-points.

2. **GPU registration / staging (`bmo_prepare_device_packed_tensors`, `bmo.cpp`)** — After host load, the runtime walks **every packed temporal matrix** per layer and builds a **`device_packed_t`** entry registered in **`ctx.packed_registry`**. Behavior splits by target:
   - **Jetson (`BMO_JETSON`)** — Packed payloads typically **stay in the same host buffers** already covered by `cudaHostRegister` on the big pool; **`cudaHostGetDevicePointer`** exposes a **device-mapped alias** so kernels read tiers without a second full copy of the weight blob. **Per-row prefix tables** (`row_c2`, `row_c4`, `row_c8`, and for v5 **`row_c16`**) are **JIT-computed on CPU** from the mask layout, then **`cudaMalloc` + `cudaMemcpy`** to device for the fused kernel’s stream offsets (tier-stream “walk” indices).
   - **Discrete CUDA (non-Jetson)** — For **packing_version &lt; 5**, **`cudaMalloc` / `cudaMemcpy`** stage `packed_weights` and `packed_mask` on true device DRAM; **v5** desktop staging may fall back to a **CPU unpack → `ggml_mul_mat`** path until Path-B CUDA is fully enabled for that build (see log branch in `bmo_prepare_device_packed_tensors`).

3. **Graph construction and execution (`bmo_compute.cpp`)** — **`bmo_reset_work_ctx`** allocates transient nodes inside a fixed host **`work_mem`** arena. **`bmo_build_temporal_graph`** assembles the 32-layer temporal transformer (embeddings, self-attention with KV read/write against **`ctx.k_cache` / `v_cache`**, FFN, logits). **`bmo_build_depth_graph`** builds the smaller **depth / depformer** stack with its own **`depth_k_cache` / `depth_v_cache`**. **`bmo_execute_graph`** applies optional input uploads, optionally **`cudaStreamSynchronize(0)`** on Jetson as a **single boundary** between eager CUDA helpers and **`ggml_graph_compute_with_ctx`**, then runs the CPU-side `ggml` scheduler for nodes not replaced by intercepts.

4. **C ABI (`bmo_api.cpp`)** — **`bmo_init`** calls **`bmo_load_model`**, which materializes GGUF weights and then invokes **`bmo_prepare_device_packed_tensors`** before returning; init continues with **`bmo_init_kv_cache`**, **`work_mem`** sizing, and handle setup. **`bmo_forward_temporal`** / **`bmo_forward_depth`** serialize access with a mutex, reset work context as needed, build the appropriate subgraph, and execute it.

### 2.2 “Block-wise loading” in this project

The phrase refers to **SEPTQ’s storage math**, not to paging individual blocks from disk on demand. Each large matrix is stored as **packed tier streams + mask + optional FP16 overrides** at **block granularity** (`block_size`, `n_blocks`). **Load time:** whole tensors are read with **`pread`** into the host pools. **Prepare time:** the registry records pointers and dimensions so **`apply_linear`** can choose a compute path. **Inference time:** custom kernels **dequantize on-the-fly inside the matvec** (or unpack to a scratch FP32 slab then `ggml_mul_mat`), so the **hot path never materializes a full dense FP16/FP32 weight matrix** for those layers.

### 2.3 Custom CUDA kernels and where they hook

| Concern | Mechanism (files) |
|--------|-------------------|
| **Fused dequant + GEMV/GEMM-style matvec** | **`launch_fused_dequant_matvec`** — `fused_dequant_matvec_kernel_v2` in `bmo_cuda_kernels.cu` (8 rows per thread block, 256 threads, shared-memory staging of tier streams). Used on **desktop CUDA** for v4-style packs. |
| **Jetson fused path + SEPTQ v5** | **`launch_fused_dequant_matvec_jetson`** in `bmo_compute.cpp` — dispatches **`launch_fused_dequant_matvec_proto`** from `bmo_cuda_kernels_proto.cu` when **`packing_version >= 5`** (per-element 2-bit mask lanes, **warp-local prefix** over stream offsets via `row_c16`); otherwise uses the **v2** fused launcher with **`row_c2/c4/c8`** tier bases. Reads weights through **mapped host pointers** or pre-uploaded device tables as configured in `device_packed_t`. |
| **RoPE (single decode token)** | **`launch_rope_interleaved`** / `rope_interleaved_kernel` in `bmo_cuda_kernels.cu` — adjacent-pair interleaved layout consistent with Moshi PyTorch. **`apply_rope_gpu_interleaved`** (`bmo_compute.cpp`) uses **mapped/pinned staging slots** (`gpu_staging_pool`) when the QKV view is **contiguous for one token**; **multi-token prefill** falls back to **`ggml_rope`** with **`GGML_ROPE_TYPE_NORMAL`**. |
| **RMSNorm, residual add, SwiGLU** | **`launch_rmsnorm`**, **`launch_residual_add`**, **`launch_swiglu_split`** in `bmo_cuda_kernels.cu`, invoked from `bmo_compute.cpp` helpers to keep hot activations on a **Jetson-friendly** path where possible. |
| **Desktop unpack fallback** | **`launch_unpack_kernel`** (non-Jetson) materializes a dense FP32 slab on device, **`cudaMemcpy`** back to host, then standard **`ggml_mul_mat`** — used when fused or proto paths are skipped. |
| **Staging pool (Jetson)** | **`gpu_staging_pool`** — fixed slots sized **`SLOT_BYTES = 22528 × sizeof(float)`** per slot (`bmo.h`, allocated in `bmo.cpp` near KV init), with **`cudaHostRegister`** on each slot so **`apply_rope_gpu_interleaved`** and related helpers can borrow **host+device aliases** without per-op allocations. |

Together, these pieces implement a **hybrid executor**: **`ggml`** supplies the graph structure, tensor shapes, and CPU fallback ops, while **BMO-specific CUDA** replaces the most memory- and bandwidth-sensitive **linear and activation** stages with **kernels that understand SEPTQ packing**—the architectural answer to “how we hook custom CUDA to block-wise quantized weights.”

### 2.4 Report-ready paragraph (paste as a subsection)

The custom inference stack is organized as a **layered runtime**: a GGUF-backed **`bmo_model`** holds **block-wise multi-tier quantized tensors** (packed byte streams, masks, and FP16 exceptions) loaded into **host memory pools**; **`bmo_prepare_device_packed_tensors`** then **registers** each packed matrix for CUDA—on **Jetson**, primarily via **host memory mapped into the GPU address space** (`cudaHostRegister` / `cudaHostGetDevicePointer`) plus **small device-resident row-index tables** for tier-stream offsets, and on **discrete GPUs** via **`cudaMemcpy`** into device buffers for the v4 layout. **Temporal and depth transformers** are expressed as **`ggml` computation graphs** (`bmo_build_temporal_graph`, `bmo_build_depth_graph`) executed inside a **fixed work-memory arena**, while **`bmo_compute.cpp`** intercepts selected linear layers to call **custom fused dequantization–matrix–vector kernels** (`bmo_cuda_kernels.cu`, **`bmo_cuda_kernels_proto.cu`** for **packing_version ≥ 5**) instead of expanding full dense weights. Auxiliary kernels implement **interleaved RoPE**, **RMSNorm**, **residual adds**, and **SwiGLU** splits, coordinated with a **pinned staging pool** on Jetson so activation tiles can move through CUDA without fragmenting unified memory. The public **`libbmo.so`** surface (`bmo_api.cpp`) exposes **temporal** and **depth** forwards that drive this stack from Python (`moshi/bmo_engine.py`, **`patch_lm_for_bmo`** in `moshi/offline.py`).

---

## 3. Memory model, Jetson ceiling, and RSS — **revised (use in report)**

**Replace** the previous paragraph that relied on an “unverified operating system `mmap` hypothesis” and the old Figure 6 caption.

### Revised (use in report)

Despite producing a highly compressed on-disk GGUF artifact (your reported ~7.7 GB class), the **current runtime does not depend on lazy kernel paging of weights**. In `bmo_load_model` (`bmo.cpp`), tensors are laid out with GGUF metadata first (`no_alloc`), then **payloads are read fully into caller-allocated pools** via `pread`: small tensors into a **scalar pool**, large tensors into a **page-aligned “big” pool**. Resident memory therefore tracks **full materialization of weights** (plus KV, graph work buffers, and CUDA resources), not an idealized demand-paged `mmap` curve. On **Jetson builds** (`BMO_JETSON`), the large pool is additionally **`cudaHostRegister`’d** for mapped / portable host memory so kernels can stream from a pinned view of weights—again assuming the tensors are already resident.

The **temporal KV cache** is allocated explicitly in host memory as FP16 K/V tensors sized by `n_ctx` and layer count (`bmo_init_kv_cache` in `bmo.cpp`). For Jetson, **`n_ctx` is capped at 1024** to stay within an ~8 GB unified-memory budget after weights; callers can still force smaller `--n-ctx` for headroom. **Depth-transformer KV** is separate, small (order of hundreds of KB), and sized for Moshi-style depth (`dep_q` positions capped at 16 in the current init block).

**Graph execution** uses a fixed **`ggml` work arena** backed by `std::vector<uint8_t> work_mem` sized at init: **`kDefaultWorkMem` is 1 GiB when `BMO_JETSON` is defined, else 2 GiB** (`bmo_api.cpp`). If a compiled graph’s transient allocations exceed that arena, **`ggml` can abort** with a failure that presents as a requested allocation larger than the pool (your historical ~2.1 GiB request vs a 2 GiB cap is consistent with this class of failure, not with a missing depth stub). Mitigation is **raising the work buffer** (rebuild), **reducing graph footprint** (kernel / fusion changes), or **smaller `n_ctx` / batch-like paths** where applicable—not turning on OS-level lazy paging of the GGUF file in the present loader.

**Figure 6 — suggested replacement caption**

**Figure 6 — RSS and allocation components (current loader, not lazy weight mmap):** schematic time series of resident set size showing dominant contributions from (i) fully loaded GGUF tensor pools (scalar + big), (ii) FP16 temporal KV proportional to `n_ctx` (Jetson-capped at 1024 in-tree), (iii) depth KV (small), (iv) the fixed `ggml` work arena (1 GiB Jetson / 2 GiB default host), and (v) CUDA staging / registered buffers, against the device’s physical RAM ceiling. The curve is **not** an on-demand page-in model for weights in the current implementation.

---

## 4. Custom C++ inference engine and CUDA kernels — **revised (use in report)**

**Replace** the paragraph that claimed a **`bmo_forward_depth` `rc=10` stub** and a permanently blocked depth path.

### Revised (use in report)

The present **custom C++ inference engine** exposes a stable C ABI (`bmo_api.cpp` / `bmo_api.h`) consumed from Python via `moshi/bmo_engine.py`. **`bmo_forward_temporal`** advances the main transformer; **`bmo_forward_depth`** is a **fully implemented** path: it resets depth KV at the start of each temporal frame for codebook index 0, rebuilds a per-step depth `ggml` graph (`bmo_build_depth_graph`), executes it (`bmo_execute_graph`), and copies **`audio_logits`** out—returning structured error codes on failure, **not** a permanent stub return. The depth stack therefore **does** participate in the same loop as PyTorch reference code when `patch_lm_for_bmo` routes `forward_depformer` to the shared library (as in `moshi/offline.py` and the **`bmo_inference.py stream`** path described in `handoff.md`).

On the **CUDA side**, the tree carries **SEPTQ-specific fused dequantization / matvec paths** and related device metadata (`device_packed_t` in `bmo.h`, kernels in `bmo_cuda_kernels.cu`, orchestration in `bmo_compute.cpp`). Jetson and discrete-GPU builds diverge in **how packed weights are staged** (pinned host pools and registration on Jetson vs device-resident packed buffers elsewhere), but the **intent is a single quantized temporal + depth runtime**, not a depthless prototype.

**Residual risk (honest):** end-to-end **GGUF + C++ parity** versus full PyTorch inference has been sensitive to **driver-side details** (token layout, `LMGen` delay ring, KV length, RoPE layout). Session notes document past **audio / text degradation** when the Python driver diverged from `moshi.offline`; the **`stream` mode rewrite** aligns with **`LMGen`** and **`step_system_prompts`**. Any remaining mismatch is an **active numerical / driver QA** topic, not evidence that depth is unimplemented.

---

## 5. Jetson “full duplex” and `dep_q` — **revised (use in report)**

**Replace** the paragraph that states full-duplex on Orin Nano is “bordering on impossible” and that cutting **`dep_q` 16→8** is the most probable fix.

### Revised (use in report)

Full-duplex conversational behavior on **Jetson-class unified memory** remains **resource-constrained** primarily by **resident weights + KV + work arena + codec (Mimi) and PyTorch overhead** in the integrated Python stack, not by a disabled depth function. The in-tree Jetson policy already **trades context length for safety** (`n_ctx` cap 1024) so that long voice-prompt prefills remain feasible without instant OOM.

Reducing **`dep_q`** (acoustic codebook depth) would change **model geometry relative to the trained PersonaPlex checkpoint** and is **not** implemented as a runtime toggle in the excerpts reviewed here; treat it as a **research-level architectural change** (new export / training assumptions), not the default mitigation for memory pressure. Prefer **KV budget tuning**, **work-buffer sizing**, **shorter prompts**, and **quantization / cache-compression research** (for example TurboQuant-class ideas in your forward-looking paragraph) before altering `dep_q`.

---

## 6. Conclusion — **revised (use in report)**

**Replace** the closing paragraph that states the C++ scaffold still has an unresolved allocation crash and that physical instantiation is unfinished **only** for that reason.

### Revised (use in report)

This project successfully constructed and evaluated a complex pipeline for adapting and compressing a continuous speech-to-speech architecture for edge deployment. The primary contributions include a specialized multi-pass zero-shot voice dataset, the empirical identification of the `self_attn.out_proj` layer as a rigid precision floor, and the development of the \(z_s\) drift metric for deep structural diagnostics. Integration of **multi-tier SEPTQ** compression and QAT recovery stabilized the network in your reported cosine regime and produced the **compressed GGUF artifact** you summarize (~9.7 GB class for the full pipeline artifact, ~7.7 GB for the GGUF-focused deployment package—**retain your measured numbers**).

On the **implementation side**, the **custom C++ / CUDA runtime and C ABI are now feature-complete for temporal and depth forwards**, with **explicit memory accounting** (weight pools, KV, fixed `ggml` work memory, Jetson `n_ctx` cap, optional `cudaHostRegister` on the weight pool). The **Python driver** for hybrid inference (`bmo_inference.py`) has a **`stream` mode aligned with `moshi.offline` + `LMGen`**, which is the supported path for **bit-accurate delay and depformer behavior** relative to the reference stack. Remaining work is **operational and scientific validation** on target hardware: **RSS profiling under real prompts**, tuning **`kDefaultWorkMem`** if graphs grow, and **continued GGUF-vs-PyTorch quality checks**—plus longer-term **KV compression** (for example TurboQuant-class methods, Zandieh et al., 2025) and, only if unavoidable, exploration of **smaller backbones** to reduce the quantization burden.

---

## Changelog vs. older report draft (for your appendix)

| Prior claim | Current codebase / project state |
|-------------|----------------------------------|
| Viability rests on unverified OS `mmap` paging of GGUF | Weights are **`pread` into explicit pools**; not lazy file-backed paging in `bmo_load_model`. |
| `bmo_forward_depth` blocked by `rc=10` stub | **`bmo_forward_depth` is implemented** end-to-end (graph build + execute + logits copy). |
| C++ engine only / no depth in C ABI loop | **`forward_depformer` → `bmo_forward_depth`** in `patch_lm_for_bmo` (`moshi/offline.py`). |
| Jetson context only implied | **`BMO_JETSON`:** `n_ctx` **≤ 1024**, work arena **1 GiB**, registered weight pool. |
| `stream` hand-rolled tokens | **`mode_stream` uses `LMGen`** parity path per `handoff.md`. |
| “C++ is only a thin stub / no CUDA integration” | **Hybrid `ggml` + custom CUDA**: `packed_registry`, fused dequant–matvec (v2 + proto v5), RoPE/RMSNorm/residual/SwiGLU hooks (`bmo_compute.cpp`, `bmo_cuda_kernels*.cu`). |

---

*Generated to align report language with the PersonaPlex / BMO repository layout described above. Adjust absolute RAM figures (5.5 vs 8 GB SKUs) to match your exact Jetson SKU and measurement campaign.*
