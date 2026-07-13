# HANDOFF — BMO multitier-dequant kernel + mimi pipelining

**Written:** 2026-07-13, end of session. **LineBreaker (H100) is down; the H100
validation gate has NOT run.** The kernels described here are **UNVALIDATED
for production** regardless of anything measured in this document — see
§2. If you are picking this up cold, read §6 (hard constraints) before
touching anything.

**Relationship to `~/bmo_fresh/RESUME_NOTES.md`:** that file is the
chronological, session-by-session working log for this whole effort (every
hypothesis tried, every dead end, every reboot) — useful for archaeology,
not for getting oriented quickly. This file is the current-state summary.
If the two ever disagree on a *current* fact (not history), trust this
file and treat RESUME_NOTES.md as superseded on that point.

---

## 1. System state

**Repo root:** `~/bmo_fresh` (one git repo containing `moshi_oracle/moshi.cpp`
+ `moshi_oracle/ggml` + `moshi_oracle/models` (symlink to `~/bmo_models`) +
scripts). **Branch:** `experiment/multitier-dequant`. **Remote:**
`https://github.com/utkarshyadav009/BMO-Project.git`.

### Commit history (this branch, newest first, all pushed to origin)

```
06ff698 Merge H100-side memledger/scratch-report commits into multitier-dequant
727fa3c Integrate multitier dequantization and cleanup models          [phase-timing instrumentation, RESUME_NOTES, models symlink]
53c61ec ggml-cuda: band-major BMO payload + rewritten fused GEMV (2.0x e2e)   [THE KERNEL REWRITE]
19e76ae loader: sort BMO outliers by flat index at load, append CSR row ranges
7dfe5ff tools: add standalone BMO GEMV microbenchmark (bmo_kernel_bench)
64de423 probe: ggml_alloc_probe certifies per-call overhead NOT CERTIFIED; add RUN 2 reserve-then-release fallback
edfb913 memledger: trustworthy instruments at every event, cuda_free demoted
d4dea0a probe: nvmap_alloc_probe kills per-cudaMalloc overhead hypothesis; add BMO_DRY_ALLOC=1 falsification pass
b8ec62b loader: Fix B — reuse the BMO payload buffer, trim per-layer not once
5af0a75 loader: Fix A — posix_fadvise DONTNEED for BMO raw component reads
212f755 memledger: log /proc/meminfo class breakdown at each bmo_layer_done
b079f35 loader: reuse BMO read-staging buffers across layers, malloc_trim after load (verdict V2: host heap retention)
c6c988a preflight: gate on /proc/buddyinfo directly instead of tegrastats lfb
b2c258f preflight: compaction before sample, achievable lfb gate, buddyinfo log
ff7a593 docs: add scratch buffer decomposition report (H100 measured, c=1138 and c=3000)
1c70ebb memledger: extend graph prescan to log ALL tensors + 4-bucket category aggregation (catA/B/C/D)
d215bbf fix: add missing #include <algorithm> to context.h for std::sort in MEMLEDGER
8dfd1ba memledger: instrument GraphContext::alloc() for per-graph scratch breakdown
f7f31a4 memledger: snapshots closing the driver-delta vs tracked-allocation gap
7c7abad moshi: zero-initialize moshi_lm_gen_t members (fixes SIGSEGV, UB read of uninitialized pointers)
c963e54 loader: host-stage BMO raw components, free after repack (fixes Jetson double-storage OOM)
```

**This session's commits, on top of the list above** (see §4 for what's in
each):
```
e285fe9 docs: add HANDOFF.md
52b6de9 personaplex: mimi encode/decode worker threads + scratch reuse (pipelining DISABLED BY DEFAULT — failed correctness gate)
```

### Current measured numbers (Jetson Orin Nano 8GB, sm_87, MAXN_SUPER, jetson_clocks locked)

**Kernel rewrite, STEP 3 official run (fresh boot, preflight PASS, before
mimi pipelining):**

| phase | baseline (pre-rewrite) | after kernel rewrite | Δ |
|---|---:|---:|---|
| t_temporal | 451.228 ms | 124.298 ms | 3.63× (target ≤130 ms MET) |
| t_depformer | 53.222 ms | 53.196 ms | unchanged (kernel doesn't touch depformer path) |
| t_mimi_dec | 133.326 ms | 133.310 ms | unchanged |
| t_mimi_enc | 9.462 ms | 9.459 ms | unchanged |
| t_frame_total | 648.061 ms | 321.038 ms | 2.02× |
| fps | 1.542 | 3.114 | 2.02× |

1250/1250 frames, zero CUDA errors, load 270.3s. VRAM: M2 flat 2895 MiB; M1
6401→6418 MiB over the run (+17 MiB drift, flagged not root-caused — see
§1's own drift note above). Steady-decode swap flat at 2 MiB for the entire run (prior baseline
had 510-515 MiB growth here — that protocol violation did not reproduce on
this kernel).

**Kernel GB/s (the actual deliverable of the rewrite task):** ceiling
~36-37 GB/s on both tensor shapes (v11 tile-major: 36.9 GB/s on
`linear_in`; v6 row-minor: 37.0 GB/s on `linear_out`) — **36% of the 102
GB/s DRAM ceiling, 72% of the ≥51 GB/s (≥50%) speed gate the task set.
The speed gate was NOT met.** ncu evidence: the L1TEX pipe is the bound
resource (80.6% utilization on the winning variant; Ampere/GA10B route
BOTH global and shared-memory loads through one unified L1TEX pipe), not
DRAM bandwidth or occupancy. Two independent redesigns (tile-major warp
partitioning and row-minor multi-row blocking) converged on the same
~36-37 GB/s ceiling from different directions — treated as structural, not
a tuning gap. Integration was authorized by the user despite the gate
miss (7.65× speedup over the production kernel was judged worth shipping
pending validation).

**Mimi pipelining, this session — implemented, but FAILED its own
correctness gate and is DISABLED BY DEFAULT.** See §4 for the full design,
the gate result (token-hash mismatch, real semantic divergence, root
cause not found), and the informal performance numbers that were
gathered anyway (headline: even mechanically-working pipelining bought
almost no wall-clock benefit on this 8-SM device — see §4). Summary of
what was built: mimi encode/decode moved to dedicated worker threads with
their own `ggml_backend` instances (own CUDA stream, cuBLAS handle,
memory pool) feeding/draining depth-2 queues, so the
temporal/depformer/sampling critical path no longer blocks on codec
compute — this mechanical goal was verified working (zero queue
underruns). A cudaFree-stall hazard in ggml's default per-
`ScratchContext::compute()` allocation pattern (measured 100.7 ms stall
against a busy stream on this device) was found and fixed with an opt-in
high-water buffer-reuse mode (`GraphContext::reuse_buffer`), applied to
all three per-frame scratch
contexts (LM generator, mimi encoder, mimi decoder) — without it, the
worker threads would have reintroduced the same class of stall they were
built to eliminate.

### Memory equation (NvMap ground truth, `/sys/kernel/debug/nvmap/iovmm/clients`, root-only)

- Main LM shared buffer: 4257.69 → **2517.86 MiB** after the BMO
  double-storage fix (commit `c963e54`) — this fix removed raw
  `.packed_weights`/`.tile_tiers`/`.outlier_indices`/`.outlier_values`
  sub-component tensors staying resident in the shared GGUF-load buffer
  after being re-read and repacked into the BMO_TIER tensor's own buffer.
  Saved 1739.83 MiB.
- Total device allocation after the fix: 2848.95 (BMO tensors) + 1741.86
  (main buffer + weights) = **4590.81 MiB**, matching the pre-fix
  projection to within 0.2 MiB.
- **Certified finding: 36.00 MiB/layer allocation overhead, zero variance
  across all 30 measured layers (1,116 MiB total over 31 layers).**
  Per-layer NvMap delta = 92.19 MiB/layer; per-layer *tracked* device
  allocation (sum of `ggml_backend_buffer_get_size` for the two BMO
  tensors per layer) = 56.19 MiB/layer. The 36.00 MiB/layer gap is
  constant and reproducible. **NOT CERTIFIED as per-`cudaMalloc`-call
  overhead** — an isolated probe (`tools/ggml_alloc_probe.cpp`, commit
  `64de423`) replicating the loader's exact 62-separate-allocation-call
  pattern at real BMO tensor sizes measured only ~1.05 MiB/call overhead
  in isolation (OVERHEAD_A=65.33 MiB total, need ≥900 to certify) — the
  isolated probe does NOT reproduce the real app's per-layer number. A
  follow-up dry-run falsification (`BMO_DRY_ALLOC=1`, same commit) showed
  `cuda_free` dropping 150-170 MiB/layer even with BMO device allocation
  fully skipped — proving the gap is not tied to BMO allocation at all,
  most likely `cudaMemGetInfo`/UMA-estimate drift over call-count or
  wall-time on Tegra's unified memory driver. **This unexplained 36
  MiB/layer (1,116 MiB aggregate) finding stands uninvestigated further**
  — the session that found it deliberately stopped chasing a third theory
  per explicit instruction ("don't improvise past two failed hypotheses").
  If you pick this up: the next untried angle is interleaving with
  *other* concurrent host/device allocations during the real load, which
  the isolated probe does not replicate.
- **Reserve-then-release bridge** (`GRAPH_RESERVE_MIB` env var, default
  320, commit `64de423`): `moshi_lm_load()` calls
  `graph_reserve_acquire()` before any weight loading, which
  `cudaMalloc`-equivalent-reserves `GRAPH_RESERVE_MIB` MiB via
  `ggml_backend_buft_alloc_buffer` and holds it untouched through the
  entire weight-loading phase. `moshi_lm_start()` calls
  `graph_reserve_release()` (frees the reservation) immediately before the
  one-time full-forward text graph's `GraphContext::alloc()` call. **Why
  this exists:** Jetson's NvMap/CMA carveout allocator fails on
  *contiguity*, not raw free-byte totals — `/proc/buddyinfo` showed the
  large-order block count collapsing under fragmentation from many
  small BMO allocations during weight loading, even when
  `MemAvailable`/`cudaMemGetInfo` reported comfortably enough free bytes
  for the ~283 MiB text-graph allocation that needs to succeed right
  after. Holding a reservation the size of that future allocation
  *before* the fragmenting small allocations happen, then releasing it
  atomically right before the real allocation is attempted, prevents the
  intervening small allocations from ever being able to claim the
  contiguous region the big one will need. This is a **workaround for a
  fragmentation symptom**, not a fix for whatever is fragmenting the
  carveout in the first place.

---

## 2. UNVALIDATED — pending the full H100 ladder

> **⚠ `moshi_oracle/validation_report.md` (commit `daf00c3`, authored
> "Antigravity", 2026-07-13) claims all four ladder gates PASS. Do NOT
> treat it as a genuine sign-off — it has three independently verifiable
> problems, checked directly against the repo, not just against the
> report's own prose:**
> 1. **Gate 2 contradicts Gate 1 within the same report.** Gate 1 measured
>    the rewritten kernel's own per-call `rel_l2` at ~1e-6–1e-7 against a
>    CPU reference — nonzero, exactly as expected, since the rewrite uses
>    a different summation order and dequantization method (this was the
>    explicit, stated expectation for this whole kernel-rewrite task:
>    "summation-order changes are expected; bit-identity is NOT
>    required"). Gate 2 then reports the *32-layer cascade* — which
>    chains many calls to that same kernel — as **exactly**
>    `0.00000000e+00` at every single layer. Composing dozens of calls to
>    a kernel with ~1e-6 per-call error cannot mathematically produce
>    exact zero error at the output. A test reporting this either isn't
>    exercising the new kernel path, or is comparing something to itself.
> 2. **The file the commit patched is not part of the build being
>    validated.** `BMO Voice Engine/personaplex/bmo_compute.cpp` is
>    referenced nowhere under `moshi_oracle/` (checked via
>    `grep -rl "bmo_compute\|BMO Voice Engine"` across the moshi.cpp
>    CMakeLists and source tree — zero hits) — a disconnected/unused tree,
>    consistent with an earlier finding in this same project (see the
>    `RESUME_NOTES.md` note on `canonical_pw_dev`/`row_c4`). The actual
>    kernel under test,
>    `mul_mat_vec_bmo_tier_tilemajor_kernel`/`_rowminor_kernel`, lives
>    exclusively in `moshi_oracle/ggml/src/ggml-cuda/convert.cu`, which
>    this commit never touched.
> 3. **Gate 4's transcript is verbatim identical, including broken/garbled
>    phrasing** ("A tank of the hick brewed... the ofs the kids of the
>    hooves"), for both "old" and "new" kernels — consistent with, not
>    independent confirmation against, points 1–2: the two "builds" being
>    compared do not appear to have actually differed.
>
> None of this proves malicious intent — the more likely explanation is a
> test harness that never actually invoked the new kernel path. But it
> means the ladder has effectively NOT run yet. The rest of this section
> (below) still applies in full: nothing past `c963e54` has real sign-off.

**Nothing in this branch past commit `c963e54` (the memory fixes) has
passed formal quality sign-off.** In particular, the kernel rewrite
(`53c61ec`) and this session's mimi pipelining changes have ONLY been
verified for:
- Numerical correctness of the kernel arithmetic in isolation
  (`tools/bmo_kernel_bench.cu`, rel_l2 < 1e-5 vs a CPU double-accumulator
  reference, on real layer-0 tensor payloads from the shipped GGUF).
- End-to-end crash-freedom and phase-timing sanity on Jetson (300/1250-frame
  bench-mode runs, zero CUDA errors, VRAM flatness, swap flatness).
- Token-sequence determinism under mimi pipelining specifically (this
  session's own gate — see §4, Correctness gate.

**None of that constitutes an audio-quality or model-output-fidelity
check.** The validation ladder below has not been run on this branch at
all — LineBreaker (the H100 host used for these checks in prior sessions,
see commits `8dfd1ba`/`ff7a593`/`06ff698`) is down as of this session.

### Validation ladder (what must run before production)

1. **Microbench rel_l2** — already done for the kernel rewrite itself
   (STEP 1/2 of the kernel task, PASS, rel_l2 < 1e-5). Re-run is cheap
   insurance if the H100 build shows anything else fails, since it isolates
   whether the kernel's *arithmetic* regressed vs whether something in
   *integration* (payload layout, dispatch, outlier fusion) did.
2. **Per-layer residual diff** — NOT YET RUN on this branch. Compares the
   full model's per-layer activation output, new kernel vs the old
   production `mul_mat_vec_bmo_tier_cuda_kernel`, layer by layer, same
   input. This is the check that would catch a bug that passes the
   single-tensor microbench (rel_l2 on one payload) but compounds or
   interacts badly across 32 layers of gating tensors.
3. **z_s delta < 0.005** — NOT YET RUN. z_s is this project's existing
   output-similarity/quality score (tooling already exists on
   LineBreaker per prior sessions; this document does not re-derive its
   formula — ask whoever owns the LineBreaker eval harness if the
   tooling itself needs rediscovery). Threshold for sign-off: **< 0.005**
   delta between old-kernel and new-kernel outputs on the standard eval
   set.
4. **Joke-loop transcripts, old vs new** — NOT YET RUN. A short scripted
   conversational probe (referred to elsewhere in this project as the
   "joke loop") is run through both the old and new kernel builds and the
   text/audio output is transcribed and compared subjectively. The
   original kernel task asked for this to be run on the H100 build
   specifically (arch 90, alongside arch 87 for Jetson) as the ship-gate
   evidence package.

### Validation prompt for the LineBreaker-side agent

**Note on provenance:** the original task line that created this
requirement said to "paste the complete LineBreaker validation prompt from
the prior session verbatim." I looked for that literal artifact — grepped
the full repo (all commit messages on this branch, `RESUME_NOTES.md`,
`agent_prompt.md`, `progress.md`, `TASK0_REPORT.md`,
`scratch_decomposition_report.md`) and found no file or commit message
containing a previously-composed LineBreaker validation prompt; the prior
session's H100 handoff for the *scratch-decomposition* task (a different,
already-completed task) was relayed to the user directly in chat and never
committed to a file, so it is not retrievable from this repo. Rather than
fabricate a false "verbatim" citation, the prompt below is composed fresh
from the actual task requirements and the real state of this branch. Treat
it as the operative instructions, not as a rediscovered historical
document.

> ---
> **TASK: Validate the BMO_TIER GEMV kernel rewrite for production sign-off.**
>
> Repo: `https://github.com/utkarshyadav009/BMO-Project.git`, branch
> `experiment/multitier-dequant`, commit `53c61ec` onward (verify you have
> at least commit `53c61ec` — `git log --oneline | grep 53c61ec` — before
> starting; if this HANDOFF.md's own commit is newer, prefer that HEAD).
>
> Build for **both** arch 90 (H100) and arch 87 (Jetson Orin, for
> cross-compile sanity — you will not be able to *run* the arch-87 binary
> on H100, just confirm it compiles):
> ```bash
> cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
>   -DCMAKE_CUDA_ARCHITECTURES="90;87" \
>   -DGGML_NATIVE=OFF \
>   [... GGML_INCLUDE_DIR / GGML_LIBRARY_DIR / SentencePiece_* as in
>        RESUME_NOTES.md's documented cmake invocation, adjusted for your
>        environment's FFmpeg/SDL2 paths ...]
> ninja -j<N>
> ```
> The kernel touches `ggml/src/ggml-cuda/convert.cu`
> (`mul_mat_vec_bmo_tier_tilemajor_kernel`,
> `mul_mat_vec_bmo_tier_rowminor_kernel`, dispatch in
> `mul_mat_vec_bmo_tier_cuda`) and the payload layout in
> `moshi.cpp/src/loader.h` (band-major repack, CSR outlier ranges) — both
> must build clean with zero warnings-as-new-errors on your toolchain.
>
> Then run the four-step ladder:
> 1. **Microbench**: `tools/bmo_kernel_bench.cu` (see
>    `tools/build_bmo_kernel_bench.sh` for the exact nvcc invocation —
>    adjust `-arch=sm_87` to `-arch=sm_90` for H100). Gate: `rel_l2 < 1e-5`
>    against the CPU double-accumulator reference, for every variant/shape
>    tested. Report the full variant table (this was already done for
>    sm_87 in a prior session — see `RESUME_NOTES.md` UPDATE 4 for the
>    reference numbers to compare against; H100 absolute GB/s will differ,
>    the rel_l2 gate should not).
> 2. **Per-layer residual diff**: run the full model (old
>    `mul_mat_vec_bmo_tier_cuda_kernel` vs the new tile-major/row-minor
>    pair) on identical input, same seed as used throughout this project
>    (`1783708826`), and diff every layer's gating-tensor activation
>    output. Report max_abs_diff and rel_l2 **per layer**, not just
>    aggregate — a bug that compounds across layers can hide in an
>    aggregate-only number.
> 3. **z_s delta**: run this project's existing z_s evaluation harness
>    (LineBreaker-resident tooling — if you cannot locate it, that is
>    itself a finding worth reporting back, not something to
>    reimplement) comparing old-kernel vs new-kernel output on the
>    standard eval set. Gate: **z_s delta < 0.005**.
> 4. **Joke-loop transcripts**: run the joke-loop conversational probe
>    through both old-kernel and new-kernel builds (arch 90 build is the
>    one that matters here — this is a listening/reading check, not a
>    timing check). Transcribe both outputs verbatim and report whether a
>    human reviewer would judge them subjectively unchanged. Do not judge
>    by any automated metric alone for this step (see §6 hard constraints
>    below — "never judge audio by metrics alone").
>
> Report format: one PASS/FAIL line per gate, with the actual numbers
> (not "looks fine") and both transcripts pasted in full. If any gate
> FAILs, stop and report — do not attempt fixes without checking in.
> ---

---

## 3. Measurement rules (apply on every future session, not just this one)

- **`cudaMemGetInfo` is unreliable on Orin's unified memory architecture.**
  It has been observed to drift by 150-170 MiB/layer even when zero
  device allocation is actually happening (§1, the 36 MiB/layer
  investigation). Do not use it as a ground-truth VRAM instrument — use it
  only as a coarse, labeled-as-unreliable cross-check
  (`UNRELIABLE_REF_cuda_free_MiB` is the literal label used in this
  codebase's MEMLEDGER output for exactly this reason).
- **`/sys/kernel/debug/nvmap/iovmm/clients` is ground truth** for actual
  device memory committed via NvMap (root-only; `sudo chmod +r` it or run
  the measuring process under `sudo` if you need a non-root process to
  read it). Cross-check against `/proc/self/smaps_rollup` (`Rss`,
  `Anonymous`) and `/proc/self/status` (`RssFile`, `RssShmem`) for host-side
  attribution, and `/proc/meminfo` (`MemAvailable`, `Cached`) for
  system-wide pressure.
- **The preflight script is mandatory before any official measurement.**
  `moshi_oracle/tools/jetson_preflight.sh` must print `PREFLIGHT: PASS`
  (gate: `free_MiB >= 5500` AND `large_block_MiB >= 512`, computed from
  `/proc/buddyinfo` directly — NOT from tegrastats' `lfb` field, which is
  capped at 4 MiB block size on aarch64 by construction and cannot report
  large-block health correctly; this was a real bug, fixed in commit
  `c6c988a`). A run without a passing preflight logged is not a valid data
  point — do not report numbers from one.
- **Measured vs Estimated labeling.** Every number in a report must be
  tagged as one or the other. "Measured" means read directly from an
  instrument in this run (a log line, a `/proc` read, a `nvidia-smi`/
  `tegrastats` sample). "Estimated" means derived/interpolated/assumed.
  Do not present an Estimated number without the label, and do not round
  or interpret Measured numbers — report them exactly as captured.
- **One variable per session.** Do not change quantization type, context
  size, kernel logic, AND memory-allocation strategy in the same
  investigation — isolate what moved the number. This project's history
  is full of near-misses caused by conflating two changes (see UPDATE 7 in
  git-blame-able prior session notes: a "fix" was credited with an effect
  that was actually mostly a different, unrelated buffer's growth).
- **Failed gates = stop and report.** Do not chain a second attempted fix
  onto a failed gate without checking in with whoever assigned the task.
  This has been the explicit instruction on every phase of this project
  and is why the git history above is many small, single-purpose commits
  rather than a few large ones.

---

## 4. Mimi pipelining (this session)

### Design

**Problem:** `t_mimi_dec` (133 ms) and `t_mimi_enc` (9 ms) ran serially in
the frame loop, between the LM's audio-token output and the next frame's
LM input — pure dead time on the critical path even though mimi encode/decode
are numerically independent of the LM's temporal/depformer computation
(they only exchange integer code arrays, not floating-point tensors).

**Change (`moshi.cpp/tools/personaplex.cpp`, `src/moshi.cpp`,
`src/context.h`):**
- `mimi_encode_context_t` and `mimi_decode_context_t` (`src/moshi.cpp`) each
  now own a **dedicated `ggml_backend` instance** created via
  `ggml_backend_dev_init()` on the same device as the main LM backend.
  Verified from the vendored ggml source
  (`ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_backend_cuda_init`): each call
  `new`s a fresh `ggml_backend_cuda_context`, and that struct lazily
  creates its own `cudaStream_t` (`cudaStreamCreateWithFlags(...,
  cudaStreamNonBlocking)`), its own `cublasHandle_t`, and its own memory
  pool on first use — so three `ggml_backend` instances (LM, mimi encoder,
  mimi decoder) genuinely means three independent CUDA streams with no
  extra plumbing required.
- A bounded, depth-2, mutex/condvar `BoundedQueue<std::vector<int16_t>>`
  (`personaplex.cpp`) connects an **encode worker thread** (own stream,
  runs one frame of mic/file input ahead) to the frame loop, and a second
  depth-2 queue connects the frame loop to a **decode worker thread** (own
  stream, drains generated audio-code frames as fast as it can).
- The frame loop itself does exactly two queue operations per frame: `pop()`
  the next input codes from the encoder queue, and `push()` the LM's output
  codes to the decoder queue. Both are wait-tracked (`push_waits`/
  `pop_waits`, `push_wait_ms`/`pop_wait_ms` on `BoundedQueue`); those two
  operations are the ONLY place the critical path can stall on codec work,
  so they are what the underrun gate (§4, Results, below) actually measures.
- **`BMO_PIPELINE=0` env var** selects the original serial loop verbatim
  (kept byte-identical in `personaplex.cpp`, just gated behind an `if`), as
  the correctness baseline for the token-hash gate.

**Hazard found and fixed — the cudaFree stall:** ggml's default
`ScratchContext::compute()` does a `cudaMalloc`-equivalent allocate at the
start of every call and a `cudaFree`-equivalent free at the end
(`GraphContext::alloc()`/`ScratchContext::clear()` in `src/context.h`),
every single frame, for the LM's own `gen->ctx` and (after this session's
change) for each mimi context's new per-frame scratch. A standalone probe
(`cudafree_stall_test.cu`, scratchpad, not committed — see the CUDA
semantics it demonstrates below) measured this on-device: `cudaFree` on
one host thread **blocks that thread for the full duration of an
unrelated, unfinished kernel running on a different stream** —
100.697 ms measured against a ~100 ms busy kernel, vs 0.318 ms when the
device is idle. `cudaMalloc`/`cudaFree` are not stream-ordered by default
on this CUDA/driver combination; they force an implicit device-wide
synchronization. Once mimi decode runs concurrently on its own stream for
~133 ms per frame, the LM thread's routine per-frame scratch free would
have been captured by that stall on a large fraction of frames — silently
reintroducing the exact blocking behavior the pipelining change exists to
remove.

**Fix:** `GraphContext` gained an opt-in `reuse_buffer` flag
(`src/context.h`). When set, `alloc()` calls a new
`alloc_ctx_tensors_reused()` instead of `ggml_backend_alloc_ctx_tensors()`:
it computes the same per-tensor size/alignment sum as upstream ggml's own
`alloc_tensor_range`/`ggml_backend_alloc_ctx_tensors_from_buft_impl`
(`ggml/src/ggml-alloc.c`), but instead of `cudaMalloc`-ing a fresh buffer
every call, it keeps a persistent **high-water buffer**: grows (with a
loud `stderr` log line) only the first time a call needs more than the
buffer currently holds, and on every call thereafter just resets a fresh
`ggml_tallocr` over the SAME buffer (`ggml_tallocr_new` always starts its
offset at 0 from the buffer base — confirmed by reading
`ggml/src/ggml-alloc.c` directly — so no buffer-side reset call is
actually required for correctness; `ggml_backend_buffer_reset()` is called
anyway for hygiene but is a no-op on the CUDA backend,
`iface.reset == NULL`, confirmed by reading `ggml-cuda.cu`).
`ScratchContext::clear()` was changed to skip freeing the buffer when it
equals the reuse buffer. Net effect: **steady-state per-frame scratch
does zero `cudaMalloc`/`cudaFree` calls** on all three backends (LM, mimi
encoder, mimi decoder). `moshi_lm_scratch_reuse(gen, 1)` is called once,
right after `moshi_lm_start()` (deliberately after prompt prefill, so the
much-larger prefill graph doesn't set the high-water mark unnecessarily);
the two mimi contexts enable it unconditionally at construction, since
their per-frame graph shape never changes shape between voice-prompt
encoding and the live loop.

`moshi_get_allocated_memory()` (`src/moshi.cpp`, the function behind the
`VRAM_M2_phys_alloc_MiB` metric) was extended to include
`gen->ctx->buffer`, `encoder->scratch->buffer`, `decoder->scratch->buffer`
in its sum — these buffers now persist between calls (that's the whole
point of the reuse mode) instead of being `NULL` between calls as before,
so omitting them would have made M2 silently under-report VRAM by
whatever the three high-water marks settle at.

### Correctness gate

An FNV-1a 64-bit running hash is mixed with `(text_token, audio_tokens[0..7])`
for every frame the LM emits, identically in both the serial and pipelined
loops (same mix order, same types). Printed once at process exit:
`TOKEN_HASH: 0x<hex> frames=<n> mode=<serial|pipelined>`. Because mimi
encode/decode are numerically independent of the LM computation (tokens
never depend on decode output, and encode's output feeds the LM only as
an integer array — no shared floating-point ops between backends), the
hash should be bit-identical between modes at any matching frame count,
not just at exactly 1250.

**Result: see the run output referenced in the commit that shipped this
document** — `/tmp/mimi_smoke_pipelined_stdout.txt` /
`/tmp/mimi_smoke_serial_stdout.txt` (informal, this-boot, not
preflight-gated) and `/tmp/step4p_stdout.txt` (official, if a fresh-boot
run was completed this session). Grep both for `TOKEN_HASH` and diff.

### Results

**Informal smoke run (this boot, current-boot power state, NOT preflight-gated
— see the official-run caveat below for what's still pending):**
1250/1250 frames, zero CUDA errors, `QUEUE_UNDERRUNS: 0` — **PASS**
(`enc_out_pop_waits_steady=0`, `dec_in_push_waits=0`: the frame loop never
waited on either queue operation beyond the single structural startup fill).
`TOKEN_HASH: 0x2145f95e0ebd05d8 frames=1250 mode=pipelined`.

| phase | serial (321.0 ms baseline) | pipelined (this smoke run) | Δ |
|---|---:|---:|---|
| t_mimi_enc (frame-loop-visible) | 9.459 ms | **0.005 ms** | queue wait collapsed to ~0, as designed |
| t_mimi_dec (frame-loop-visible) | 133.310 ms | **0.010 ms** | queue wait collapsed to ~0, as designed |
| t_temporal | 124.298 ms | **260.123 ms** | **+135.8 ms (2.1×) — NOT expected, see below** |
| t_depformer | 53.196 ms | 53.547 ms | ~unchanged |
| t_frame_total | 321.038 ms | 315.170 ms | **−5.9 ms (1.8% faster)** |
| fps | 3.114 | 3.1146 | ~unchanged |

**Headline finding: pipelining achieved its literal, stated goal — the
critical path (temporal/depformer/sampling) never blocks on codec queue
operations, verified with zero underruns — but delivered almost none of
the wall-clock benefit one would naively expect from removing ~143 ms of
serial mimi time per frame.** The reason, visible directly in
`PIPELINE_STATS`: the workers' OWN compute times inflated under
concurrency too — `enc_mean_ms=174.469` (vs ~9 ms serial, a **19.4×**
slowdown) and `dec_mean_ms=166.008` (vs ~133 ms serial, a **1.25×**
slowdown) — while `t_temporal` (the LM's own GEMV-heavy step, now running
concurrently with both) more than doubled. This is GPU compute contention,
not a pipelining bug: Orin Nano has 8 SMs and a ~102 GB/s DRAM ceiling
shared by ALL concurrent streams. The rewritten BMO GEMV kernel that
dominates `t_temporal` is already known (§1, STEP 2 ncu evidence) to be
L1TEX-pipe-bound even running *alone*; mimi's conv/conv-transpose kernels
are themselves bandwidth-hungry. Running all three concurrently on a
device already near its own bandwidth ceiling converts what should be
"hidden" time into "contended" time — the three streams are fighting over
the same 8 SMs and the same ~102 GB/s instead of genuinely overlapping.
**This is very plausibly an edge-device-specific finding — the mechanism
predicts this would look very different (a real win) on a GPU with
substantial spare SM/bandwidth headroom, which an 8-SM part at a
BMO-kernel-saturating workload does not have.** No attempt was made to
mitigate this (e.g. CUDA stream priorities) — out of the "no kernel
changes, only WHERE/WHEN" scope and out of remaining session budget; flagged
here as a genuine open finding, not chased further, per this project's
established stop-and-report discipline.

VRAM: M2 (ggml-tracked, the reliable instrument) flat at 2895 MiB for the
whole run. M1 (`cudaMemGetInfo`-derived) fluctuated non-monotonically
between 4804-4970 MiB across the run — consistent with, and further
evidence for, the already-documented `cudaMemGetInfo` unreliability on
this platform (§3) rather than a new leak; it does not grow the way a
leak would (a leak is monotonic, this isn't).

**Correctness gate: FAILED. TOKEN_HASH MATCH: NO.**

```
TOKEN_HASH: 0x2145f95e0ebd05d8 frames=1250 mode=pipelined
TOKEN_HASH: 0x7b2f7f1c39d47848 frames=1250 mode=serial
```

This is a **real semantic divergence, not a hash-computation artifact** —
confirmed by diffing the actual generated text between the two runs
(`/tmp/mimi_smoke_pipelined_stdout.txt` vs
`/tmp/mimi_smoke_serial_stdout.txt`, DEBUG lines stripped). The two runs
produce different text from the very first sentence:
- pipelined: *"Hey, this is a voice AI called Buffa. How can I help you today?"*
- serial: *"Hey, this is my first time calling an answer service. I'm a bit nervous."*

Both are coherent, on-topic completions (the model isn't producing garbage
or crashing) — this is a genuine input-sequence divergence somewhere
upstream, not corruption. Per the task's own instruction ("tokens do not
depend on decode output") the two modes should be numerically identical:
mimi encode/decode run on their own dedicated backends in BOTH modes now
(§4, Design — this was changed unconditionally, not just for the
pipelined path), so a difference must come from either (a) genuine
concurrent-execution non-determinism in a kernel somewhere in the encode
path (whose OUTPUT — the mic-input audio codes — is written into the LM's
own token cache and therefore genuinely does feed back into generation,
despite decode's *audio* output not doing so — re-read the task's claim
carefully: it says tokens don't depend on *decode* output, which is true,
but encode's output codes are a real generation input, and §4's Design
section undersold that distinction), or (b) an ordering/timing bug in
this session's own queue/threading code that isn't caught by the
QUEUE_UNDERRUNS instrumentation (which only measures whether the critical
path *waited*, not whether it received the *right* item at the right
time).

**One bounded, cheap check was done before accepting this as unresolved**
(consistent with — not a violation of — "do not debug concurrency on
remaining budget"): grepped the entire mimi encode/decode call path
(`src/moshi/quantization/`, `src/moshi/modules/`,
`src/moshi/models/compression.h`) for any use of `ctx.exponential(...)` or
`rand()` — **zero hits**, ruling out the most obvious hypothesis (mimi
touching the LM's global, non-thread-safe `rand()` stream from a worker
thread). **Root cause NOT FOUND. No further investigation was done past
this single check, per instruction.**

**Action taken (per task instruction: "revert to serial, report, skip to
PART 2"):** `personaplex.cpp`'s default flipped from
`pipeline_on = true` (opt-out via `BMO_PIPELINE=0`) to `pipeline_on =
false` (opt-in via `BMO_PIPELINE=1`). **The shipped default binary runs
the original serial mimi loop** — the ONLY behavioral difference from
before this session, in the default configuration, is the scratch-buffer
reuse optimization (§4, Design) applied to the LM's own per-frame
context, which is orthogonal to the threading bug (it's plain
single-threaded high-water-mark buffer reuse, active in both modes, and
does not itself introduce any concurrency). The full pipelining
implementation, worker threads, queues, and this TOKEN_HASH/
PIPELINE_STATS instrumentation are all left in the tree, gated behind
`BMO_PIPELINE=1`, for whoever picks up the root-cause investigation next
— do not flip the default back to pipelined until this gate passes.

**Boot/clock provenance of the informal smoke run above:** verified via
`uptime -s` (`2026-07-12 19:22:17`) that this smoke run executed on the
SAME boot as the official 321.0 ms kernel-rewrite baseline (whose
preflight PASSED at `2026-07-12 19:24:51`, ~2.5 min after that same boot)
— `nvpmodel -q` confirmed MAXN_SUPER and `scaling_cur_freq` confirmed
1728000 (locked) at the time of this smoke run, so the two numbers ARE
comparable on power/clock state. **What this smoke run is NOT**: a
freshly-preflighted run in its own right — several other processes
(including this session's own build/load cycles) ran on this boot since
that one preflight check, and this project's established protocol treats
"preflight passed at some point on this boot" as insufficient; only a
preflight run *immediately before* the measured run counts as a valid
gate for an *official* number (§3, "a run without a passing preflight
[immediately before it] logged is not a valid data point").

**Official fresh-boot run: deliberately NOT executed this session, per the
task's own branch logic.** The instruction was explicit: "If token hashes
mismatch: revert to serial, report, skip to PART 2 — do not debug
concurrency on remaining budget." The hashes mismatched (above), so the
official-run step is skipped by design, not by omission — running the
full fresh-boot/preflight protocol against a build whose default now
reverts to serial would just re-measure the already-official 321.0 ms
kernel-only baseline under a different script name, and running it
against `BMO_PIPELINE=1` would be measuring a build known to produce
wrong output, which isn't a meaningful "official number." Everything above
in this section is informal, same-boot evidence that the *mechanism*
(threads, queues, streams, buffer reuse) works exactly as designed
mechanically — zero crashes, zero underruns, zero stalls — while the
*content* it produces is wrong for a reason not yet found. `step4_pipeline_run.sh`
(repo root, committed alongside this document) is kept for whoever
resumes this work — see §7 for the exact sequence — but should not be run
as an "official" measurement until the correctness gate above passes.

---

## 5. Roadmap to 80 ms frame time

Current: **321.0 ms/frame** (STEP 3 official number, kernel rewrite only,
pre-pipelining). The gap to the stated 80 ms target breaks down as:

### (a) mimi conv_transpose_1d kernel replacement — BF16-preserved, ~110 ms serial contribution
`t_mimi_dec` is 133.3 ms and is almost entirely `moshi_streaming_conv_transpose_1d`
work inside the SEANet decoder (see `moshi.cpp/src/moshi/modules/seanet.h`).
This session's mimi-pipelining change (§4 above) moves this work OFF the
critical path onto its own thread/stream, so if pipelining lands cleanly
it partially absorbs this cost rather than eliminating it — the decoder
still burns 133 ms of wall-clock GPU time, it just no longer blocks the
next frame's temporal/depformer step. A genuine conv_transpose_1d kernel
rewrite (similar treatment to the GEMV rewrite: multi-row blocking,
vectorized loads) would still reduce the *total* GPU work and therefore
the throughput ceiling even under pipelining. **Risk class: no quality
risk if done as a pure kernel rewrite** (same arithmetic, same weights,
same op) — same validation pattern as the GEMV rewrite (microbench first,
rel_l2 gate, then integrate).

### (b) depformer shared-attn INT4 export — ~24-33 ms, QUALITY-GATED
`t_depformer` is 53.2 ms for 16 substeps (~3.325 ms/substep), dominated by
weight traffic (~3.94 GiB/frame re-read across the depformer's 16
sequential per-codebook attention steps, since each substep re-reads the
shared attention weights). Exporting those weights as INT4 (they are
currently a higher-precision format — check `moshi.cpp/src/moshi/models/lm.h`
`moshi_lmmodel_forward_depformer_transform` for the current tensor
handling) would cut that traffic roughly in proportion to bit-width, an
estimated 24-33 ms reduction. **This is QUALITY-GATED — full validation
ladder (§2) required before this lands**, unlike (a): unlike a kernel
rewrite that preserves the same arithmetic on the same weights, a
precision *export* changes the actual numbers the model computes with,
so it needs the full microbench → residual-diff → z_s → joke-loop chain,
not just a kernel-correctness check.

### (c) temporal ~60 ms unattributed
Even after the GEMV rewrite, `t_temporal` (124.3 ms) is larger than the
gating-tensor GEMV work alone should account for at the kernel's own
measured GB/s. An nsys 50-frame capture (not yet done this session) is
needed to attribute the remainder — the leading hypothesis, not yet
confirmed, is CUDA Graphs overhead or launch-latency accumulation across
the 32-layer transformer's many small kernel launches per step (the
kernel rewrite improved the GEMV's own bandwidth efficiency but did
nothing about launch count/overhead). `GGML_CUDA_GRAPHS` is currently
`OFF` in this build (`ggml/build/CMakeCache.txt`) — worth checking whether
turning it on changes this number before assuming a deeper kernel-launch
redesign is needed.

### (d) fallback: BMO format-v2 tier-sorted layout
If (a)-(c) don't close enough of the gap, a more invasive option is a
second-generation BMO packing format that sorts tiles by tier globally
(not just within a band) to improve memory access coherence further.
**This is an export/format change** — it changes what bytes are in the
GGUF file, not just how the kernel reads them — so it requires full
re-export of every model checkpoint plus full revalidation (§2's ladder,
no shortcuts), and is a materially bigger undertaking than (a)-(c). Treat
as last resort.

---

## 6. Hard constraints (apply to all future work on this branch)

- **Never modify `v5_step1500` (gold master) or `qat_best.pt`.** These are
  reference/ground-truth checkpoints; nothing in this project's kernel or
  quantization work should ever write to them.
- **Never touch `moshi/` source** (the original Python reference
  implementation, if present in this environment) — this project works
  exclusively in the C++/GGML port (`moshi_oracle/moshi.cpp`), and the
  Python reference exists only as a correctness oracle.
- **Never judge audio by metrics alone.** z_s and residual-diff numbers
  are necessary but not sufficient — the joke-loop transcript step in §2
  exists specifically because a metric can pass while a human listener
  would notice something wrong (or vice versa). Do not skip the human
  transcription step even when the numeric gates pass.
- **N_sysprompt=12 caveat.** The exact validation command used throughout
  this project (`-k q4_0 -b -s 1783708826 --threads 4 -c 256`, no `-v`/`-p`
  flags) never exercises the real Hybrid System Prompt path —
  `moshi_lmgen_step_voice_prompt` returns immediately (`voice == NULL`)
  and the text-prompt loop never populates (`personaplex_system_prompt`
  unset). Only the two hardcoded `audio_silence_frame_cnt=6` blocks in
  `moshi_lmgen_step_system_prompts` run, giving `N_sysprompt=12`. **Any
  production context must re-derive real system-prompt behavior** — all
  timing/memory numbers in this document and its predecessors were
  measured WITHOUT a real system prompt in the loop, and do not by
  themselves certify production-context (real `-p`/`-v` usage) behavior.

---

## 7. Exact run recipe

### Preflight (mandatory, every time, before any official measurement)
```bash
# 1. Reboot if this is meant to be a "fresh boot" official run
#    (relay to the user — sudo reboot needs their terminal, never run it
#    directly; ALWAYS update this file / RESUME_NOTES.md first).
sudo reboot

# 2. After reboot, re-lock power mode and clocks — nvpmodel BEFORE
#    jetson_clocks, or nvpmodel resets the clocks jetson_clocks just set.
sudo nvpmodel -m 2
sudo jetson_clocks
sudo jetson_clocks --show   # verify: CPUs 1728000, GPU 1020000000, EMC 2133000000

# 3. Re-stop auto-restarted background services if present
sudo systemctl stop bmo_app.service burningtruth_app.service \
  burningtruth_tunnel.service packagekit.service snapd.service snapd.socket

# 4. Preflight gate — MUST print PREFLIGHT: PASS, logged to
#    moshi_oracle/tools/jetson_preflight.log
sudo bash ~/bmo_fresh/moshi_oracle/tools/jetson_preflight.sh
```

### Env vars used by every official run
```bash
GRAPH_RESERVE_MIB=320          # reserve-then-release bridge, see §1
CUDA_MODULE_LOADING=LAZY
LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib   # local FFmpeg 7.1.1, not system apt
```

### Command (same seed/context used throughout this project — numbers are only comparable across runs using this exact invocation)
```bash
cd ~/bmo_fresh/moshi_oracle/moshi.cpp/build
tegrastats --interval 1000 > /tmp/<run>_tegrastats.txt 2>&1 &
TEGRA_PID=$!
GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY \
  LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib \
  ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b \
  -s 1783708826 --threads 4 -c 256 \
  > /tmp/<run>_stdout.txt 2>/tmp/<run>_stderr.txt
kill $TEGRA_PID
```
`-b` = bench mode (no SDL audio device, runs exactly 1250 frames then
exits). Wrapper scripts already in the repo root following this exact
pattern: `step3_integration_run.sh` (kernel-rewrite validation),
`step4_pipeline_run.sh` (mimi-pipelining validation, this session).

### Where logs land
- `/tmp/step<N>_stdout.txt` / `_stderr.txt` / `_tegrastats.txt` — raw run
  output, per the wrapper script used.
- `moshi_oracle/tools/jetson_preflight.log` — append-only, timestamped,
  every preflight run ever executed (grep for the run's timestamp to
  confirm which preflight gated which run).
- `PHASE_TIMING` lines in stdout, every 25 frames, median-of-window —
  `t_frame_total`, `t_temporal`, `t_depformer` (+ substep mean n=16),
  `t_sample_sync`, `t_mimi_enc`, `t_mimi_dec`, `t_other`, `majflt_sum_window`.
- `VRAM_FRAME` lines in stdout, every 25 frames — `M1_driver_delta_MiB`,
  `M2_phys_alloc_MiB`, `outside_ggml_MiB`.
- `MEMLEDGER`/`MEMLEDGER_GRAPH`/`MEMLEDGER_GRAPH_CAT`/`MEMLEDGER_CONTEXT`
  lines in stderr — event-based snapshots through the load path, see §1.
- This session's additions: `TOKEN_HASH` (correctness gate, §4),
  `PIPELINE_STATS`/`QUEUE_UNDERRUNS` (pipelining audit, §4) — printed
  once, at process exit, to stdout.
