# BMO-Project Jetson benchmark — resume notes (post-reboot)

State as of just before the reboot triggered to clear NvMap/CUDA carveout fragmentation.

## Repo state (all done, don't redo)
- Fresh clone at `~/bmo_fresh/moshi_oracle` (NOT `~/bmo_fresh/BMO-Project/moshi_oracle` — clone target dir IS the repo root)
- Branch `experiment/multitier-dequant` checked out, commit `eaad94d`
- `~/bmo_fresh/moshi_oracle/models` -> symlink to `~/bmo_models` (repo's own tracked `models/` dir was removed first, since it only had placeholder configs, no weights — verified via git status it's fully tracked/recoverable)

## Pre-flight done (verify still true after reboot, re-set if not)
- Headless: `multi-user.target`, no GDM/Xorg — was already true before any of this
- Power mode: **MAXN_SUPER** (mode 2), NOT the originally-instructed mode 0/15W — user changed it manually. Re-run after reboot:
  `sudo nvpmodel -m 2 && sudo jetson_clocks && sudo jetson_clocks --show`
  Expect: CPUs pinned 1728000, GPU pinned 1020000000, EMC pinned 2133000000.
  IMPORTANT: nvpmodel must run BEFORE jetson_clocks, or nvpmodel resets the clocks jetson_clocks just set.
- Background services stopped (may restart on boot, check and re-stop if needed):
  `bmo_app.service burningtruth_app.service burningtruth_tunnel.service packagekit.service snapd.service snapd.socket`
  Note: also running but NOT stopped (pre-existing, unrelated): cloudflared tunnel for BMO-LabelData (PID varies), `python3 -m http.server 8080` for BMO-Landing, jtop x2, pulseaudio.
- sudo caching: the Bash tool's shell has no TTY, so cached sudo doesn't apply there. Any sudo command must be run by the user via the `!` prefix in their own terminal.

## Build (COMPLETE, all binaries linked — do not rebuild unless files are gone)
Binary at `~/bmo_fresh/moshi_oracle/moshi.cpp/build/bin/personaplex` (179MB, built successfully).

Build required 3 non-obvious fixes beyond the original protocol's single cmake command:
1. **GGML and SentencePiece must be built first**, separately, then moshi.cpp linked against them via `-DGGML_INCLUDE_DIR/-DGGML_LIBRARY_DIR/-DSentencePiece_INCLUDE_DIR/-DSentencePiece_LIBRARY_DIR` (all as **absolute paths** — relative paths resolve wrong, against source dir not CWD). Already built at:
   - `~/bmo_fresh/moshi_oracle/ggml/build` (ninja, GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=87)
   - `~/bmo_fresh/moshi_oracle/sentencepiece/build` (ninja)
2. **FFmpeg 4.4 (system/apt, including NVIDIA's L4T-pinned version) is too old.** The branch's `personaplex.cpp`/`common_av.h` needs FFmpeg 7.1+ (`avcodec_get_supported_config`, `AV_CODEC_CONFIG_*`, introduced FFmpeg 7.1 Sept 2024). Built FFmpeg 7.1.1 from source, local prefix only (did NOT touch system packages, to avoid breaking NVIDIA's hardware-accelerated ffmpeg used elsewhere):
   - Source + install at `~/ffmpeg_local/` (`~/ffmpeg_local/install/lib/pkgconfig` has the .pc files)
   - Must set `PKG_CONFIG_PATH=$HOME/ffmpeg_local/install/lib/pkgconfig` when running cmake for moshi.cpp
   - Must set `LD_LIBRARY_PATH=$HOME/ffmpeg_local/install/lib` at RUNTIME (shared libs, non-standard prefix, not copied into bin/)
3. **Missing link flags**: building against raw `.a` files (not GGML's own exported CMake package) means CUDA runtime/driver/cublas and OpenMP aren't pulled in transitively. Needed:
   - `-DCMAKE_EXE_LINKER_FLAGS="-fopenmp"`
   - `-DCMAKE_CXX_STANDARD_LIBRARIES="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcuda"` (must be CXX_STANDARD_LIBRARIES, not EXE_LINKER_FLAGS, so it's appended AFTER the .a archives on the link line — ld is order-sensitive for static libs)

Full working cmake invocation (from `~/bmo_fresh/moshi_oracle/moshi.cpp/build`):
```bash
GGML_ROOT=$(cd ~/bmo_fresh/moshi_oracle/ggml && pwd)
SPM_ROOT=$(cd ~/bmo_fresh/moshi_oracle/sentencepiece && pwd)
export PKG_CONFIG_PATH=$HOME/ffmpeg_local/install/lib/pkgconfig
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_NATIVE=OFF \
  -DGGML_INCLUDE_DIR=${GGML_ROOT}/include \
  -DGGML_LIBRARY_DIR=${GGML_ROOT}/build/src \
  -DSentencePiece_INCLUDE_DIR=${SPM_ROOT}/src \
  -DSentencePiece_LIBRARY_DIR=${SPM_ROOT}/build/src \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcuda"
ninja -j2
```
Also required (installed via apt, system packages, already done): `libsdl2-dev libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev` (the dev headers were needed at configure/compile time even though the actual FFmpeg *linked* is the local 7.1.1 build).

## Numbers recorded so far
- STAGE0_free_MiB: literal `free`=368 (FAILs literal 3000 gate), `available`=5916 (PASSES) — user chose to score PASS using `available`, since `free` is not the correct Linux memory-pressure metric when buff/cache is reclaimable.
- STAGE1_overhead_MiB = 51, budget 479 → PASS
- STAGE2: build succeeded, but **runtime FAILS** — `cudaMalloc`/`cudaMallocManaged` both fail allocating the 4257.69 MiB weight tensor buffer with `NvMapMemAllocInternalTagged error 12` (ENOMEM at the NvMap/carveout level), despite `ggml_backend_cuda_get_available_uma_memory` reporting 4.6-5.1 GB nominally available. `tegrastats` showed `lfb` (largest free block) capped at only 4MB blocks even after `sudo sysctl vm.compact_memory`-style compaction (`echo 1 | sudo tee /proc/sys/vm/compact_memory`) — meaning fragmentation in the NvMap/CMA carveout specifically, not ordinary page cache, which normal compaction doesn't reach. Tried `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (existing env var in this checked-out ggml-cuda.cu, switches to `cudaMallocManaged`) — same failure, confirming even managed allocations route through the same fragmented NvMap pool on Tegra.
- **User chose: reboot to reset the GPU carveout.** This is the action being taken now.

## Exact command to retry Stage 2 after reboot + clock re-lock
```bash
cd ~/bmo_fresh/moshi_oracle/moshi.cpp/build
CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=$HOME/ffmpeg_local/install/lib \
./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 \
2>/tmp/jetson_stderr.txt 1>/tmp/jetson_stdout.txt
echo "EXIT: $?"
grep "BMO_TENSOR_LOAD" /tmp/jetson_stderr.txt | head -5
grep "BMO_TENSOR_LOAD" /tmp/jetson_stderr.txt | awk -F'n_tiles_col=' '{print $2}' | awk '{print $1}' | sort -n | tail -1
```
If it OOMs again immediately post-reboot before any of our build activity has had a chance to re-fragment memory, that would mean the model genuinely doesn't fit in this device's real budget — a materially different conclusion than "was fragmented," worth flagging clearly rather than assuming another reboot will help.

## Also still pending after Stage 2 succeeds
Stage 2 remaining checks (VRAM_M1/M2 at load, swap during load), then Stage 3 (1250-frame growth curve, frame rate gate ≥12.5fps), Stage 4 (conditional on Stage 3 failing), and the final report template — none of these have been run yet.

---

## UPDATE — MEMLEDGER investigation (new task, separate from the original protocol)

After the first post-reboot Stage 2 retry, the run got MUCH further (25/32 transformer
layers loaded, clean unfragmented boot) before OOMing on a plain 27 MiB allocation.
This ruled out fragmentation as the cause and pointed at a real capacity/accounting
problem. User's follow-up task: **measure only, do not fix, do not change quant/context**.
Build a memory ledger of the load path to find where ~2GB of "missing" memory goes
(expected ~5.0GB total, only 1.4GB of BMO payload was logged before crash).

### Source findings already confirmed by reading code (not yet runtime-verified)
1. **Q_D answered, high confidence, no instrumentation needed**: `BMO Voice Engine/personaplex/bmo_compute.cpp`
   (canonical_pw_dev / row_c4 etc.) is a COMPLETELY SEPARATE, disconnected directory tree
   from `moshi_oracle/moshi.cpp` (the actual build we run). None of those symbols appear
   anywhere in `moshi_oracle/`; the only substring hits were unrelated Apple Metal GPU
   kernel names (`row_c4_fuse`) in ggml's Metal backend, which isn't even compiled in
   (GGML_CUDA=ON only). There is no second BMO weight copy from that file — it's not linked.
2. **Likely major lead (needs runtime confirmation)**: in `loader.h`, `load_gguf()` loads
   EVERY raw named GGUF tensor into ONE shared buffer first — including each BMO tensor's
   raw `.packed_weights`/`.tile_tiers`/`.outlier_indices`/`.outlier_values` component
   tensors (they're ordinary named GGUF tensors at that stage). THEN `build_custom_ffn_tensor()`
   reads those back via `read_bytes()` (a GPU→CPU `ggml_backend_tensor_get` copy) and
   re-uploads a repacked copy into a BRAND NEW separate buffer. Nothing in the code visibly
   frees the original raw component tensors afterward — they stay resident in the shared
   buffer for the WeightLoader's lifetime. This is a plausible near-doubling of BMO payload
   bytes on the GPU. NOT YET CONFIRMED WITH NUMBERS — that's what the ledger is for.
3. **Structural fact**: BMO tensors each get their OWN separate `ggml_backend_buffer_t`
   (one `cudaMalloc`-equivalent per tensor, via `ggml_backend_alloc_ctx_tensors(custom_ctx, backend)`
   called once per BMO tensor at loader.h — previously the returned buffer handle was
   discarded, now captured for the ledger). Non-BMO tensors share ONE bulk buffer for all
   of them. This asymmetry means BMO loading incurs many more separate allocation calls
   (possibly hundreds), each with its own NvMap overhead/fragmentation exposure — contrast
   with the single-big-block theory from before the last reboot, which this data will refute.
4. **`moshi_get_allocated_memory()` (src/moshi.cpp ~line 993, the function behind the
   protocol's own `VRAM_M2_phys_alloc_MiB` metric) only sums `lm->weights->buffer`** — the
   ONE buffer field on WeightLoader, set only by `load_gguf()`'s single bulk allocation.
   It NEVER sees the per-BMO-tensor buffers from `build_custom_ffn_tensor()`, because those
   buffer handles were never stored anywhere retrievable (until this session's instrumentation).
   **This means the Stage 2/3 protocol's own official M2 number has always undercounted
   actual VRAM usage by the full sum of all BMO tensor allocations.** This is a structural
   fact independent of the OOM investigation, but it means Stage 2/3's VRAM_M2 numbers
   (if reached) should be treated as unreliable without a fix — NOT applied in this session
   per the "measure only" instruction, just flagged.

### Instrumentation added (MEMLEDGER) — already built successfully
Format: `MEMLEDGER event=<e> name=<n> type=<t> payload_B=<> alloc_B=<> nbytes_B=<> cuda_free_MiB=<> rss_MiB=<>`

- `moshi.cpp/tools/personaplex.cpp`: `process_start` (top of main, before any init — RSS
  baseline only, cuda_free_MiB=0/NA since no backend yet), `after_ggml_init` (right after
  `init_ggml()` — CUDA context floor, analogous to Stage 1's cublas overhead number).
- `moshi.cpp/src/moshi.cpp`: `after_gguf_parse` in `moshi_lm_from_files()` (file opened,
  `no_alloc=true`, zero tensor data uploaded yet — the literal pre-upload baseline the
  task asked for). `after_kv_cache` in `moshi_lm_start()`, right after `state_ctx->init()`
  — NOTE: this event will NOT appear in a run that OOMs during weight loading, since
  `moshi_lm_start()` runs much later (after mimi encoder/decoder alloc), well past the
  crash point. Don't expect to see it in the captured log — that's expected, not a bug.
- `moshi.cpp/src/loader.h`: `after_gguf_buffer_alloc` (the ONE bulk buffer alloc for all
  non-BMO tensors, right after it happens, before any tensor data copied — reports the
  total shared-buffer size via `ggml_backend_buffer_get_size`). `per_tensor_regular` for
  every non-BMO/raw GGUF tensor upload (alloc_B is deliberately 0 — no separate per-tensor
  buffer exists in the shared-buffer path, this is documented as intentional in a comment,
  not a bug). `after_load_gguf_raw` milestone once the whole `load_gguf()` loop finishes
  (all raw tensors including BMO raw components resident). `per_tensor_bmo` for every BMO
  tensor in `build_custom_ffn_tensor()`, with alloc_B now genuinely captured from the
  previously-discarded buffer handle (real fix to a real blind spot, not a guess).

Helper functions (`memledger_cuda_free_mib`, `memledger_rss_mib`, `memledger_log`) added
near the top of `loader.h`; `personaplex.cpp` has its own small standalone duplicate
(`memledger_rss_mib`, `memledger_log_simple`) since it's a separate translation unit and
already has `device_memory_free()` available via `common_ggml.h` for CUDA free bytes.

**Explicitly did NOT touch**: quantization type, context size (`-c` flag), the
`std::vector<char*> data;` line in `load_gguf()`'s read loop (loader.h, in the per-tensor
loop) — this looks like a real bug (vector of 8-byte pointers being resized/fread'd as if
it were a byte buffer, i.e. an 8x host-RSS over-allocation for the raw tensor staging
buffer, reused/grown across the whole loop) but per the "measure only, no fixes" instruction
it was left completely alone. Worth flagging prominently in the report and checking
whether the RSS ledger numbers corroborate an 8x-driven spike — if they do, this buffer
bug is a strong candidate as the dominant unaccounted consumer, separate from the
double-storage hypothesis above (one is host RSS pressure competing with GPU allocation
on this Jetson's unified memory; the other is direct GPU buffer duplication).

### Rebuild status
Full rebuild succeeded (all binaries linked cleanly, `bin/personaplex` timestamp confirms
new code is in the binary) using the exact same cmake configuration documented above
(no changes to that recipe were needed — just `ninja -j2` after editing sources).

### Next step (Step 3 of the MEMLEDGER task) — NOT YET DONE
Needs a fresh reboot first (same rationale as before — eliminate build-session memory
clutter as a confound), then re-lock clocks / re-stop services per the pre-flight section
above, then run:
```bash
cd ~/bmo_fresh/moshi_oracle/moshi.cpp/build
tegrastats --interval 1000 > /tmp/memledger_tegrastats.txt 2>&1 &
TEGRA_PID=$!
CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=$HOME/ffmpeg_local/install/lib \
./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 \
2>/tmp/memledger_stderr.txt 1>/tmp/memledger_stdout.txt
kill $TEGRA_PID
grep MEMLEDGER /tmp/memledger_stderr.txt > /tmp/memledger_only.txt
```
Then analyze `/tmp/memledger_only.txt` against the Q_A–Q_E questions and build the
Q_E table (sum(payload_B), sum(alloc_B), RSS, cuda_free delta, tegrastats RAM/swap at
crash point), per the strict evidentiary bar in the task (quote actual log lines, don't
infer, FAIL LOUDLY on anything that can't be measured from the logs).

---

## UPDATE 2 — BMO double-storage FIX implemented, needs on-device verification

Root cause fully diagnosed (see full MEMLEDGER report above): BMO gating tensors'
raw sub-components (`.packed_weights`, `.tile_tiers`, `.outlier_indices`,
`.outlier_values` — the 4 LARGE fields, NOT the tiny scalar metadata) were loaded
into the shared device buffer by `load_gguf()`, then read back via
`ggml_backend_tensor_get()` and re-uploaded as a repacked BMO_TIER tensor in a
SEPARATE buffer — original never freed. Measured: 1739.85 MiB of pure redundant
storage (layers 0-30 only; layer 31 is a single plain F16 tensor, untouched).

### Fix implemented in `moshi.cpp/src/loader.h` (uncommitted — commit message
per the task should be: "loader: host-stage BMO raw components, free after
repack (fixes Jetson double-storage OOM)")

Mechanism: `ggml_backend_alloc_ctx_tensors` (ggml-alloc.c:1186) skips any tensor
where `t->data != NULL`. `load_gguf()` now pre-marks the 4 large BMO sub-component
tensors (layers 0-30 only, via `is_bmo_big_subcomponent()` — matches `_gating_
linear_in_weight.` / `_gating_linear_out_weight.` + one of the 4 big suffixes;
requires a trailing dot, so layer 31's plain single tensor never matches) with a
sentinel `data = (void*)1` BEFORE the bulk buffer allocation, excluding them from
the shared device buffer entirely. `build_custom_ffn_tensor()` now reads those 4
fields directly from the GGUF file (`read_raw_bytes_from_gguf_file()`, new method:
`gguf_find_tensor` + `gguf_get_data_offset`/`gguf_get_tensor_offset`/
`gguf_get_tensor_size` + fresh `fopen`/`fseek`/`fread`/`fclose` per call) instead
of `ggml_backend_tensor_get()`. Small scalar siblings (`.rows`,`.cols`,
`.n_outliers`,`.scale_*`,`.zp_*`,`.n_tiles`,`.tier_offsets`,`.packing_version`,
4-20 bytes each) deliberately left on the normal device path — negligible size,
not worth the extra risk.

**Important side-fix**: the existence guard in `build_custom_ffn_tensor()`
(`if (tensors.find(name_pw) == tensors.end()) return NULL;`) had to change to
`if (gguf_find_tensor(gguf, name_pw.c_str()) < 0) return NULL;`, since
`.packed_weights` is now deliberately excluded from the `tensors[]` map — the
old check would have always returned NULL (false), breaking BMO detection
entirely, if left unchanged.

Peak host memory is bounded to ~one tensor (~24-36 MiB) automatically via C++
scope — `read_raw_bytes_from_gguf_file()`'s returned vectors are locals inside
`build_custom_ffn_tensor()`, freed when that call returns. No manual per-layer
buffer/free bookkeeping was needed (simpler than the task's draft design implied).

New MEMLEDGER events added: `bmo_subcomponents_excluded` (prescan count),
`bmo_subcomponent_skipped` (per skipped raw tensor), `bmo_layer_done`
(once per layer, after both gating tensors repacked — tracked via new
`memledger_bmo_layer_tensor_count` map member on WeightLoader).

**Build status: succeeded cleanly**, `bin/personaplex` rebuilt (timestamp Jul 11
17:24), no compile errors.

**Git status: NOT YET COMMITTED.** Do not commit until on-device verification
(Step 6/7 below) passes — task says commit as one PR only after verifying.

### Step 5's attention-retention finding (no fix, per task scope) — already answered
No `per_tensor_bmo`-style conversion evidence exists anywhere in the ledger for
any `self_attn` tensor. Combined with the fact the raw split components
(i8 packed_weights + f32 scales/zeros, confirmed via byte-math to be exactly
4.5 bits/weight = already-compact INT4 density) are the ONLY resident copy
observed, the most likely read: attention Format-2 tensors are used directly
at runtime in their raw split form, with no second (converted) copy — but this
was not directly instrumented (no log call was ever added to
`build_quantized_attn_tensor()`), so this is an evidence-based inference from
absence, not a certified measurement.

### UPDATE 3 — bug found + fixed, correctness now confirmed; ONE MORE fresh-reboot run needed for authoritative 6a-e numbers

**Bug found (self-caught during first verification attempt):** the fix's first build broke BMO tensor
resolution entirely (`per_tensor_bmo` count = 0 for all 32 layers — every gating tensor silently
resolved to NULL, "done loading" still printed with no crash, since NDEBUG builds silently accept
NULL — a functionally broken model that would have looked like a clean success from stdout alone).
Root cause: `get_tensor()` (loader.h ~line 885) has its OWN existence guard
`if (tensors.find(name_pw) != tensors.end())` gating whether `build_custom_ffn_tensor()` is ever
called at all — I had only fixed the guard INSIDE `build_custom_ffn_tensor()` itself, not this outer
one. Since `.packed_weights` is deliberately excluded from `tensors[]`, this outer check always
returned false. **Fixed**: changed to `gguf_find_tensor(gguf, name_pw.c_str()) >= 0` (same pattern
as the other fix), which is correct for BOTH the BMO path (packing_version==6) and the attention
path (packing_version==10, untouched by the exclusion, unaffected by this change).

**Re-verified after fix, still not a fresh reboot (reused state after the first attempt's clean exit):**
- `per_tensor_bmo`=62 (31×2, correct), `bmo_layer_done`=31 (correct), zero weight-lookup NULLs for
  gating (only pre-existing `_bias` NULLs remain — bias-less linear layers, unrelated/expected).
- Main LM shared buffer: 4257.69 → **2517.86 MiB** (saved 1739.83 MiB, matches measured redundant
  total almost exactly). Total device allocation: 2848.95 + 1741.86 = **4590.81 MiB**, matches the
  task's own projection "2,849 + 1,742 ≈ 4,591" to within 0.2 MiB.
- `done loading. 265.048004` printed — full load success, no OOM, no crash.
- BUT: not a fresh reboot, so peak RAM (6947/7620 MiB) and swap (nonzero, 1→21 MB throughout the
  whole tegrastats capture) are ambiguous against the task's "swap during steady decode must be 0"
  rule — can't tell if the persisting swap at the end is residual from peak-load or genuine
  steady-decode swapping, since the process appeared to have already exited by the final samples.

**User decision: redo with a genuine fresh reboot before trusting 6a-e's numbers or proceeding to
Stage 3 / committing.** This is the very next step.

### UPDATE 4 — crash root-caused via gdb, fixed, three commits landed

Ran the fresh-reboot verification. 6a/6b confirmed again exactly (per_tensor_bmo=62,
bmo_layer_done=31, allocations match projection). But process died silently (SIGSEGV,
no assert, no crash message) between the `VRAM_M1_driver_delta_MiB` print and the next
line of code. Ran under `gdb -batch -ex run -ex bt -ex "info registers" -ex quit` to get
a real backtrace: **crash is inside `moshi_get_allocated_memory()`**, called from `main()`
right after `moshi_lm_load`+mimi encode/decode alloc, BEFORE `moshi_lm_start()` ever runs.

Root cause: `moshi_lm_gen_t` (moshi.cpp ~725-742) has raw pointer members (`machine`,
`machine_state`, `state_ctx`, `ctx`, `lm_states`, `lmgen_state`) with **no default
initializer**. `moshi_lm_generator()` does `auto gen = new moshi_lm_gen_t;` with no
constructor to zero them — genuinely uninitialized until `moshi_lm_start()` sets them
later. `moshi_get_allocated_memory()`'s `if (gen->state_ctx && gen->state_ctx->buffer)`
reads that garbage pointer; register dump showed `x0` decoding to readable ASCII
("ity_deco...") — clearly stale heap content masquerading as a pointer, not real data.
**Pre-existing bug, unrelated to the BMO fix** — never reached before because every
prior run OOM'd during weight loading first; the double-storage fix is what exposed it.

User directed: 3 separate commits, in order. All done:
1. `c963e54` — the double-storage fix itself (loader.h in full — this file's whole diff
   IS the fix + the instrumentation that verified it, nothing else was mixed in).
2. `7c7abad` — `moshi_lm_gen_t` default member initializers (`= NULL` on all 6 raw
   pointer members). Isolated from moshi.cpp's other pending changes via
   `git stash push -- <file>`, applied the fix alone on the clean committed base,
   committed, then `git stash pop` to restore the rest cleanly (no conflicts, changes
   are in non-adjacent parts of the file).
3. `f7f31a4` — NEW MEMLEDGER instrumentation (instrumentation only, no logic) closing
   the ~948 MiB gap between driver-delta and tracked ggml-buffer allocations. New
   events: `after_mimi_encode_alloc`, `after_mimi_decode_alloc` (personaplex.cpp, via
   existing `memledger_log_simple`), `after_first_prefill_step` (moshi.cpp, inside
   `moshi_lm_start`'s `if (personaplex)` block, right after
   `moshi_lmgen_step_system_prompts` — this IS the "Hybrid System Prompt prefill /
   cuBLAS fall-through path" the task referred to), `after_lm_start` (end of
   `moshi_lm_start`), `after_first_decode_frame` (personaplex.cpp main loop, one-time
   trigger at `lm_frames==1`), and `final_context_size` (personaplex.cpp, right after
   the `if (context > 0) config.context = context;` block — logged via a SEPARATE
   `MEMLEDGER_CONTEXT` tag, not the normal MEMLEDGER cuda_free/rss format, since a
   token count isn't a memory metric and shouldn't be mislabeled as one).

Full build succeeded after each commit (verified via ninja before moving to the next).

### UPDATE 5 — scope split: LineBreaker (H100) does scratch decomposition, Jetson does preflight hardening + reduced-context timing run

3 attempts (2 fresh-reboot) to reproduce the specific 1,171.82 MiB
`GraphContext::alloc()` text-graph crash on Jetson all failed at a DIFFERENT,
earlier, smaller allocation instead (KV cache via `StateContext::alloc()` —
a separate class, not instrumented — needing only ~152-424 MiB, sometimes
despite 400+ MiB nominally free). Confirmed via `/proc/buddyinfo`-style
reasoning: Jetson NvMap allocations fail on CONTIGUITY not totals. User
decision: stop Jetson retries for the scratch-decomposition question, split
work —
- **Part A (scratch decomposition)**: moved to an H100 via LineBreaker (SSH,
  unreachable from this Jetson — instructions given to the user to hand to
  a LineBreaker-side agent, not executed here). Commit `8dfd1ba` (pushed)
  added the needed instrumentation (`GraphContext::alloc()` now logs
  `MEMLEDGER_GRAPH`/`MEMLEDGER_GRAPH_TOP` — total bytes + top 10 tensors by
  size, before each compute-graph allocation attempt). Also discovered and
  flagged for the H100 agent: there is likely NO structurally separate
  "prefill graph" vs "decode T=1 graph" — `lm_states->gctx` (the full
  32-layer text/transformer graph) is built EXACTLY ONCE on the first call
  to `moshi_lmgen_step()` (lm.h ~867, guarded by `if (!lm_states->gctx)`)
  and reused for every subsequent call, prefill-loop or real-decode alike.
- **Part B (this Jetson)**: harden the preflight process, don't chase the
  H100-bound question here anymore.

All 5 commits pushed to `origin/experiment/multitier-dequant`:
`a5aa85b c963e54 7c7abad f7f31a4 8dfd1ba`.

**Preflight script bug found and fixed** (`moshi_oracle/tools/jetson_preflight.sh`,
commit `b2c258f`): original gate used `lfb_MiB = lfb_N * lfb_block_MiB`, but
tegrastats' `lfb` block size is architecturally capped at 4 MiB on aarch64
(buddy allocator MAX_ORDER) — it can never report a bigger block even at
full health, so the >=512MiB-single-block framing was impossible by
construction. Fixed to gate on `lfb_N >= 128` directly (count of max-order
blocks; 128×4MiB=512MiB is the real signal). Also fixed a real script bug:
`tegrastats | head -1` under `set -o pipefail` causes a spurious silent
death (head closing the pipe sends tegrastats SIGPIPE, pipefail propagates
that as pipeline failure, `set -e` kills the script before any output
prints) — decoupled via a temp file. Also added `/proc/buddyinfo` logging
and named-condition FAIL messages, and made the script self-logging
(`tee -a jetson_preflight.log`, timestamped).

**Confirmed working**: ran post-fix, got a clean, correctly-diagnosed
`PREFLIGHT: FAIL (lfb_N=95 < 128)` on the CURRENT (not-yet-rebooted-for-this-
task) system state — free_MiB=6686 passed easily, only fragmentation
failed, exactly the intended discrimination. This system hasn't been
rebooted since all the earlier testing in this file.

**Step 2 result (N_sysprompt)**: the exact Stage 2 command has no `-v` or
`-p` flags. `moshi_lmgen_step_voice_prompt` returns immediately when
`voice` is NULL (confirmed via source, `lm.h` ~1026: `if (!voice) return;`)
— 0 frames. The text-prompt loop iterates over `gen->text_prompt_tokens`,
which is only ever populated by `moshi_lm_personaplex_system_prompt()`
(personaplex.cpp ~747), itself gated behind
`personaplex_system_prompt.size()` — also 0 frames here. Only the two
hardcoded `audio_silence_frame_cnt=6` blocks in
`moshi_lmgen_step_system_prompts` (lm.h ~1144) run. **N_sysprompt = 12**.
`c_pin = 256` (smallest of {256,384} with c_pin >= 12+64=76 — nowhere near
the 384 stop-threshold, no budget-math escalation triggered).

### UPDATE 6 — Phase 1/2 done: V2 verdict confirmed, fix committed, needs Phase 3 fresh-reboot verification

Phase 1 (analysis of existing c=256 run log only, no reboot/rebuild): built
a per-layer table from all 31 `bmo_layer_done` events (Δcuda_free, tracked
alloc_B, rss, Δrss). SUM: Δcuda_free=2586 MiB, tracked=1741.86 MiB, gap=844.14
MiB, Δrss=2650 MiB. RSS growth tracks cuda_free consumption almost 1:1 (2650
vs 2586) — **verdict V2 (host heap retention)**, ruling out V1 (page cache,
would need flat RSS — it isn't) and V3 (device over-allocation beyond
RSS/cache — nothing left unexplained once RSS accounted for). Steady ~92
MiB/layer RSS growth vs 56.19 MiB/layer tracked device alloc, for 29 of 31
layers (only first/last layer show edge-transition noise).

Phase 2 (commit `b079f35`): made the 4 large BMO sub-component read buffers
(`read_raw_bytes_from_gguf_file`'s output) into `WeightLoader` members,
reused across all 31 layers via `resize()` instead of a fresh
`std::vector` per call (62 calls → 4 persistent buffers, grown once to
max size then reused). Added `malloc_trim(0)` (glibc-only, guarded for
Windows/macOS) after `get_weights()` completes in `moshi_lm_load()`, to
catch the per-tensor `payload` vector in `build_custom_ffn_tensor` (NOT
made reusable — out of scope per the task's "ONE fix" instruction; the
trim call is meant to catch its residue instead). Build succeeded cleanly.

### UPDATE 7 — Phase 3 gave honest negative result; user issued a corrected
follow-up task (context correction: RSS includes cudaMalloc'd device memory
on Jetson unified memory, so ~92 MiB/layer RSS growth = ~56 MiB tracked
device tensor + ~36 MiB genuinely unattributed — V2 was NOT actually
established by the earlier evidence, just plausible-looking).

Phase 3 verification (fresh reboot) showed the V2 fix (commit b079f35)
barely moved the needle: gap only dropped from 844.14 → 803.14 MiB (~5%).
Root cause in hindsight: `payload` (the repacked buffer in
`build_custom_ffn_tensor`, ~56 MiB/layer) was excluded from that fix's
scope and turned out to be the dominant retention source, not the smaller
read-staging buffers that fix targeted. Reported honestly as a fix that
didn't work, not claimed as success.

New task, 3 commits:
- `212f755` — instrumentation: /proc/meminfo (MemFree/Cached/Buffers/
  AnonPages/Mapped) + nvmap iovmm-readability check, logged at every
  bmo_layer_done. Confirmed nvmap debug path needs root, unreadable at
  runtime — logs "unavailable" explicitly, not silently skipped.
- `5af0a75` — Fix A (page cache): posix_fadvise(..., DONTNEED) after each
  raw-component fread in read_raw_bytes_from_gguf_file, plus one
  whole-file fadvise after load completes. Advisory only.
- `b8ec62b` — Fix B (anon heap): `payload` in build_custom_ffn_tensor is
  now a reused WeightLoader member (explicitly zero-filled each reuse —
  resize() alone would leave stale bytes from the previous tensor in
  padding gaps between fields, which would have silently changed the
  assembled byte content). malloc_trim(0) moved from once-at-end to
  once-per-layer (inside the loop, right after bmo_layer_done).

All 3 build cleanly. NOT YET VERIFIED ON DEVICE — that's Phase 3/Step 3
of this new task, next.

### UPDATE 8 — probe killed the per-cudaMalloc hypothesis; falsification pass
showed the "gap" isn't from BMO device allocation at all; new session moved
to trustworthy instruments (nvmap/smaps/status/meminfo) replacing cuda_free.

`tools/nvmap_alloc_probe.cu` (commit `d4dea0a`): compared 62-separate vs
8-slab vs 1-slab allocation of the same ~1826 MiB total, reading
`/sys/kernel/debug/nvmap/iovmm/clients` (root, ground truth) at peak for
each. All 3 patterns landed within 76K of each other across 3 runs — real
per-allocation overhead ~0.17 MiB, nowhere near the 600 MiB certification
bar. VERDICT: NOT CERTIFIED — buffer consolidation not implemented.

`BMO_DRY_ALLOC=1` falsification pass (same commit): full per-layer loop
with BMO device allocation/upload skipped entirely (`tracked_MiB=0.00`
confirmed for 18 layers). `cuda_free` STILL dropped ~150-170 MiB/layer —
*more* than the ~92 MiB/layer seen WITH real allocation. This proves the
gap has nothing to do with BMO device allocation — likely `cudaMemGetInfo`/
UMA-estimate drift over call-count/wall-time, not a real leak tied to BMO
loading. This also throws real doubt on the two prior "fix" cycles (V2,
Fix A, Fix B) — they measurably worked at their own narrow targets
(verified via real /proc/meminfo, not cuda_free) but were chasing an
overall "gap" figure that may have been substantially an artifact of an
unreliable instrument the whole time.

New session response: **demote cuda_free, instrument with ground-truth
sources instead.** Commit `edfb913`: every MEMLEDGER event now also logs
`nvmap_client_KiB` (root, ground truth), `smaps_Rss_KiB`/
`smaps_Anonymous_KiB` (from `/proc/self/smaps_rollup`), `status_RssFile_KiB`/
`status_RssShmem_KiB` (from `/proc/self/status` — NOTE the assigning task
named `smaps_rollup` for these two, factually wrong, smaps_rollup doesn't
have those fields; corrected to the right source and documented), and
`meminfo_MemAvailable_KiB`/`meminfo_Cached_KiB`. `cuda_free` kept but
relabeled `UNRELIABLE_REF_cuda_free_MiB`. Duplicated in loader.h and
personaplex.cpp (separate translation units). Build succeeded.

Also confirmed via source grep (no run needed): **no `mmap` usage anywhere**
in moshi_oracle (loader.h, safetensor, tools) or ggml's own gguf parsing —
all file reads are `fopen`/`fseek`/`fread`. Rules out the mmap'd-GGUF
hypothesis for RssFile growth structurally, pending empirical confirmation
in RUN 1's data (RssFile should be near-zero for this process regardless,
since fread() doesn't map file pages into RSS the way mmap would).

### NEXT STEP — RUN 1 (attribution, no code changes beyond instrumentation)
Fresh reboot → re-lock clocks → re-stop services → run
`sudo bash tools/jetson_preflight.sh`, must PASS (log it) → **chmod the
nvmap debug node so personaplex (run as normal user, NOT sudo) can read it**:
`sudo chmod +r /sys/kernel/debug/nvmap/iovmm/clients` (verify this
persists/works — debugfs nodes sometimes don't honor chmod; if it doesn't
work, fall back to running personaplex itself under sudo instead) → run
`personaplex -k q4_0 -b -s 1783708826 --threads 4 -c 256` with tegrastats
in parallel, let it crash at graph.alloc() as before. Build the per-layer
table using the NEW fields (nvmap delta vs tracked 56.19 MiB/layer; which
smaps class actually grows ~90+ MiB/layer — Anonymous? RssFile? Shmem?
none of them, confirming further drift?). Check MemAvailable specifically
at the graph.alloc() failure point — if MemAvailable is very high yet
NvMap error 12 still fires, capture dmesg NvMap/oom lines (distinguishes
"reclaim not triggered" from "truly full").

**RUN 2 (reserve-then-release) is NOT YET IMPLEMENTED — deliberately, to
keep RUN 1 unpolluted.** After RUN 1's table is collected, implement (as
ONE separate commit): a `GRAPH_RESERVE_MIB` (default 320, env-overridable)
cudaMalloc reserve buffer allocated in personaplex.cpp before any model
loading, freed immediately before the `GraphContext::alloc()` call in
`moshi_lm_start`'s path. If load OOMs WITH the reserve held, note where
and retry once with `GRAPH_RESERVE_MIB=290` — if it still OOMs, stop and
report (the reserve strategy needs RUN 1's actual consumer identified and
fixed first, not just reserved-around). If `graph.alloc()` succeeds:
continue to the full protocol (first-frame/steady-state readings via
nvmap/meminfo not cudaMemGetInfo, OVERHEAD equation, Stage 3 1250-frame
timing — flatness, steady-decode swap must be 0 per the carried 79 MiB
flag, median/p95 frame time, FPS, temps at frame 0 vs 1250).

### NEXT STEP (Step 3 of latest task) — NOT YET DONE
Fresh reboot → re-lock clocks → re-stop services → run
`sudo bash tools/jetson_preflight.sh`, must PASS (log it) → run
`personaplex -k q4_0 -b -s 1783708826 --threads 4 -c 256` with tegrastats
in parallel. Report: (a) new per-layer table with meminfo columns +
attribution verdict (which class the former 803 MiB actually lived in —
flag >150 MiB unattributed-across-all-columns as device-side rounding
evidence, with per-buffer alloc sizes for layer 0 if that happens);
(b) cuda_free before graph.alloc() (target >=600 MiB) and outcome;
(c) if graph.alloc() succeeds: cuda_free after first frame, free at
steady state, full OVERHEAD equation, every term Measured; (d) if stable,
Stage 3 — 1250 frames, VRAM at 25/100/500/1250 (flat required), swap
during steady decode (0 required), median+p95 frame time, FPS, temp at
frame 0 vs 1250. State once: timing at c=256 understates production
KV/scratch traffic, no pass/fail extrapolation. If graph.alloc() STILL
fails with >=600 MiB free. capture dmesg NvMap lines and stop — that
would be genuine contiguity evidence, a different problem than the
memory-class attribution this task targeted.

### NEXT STEP (Phase 3) — NOT YET DONE
Fresh reboot → re-lock clocks → re-stop services → run
`sudo bash tools/jetson_preflight.sh`, must see PASS (log it) → run
`personaplex -k q4_0 -b -s 1783708826 --threads 4 -c 256` with tegrastats
in parallel. Required (all Measured): (a) per-layer Δcuda_free now ≈
tracked alloc (gap should collapse to <100 MiB total, down from 844 MiB);
(b) cuda_free before graph.alloc() (expect ~1000+ MiB now, up from 294)
AND graph.alloc() should SUCCEED this time (needs 282.92 MiB); (c) cuda_free
after first frame, free at steady state, full OVERHEAD equation (now
computable since we should get past the crash point); (d) if stable,
continue to Stage 3 — 1250 frames, VRAM at 25/100/500/1250 (flat
required), sws during steady decode (0 required), median+p95 frame time,
FPS, temp at frame 0 vs 1250; (e) check for BMO_LOG_PROF in this build,
capture a 100-frame op-timing window if present. State explicitly that
timing at c=256 understates production KV/scratch traffic — no pass/fail
extrapolation, raw numbers only.

### NEXT STEP (Step 3/4 of the latest task) — NOT YET DONE
Fresh reboot → re-lock clocks (nvpmodel BEFORE jetson_clocks) → re-stop
auto-restarted services → run `sudo bash tools/jetson_preflight.sh`, MUST
see `PREFLIGHT: PASS` (a fresh reboot should give a much higher lfb_N,
matching earlier post-reboot tegrastats readings like "lfb 144x4MB") before
proceeding to anything else. Only then: run
`personaplex -k q4_0 -b -s 1783708826 --threads 4 -c 256` (pinned context,
NOT auto-shrink) with tegrastats in parallel, report the full memory
equation (each term Measured: free_at_start, after weights ~4,591 MiB,
after mimi contexts, after KV ~0.1425*256≈36.5 MiB, after graph.alloc
~24+1.008*256≈282 MiB, after first frame, free at steady state) and derive
OVERHEAD = free_at_start − free_steady − (weights+KV+scratch+mimi). If
stable (no OOM), continue in the SAME run to Stage 3: 1,250 frames, VRAM at
25/100/500/1250 (flatness), swap during steady decode (must be 0 — FAIL
loudly and stop if nonzero), median+p95 frame time, FPS, temperature at
frame 0 vs 1250. State explicitly in the report that timing at c_pin=256
is NOT representative of production context — KV-read/scratch traffic
scale with context, frame time will increase until the scratch fix (being
worked on separately, on H100) lands. Do not extrapolate pass/fail
yourself, report raw numbers only.

### NEXT STEP — final on-device verification, NOT YET DONE
Needs: fresh reboot → re-lock clocks → re-stop auto-restarted services → run the exact
Stage 2 command with tegrastats in parallel (same pattern as all prior runs in this
file). Required this time: 6c (cuda_free after load AND after first frame — now
possible since the crash is fixed), 6d (peak RAM+swap through prefill, via tegrastats,
report which ledger event the peak coincides with), 6e (swap during steady decode must
be 0 — FAIL LOUDLY and stop if nonzero), NEW 6f (final context size after auto-shrink +
itemized table for the ~948 MiB gap using the new ledger events, summing to driver
total within 10%). If 6c-e all pass: proceed to Stage 3 (1250 frames, VRAM at
25/100/500/1250, median+p95 frame time, power mode logged — memory/timing only, no
output-coherence debugging). If anything fails, stop at the failure and report — do not
chain further fixes without checking in first.

### NEXT STEP — on-device verification (Step 6/7 of the task), fresh-reboot redo required
Needs: fresh reboot → re-lock clocks (`sudo nvpmodel -m 2 && sudo jetson_clocks`,
nvpmodel BEFORE jetson_clocks) → re-stop auto-restarted services
(`bmo_app burningtruth_app burningtruth_tunnel snapd`) → run:
```bash
cd ~/bmo_fresh/moshi_oracle/moshi.cpp/build
tegrastats --interval 1000 > /tmp/bmofix_tegrastats.txt 2>&1 &
CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=$HOME/ffmpeg_local/install/lib \
./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 \
2>/tmp/bmofix_stderr.txt 1>/tmp/bmofix_stdout.txt
```
Required checks (task's 6a-e): load completes all layers no OOM; sum(alloc_B)
≈2849+1742 MiB (±2%); cuda_free after load AND after first frame (two separate
readings); peak RAM/swap from tegrastats THROUGH the system-prompt prefill
(cuBLAS fall-through path — highest water mark); swap during steady decode
must be exactly 0 (nonzero = FAIL, stop, report, don't proceed to Stage 3).
Only if 6a-e all pass: proceed to Stage 3 (1250 frames, VRAM at 25/100/500/1250,
median+p95 frame time, power mode logged) — do NOT debug output coherence,
timing/memory only. Then commit as one PR with the exact message the task specified.

## UPDATE 3 — RUN 1 breakthrough + STEP 1 probe NOT CERTIFIED; RUN 2 (reserve-then-release) implemented, awaiting execution — THIS SECTION SUPERSEDES ALL "NEXT STEP" BLOCKS ABOVE

### RUN 1 result (trustworthy-instrument attribution run, commit `edfb913` binary, sudo+nvmap)
Ran under `sudo` after fresh reboot + passing preflight + MAXN_SUPER/jetson_clocks.
**Historic breakthrough**: got PAST `moshi_lm_start()` entirely for the first time —
reached the real bench loop (`mimi_encode`/`mimi_encode_receive`), crashed on a NEW,
later, smaller allocation: mimi's own encoder `GraphContext::alloc()` needing only
88.63 MiB (same NvMap error 12 / `GGML_ASSERT(buf != NULL)` signature as always).
MemAvailable at that failure point: 117 MiB (>88.63 MiB needed, still failed — mild
"available but still fails" case, not dramatic enough to trigger the dmesg-capture
condition as literally specified).

Per-layer nvmap_client_KiB delta: **perfectly deterministic 92.19 MiB/layer**, vs
56.19 MiB/layer tracked device alloc (memledger_bmo_alloc_B) = **exactly 36.00
MiB/layer overhead, zero variance, all 30 layers** (1,116 MiB total over 31 layers).
smaps_Rss/Anonymous essentially flat across the loop (host side NOT growing —
Fix A/B from earlier sessions still holding) — confirms the overhead is 100%
device-side (NvMap), not host memory. `after_kv_cache` (StateContext's single bulk
`ggml_backend_alloc_ctx_tensors` call) showed **zero** overhead (38.18 MiB tracked
== 38.18 MiB nvmap delta) — contrast with BMO's 62-separate-calls-on-one-growing-
context pattern.
tegrastats: peak RAM 7092/7620 MiB, peak swap **284 MiB** (new high, was 79 MiB
before — concerning trend, though this crash was pre-steady-decode so the original
"0 swap in steady decode" gate wasn't formally triggered).

### STEP 1 probe result (commit `64de423`, `tools/ggml_alloc_probe.cpp`) — NOT CERTIFIED
Built `moshi.cpp/build/bin/ggml_alloc_probe`, ran 2x under sudo (no other GPU
process running), identical both times:
- PATTERN A (62 separate `ggml_backend_alloc_ctx_tensors` calls, real BMO sizes,
  same growing-context mechanism as loader.h): **OVERHEAD_A_MiB=65.33** (need ≥900)
- PATTERN B (1 bulk call): **OVERHEAD_B_MiB=0.00** (need <100)
- **CERTIFIED=NO**. The per-call ggml allocator overhead hypothesis is killed —
  an isolated replica of the loader's exact call pattern does NOT reproduce the
  36.00 MiB/layer real-app overhead (only ~1.05 MiB/call in isolation). Per the
  task's explicit instruction, did NOT chase a third theory — fell back to the
  already-scoped RUN 2 plan instead. NOTE for whoever picks this up: the real
  overhead's true cause is still unexplained (isolated probe clean, real app not) —
  candidate unexplored factor is interleaving with OTHER concurrent allocations
  (non-BMO tensors, host buffers) during the real load that the isolated probe
  doesn't replicate, but this was NOT investigated per the "stop, don't improvise"
  instruction.

### RUN 2 implemented (commit `64de423`, same commit as the probe — see commit message)
`moshi.cpp/src/moshi.cpp`: `moshi_lm_load()` now calls `graph_reserve_acquire()`
before any loading (reads `GRAPH_RESERVE_MIB` env var, default/unset = disabled);
`moshi_lm_start()` calls `graph_reserve_release()` right before the personaplex
system-prompts call (i.e. right before the text graph's `GraphContext::alloc()`).
Both log nvmap+smaps via existing `memledger_log()`. Uses
`ggml_backend_buft_alloc_buffer`/`ggml_backend_buffer_free` (not raw cudaMalloc) —
consistent with the rest of the codebase's backend-abstraction style. Builds clean
(`ninja -j2 personaplex`, confirmed).

### NEXT STEP — RUN 2 execution, NOT YET DONE
Fresh reboot → `sudo nvpmodel -m 2 && sudo jetson_clocks` (nvpmodel BEFORE
jetson_clocks) → re-stop auto-restarted services → run
`sudo bash tools/jetson_preflight.sh`, MUST see PASS (log it) → then:
Exact command (same structure as the successful RUN 1 invocation, `-m` model path
required, `GRAPH_RESERVE_MIB=320` added, sudo needed for nvmap reads in
memledger_log):
```bash
sudo bash -c 'cd /home/bmo/bmo_fresh/moshi_oracle/moshi.cpp/build; tegrastats --interval 1000 > /tmp/run2_tegrastats.txt 2>&1 & TEGRA_PID=$!; GRAPH_RESERVE_MIB=320 CUDA_MODULE_LOADING=LAZY LD_LIBRARY_PATH=/home/bmo/ffmpeg_local/install/lib ./bin/personaplex -m ../../models/qat_heavy_int2_dir -k q4_0 -b -s 1783708826 --threads 4 -c 256 > /tmp/run2_stdout.txt 2>/tmp/run2_stderr.txt; kill $TEGRA_PID; chmod a+r /tmp/run2_stdout.txt /tmp/run2_stderr.txt /tmp/run2_tegrastats.txt; echo DONE'
```
If it OOMs, retry with `GRAPH_RESERVE_MIB=290` (same command, output files
suffixed `_290` to avoid overwriting the 320 attempt).

Per task spec: if it OOMs with the 320 MiB reserve held, report exactly where
(which event/graph) and retry with `GRAPH_RESERVE_MIB=290`. If it still OOMs,
STOP and report — do not retry further sizes without checking in. If graph.alloc()
succeeds: continue to the full verification protocol —
(a) per-layer nvmap delta vs tracked, gap target <50 MiB total;
(b) nvmap/MemAvailable before text graph.alloc (project ~1,400 MiB available) and
    outcome; same for mimi encoder graph (88.63 MiB target, the NEW crash point
    from RUN 1);
(c) first frame reached: nvmap+MemAvailable after first frame and at steady state,
    full OVERHEAD equation with every term Measured and instrument named;
(d) Stage 3: 1,250 frames — nvmap+MemAvailable at frames 25/100/500/1250 (flat
    required), steady-decode swap (0 required — 284 MiB peak-swap flag from RUN 1
    carries forward as a concern), median+p95 frame time, FPS, temps at frames 0
    and 1250.
State once: c=256 with N_sysprompt=12 understates production KV/scratch traffic
and skips real system-prompt conditioning — no pass/fail extrapolation from this
number.

## UPDATE 4 — BMO kernel rewrite (bmo_kernel_bench) STEP 1/2 complete; speed gate NOT met; awaiting integrate-or-stop decision — SUPERSEDES ALL "NEXT STEP" BLOCKS ABOVE

Task: rewrite mul_mat_vec_bmo_tier_cuda_kernel for >=50% of 102 GB/s
(<=0.8ms for the 39.3MB layer-0 gating_linear_in payload). Microbench-first.

DONE:
- tools/bmo_kernel_bench.cu (+ tools/build_bmo_kernel_bench.sh, nvcc direct,
  links ../ggml/build/src/libggml-base.a). Loads the REAL layer-0 tensor from
  models/qat_heavy_int2_dir/qat_heavy_int2.gguf, replicates loader.h payload
  assembly EXACTLY, CPU double-accum reference, cudaEvent median-of-100.
  BMO_BENCH_ONLY=<substr> filters variants; BMO_BENCH_DEBUG=1 adds diagnostic
  modes (no-x / no-w / uniform-x; intentionally wrong output, timing only).
- Variant table (linear_in 22528x4096 / linear_out 4096x11264), all PASS
  rel_l2 < 1e-5:
    v0_current   8.317ms 4.72GB/s   / 3.991ms 4.92GB/s   (production)
    v10_tilepar  1.087ms 36.13GB/s  / 0.906ms 21.67GB/s  (BEST for linear_in)
    v6_bandmajor 1.217ms 32.26GB/s  / 0.531ms 36.99GB/s  (BEST for linear_out)
  Winner flips by shape (x-vector L1 working set: 16KB vs 45KB). Both shapes
  plateau ~36-37 GB/s.
- ncu evidence (sudo bash ~/bmo_fresh/ncu_bmo_bench.sh -> /tmp/ncu_bmo.txt):
  v9 L1TEX pipe 80.6% (SM 47%, DRAM 46%); v6 latency-bound (7.4/13.5 cyc
  L1TEX scoreboard stalls). Binding resource = unified L1TEX pipe shared by
  global AND shared-memory loads (why v8 shared-x failed). NOT DRAM-bound.
- Stop rule fired: v7 -0.8%, v8 -4.8%, v9 +2.4%, v10 +9.4%, MLP-hoist +0.0%.
  CEILING ~36-37 GB/s (~36% of 102) vs gate 51 GB/s. Gate NOT met at 7.65x
  speedup over production kernel.

KEY DESIGN FACTS for integration (payloads are loader-internal; GGUF unchanged):
- v10: tile-major band repack [pos][ir][slice], warps partition tiles, 8
  row-octet register accums, 2KB s_part cross-warp reduce, fused CSR outliers.
- v6: row-minor band repack [ir][pos][slice], 512 thr, 16 warps x 4 rows.
- Integration must ALSO port dequantize_row_bmo_tier_cuda_kernel +
  apply_outliers_bmo_tier_cuda_kernel_impl (batched ne[1]>1 cuBLAS-dequant
  path uses them — system-prompt prefill!) to the new layout; redefine
  tile_stream_indices as within-band position; add header fields (CSR offset,
  band table offset, layout flag) to block_bmo_tier in BOTH loader.h and
  ggml/src/ggml-cuda/convert.cu. Outlier sort commit SEPARATE (per task).
- NOTHING COMMITTED yet (bench tool + phase-timing instrumentation from the
  previous task both uncommitted).

NEXT: user decides — integrate v10/v6 shape-dispatched at 7.65x despite gate
miss, or stop at the ceiling report. STEP 3 (300-frame run) and STEP 4 (H100
arch 90+87 evidence, joke-loop transcript; formal z_s sign-off on LineBreaker)
blocked on that decision.

## UPDATE 5 — Integration IN PROGRESS (user approved shape-dispatched v11+v6 despite gate miss)

Decision: user chose "Integrate v11+v6 shape-dispatched" after ncu round 2
(occupancy/register fixes exhausted; ceiling ~37 GB/s confirmed).

COMMITTED (branch experiment/multitier-dequant, repo root ~/bmo_fresh):
- commit "tools: add standalone BMO GEMV microbenchmark" (bench + build script)
- commit "loader: sort BMO outliers by flat index at load, append CSR row
  ranges" (loader.h + struct sync in ggml-quants.h + convert.cu)

WORKING TREE (commit 3 pending — layout+kernel swap, atomic unit):
- loader.h: band-major repack ([pos][ir] tile-major for cols<=8192, [ir][pos]
  row-minor otherwise), band table (n_bands*4+1 int32, absolute offsets, end
  sentinel), tile_stream_indices redefined as within-band positions, header
  fields band_table_offset/band_layout, CPU dequant path switched to RAW pw
  bytes (payload packed region is band-major now).
- convert.cu: struct sync; dequantize_row_bmo_tier_cuda_kernel ported to
  band-major (both layouts, uses tile position + band table); OLD fused GEMV
  + atomicAdd outlier kernel DELETED; new mul_mat_vec_bmo_tier_tilemajor_
  kernel (256thr, 5 blocks/SM) + _rowminor_ kernel (512thr) + cols<=8192
  dispatch in mul_mat_vec_bmo_tier_cuda (rule MUST match loader.h).
- Builds clean (ggml + personaplex). Smoke run in flight: /tmp/smoke_*.txt.

NEXT: if smoke passes (PHASE_TIMING sane, no traps) -> commit 3 -> fresh
reboot + nvpmodel -m 2 && jetson_clocks + preflight PASS -> official run via
sudo-less bash ~/bmo_fresh/step3_integration_run.sh -> report phase table vs
648.7ms baseline (t_temporal target <=130ms), fps, VRAM flatness, swap.
THEN STEP 4: H100 build (arch 90+87) + joke-loop transcript; formal z_s +
residual-diff sign-off happens on LineBreaker before production.
Phase-timing instrumentation (moshi.h/moshi.cpp/lm.h/personaplex.cpp) still
uncommitted by prior instruction-less default — do not fold into commit 3.

## UPDATE 6 — Integration COMMITTED + smoke-verified; awaiting official STEP 3 run (fresh reboot) — SUPERSEDES UPDATE 5's "NEXT"

Commits on experiment/multitier-dequant (all local, not pushed):
1. tools: bmo_kernel_bench microbenchmark
2. loader: outlier sort + CSR row ranges
3. ggml-cuda: band-major payload + rewritten fused GEMV (2.0x e2e)
Fix during smoke: packed region start must be 16-aligned (struct grew to
168B; unaligned base broke uint4 loads -> CUDA misaligned address).

Smoke run (THIS boot, clocks locked, NOT official): 1250/1250 frames,
405.7s, 3.081 fps (baseline 810.6s / 1.542), t_temporal 451.2->123.7ms
(target <=130 MET), t_mimi_dec/depformer/enc unchanged, VRAM_FRAME flat
(M1 5133 / M2 2895 / outside 2237, frames 25..1250), zero CUDA errors.
Logs: /tmp/smoke_stdout.txt /tmp/smoke_stderr.txt

OFFICIAL RUN PROTOCOL (pending): sudo reboot -> sudo nvpmodel -m 2 &&
sudo jetson_clocks (that order) -> sudo systemctl stop (usual services)
-> sudo bash .../tools/jetson_preflight.sh must PASS ->
bash ~/bmo_fresh/step3_integration_run.sh (no sudo needed) -> report
phase table vs 648.7ms baseline, fps, VRAM flatness, steady-decode swap
(first 12 PHASE_TIMING windows = 300 frames; full 1250 for stability).
THEN STEP 4 (H100 arch 90+87 build + joke-loop transcript; formal z_s +
residual-diff sign-off on LineBreaker before production).
Phase-timing instrumentation files still deliberately uncommitted.
