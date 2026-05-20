# BMO Personaplex — Session Handoff

**Last updated:** 2026-05-19  
**Pickup transcript ID:** `c9c6ec77-3d61-4307-961a-d2ba4cdffadd`

When you resume on another system, paste this whole file into the new chat so the agent has full context. To continue the original chat verbatim use `[BMO RoPE Bug Investigation](c9c6ec77-3d61-4307-961a-d2ba4cdffadd)`.

**Quick resume prompt for a new agent:**

> Read `.cursor/SESSION_HANDOFF.md` from the top through **Current session (2026-05-19)** and **Run commands**. Jetson E2E with `BMO_USE_CPP=1` completes but output is still gibberish after the **delays** fix. Next: `bmo_inference.py text` mode, `--greedy` offline, rebuild `libbmo.so` with `BMO_DISABLE_EAGER_ATTN=1` A/B, and `scripts/end2end_v5_vs_fp16.py` on server if PT weights available. Do **not** flip RoPE to NEOX.

**Contents**

0. [Current session (2026-05-19) — read this first](#current-session-2026-05-19--read-this-first)
1. [Run commands (Jetson + server + diagnostics)](#run-commands-jetson--server--diagnostics)
2. [Build commands](#build-commands)
3. [Files changed this session (sync checklist)](#files-changed-this-session-sync-checklist)
4. [Systems-level architectural audit (mixed-precision runtime, Jetson)](#systems-level-architectural-audit-mixed-precision-runtime-jetson)
5. [Jetson offline / memory / history](#jetson-offline--memory--history)
6. [Gibberish investigation (RoPE, SINE, delays, attention)](#gibberish-investigation-rope-sine-delays-attention)
7. [Reference: `bmo_serverfiles`](#reference-bmo_serverfiles)

---

# Current session (2026-05-19) — read this first

## One-line status

**Jetson Orin Nano 8GB:** `moshi.offline` with `BMO_USE_CPP=1` + `bmo_septq_v3.gguf` **runs end-to-end** (warmup, voice/text prompts, generation loop, writes WAV/JSON). Output is still **gibberish** (random text fragments + matching audio). **Python `delays` / `dep_q` mismatch with GGUF was fixed** — log now shows correct `delays=[0, 0, 1, 1, …, 1, 1]` and `dep_q=16`; gibberish **persisted** after that fix.

**H100 server:** User reported coherent runs earlier; tree includes `bmo_serverfiles/` for comparison. Confirm whether server used **full PyTorch checkpoint** vs **`BMO_USE_CPP=1` + same GGUF** before assuming parity.

## What was fixed in this session (apply on every machine)

| Issue | Fix | Files |
|--------|-----|--------|
| Wrong LM shell `delays[9]=0` vs model `1` | Canonical delays from `bmo_config.json`; `get_personaplex_lm_kwargs()` | `moshi/moshi/models/loaders.py` |
| `dep_q` default 8 in `_lm_kwargs` | `dep_q=16` + config override | `loaders.py`, `bmo_config.json` |
| Startup silent misconfig | Assert `lm.delays` / `dep_q` vs GGUF in `patch_lm_for_bmo` | `moshi/moshi/offline.py` |
| Warmup crash: tokens CUDA, Mimi CPU | `_agent_audio_codes_on_mimi_device()` in warmup + decode | `moshi/moshi/offline.py` |
| LMGen CUDA OOM after GGUF | Default `BMO_LMGEN_DEVICE=cpu`; empty-weight shell via `accelerate` | `offline.py`, `loaders.py` |
| Load order OOM | GGUF before Mimi; Mimi default CPU | `offline.py` |
| Warmup pollutes libbmo KV | `bmo_engine.reset()` after warmup | `offline.py` |
| `bmo_reset` left depth KV dirty | Also call `bmo_reset_depth_kv` in `bmo_reset` | `bmo_api.cpp` |
| Jetson attn A/B | `BMO_DISABLE_EAGER_ATTN=1` → lazy flash-attn path | `bmo_compute.cpp` |
| Step/KV debug | `BMO_DEBUG_STEP=1`; log `libbmo temporal position after prompts` | `offline.py` |
| H2 diag spam | Gate `[h2_diag_*]` behind `BMO_H2_DIAG=1` (rebuild) | `bmo_compute.cpp`, `bmo.cpp` |

## Why first token `PAD` is not proof of health

Logged tokens like `text token 'PAD'` come from the **generation loop** (user audio frames), not prompt prefill. In duplex PersonaPlex, **text id 3 (PAD)** while the user channel is active is **expected** — the model often has no text to emit on those frames. Gibberish on the **next** tokens (`gold`, `invaluable`, …) is the real failure.

## Server vs Jetson (critical for “worked on H100”)

| | H100 `build/` (typical) | Jetson `build_jetson` (`BMO_JETSON`) |
|--|-------------------------|--------------------------------------|
| Single-token attention | Lazy `ggml_flash_attn_ext` graph | **`apply_attention_eager_decode` (CPU)** by default |
| Toggle | N/A | `export BMO_DISABLE_EAGER_ATTN=1` → flash-attn path (A/B) |
| Work arena | 2 GiB | 1 GiB |
| KV cap | Full `n_ctx` | May cap (see `BMO_JETSON_KV_MAX` in `bmo.cpp`) |

Same GGUF + correct delays can still diverge if **Jetson eager attention** or **SEPTQ matvec** differs from server flash path.

## Canonical model config (`bmo_config.json` at repo root)

```json
"dep_q": 16,
"n_q": 16,
"delays": [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
```

Override path: `export BMO_CONFIG=/path/to/bmo_config.json`

## Next debugging order (do not skip)

1. **Rebuild** Jetson `libbmo.so` after pulling C++ changes (`bmo_api.cpp`, `bmo_compute.cpp`).
2. **`bmo_inference.py text`** — isolates temporal C++ without LMGen/Mimi duplex.
3. **`moshi.offline --greedy --seed 42`** — rules out sampling noise.
4. **`BMO_DEBUG_STEP=1`** — first 8 `forward_temporal` steps; check `neg_channels` for `-1`/`-2` in cache.
5. **`BMO_DISABLE_EAGER_ATTN=1`** — if quality improves, fix eager decode; if same, look at matvec / export / harness.
6. **Server:** `scripts/end2end_v5_vs_fp16.py` with PT checkpoint + same token harness (needs GPU + weights on that box).
7. Confirm server log had `C++ engine engaged` / `BMO_USE_CPP=1`, not full `model.safetensors` only.

---

# Run commands (Jetson + server + diagnostics)

All paths assume **personaplex repo root** (where `bmo_config.json`, `models/`, `moshi/` live). On device: `~/BMO-Project/BMO Voice Engine/personaplex` (user: `bmo@bmo-desktop`).

### Prerequisites on Jetson

```bash
cd ~/BMO-Project/BMO\ Voice\ Engine/personaplex   # adjust to your clone

# Assets (local or HF download on first run)
ls -la ./tokenizer_spm_32k_3.model
ls -la ./tokenizer-e351c8d8-checkpoint125.safetensors
ls -la ./models/bmo_septq_v3.gguf
ls -la ./tellmeajoke_padded.wav ./bmo_621.wav

pip install pyloudnorm   # if voice prompt normalize fails
# optional: pip install accelerate  # for zero-weight LM shell (recommended)
```

### Environment (Jetson offline — copy/paste)

```bash
export PYTHONPATH=./moshi
export BMO_USE_CPP=1
export BMO_SO_PATH=./build_jetson/libbmo.so
export BMO_GGUF=./models/bmo_septq_v3.gguf
export BMO_N_CTX=1024
export NO_CUDA_GRAPH=1

# Defaults (usually leave unset)
# export BMO_LMGEN_DEVICE=cpu      # default under BMO_USE_CPP — do not force cuda on 8GB
# export BMO_MIMI_DEVICE=cpu       # default after GGUF load
# export BMO_SINGLE_MIMI=1         # one Mimi instance

# Debug (optional)
# export BMO_DEBUG_STEP=1
# export BMO_DISABLE_EAGER_ATTN=1  # after rebuild — A/B flash vs eager attn
# export BMO_H2_DIAG=1             # verbose matvec logs — rebuild required
# export BMO_LOG_KV=1
# export BMO_LOG_EMBED=1
# export BMO_LOG_ATTN=1
```

### Primary E2E — `moshi.offline` (Jetson)

```bash
PYTHONUNBUFFERED=1 python -u -m moshi.offline \
  --input-wav ./tellmeajoke_padded.wav \
  --output-wav ./outputs/offline_gguf_test.wav \
  --output-text ./outputs/offline_gguf_test.json \
  --text-prompt "You are BMO. Tell me a joke." \
  --voice-prompt bmo_621.wav \
  --voice-prompt-dir . \
  --tokenizer ./tokenizer_spm_32k_3.model \
  --mimi-weight ./tokenizer-e351c8d8-checkpoint125.safetensors \
  --device cuda
```

**Success log markers:**

- `delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- `C++ engine engaged: ... dep_q=16 n_codebooks=17`
- `reset libbmo KV/position after warmup`
- `libbmo temporal position after prompts: <N>` (should be ≪ `BMO_N_CTX`)

**Greedy / reproducible:**

```bash
PYTHONUNBUFFERED=1 python -u -m moshi.offline \
  ...same args... \
  --greedy --seed 42
```

### Text-only C++ smoke (no Mimi — isolates temporal)

```bash
export PYTHONPATH=./moshi
export BMO_SO_PATH=./build_jetson/libbmo.so

python bmo_inference.py text \
  --gguf ./models/bmo_septq_v3.gguf \
  --tokenizer ./tokenizer_spm_32k_3.model \
  --text-prompt "You are BMO. Tell me a joke." \
  --n-generate 40 \
  --n-ctx 512 \
  --so-path ./build_jetson/libbmo.so
```

Coherent decoded text here → temporal GGUF mostly OK; bug likely in **LMGen + depth + Mimi** path. Still gibberish → **GGUF / SEPTQ / Jetson attention**.

### `bmo_inference.py` stream (LMGen parity with offline)

```bash
export PYTHONPATH=./moshi
export BMO_USE_CPP=1
export BMO_SO_PATH=./build_jetson/libbmo.so
export BMO_GGUF=./models/bmo_septq_v3.gguf
export BMO_N_CTX=2048
export NO_CUDA_GRAPH=1

python bmo_inference.py stream \
  --gguf ./models/bmo_septq_v3.gguf \
  --mimi ./tokenizer-e351c8d8-checkpoint125.safetensors \
  --tokenizer ./tokenizer_spm_32k_3.model \
  --voice-prompt ./bmo_621.wav \
  --text-prompt "You are BMO. Tell me a joke." \
  --input-wav ./tellmeajoke_padded.wav \
  --n-frames 125 \
  --n-ctx 2048 \
  --output-wav ./outputs/bmo_stream_test.wav \
  --output-text ./outputs/bmo_stream_test.json \
  --so-path ./build_jetson/libbmo.so
```

(`bmo_inference.py` uses **SINE_TOKENS** on user channel during prefill in its manual path; `moshi.offline` uses **LMGen** which already calls `_encode_sine_frame()` for user input during prompts — do not confuse the two drivers.)

### Matvec / load smoke (no audio)

```bash
export PYTHONPATH=./moshi BMO_SO_PATH=./build_jetson/libbmo.so
python bmo_inference.py smoke \
  --gguf ./models/bmo_septq_v3.gguf \
  --n-steps 50 \
  --n-ctx 128 \
  --so-path ./build_jetson/libbmo.so
```

### Layer harness (server with PT weights + GPU)

```bash
export PYTHONPATH=.:./moshi
export BMO_SO_PATH=./build/libbmo.so   # or build_jetson on device

python scripts/end2end_v5_vs_fp16.py \
  --gguf ./models/bmo_septq_v3.gguf \
  --so-path "$BMO_SO_PATH" \
  --checkpoint /path/to/model.pt \
  --harness-input harness_input.json \
  --out path_b_day3_e2e_report.txt
```

Optional Jetson residual dumps: `BMO_H3_DUMP_RESIDUAL_BINS=1` + rebuild, then milestone table in script.

### Environment (H100 server — reference)

```bash
cd /home/jovyan/work/BMO-Project/personaplex_repo   # or your path
export PYTHONPATH=$PWD:$PWD/moshi
export BMO_SO_PATH=$PWD/build/libbmo.so
export BMO_USE_CPP=1
export BMO_GGUF=$PWD/models/bmo_septq_v3.gguf
export CUDA_VISIBLE_DEVICES=1
export NO_CUDA_GRAPH=1
```

---

# Build commands

### Jetson Orin (`build_jetson`)

```bash
cd ~/BMO-Project/BMO\ Voice\ Engine/personaplex

cmake -B build_jetson -S . \
  -DBMO_ENABLE_CUDA=ON \
  -DBMO_TARGET_JETSON=ON \
  -DBMO_BUILD_TESTS=OFF

cmake --build build_jetson -j$(nproc) --target bmo_shared

# Verify symbols
nm -D --defined-only build_jetson/libbmo.so | grep ' T bmo_'
```

Defines **`BMO_JETSON`**: eager CPU attention on `n_token==1`, pinned staging pool, 1 GiB work mem, optional KV cap.

### H100 / discrete GPU (`build`)

```bash
cmake -B build -S . -DBMO_CUDA_ARCHS="87;90"
cmake --build build -j"$(nproc)" --target bmo_shared
```

No `BMO_JETSON` → single-token path uses **lazy flash-attn** (unless you set `BMO_DISABLE_EAGER_ATTN` on a Jetson build only).

### After C++ edits

Always rebuild the `.so` matching `BMO_SO_PATH` before re-running offline/text tests.

---

# Files changed this session (sync checklist)

Copy these to the Jetson (or `git pull`) before claiming a fix is deployed:

| File | Why |
|------|-----|
| `moshi/moshi/models/loaders.py` | `bmo_config.json` overrides, `dep_q=16`, fixed `delays` |
| `moshi/moshi/offline.py` | KV reset, Mimi device, LMGen CPU, patch asserts, debug logs |
| `moshi/moshi/models/lm.py` | `_bmo_activation_device` on shell (if not already) |
| `bmo_api.cpp` | `bmo_reset` clears depth KV |
| `bmo_compute.cpp` | `BMO_DISABLE_EAGER_ATTN`, H2 diag gate |
| `bmo.cpp` | H2 diag gate (if touched) |
| `bmo_config.json` | Canonical delays / dep_q |
| `.cursor/SESSION_HANDOFF.md` | This file |

**Not required for delays fix:** rebuild if only Python changed. **Required** for `bmo_reset` depth + eager-attn toggle: **rebuild `libbmo.so`**.

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
4. Self-attention — decode: **CPU eager** `apply_attention_eager_decode` (Jetson default); prefill: lazy `ggml_flash_attn_ext` path.
5. **Out proj** — `self_attn_out_proj_weight`: `[4096, 4096]` (packed or dense per export).
6. Residual add (GPU `apply_residual_add_gpu` on Jetson).
7. Pre-norm RMSNorm (`norm2_weight`).
8. **SwiGLU FFN** — `gating_linear_in_weight`: `[2×d_ff, 4096]` with `d_ff=11264` → `22528` rows (matches `gpu_staging_pool::SLOT_BYTES / sizeof(float)`).
9. `gating_linear_out_weight`: `[4096, d_ff]`.
10. Residual add.

**Final temporal:** `out_norm_weight` (RMSNorm γ, F32), then `text_linear` `[32000×4096]` F16 (`text_linear_bias=MISSING` in current GGUF log).

**Depth block (per `depformer_layers_{i}`, dense F16):** same pattern at 1024-dim; **no** `packed_weights` in export — `ggml_mul_mat` on F16 weights.

**KV caches:**

- Temporal: FP16, shape `[head_dim, n_ctx, n_heads, n_layers]` → **512 MB** at `n_ctx=1024` (logged).
- Depth: FP16, `dep_ctx=16` (codebook steps), **384 KB** total.
- Reset: `bmo_reset` zeroes temporal K/V + **`bmo_reset_depth_kv`**; depth also reset at `cb_index==0` each frame.

**Streaming / scheduling:**

- Frame rate **12.5 Hz** (80 ms/frame).
- Per frame: Mimi encode → temporal step(s) → for each codebook step `k`, depth forward → sample audio token → Mimi decode agent channels.
- `LMGen` + `streaming_forever`; C++ path: patched `LMModel.forward_codes` → `bmo_engine.forward_temporal(tokens_np)` with internal `pos` from `BMOEngine._pos`.
- `NO_CUDA_GRAPH=1` on Jetson offline — **no** CUDA graphs for temporal; eager kernels + deferred `cudaStreamSynchronize(0)` at layer boundaries.

**LMGen token layout (17 codebooks):**

- `tokens[0]` = text
- `tokens[1..8]` = agent (moshi) audio — Mimi `set_num_codebooks(8)` for decode uses `tokens[:, 1:9]`
- `tokens[9..16]` = user audio (8 codebooks)
- `AUDIO_TOKENS_PER_STREAM = 8` in `lm.py`
- **SINE_TOKENS** on user channel during prompts via `_encode_sine_frame()`; **SILENCE_TOKENS** for agent during silence spacers

---

## SECTION 2–10 — (unchanged technical depth)

Sections 2–10 below retain the full quantization pipeline, kernel architecture, memory layout, INT2 details, bottleneck analysis, optimization table, codebase map, and raw code snippets from the 2026-05-17 audit. Still accurate for SEPTQ v5 / Jetson matvec path.

### Quick pointer — hottest files

```
personaplex/
  bmo.h, bmo.cpp, bmo_compute.cpp, bmo_api.cpp
  bmo_cuda_kernels.cu, bmo_cuda_kernels_proto.cu
  export_bmo_gguf.py, bmo_config.json
  bmo_inference.py
  moshi/moshi/offline.py, models/loaders.py, models/lm.py, bmo_engine.py
  scripts/end2end_v5_vs_fp16.py
  bmo_serverfiles/          # snapshot from server (compare, not auto-synced)
```

---

## SECTION 2 — QUANTIZATION PIPELINE

### Assignment (export: `export_bmo_gguf.py` → `create_packed_layer`)

**Static at export time** — not runtime-dynamic.

**Sources (priority):**

1. **v5 per-element:** `tier_masks_uint2` from PTQ/QAT checkpoint, byte length `(rows × padded_cols + 3) / 4`. Serialized **verbatim** — no re-tiering.
2. **v4 per-block:** `block_tier_map` or `max(abs)` thresholds / ratio ranking inside 32-wide blocks.
3. **Tier encoding in `packed_mask`:** 2 bits per slot, little-endian nibble order:  
   `0=FP16`, `1=INT8`, `2=INT4`, `3=INT2` (mask value `m` → storage tier `3-m` when deriving from external maps).

**Affine dequant (per layer, global — not per-block scale in kernel):**

```
w = (q - zp_low)   * scale_low    # INT2, q ∈ [0,3]
w = (q - zp_int4)  * scale_int4   # INT4, q ∈ [0,15]
w = (q - zp_int8)  * scale_int8   # INT8
w = fp16_values[i]                 # tier 0
```

---

## SECTION 3 — RUNTIME INFERENCE PIPELINE

### End-to-end (Jetson, `BMO_USE_CPP=1`)

```
Audio PCM → Mimi.encode (CPU) → user codes [8] per frame
  → LMGen.prepare_step_input (delay ring) → cache [17] @ model_input_position
  → patched forward_codes → bmo_forward_temporal(tokens, pos)
  → depformer_step × dep_q (16) → patched forward_depformer
  → sample text + agent audio → Mimi.decode(agent codes)
```

**Per temporal layer, per token (`n_token==1`, Jetson default):**

1. RMSNorm GPU → fused SEPTQ matvec QKV → sync → RoPE GPU → **CPU eager attention** → out proj matvec → residual GPU → FFN → residual.

**Toggle:** `BMO_DISABLE_EAGER_ATTN=1` uses multi-token flash-attn graph even for `n_token==1` (server-like; may be racy on staging — use for A/B only).

---

## SECTION 4–7 — CUDA / INT2 / bottlenecks

(See prior audit: v5 proto matvec with ballot divergence; ~0% Tensor Cores on temporal path; CPU attention cost scales with `kv_len`; gate `BMO_H2_DIAG` for production.)

---

## SECTION 8 — CURRENT OPTIMIZATION ATTEMPTS

| Attempt | Outcome |
|---------|---------|
| SEPTQ + GGUF export | Works; ~1.8 GB temporal weights |
| Pinned `cudaHostRegister` weight pool | Works on Jetson |
| Eager GPU norms / RoPE / SwiGLU | Correctness on Jetson |
| Eager CPU attention | Correctness; perf hit; A/B via `BMO_DISABLE_EAGER_ATTN` |
| Empty LM shell + LMGen on CPU | Fits 8 GB unified memory |
| Mimi on CPU after GGUF | Fits memory; token `.to(mimi.device)` in offline |
| **delays + dep_q aligned with `bmo_config.json`** | **Applied 2026-05-19**; E2E still gibberish → not root cause alone |
| RoPE NeoX flip | **Rejected** — wrong layout |
| SINE vs SILENCE on user (bmo_inference manual path) | Fixed in `bmo_inference.py`; **offline uses LMGen sine** already |

---

## SECTION 9–10 — CODEBASE + SNIPPETS

(Unchanged — tier mask extract, INT2 decode, `launch_fused_dequant_matvec_jetson` dispatch, Jetson `cudaHostGetDevicePointer` for weights.)

---

# Jetson offline / memory / history

## What works on Jetson (2026-05-19)

- `libbmo.so` builds (`build_jetson`, `bmo_shared`).
- GGUF load + prepare (~6.5 GB VmRSS, ~200–260 MB MemAvail after load).
- Warmup completes; **`bmo_engine.reset()` after warmup**.
- Voice/text prompt phases complete (`Done loading voice prompt` / `text prompt` / silence).
- Generation loop runs; **WAV + JSON written** (quality still bad).
- Log shows **`delays=[0, 0, 1, 1, …, 1, 1]`** and **`dep_q=16`**.

## Resolved blockers (do not re-debug as open)

| Blocker | Resolution |
|---------|------------|
| Warmup device mismatch (CUDA tokens, CPU Mimi) | `_agent_audio_codes_on_mimi_device()` |
| LMGen on GPU OOM after GGUF | `BMO_LMGEN_DEVICE=cpu`, empty shell |
| Mimi before GGUF OOM | Load order in `offline.py` |
| Warmup polluting KV | `bmo_engine.reset()` after warmup |
| Wrong `delays[9]=0` | `loaders.py` + `bmo_config.json` |
| Missing `pyloudnorm` | pip install / deps |

## Open issue

**Output quality:** gibberish text/audio despite correct delays and successful E2E. Suspect **Jetson eager attention vs server flash**, **SEPTQ numerical drift**, or **harness mismatch** — use diagnostic commands in [Run commands](#run-commands-jetson--server--diagnostics).

## Typical load log (healthy)

```
[bmo_load_model] dep_q=16 num_codebooks=17 ...
[bmo_init_kv_cache] Allocated KV cache: 512 MB
[Info] C++ engine engaged: ... delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[Info] constructing LMGen (device=cpu)
[Info] reset libbmo KV/position after warmup (BMO_USE_CPP)
[Info] libbmo temporal position after prompts: <N> (n_ctx=1024)
```

SciPy warning `NumPy 1.26.4` vs SciPy wanting `<1.25` is noisy but non-fatal in observed runs.

---

# Gibberish investigation (RoPE, SINE, delays, attention)

## Ruled out or fixed

- ✅ Weights/scales/masks vs PTQ export (per prior server validation).
- ✅ RoPE **NORMAL** (interleaved) — **do not** switch to NEOX.
- ✅ `rope_theta=10000` matches Moshi.
- ✅ **delays** and **dep_q** match GGUF / `bmo_config.json` (2026-05-19).
- ✅ Offline prompt path uses **SINE** on user via LMGen (not the old `bmo_inference` SILENCE bug).

## Still open

- ❌ Gibberish after delay fix on Jetson `moshi.offline`.
- ❌ Whether H100 “good” run used **same** `BMO_USE_CPP` + GGUF path.
- ❌ Jetson **`apply_attention_eager_decode`** vs server flash-attn parity.
- ❌ Full **`end2end_v5_vs_fp16.py`** on device with PT checkpoint (usually server-only).

## SINE_TOKENS reference (`moshi/models/lm.py`)

```python
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]  # agent silence spacers
SINE_TOKENS    = [430, 1268, 381, 1611, 1095, 1495, 56, 472]   # user channel @ prompts
```

`bmo_inference.py` manual prefill must use **SINE** on user; generation fallback may use **SILENCE** when input wav exhausted.

## RoPE decision tree (if text mode still fails)

| rope_consistency_test | K-cache cosine | Conclusion |
|---|---|---|
| FAIL | n/a | RoPE GPU vs ggml mismatch — **not** NEOX flip |
| PASS | < 0.99 | KV write/position/mask during prefill |
| PASS | ≥ 0.99 | Downstream: matvec, out_proj, depth, sampler |

Worker scripts (`rope_consistency_test`, `dump_kcache_after_prefill`) were **never delivered** — still valid plan if eager-attn A/B fails.

## Hard rules

1. **Do not** flip `GGML_ROPE_TYPE_NORMAL` → `NEOX`.
2. **Do not** assume first `PAD` token means success.
3. Rebuild `libbmo.so` after C++ changes before concluding a fix failed.
4. Compare server vs Jetson **build flags** (`BMO_JETSON`) when comparing audio quality.
5. SEPTQ-only — no Q4_K_M llama.cpp quant fallback.

---

# Reference: `bmo_serverfiles`

Path: `personaplex/bmo_serverfiles/` — user-added snapshot from H100 box.

Contains older copies of `bmo_compute.cpp`, `bmo.cpp`, `moshi/moshi/offline.py`, `loaders.py`, etc. **Not auto-synced** with repo root. Server `loaders.py` still had `delays[9]=0` in `_lm_kwargs` like pre-fix tree — do not treat as canonical without diff.

Use for: diff Jetson vs server C++, or recovering server `offline.py` load order notes.

---

## Report / docs

- `.cursor/SESSION_HANDOFF.md` — this file
- `REPORT_C_CPP_JETSON_CURRENT_STATE.md` — narrative report
- `scripts/generate_report_figures.py` — tier heatmaps / figures

---

## Git / sync note

Repo git root may be `BMO-Project` parent; personaplex is nested. On Jetson, `git pull` from the branch you develop on, then rebuild `build_jetson`, then re-run [Primary E2E](#primary-e2e--moshioffline-jetson).

**Minimum pull set for 2026-05-19 Python fixes:** `loaders.py`, `offline.py`, `bmo_config.json`.  
**Minimum rebuild set:** `bmo_api.cpp`, `bmo_compute.cpp` (+ `bmo.cpp` if H2 diag changed).
