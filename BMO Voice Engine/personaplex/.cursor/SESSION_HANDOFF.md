# BMO C++ Runtime Gibberish — Session Handoff

**Last updated:** 2026-05-11 23:14 UTC+1
**Pickup transcript ID:** `c9c6ec77-3d61-4307-961a-d2ba4cdffadd`

When you resume on another system, paste this whole file into the new chat so the agent has full context. To continue the original chat verbatim use `[BMO RoPE Bug Investigation](c9c6ec77-3d61-4307-961a-d2ba4cdffadd)`.

---

## The one-line status

GGUF + C++ runtime produces gibberish audio. PT (.pt / .safetensors) produces coherent audio. Text logits in C++ output are **static across frames** — canonical symptom of attention being unable to see context.

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

## What's currently in flight

Background worker (already launched, will land in chat when done):

- **Goal:** localize the actual bug with two scripts, no code changes to RoPE math.
- **Deliverable 1:** standalone CUDA/C++ test `tests/rope_consistency_test.cpp` + CMake target `bmo_rope_consistency_test`. Compares `ggml_rope(NORMAL)` output to `launch_rope_interleaved` output for an identical synthetic input. Exits 0 if max_abs_diff < 1e-5.
- **Deliverable 2:** `scripts/dump_kcache_after_prefill.py` + a new debug C-API entry `bmo_get_kcache_layer(int layer, float* out, int max_elements)` exposed through `bmo_inference.py`. Runs identical small prefill in PT and C++, dumps K-cache layer 0 (and 16), computes cosine + per-head breakdown.
- **Deliverable 3:** exact server-side commands + a decision tree for interpreting the results.

When the worker lands, the chat will have build + run commands ready to paste on the server.

## Decision tree (apply once worker returns results)

| rope_consistency_test | K-cache cosine | Conclusion |
|---|---|---|
| FAIL (exit 1) | n/a | RoPE math diverges between multi-token ggml fallback and single-token CUDA kernel. Fix is NOT NORMAL→NEOX. Look for stride/position mismatch in `apply_rope_gpu_interleaved` (`bmo_compute.cpp:542-599`). |
| PASS | < 0.99 | Prefill produces a bad K cache. Inspect: positions passed during prefill, KV cache write path, attention mask. Start at `bmo_build_temporal_graph` around `bmo_compute.cpp:2100-2140`. |
| PASS | ≥ 0.99 | Prefill is fine. Bug is downstream: decode-side Q computation, attention compute, out_proj, depth transformer, or sampler. |

## Hard rules for the next session

1. **Do not** apply Gemini's `GGML_ROPE_TYPE_NORMAL` → `GGML_ROPE_TYPE_NEOX` flip.
2. **Do not** change `rope_theta` handling. The model uses 10000 and `ggml_rope`'s default is 10000.
3. **Do not** restart end-to-end debugging from scratch. Use the worker's two scripts to localize first.
4. **Do not** chase Q4_K_M / generic llama.cpp quant fallback paths — user has explicitly ruled this out repeatedly. SEPTQ multi-tier is the only quantization path.

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

## If the worker takes forever or you need to start fresh on the new system

Re-prompt the new chat with:

> Read `.cursor/SESSION_HANDOFF.md`. The C++ runtime produces gibberish but PT works. A worker was previously launched to write a RoPE consistency test and a K-cache dump+compare script — pick up from that plan. Do not change any RoPE constants. Verify the prior worker's output if available, otherwise re-launch it.

Sleep well.
