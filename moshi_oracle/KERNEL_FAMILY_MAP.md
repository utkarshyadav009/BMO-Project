# KERNEL_FAMILY_MAP.md

**Purpose:** map each of four CUDA GEMV/copy kernel families to the tensor
families in the BMO/PersonaPlex model that actually dispatch to them, and
derive a per-frame call count from the real graph structure and config.
**Method:** pure static analysis — reading `moshi_oracle/ggml/src/ggml-cuda/*`,
`moshi_oracle/moshi.cpp/src/*`, plus one read-only data inspection of the
actual shipped weight file (`qat_heavy_int2.gguf`, via Python's `gguf`
package, header/tensor-metadata only, no tensor data touched, no GPU, no
build, no run of any project binary). Every number below is labeled
**Measured** (read from an instrument in a prior, already-completed run —
i.e. quoted from `HANDOFF.md`) or **Estimated** (derived arithmetically from
source code + config + gguf tensor metadata in this session, not measured on
a GPU this session).

Repo state analyzed: branch `experiment/multitier-dequant`, working tree as
found (HEAD `803ee57` plus the uncommitted `git status` shown at session
start). No file under `moshi_oracle/` was modified by this task — see
**Files touched** at the end.

---

## 0. Config numbers actually used (verified, not assumed)

Source: `moshi_oracle/models_h100_actual/qat_heavy_int2_dir/personaplex-config.json`
(the file the task pointed at — confirmed identical in spirit to
`moshi_oracle/models_h100_actual/personaplex/personaplex-config.json`, not
re-diffed line by line since the `qat_heavy_int2_dir` copy is the one paired
with the actual weights file used).

| field | value | used for |
|---|---:|---|
| `num_layers` | 32 | temporal transformer depth |
| `num_heads` | 32 | temporal attention heads |
| `dim` | 4096 | temporal hidden size → dim_per_head = 4096/32 = **128** |
| `context` | 3000 | temporal self-attn KV-cache capacity |
| `dep_q` | 16 | depformer substeps/frame (see §3.0 for why the `dep_q=8` string found elsewhere in source does **not** override this) |
| `n_q` | 16 | audio codebooks; `num_codebooks = n_q+1 = 17` (`lm_default.h:217`) |
| `depformer_num_layers` | 6 | depformer transformer depth |
| `depformer_num_heads` | 16 | depformer attention heads |
| `depformer_dim` | 1024 | depformer hidden size → dim_per_head = 1024/16 = **64** |
| `depformer_context` | 8 | depformer self-attn KV-cache capacity |
| `depformer_multi_linear` | true | enables the per-substep `depformer_in[cb_index]` selection |
| `depformer_weights_per_step` | true | (schedule itself is absent from the JSON → schedule array size 0, see §3.0) |
| `cross_attention` | false | confirms **no cross-attention layers exist at all** in this deployment — `moshi_smha_state`'s cross-attention branch (`transformer.h:396-430ish`) is dead code for this model |
| `model_type` | `"personaplex"` | sets `lmmodel->personaplex = true` (`lm_default.h:223`) |

KV-cache dtype is **not** in this JSON — it's a CLI flag. The run recipe
used throughout this project (`HANDOFF.md` §7): `personaplex -k q4_0 ...`.
`-k q4_0` → `config.kv_cache_type = GGML_TYPE_Q4_0` (`personaplex.cpp:574`)
→ `lm->kv_cache_type` (`moshi.cpp:697`) → `gen->state_ctx->kv_cache_type`
(`moshi.cpp:980`), a **single global field** shared by every self-attention
layer built from that `StateContext*` — temporal **and** depformer alike
(`moshi_lmmodel_states()`, `lm.h:434-444`, passes the same `state_ctx` to
both). All per-frame counts below assume this standard `-k q4_0` invocation.

---

## 1. `mul_mat_vec_q<GGML_TYPE_Q4_0>` (type=2)

**Verified enum/dispatch:** `ggml/include/ggml.h:387` `GGML_TYPE_Q4_0 = 2`.
Dispatch in `ggml/src/ggml-cuda/mmvq.cu`: `vec_dot_q4_0_q8_1`/`VDR_Q4_0_Q8_1_MMVQ`
selected at lines 12/38 (per-type function tables) and the actual kernel
launch switch at `mmvq.cu:479` (`case GGML_TYPE_Q4_0:`).

This family has **two structurally different sources** in the graph — real
weight matrices (native Q4_0 in the GGUF) and the Q4_0-quantized KV cache
being read as the attention-score operand. Verified by reading the actual
`qat_heavy_int2.gguf` tensor list (Python `gguf.GGUFReader`, read-only,
1839 tensors total — no Q4_K or BMO_TIER appear as native `tensor_type`
because those two are built at **load time** from raw packed bytes, see
§2/out-of-scope §5).

### 1a. Weight-linear GEMVs (native Q4_0 tensors in the GGUF)

| tensor family | GGUF names | dtype | calls/frame | arithmetic |
|---|---|---:|---:|---|
| depformer self-attn `in_proj` | `depformer_layers_{0..5}_self_attn_in_proj_weight` (1 tensor/layer, **view**-sliced 16 ways) | Q4_0 | 96 | 16 substeps × 6 layers |
| depformer self-attn `out_proj` | `depformer_layers_{0..5}_self_attn_out_proj_weight` (view-sliced 16 ways) | Q4_0 | 96 | 16 × 6 |
| depformer gating `linear_in` | `depformer_layers_{0..5}_gating_{0..15}_linear_in_weight` (16 **distinct physical tensors**/layer) | Q4_0 | 96 | 16 × 6 |
| depformer gating `linear_out` | `depformer_layers_{0..5}_gating_{0..15}_linear_out_weight` | Q4_0 | 96 | 16 × 6 |
| depformer per-substep input proj | `depformer_in.{0..15}.weight` | Q4_0 | 16 | 1 per substep |
| temporal text output head | `text_linear.weight` | Q4_0 | 1 | once/frame |
| **subtotal** | | | **401** | |

Arithmetic detail for the depformer rows: `moshi_lmmodel_depformer_step`
(`lm.h:489-542`) builds **one persistent GGML graph, once**, containing a
`for (cb_index = 1; cb_index < lm->dep_q; cb_index++)` loop
(`lm.h:523`) chained onto an initial `cb_index=0` call — **16 total**
`moshi_lmmodel_forward_depformer_transform` invocations baked into the
graph. `lm->dep_q` is set from `config->dep_q` = 16 verbatim
(`lm_default.h:154`). That persistent graph's `ctx.compute()` (`lm.h:558`)
re-executes **every frame**, outside the one-time build guard — so all 16
substeps' worth of matmuls really do run every single frame, not just once
at startup. Each `moshi_lmmodel_forward_depformer_transform` call runs the
full 6-layer depformer transformer (`moshi_streaming_transformer`,
`lm.h:480-481` → `transformer.h:1360-1381`, one `moshi_streaming_transformer_layer`
call per layer, no early-exit).

Per depformer layer, each substep does exactly one `in_proj` mul_mat and one
`out_proj` mul_mat — confirmed via `get_weights(WeightLoader*, string,
moshi_smha_t*)` (`transformer.h:899-929`): because
`config->depformer_weights_per_step_schedule` is **absent** from the JSON
(`.size()==0`), `depformer_num_weights = config->dep_q = 16`
(`lm_default.h:73`), and `attn->in_projs.size()==16 > 1` takes the
**view-slicing branch** (`transformer.h:910-921`): all 16 `in_projs[i]->weight`
are `ggml_view_2d` slices of the **one** GGUF tensor
`..._self_attn_in_proj_weight` (confirmed dtype-preserving — a view keeps
its parent's `ggml_type`). `moshi_apply_weights_per_step_linear`
(`transformer.h:55-99`) then does exactly one `ggml_mul_mat` per call
(T=1 per substep) selecting `modules[t+offset]` — one physical Q4_0 GEMV,
not sixteen. Gating is different: `layer->gating[j]` (`lm_default.h:136-146`)
are **16 real separate `moshi_activation_gating_t` objects**, each with its
own `linear_in`/`linear_out` GGUF tensor (confirmed distinct names in the
gguf dump: `gating_0_linear_in_weight` … `gating_15_linear_in_weight`) — 2
GEMVs (`moshi_activation_gating`, `gating.h:12-37`) per (substep, layer).

`text_linear` (`lm.h:674`/`697`/`729`, all three text-forward variants) runs
once per frame — the temporal transformer (`lm->transformer`, 32 layers) is
called exactly once per frame (T=1) in all of `moshi_lmmodel_forward_text`,
`_build`, `_step`, `moshi_lmmodel_forward_embedding`.

### 1b. KV-cache-read GEMVs (attention Q·Kᵀ against the quantized cache)

`moshi_streaming_multihead_attention` (`transformer.h:501` and its duplicate
at `transformer.h:679`, identical logic — see §3) computes
`attn_weight = ggml_mul_mat(ctx, key, query)` (`torch.h:158`, called via
`torch_nn_functional_scaled_dot_product_attention_custom`) where `key` is
the **full KV-cache tensor returned by `moshi_kv_cache_insert_kv`**
(`transformer.h:258-296`, the `ggml_set_rows`-based overload — confirmed
this is the overload actually called at `transformer.h:618`/`790`, which
pass an `indices` tensor, not an int index; the int-index overload at
`transformer.h:194-256`, gated by `#define CACHE_BF16` at line 169, is
dead code for this call site). Since `key`'s dtype = `state_ctx->kv_cache_type`
= Q4_0 and `query`'s dtype = F32, and the query has a single column
(T=1 per frame/substep), GGML's CUDA mul_mat dispatch routes this to
`mul_mat_vec_q<GGML_TYPE_Q4_0>` directly on the **still-quantized** K
tensor — no dequant copy for K (see §4 for why V is different).

| attention site | capacity (ne1 of cache) | dim_per_head | num_heads | calls/frame |
|---|---:|---:|---:|---:|
| temporal self-attn | 3000 (`context`) | 128 | 32 | 32 (1/layer × 32 layers) |
| depformer self-attn | 8 (`depformer_context`) | 64 | 16 | 96 (1/(substep,layer) × 16×6) |
| **subtotal** | | | | **128** |

Both temporal and depformer KV caches are quantized here because
`state_ctx->kv_cache_type` is one shared field (§0); `moshi_smha_state`
(`transformer.h:382-393`) calls `moshi_kv_cache_state` unconditionally for
every self-attention layer, `ggml_is_quantized(Q4_0)==true` for both
dim_per_head=128 and dim_per_head=64 (block size 32 divides both cleanly,
so the `dim_per_head % blck_size` fallback-to-Q4_0 warning path,
`transformer.h:180-190`, never triggers — the type stays Q4_0 as
configured, it isn't already-Q4_0-by-fallback).

### Family 1 total: **401 + 128 = 529 `mul_mat_vec_q<Q4_0>` calls/frame** (Estimated, derived from source + config, not measured on GPU this session)

---

## 2. `mul_mat_vec_q<GGML_TYPE_Q4_K>` (type=12)

**Verified enum/dispatch:** `ggml/include/ggml.h:397` `GGML_TYPE_Q4_K = 12`.
Dispatch in `mmvq.cu`: `vec_dot_q4_K_q8_1`/`VDR_Q4_K_Q8_1_MMVQ` (lines 20/46),
kernel launch switch at `mmvq.cu:527` (`case GGML_TYPE_Q4_K:`).

**This type is never baked into the GGUF file as native `GGML_TYPE_Q4_K`
blocks.** Confirmed by the tensor-type census of the actual weight file
(`qat_heavy_int2.gguf`): only `I32`, `F32`, `Q4_0`, `I8`, `F16` appear as
native `tensor_type` values (744/573/254/186/82 tensors respectively, 1839
total) — no `Q4_K`. Instead, the **temporal transformer's self-attention
in/out projection weights for layers 0-30** (31 of the 32 layers) are stored
as a custom group-quantized INT4 format: `..._weight.packed_weights` (I8),
`.scales`/`.zeros` (F32), `.group_size`/`.n_groups`/`.rows`/`.cols`/
`.packing_version` (I32, `packing_version==10` for these). At load time,
`WeightLoader::get_tensor` (`loader.h:1275-1290`) detects the
`.packed_weights` suffix via `gguf_find_tensor` and, for
`packing_version==10`, calls `build_quantized_attn_tensor`
(`loader.h:641-684`): it dequantizes the packed INT4+scale+zero data to an
F32 CPU buffer (`dequantize_attn_to_f32`, `loader.h:607-639`), then
**re-quantizes to `GGML_TYPE_Q4_K`** via `ggml_quantize_chunk(target_qtype,
...)` with `target_qtype = GGML_TYPE_Q4_K` hardcoded at `loader.h:662`.
Layer 31 (the 32nd, last temporal layer) has **no** `.packed_weights` for
its attention tensors — `transformer_layers_31_self_attn_{in,out}_proj_weight`
are plain native `F16` in the gguf, so `get_tensor` falls through to a
direct lookup and layer 31's attention runs in F16, **outside all four
families** in this task (noted, not counted).

| tensor family | GGUF source | dtype after load | calls/frame |
|---|---|---:|---:|
| temporal self-attn `in_proj`, layers 0-30 | `transformer_layers_{0..30}_self_attn_in_proj_weight.packed_weights` (+scales/zeros) → `build_quantized_attn_tensor` → Q4_K | 31 |
| temporal self-attn `out_proj`, layers 0-30 | `transformer_layers_{0..30}_self_attn_out_proj_weight.packed_weights` → Q4_K | 31 |
| **subtotal** | | | **62** |

Arithmetic: 31 layers (0-30) × 2 GEMVs (in_proj + out_proj) = 62. In_proj
here is the SAME single combined-QKV tensor pattern as the depformer (one
`ggml_mul_mat` produces q,k,v together via `ggml_view_3d` splits inside
`moshi_streaming_multihead_attention`, `transformer.h:559-608`ish) — 1 call,
not 3, per layer.

### Family 2 total: **62 `mul_mat_vec_q<Q4_K>` calls/frame** (Estimated)

**Out-of-scope reminder (per task instructions):** the temporal FFN gating
weights for these same layers 0-30 (`..._gating_linear_in_weight` /
`..._gating_linear_out_weight`, `packing_version==6`,
`build_custom_ffn_tensor` → `GGML_TYPE_BMO_TIER`, `loader.h:875`) dispatch
to `mul_mat_vec_bmo_tier_*`, a separate custom kernel — **62 more calls/frame
(31×2) that are deliberately NOT counted into Q4_K or Q4_0 above.**

---

## 3. `mul_mat_vec_f<f32,f32>`

**Verified:** `ggml/src/ggml-cuda/mmvf.cu`. Dispatch guard
`ggml_cuda_should_use_mmvf` (`mmvf.cu:719-744`): for `src0->type==F32`, the
CUDA path is taken only when `ne11` (src1's column count) is small (`<=3`
on Ampere-MMA-capable GPUs, `<=4` on Turing+) — i.e. a mat-vec shape, not a
GEMM. Kernel switch case `GGML_TYPE_F32:` at `mmvf.cu:635`/`694`.

### 3.0 Source: the Hadamard rotation on Q/K/V

`moshi_smha_state` (`transformer.h:382-393`): `if
(ggml_is_quantized(state_ctx->kv_cache_type)) state_ctx->add_hadamard(dim_per_head);`
— since `kv_cache_type=Q4_0` is quantized (§0/§1b), **every** self-attention
layer (temporal and depformer) gets a Hadamard matrix attached.
`StateContext::add_hadamard` (`context.h:815-851`) builds `hadamard64`
(64×64) or `hadamard128` (128×128) as **`GGML_TYPE_F32`** tensors
(`context.h:875-888`, the `std::vector<float>`-taking `new_tensor` overload
hardcodes `GGML_TYPE_F32`) — one singleton each, shared/reused across all
layers with that head dimension (guarded by `if (hadamard64 != NULL)
return;`).

In `moshi_streaming_multihead_attention` (`transformer.h:622` / the
duplicate at `794`):
```cpp
ggml_tensor * hadamard = state->hadamard ? *state->hadamard : NULL;
if ( hadamard ) {
    q = ggml_mul_mat( ctx, hadamard, ggml_cont( ctx, q ) );
    k = ggml_mul_mat( ctx, hadamard, ggml_cont( ctx, k ) );
    v = ggml_mul_mat( ctx, hadamard, ggml_cont( ctx, v ) );
}
```
`ggml_mul_mat` always allocates its result as `GGML_TYPE_F32`
(`ggml.c:3189`), and q/k/v are F32 going in (post-linear-projection,
pre-cache-write) — so src0=hadamard(F32), src1=q/k/v(F32), both operands
F32. Each call's `ne11` (src1 columns) = `T` = 1 per frame/substep during
steady-state streaming decode (`states->offset += T` bookkeeping throughout
`transformer.h`/`lm.h` confirms T=1 outside the initial prompt-prefill
pass) — satisfies the `ne11<=3/4` mmvf gate, so this dispatches to
`mul_mat_vec_f<float,float,...>`, not cuBLAS.

| attention site | dim_per_head (hadamard size) | attention calls/frame | ×3 (q,k,v) |
|---|---:|---:|---:|
| temporal self-attn | 128 (hadamard128) | 32 | 96 |
| depformer self-attn | 64 (hadamard64) | 96 (16×6) | 288 |
| **subtotal** | | 128 | **384** |

### Family 3 total: **384 `mul_mat_vec_f<f32,f32>` calls/frame** (Estimated)

No other F32×F32 mul_mat was found: the depformer per-substep output head
(`lm->linears[cb_index]`, GGUF names `linears.{0..15}.weight`) is native
**F16**, not F32 — confirmed in the gguf census — so it's a different
`mul_mat_vec_f<half,...>` instantiation, out of scope for this family (16
calls/frame, noted not counted). Biases are `NULL` for every linear in the
attention/gating path actually populated (`in_projs[i]->bias = NULL` etc.,
`transformer.h:920`/`927`/`940`/`947`), so no `ggml_add`-driven F32 GEMV
hides there either.

---

## 4. `cpy_q_f32` — KV-cache dequant copy (`ggml_cpy_q4_0_f32_cuda`)

**Verified:** `ggml_cast(ctx, a, type)` (`ggml.c:3440-3452`) is *not* a
distinct op — it constructs a `GGML_OP_CPY` node (`result->op = GGML_OP_CPY`).
`GGML_OP_CPY` dispatches in `ggml-cuda.cu` to `ggml_cuda_cpy` (also reached
via `GGML_OP_CONT`/`GGML_OP_DUP` → `ggml_cuda_dup` → `ggml_cuda_cpy`,
`cpy.cu:608-611`). The type-pair dispatch table in `ggml_cuda_cpy`
(`cpy.cu:460-605`) has an explicit branch at `cpy.cu:503-505`:
```cpp
} else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
    ggml_cpy_q4_0_f32_cuda(...)
```
This is the kernel this task calls `cpy_q_f32`, specialized for Q4_0 source
— confirmed reachable because `kv_cache_type=Q4_0` (§0).

### 4.0 Where it fires: V only, not K — and only for Q4_0/Q4_K/etc., not uniformly for all quantized types

In `moshi_streaming_multihead_attention` (`transformer.h:636-660`ish, both
copies), **K and V are handled asymmetrically**:
```cpp
if ( k->type != q->type ) {
    if ( ggml_is_quantized( k->type ) ) {
        // only Q2_K gets dequantized here:
        if ( k->type == GGML_TYPE_Q2_K ) {
            ggml_tensor * k_f32 = ggml_cast( ctx, k, GGML_TYPE_F32 );
            ...
        }
        // Q4_0 (and any other quantized type): falls through, K STAYS QUANTIZED
    } else {
        k = ggml_cast( ctx, k, q->type );
    }
}
if ( v->type != q->type ) {
    if ( ggml_is_quantized( v->type ) ) {
        ggml_tensor * v_f32 = ggml_cast( ctx, v, GGML_TYPE_F32 );   // ALWAYS, any quantized type
        v = (q->type == GGML_TYPE_F32) ? v_f32 : ggml_cast( ctx, v_f32, q->type );
    } else {
        v = ggml_cast( ctx, v, q->type );
    }
}
```
So with `kv_cache_type=Q4_0`: **K is read directly by `mul_mat_vec_q<Q4_0>`
(§1b), never dequantized.** **V is unconditionally dequantized to F32 via
`ggml_cast`→`ggml_cpy_q4_0_f32_cuda` every single attention call**, before
`value = ggml_cont(ctx, ggml_transpose(ctx, value))` (`torch.h:160`) and the
second `ggml_mul_mat(ctx, value, attn_weight)`. This asymmetry is structural,
not incidental: V must be transposed (capacity dimension becomes the fast
dimension) before the second matmul, and a block-quantized tensor cannot be
transposed+made-contiguous without dequantizing first — there is no
`Q4_0→Q4_0` non-contiguous branch in the `ggml_cuda_cpy` dispatch table
(`cpy.cu:460-605`); K needs no transpose, so it never hits this problem.

Also note (relevant to the feasibility question): if the CLI used `-k q2_k`
instead, **both** K and V would dequantize via cpy (`ggml_cpy_q2_K_f32_cuda`,
since Q2_K is the one type explicitly special-cased at `transformer.h:642`/
`815`) — this Q4_0 config is actually the *cheaper* of the two, because K
already bypasses the dequant copy entirely.

### 4.1 Call count and byte volume

| attention site | capacity | dim_per_head | num_heads | calls/frame | elements/call | elements/frame |
|---|---:|---:|---:|---:|---:|---:|
| temporal self-attn | 3000 | 128 | 32 | 32 | 12,288,000 | 393,216,000 |
| depformer self-attn | 8 | 64 | 16 | 96 | 8,192 | 786,432 |
| **total** | | | | **128** | | **394,002,432** |

The **full cache capacity** is dequantized every frame, not just the newly
written row(s): `moshi_kv_cache_insert_kv`'s `ggml_set_rows`-based overload
(`transformer.h:258-296`) returns the whole `values` tensor (ne1=capacity)
unchanged in shape; nothing downstream slices it to the valid length before
the transpose+cont+mul_mat — validity is enforced only via the additive
`attn_bias` mask in `ggml_soft_max_ext` (`get_attn_bias`/`calculate_attn_bias`,
`transformer.h:450-499`), not by shrinking the tensor. So even at frame 1,
all 3000 (temporal) / 8 (depformer) cache rows get dequantized.

**Estimated byte volume** (Q4_0 block = 32 elements / 18 bytes = 0.5625
bytes/element; F32 write = 4 bytes/element — arithmetic only, not measured):

| | read (Q4_0) | write (F32) |
|---|---:|---:|
| temporal (32 calls/frame) | ≈ 210.9 MiB/frame | ≈ 1,464.8 MiB/frame (≈1.43 GiB) |
| depformer (96 calls/frame) | ≈ 0.42 MiB/frame | ≈ 3.0 MiB/frame |

Combined with the **Measured** (from `HANDOFF.md` §1, Jetson STEP-3 run)
post-kernel-rewrite throughput of 3.114 fps, the temporal V-dequant alone
implies an *order-of-magnitude* F32-write rate of roughly 1.43 GiB ×
3.114 ≈ **4.5 GiB/s** sustained just for this one copy. This is a mixed
Measured(fps)×Estimated(bytes/frame) figure — flagged explicitly as
illustrative, not a measured bandwidth number; nothing was run this session
to confirm it against `nsys`/`ncu`.

### Family 4 total: **128 `ggml_cpy_q4_0_f32_cuda` calls/frame** (Estimated call count; byte volumes Estimated from config-derived shapes)

---

## 5. Explicitly out of scope / not double-counted

| what | dispatch | calls/frame | why excluded |
|---|---|---:|---|
| temporal FFN gating, layers 0-30 | `mul_mat_vec_bmo_tier_*` | 62 (31×2) | task-scoped out; separate custom kernel, `GGML_TYPE_BMO_TIER` |
| layer 31 self-attn in/out proj | plain F16 mul_mat | 2 | not Q4_0/Q4_K, native F16 in gguf |
| layer 31 FFN gating | plain F16 mul_mat | 2 | same |
| depformer per-substep output head `linears.{0..15}` | F16 mul_mat (→ `mul_mat_vec_f<half,...>`, not the f32/f32 family) | 16 | native F16 in gguf |
| all embedding lookups (`emb.*`, `text_emb`, `depformer_emb.*`, `depformer_text_emb`) | `ggml_get_rows` (dedicated CUDA get-rows kernel, dequantizes rows on the fly) | ≈33 | different GGML op entirely, never touches `mmvq.cu`/`mmvf.cu`/`cpy.cu`'s mul_mat/cpy paths |
| mimi encoder/decoder transformer (if any self-attention exists there) | unknown, but **not quantized** | n/a | `mimi_encode_context_t`/`mimi_decode_context_t` each construct their **own** `StateContext` (`moshi.cpp:219`,`281`), which is never touched by the `gen->state_ctx->kv_cache_type = ...` assignment (`moshi.cpp:980`) — mimi's KV cache (if present) stays at the `StateContext` default, `GGML_TYPE_BF16` (`context.h:811`), so `ggml_is_quantized()==false` there: no hadamard, no Q4_0 mmvq, no cpy dequant. Also consistent with `HANDOFF.md`'s own finding that `t_mimi_dec`/`t_mimi_enc` are dominated by `moshi_streaming_conv_transpose_1d`/conv work, not attention. |

---

## 6. Summary table

| family | kernel | calls/frame (Estimated) | dominant tensor families |
|---|---|---:|---|
| 1 | `mul_mat_vec_q<GGML_TYPE_Q4_0>` (type=2) | **529** (401 weight-linear + 128 KV-read) | depformer self-attn + gating (weights-per-step), `depformer_in`, `text_linear`; KV-cache Q·Kᵀ for both temporal (cap 3000) and depformer (cap 8) |
| 2 | `mul_mat_vec_q<GGML_TYPE_Q4_K>` (type=12) | **62** | temporal self-attn in/out proj, layers 0-30 only (layer 31 is F16) |
| 3 | `mul_mat_vec_f<f32,f32>` | **384** | Hadamard rotation of Q,K,V, applied whenever `kv_cache_type` is quantized — both temporal (128-dim) and depformer (64-dim) |
| 4 | `ggml_cpy_q4_0_f32_cuda` (KV dequant) | **128** | V-cache dequant only (K stays quantized, read directly by family 1's KV-read GEMVs) — full cache capacity every frame, both temporal (3000) and depformer (8) |

All four numbers are **Estimated**: derived from reading the graph-building
source (`transformer.h`, `lm.h`, `lm_default.h`, `loader.h`, `torch.h`),
cross-checked against the real `personaplex-config.json` values and the
real `qat_heavy_int2.gguf` tensor-type census — not from a GPU profiler run
this session (none was permitted/performed, per task constraints).

---

## 7. Feasibility note — eliminating the KV-cache dequant-copy step

**Not feasible as a drop-in swap onto the existing `fattn.cu` quantized-KV
cases, because this model's attention does not go through `fattn.cu` at
all.** `torch_nn_functional_scaled_dot_product_attention_custom`
(`moshi.cpp/src/torch.h:236-247`) is a hand-rolled SDPA built from three
primitive ops — `ggml_mul_mat(key, query)` → `ggml_soft_max_ext` →
`ggml_cont(ggml_transpose(value))` → `ggml_mul_mat(value, attn_weight)` —
with no call anywhere in `moshi.cpp/src/` to `ggml_flash_attn_ext` or
anything in `ggml/src/ggml-cuda/fattn.cu` (confirmed by grep: `fattn`/
`flash_attn` appear only inside the `ggml/` library itself, never from the
`moshi.cpp` call sites). So the `FATTN_VEC_CASES_ALL_D` entries for
`(GGML_TYPE_Q4_0, GGML_TYPE_F16)` and `(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)`
K/V-dtype combinations in `fattn.cu` are simply unreachable code for this
model as it stands — adopting them would mean swapping the entire attention
implementation from this manual three-op form to real flash-attention, a
materially larger and riskier change than "dispatch V through mmvq," with
its own numerical-behavior delta on top of the one already found
unvalidated in `HANDOFF.md` §2. That said, the narrower opportunity the task
is really asking about — reading K *directly* from quantized storage via
`mul_mat_vec_q` instead of an f32 scratch copy — is **already what happens
today** for K (§4.0): only V is forced through the dequant copy, and it is
forced there for a structural reason (a quantized tensor cannot be
transposed-then-made-contiguous without dequantizing first; `ggml_cuda_cpy`
has no non-contiguous `Q4_0→Q4_0` branch, `cpy.cu:460-605`), not an
oversight. Eliminating it without adopting flash-attention would require
either restructuring the second matmul to avoid needing V pre-transposed
(e.g. computing it as `attn_weightᵀ · V` in a layout where V's quantized
blocks stay row-major — a real kernel-design change, not a dispatch
change), or writing a dedicated "quantized-V, transposed-read" GEMV variant
analogous to what `mmvq.cu` already does for K — genuine kernel work either
way, not a config flip, and it would need the same rel_l2 →
per-layer-residual → z_s → joke-loop validation ladder `HANDOFF.md` §2
already has open and failing on the *existing* kernel rewrite before
touching a second one.

---

## Files touched by this task

**None under `moshi_oracle/` were modified except the creation of this
report file itself:**

- Created: `moshi_oracle/KERNEL_FAMILY_MAP.md` (this file).

Files **read** (no writes) during analysis, for reference:
- `moshi_oracle/ggml/include/ggml.h`
- `moshi_oracle/ggml/src/ggml-cuda/mmvq.cu`, `mmvq.cuh`
- `moshi_oracle/ggml/src/ggml-cuda/mmvf.cu`
- `moshi_oracle/ggml/src/ggml-cuda/cpy.cu`, `cpy.cuh`
- `moshi_oracle/ggml/src/ggml-cuda/ggml-cuda.cu` (dispatch-decision excerpts only)
- `moshi_oracle/ggml/src/ggml.c` (`ggml_mul_mat`, `ggml_cast`, `ggml_cont`)
- `moshi_oracle/moshi.cpp/src/loader.h`
- `moshi_oracle/moshi.cpp/src/moshi.cpp`
- `moshi_oracle/moshi.cpp/src/context.h`
- `moshi_oracle/moshi.cpp/src/torch.h`
- `moshi_oracle/moshi.cpp/src/config.h`
- `moshi_oracle/moshi.cpp/src/moshi/modules/transformer.h`
- `moshi_oracle/moshi.cpp/src/moshi/modules/gating.h`
- `moshi_oracle/moshi.cpp/src/moshi/models/lm.h`
- `moshi_oracle/moshi.cpp/src/moshi/models/lm_default.h`
- `moshi_oracle/moshi.cpp/tools/personaplex.cpp`
- `moshi_oracle/models_h100_actual/qat_heavy_int2_dir/personaplex-config.json`
- `moshi_oracle/HANDOFF.md`
- `moshi_oracle/models_h100_actual/qat_heavy_int2_dir/qat_heavy_int2.gguf`
  (read-only tensor-metadata census via Python `gguf.GGUFReader` — header
  and per-tensor name/type/shape only; no tensor data payload was read,
  no GPU involved, this is the file the run recipe in `HANDOFF.md` §7
  actually loads, resolved through the repo's own symlink chain
  `moshi_oracle/models` → `models_h100_actual` →
  `qat_heavy_int2_dir/qat_heavy_int2.gguf` →
  `/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2.gguf`)

**Hard constraints respected:** `v5_step1500_split.safetensors` and
`qat_best.pt` were never opened. No file under `moshi/` (the Python
reference package) was read or touched. No `git commit`/`git add`/`git push`
was run. No build or GPU run was performed — this whole task was static
source/config/metadata analysis.
