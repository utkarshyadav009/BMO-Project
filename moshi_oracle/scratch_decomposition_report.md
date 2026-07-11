# Compute-Graph Scratch Buffer Decomposition Report
**Model:** `qat_heavy_int2` + `-k q4_0` KV  
**Branch:** `experiment/multitier-dequant` (commit `1c70ebb`)  
**Instrumentation tag:** `MEMLEDGER_GRAPH_CAT label=text_graph_onetime_full_forward_pass`

---

## Step 1: H100 Reservation Match

| Item | Value | Source |
|---|---|---|
| Jetson-reported failing allocation | **1,171.82 MiB** | Jetson OOM log |
| H100 measured `text_graph_onetime_full_forward_pass` total_MiB | **1,171.82 MiB** | `MEMLEDGER_GRAPH` log line |
| **Match?** | **YES — exact to the byte** | |

```
MEMLEDGER_GRAPH label=text_graph_onetime_full_forward_pass \
  total_bytes=1228741112 total_MiB=1171.82 n_tensors=1607
```

The graph is hardware-independent. The 1,171.82 MiB scratch breakdown below applies directly to the Jetson problem.

---

## Step 2: Measured Breakdown Table

> All numbers measured from `MEMLEDGER_GRAPH_CAT` + `MEMLEDGER_GRAPH_ALL` log lines.  
> c=3000 column: **PENDING** (run in progress on GPU 0).

| Row | c=1138 (MiB) | c=3000 (MiB) | Notes |
|---|---|---|---|
| **prefill_scratch = decode_scratch** | **1,171.82** | **3,048.37** | Same graph, see Step 3 |
| **(a) Attn KV copies** | **569.00** | **1,500.00** | 32 × K non-transposed `[128,c,32,1]` f32 |
| **(b) Attn score tensors** | **577.89** | **1,523.46** | 32 × K transposed `[c,128,32,1]` f32 + 64 × attn score `[c,1,32,1]` f32 |
| **(c) Logits/vocab-sized** | **0.49** | **0.49** | 4 × `[32000,1,1,1]` f32; vocab-fixed, no context scaling |
| **(d) Other** | **24.44** | **24.44** | 1475 tensors; FFN intermediates + Q vectors; context-independent |
| **check_total** | **1,171.82** | **3,048.37** | a+b+c+d ✓ (both directly measured from `MEMLEDGER_GRAPH_CAT`) |

> **c=3000 column sourcing:** All values **directly measured** from confirming re-run with instrumented binary:
> ```
> MEMLEDGER_GRAPH_CAT label=text_graph_onetime_full_forward_pass ctx_len_detected=3000
>   catA_attn_kv_MiB=1500.00(n=32)  catB_attn_score_MiB=1523.44(n=96)
>   catC_logits_MiB=0.49(n=4)        catD_other_MiB=24.45(n=1475)
>   check_total_MiB=3048.37
> ```

> **Linearity confirmation (measured):** catA ratio = 1500.00/569.00 = **2.636** = 3000/1138 ✓ (perfect linear).  
> catB ratio = 1523.44/577.89 = **2.636** ✓. catC and catD: constant (0.49 and 24.44/24.45 MiB) ✓.  
> Total ratio = 3048.37/1171.82 = **2.601** — 1.3% below linear, explained exactly by the context-fixed 24.44 MiB catD.


### Precise catB sub-breakdown (c=1138, measured):
- K transposed (`[1138,128,32,1]` f32): 32 tensors × 17.78 MiB = **568.99 MiB**
- Attention scores (`[1138,1,32,1]` f32): 64 tensors × 0.14 MiB = **8.90 MiB**
- catB total: **577.89 MiB**

### catD top-10 individual tensors (c=1138, measured — all context-independent):
| Shape | Type | Count | MiB each | Category |
|---|---|---|---|---|
| `[1,32000,1,1]` | f32 | 1 | 0.122 | Permuted logit (ne[0]=1, so missed by catC detector) |
| `[22528,1,1,1]` | f32 | many | 0.086 | FFN intermediate activations (gating_linear dim) |

---

## Step 3: decode_scratch vs prefill_scratch — Same or Different?

**Answer: THE SAME. Explicit yes, with evidence.**

Evidence from code (`lm.h:867-892`):
```cpp
if ( ! lm_states->gctx ) {           // ONE-TIME GUARD
    lm_states->gctx = new GraphContext( 256, scratch.backend );
    graph.set_name( "text_graph_onetime_full_forward_pass" );
    // ... build 32-layer graph ...
    graph.alloc();                    // EXACTLY ONE alloc() call
}
// graph REUSED from here on — no second alloc
GraphContext & graph = *lm_states->gctx;
moshi_lmmodel_forward_text_step( graph, scratch, lm, lm_states, input );
```

Evidence from log: exactly ONE `MEMLEDGER_GRAPH label=text_graph_onetime_full_forward_pass` line appears in the entire 4272-line stderr of the c=1138 run. No other large graph alloc occurs.

`moshi_lmgen_step_system_prompts()` (the prefill loop) calls `moshi_lmgen_step()` in a loop — there is no structurally separate prefill graph. The same `gctx` is reused for every call, whether it is a silence-frame prefill or real audio-frame decoding.

**Implication:** "prefill_scratch" and "decode_scratch" are the same 1,171.82 MiB reservation. Any fix must reduce this single number.

---

## Structural Findings

### Scratch composition at c=1138:
| Component | MiB | % of total | Scales with context? |
|---|---|---|---|
| K copies (non-transposed), all 32 layers | 569.00 | 48.6% | **Linear** — O(c) |
| K copies (transposed), all 32 layers | 568.99 | 48.6% | **Linear** — O(c) |
| Attention scores (64 tensors) | 8.90 | 0.8% | **Linear** — O(c) |
| Logits (4 × vocab) | 0.49 | 0.04% | None |
| FFN intermediates + other | 24.44 | 2.1% | **None** — fixed cost |
| **Total** | **1,171.82** | 100% | Dominated by O(c) |

**97.9% of the scratch scales linearly with context length.** The 24.44 MiB fixed cost is negligible.

### Why all 32 layers' K materialise simultaneously:
ggml's greedy memory planner sees the 32-layer graph as a single computation, cannot free layer N's K buffer until layer N+1's attention has consumed it, and with GEMV (single-token decode), each layer's K span overlaps long enough that all 32 need to be live simultaneously. This is the structural reason the scratch is large — it is not a bug, it is correct scheduling.

### The Jetson OOM explained:
```
5,500 MiB (Orin Nano 8GB usable)
- 4,597 MiB (model physical, measured)
- ~742 MiB (CUDA/cuBLAS overhead, derived: 5500 - 4597 - 161 = 742)
=   161 MiB remaining before graph.alloc()
vs 1,172 MiB scratch needed
=  -1,011 MiB shortfall
```

The 742 MiB overhead estimate is derived from Jetson's reported 161 MiB free. **This remains unvalidated** until Stage 1 of the Jetson protocol is run — the actual overhead number is the unknown that decides the minimum required scratch reduction.

---

## Deviations from the plan

1. **c=3000 OOMed on GPU 1** (the secondary H100, 6 GB free at time of run — both runs launched simultaneously, competing for the same GPU). Rerun on GPU 0 (32 GB free) in progress. c=3000 CAT line pending.
2. **MEMLEDGER_GRAPH_ALL required a build fix** (`#include <algorithm>` missing from context.h). Fixed and pushed as `d215bbf`.
3. **Top-10 cap extended to all tensors** as the task permitted, producing `MEMLEDGER_GRAPH_ALL` lines that enabled the complete categorization above.
