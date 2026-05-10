#!/usr/bin/env python3
"""dump_pt_tensors.py — PyTorch side of the Layer-0 numerical harness.

Dump-point semantics (must stay aligned with `bmo_compute.cpp` + `compare_divergence.py`):

  - **embed_sum**: Raw sum of temporal embedding rows for the K codebooks (same recipe as
    `LMModel.embed_codes`), shape [B, T, D] flattened. Matches `bmo_embed_input_tokens`:
    text_emb[token0] + sum_k audio_emb[k][token_{k+1}]. No codec delays — delays only shift
    tokens over time in streaming inference; at a single explicit step we feed the harness ids
    directly.

  - **layer0_post_norm1**: Output of transformer layer 0 **norm1** (RMSNorm/LayerNorm as configured).

  - **layer0_q_pre_rope / layer0_k_pre_rope**: Q and K after fused `in_proj` / rearrange to
    `[B, H, T, head_dim]`, **before** `RotaryEmbedding.apply_rope`.

  - **layer0_q_post_rope / layer0_k_post_rope**: After RoPE (same layout as pre-rope).

  - **layer0_v**: V projection output (no RoPE), `[B, H, T, head_dim]`.

  - **layer0_post_attn**: Output of `self_attn.out_proj` **before** the attention residual add.

  - **layer0_post_norm2**: Output of **norm2** applied to `(x + attn_out)` — i.e. the normalized
    sublayer input to the FFN block.

  - **layer0_post_ffn**: Output of the FFN **last** linear (`ActivationGating.linear_out` / down
    projection equivalent) **before** the FFN residual add.

  - **layer0_residual_out**: Hidden state at the exit of layer 0 (after FFN residual).

  Deep harness (same script; additional `pt_*.bin`):

  - **layer{L}_x_in** (L ∈ {0,1,8,15,23,31}): Residual stream entering layer L (`forward_pre_hook` on
    `transformer.layers[L]`); layer 0 matches **embed_sum** numerically.

  - **layer{L}_post_attn**: Attention **out_proj** output before the attention residual add.

  - **layer{L}_post_attn_residual**: `x_in + post_attn` (diagnostic sum; assumes attention path matches
    C++ `residual + attn_out` aside from optional LayerScale).

  - **layer{L}_post_ffn**: FFN last linear output (**gating** module output) before the FFN residual add.

  - **layer{L}_residual_out**: Full layer output after FFN residual (`transformer.layers[L]` forward output).

  - **post_out_norm**: Output of `LMModel.out_norm` applied to the temporal backbone output (matches
    `forward_embeddings` before `text_linear`).

  - **final_logits**: `text_linear(post_out_norm)` for the text head, flattened `[text_vocab]` (32000
    or 32001 depending on checkpoint `extra_text` / padding slot).

Tensor layout for flat binaries: `float32` little-endian, C-contiguous flatten of the active tensor
(in context B=1, T=1). Q/K/V use head-major flatten `[H, D]` per token after squeezing batch/time.

Deterministic harness ids are shared via `harness_input.json` (see `--emit-input-only`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# moshi package resolution (same pattern as bmo_inference.py)
_HERE = Path(__file__).resolve().parent
_MOSHI_PKG = _HERE / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

from moshi.models.loaders import get_moshi_lm  # noqa: E402
from moshi.models.lm import SILENCE_TOKENS, TemporalProjectedTransformer  # noqa: E402
from moshi.modules.transformer import KVCacheResult  # noqa: E402


HARNESS_JSON = "harness_input.json"

DUMP_NAMES = [
    "embed_sum",
    "layer0_post_norm1",
    "layer0_q_pre_rope",
    "layer0_q_post_rope",
    "layer0_k_pre_rope",
    "layer0_k_post_rope",
    "layer0_v",
    "layer0_post_attn",
    "layer0_post_norm2",
    "layer0_post_ffn",
    "layer0_residual_out",
]

DUMP_NAMES_DEEP_TAIL = [
    "post_out_norm",
    "final_logits",
]

DEEP_LAYERS = [0, 1, 8, 15, 23, 31]
SUB_STAGES = ["x_in", "post_attn", "post_attn_residual", "post_ffn", "residual_out"]


def deep_layer_stage_names() -> list[str]:
    return [f"layer{L}_{s}" for L in DEEP_LAYERS for s in SUB_STAGES]


TEXT_PAD_ID = 3


def default_harness_ids(num_codebooks: int) -> list[int]:
    if num_codebooks != 17:
        raise ValueError(f"This harness expects K=17 codebooks, got {num_codebooks}")
    sil = [int(SILENCE_TOKENS[i]) for i in range(8)]
    return [TEXT_PAD_ID] + sil + sil


def load_or_create_harness(path: Path, num_codebooks: int) -> list[int]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload.get("input_ids")
        if not isinstance(ids, list) or len(ids) != num_codebooks:
            raise ValueError(
                f"{path}: expected input_ids list of length {num_codebooks}, got {ids!r}"
            )
        return [int(x) for x in ids]
    ids = default_harness_ids(num_codebooks)
    payload = {
        "input_ids": ids,
        "num_codebooks": num_codebooks,
        "notes": "TEXT_PAD_ID=3 + SILENCE_TOKENS x2 (moshi lm.py); shared with C++ run.",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote default harness to {path}")
    return ids


def save_tensor_pt(t: torch.Tensor, name: str) -> None:
    arr = t.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    out = Path(f"pt_{name}.bin")
    arr.tofile(out)


def _flatten_qkv(t: torch.Tensor) -> torch.Tensor:
    """[B, H, T, D] with B=T=1 -> contiguous [H*D] matching head-major layout."""
    if t.dim() != 4:
        raise ValueError(f"expected Q/K/V [B,H,T,D], got {tuple(t.shape)}")
    return t.squeeze(0).squeeze(2).contiguous().reshape(-1)


def streaming_mha_forward_capture(
    attn: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bucket: dict,
) -> torch.Tensor:
    """Single-shot clone of `StreamingMultiheadAttention.forward` with tensor captures."""
    state = attn._streaming_state
    T = query.shape[1]
    if state is None:
        offset = torch.zeros(1, device=query.device, dtype=torch.long)
        offset_cpu = 0
    else:
        assert attn.causal, "Streaming only available for causal"
        offset = state.offset
        offset_cpu = state.offset_cpu

    if attn.weights_per_step:
        from moshi.modules.transformer import multi_linear

        projected = multi_linear(attn.weights_per_step, attn.in_proj_weight, query, offset_cpu)
    else:
        projected = F.linear(query, attn.in_proj_weight)

    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=attn.num_heads)
    bucket["layer0_q_pre_rope"] = q.detach().clone()
    bucket["layer0_k_pre_rope"] = k.detach().clone()
    bucket["layer0_v"] = v.detach().clone()

    if attn.rope:
        q, k = attn.rope(q, k, offset, time_before_heads=False)
    bucket["layer0_q_post_rope"] = q.detach().clone()
    bucket["layer0_k_post_rope"] = k.detach().clone()

    prior_cache_len = 0
    if state is not None and attn.compact_kv_cache:
        prior_cache_len = int(state.kv_cache.end_offset.item())

    kv_res = attn._complete_kv(k, v)

    if state is not None and attn.compact_kv_cache and prior_cache_len == 0:
        k, v, pos_k = KVCacheResult.from_kv(k, v)
    else:
        k, v, pos_k = kv_res
        if state is not None and attn.compact_kv_cache:
            valid = pos_k >= 0
            if bool(valid.any()):
                if not bool(valid.all()):
                    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    k = k.index_select(2, valid_idx)
                    v = v.index_select(2, valid_idx)
                    pos_k = pos_k.index_select(0, valid_idx)
            else:
                k = k[:, :, :1, :]
                v = v[:, :, :1, :]
                pos_k = torch.zeros((1,), device=q.device, dtype=torch.long)

    if attn.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if attn.context is not None:
            attn_bias = attn_bias & (delta < attn.context)
    else:
        attn_bias = None

    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    x = rearrange(x, "b h t d -> b t (h d)")
    if attn.weights_per_step:
        from moshi.modules.transformer import multi_linear

        x = multi_linear(attn.weights_per_step, attn.out_proj.weight, x, offset_cpu)
    else:
        x = attn.out_proj(x)

    if state is not None:
        state.offset.add_(T)
        state.offset_cpu += T
    return x


def install_deep_hooks(model: torch.nn.Module, layers: torch.nn.Module, cap: dict) -> list:
    hooks = []

    def grab(name: str):
        def fn(_m, _inp, out):
            cap[name] = out.detach().clone()

        return fn

    n_layers = len(layers)
    if n_layers < 32:
        raise RuntimeError(
            f"Deep dump expects at least 32 temporal layers (indices 0..31); got n_layers={n_layers}."
        )

    for L in DEEP_LAYERS:
        layer_mod = layers[L]

        def capture_x_in(li: int):
            def fn(_m, inp):
                cap[f"layer{li}_x_in"] = inp[0].detach().clone()

            return fn

        hooks.append(layer_mod.register_forward_pre_hook(capture_x_in(L)))
        hooks.append(layer_mod.self_attn.out_proj.register_forward_hook(grab(f"layer{L}_post_attn")))
        if layer_mod.gating is None:
            raise RuntimeError(f"Expected SwiGLU/SiLU gating on layer {L}; linear-only FFN not wired.")
        if isinstance(layer_mod.gating, torch.nn.ModuleList):
            for gate in layer_mod.gating:
                hooks.append(gate.register_forward_hook(grab(f"layer{L}_post_ffn")))
        else:
            hooks.append(layer_mod.gating.register_forward_hook(grab(f"layer{L}_post_ffn")))
        hooks.append(layer_mod.register_forward_hook(grab(f"layer{L}_residual_out")))

    # Matches LMModel.forward_embeddings: transformer backbone -> out_norm -> text_linear.
    hooks.append(model.out_norm.register_forward_hook(grab("post_out_norm")))
    hooks.append(model.text_linear.register_forward_hook(grab("final_logits")))
    return hooks


def install_layer0_hooks(layer0: torch.nn.Module, cap: dict) -> list:
    hooks = []

    def grab(name: str):
        def fn(_m, _inp, out):
            cap[name] = out.detach().clone()

        return fn

    hooks.append(layer0.norm1.register_forward_hook(grab("layer0_post_norm1")))
    hooks.append(layer0.self_attn.out_proj.register_forward_hook(grab("layer0_post_attn")))
    hooks.append(layer0.norm2.register_forward_hook(grab("layer0_post_norm2")))
    if layer0.gating is not None:
        hooks.append(layer0.gating.register_forward_hook(grab("layer0_post_ffn")))
    else:
        raise RuntimeError("Expected SwiGLU/SiLU gating on layer 0; linear-only FFN not wired.")

    def layer_out_hook(_m, _inp, out):
        cap["layer0_residual_out"] = out.detach().clone()

    hooks.append(layer0.register_forward_hook(layer_out_hook))
    return hooks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump Layer-0 + deep PT tensors for numerical harness.")
    p.add_argument("--ckpt", type=str, required=True, help="LM checkpoint (.pt / .pth).")
    p.add_argument("--mimi", type=str, default="", help="Unused here; accepted for CLI symmetry.")
    p.add_argument("--tokenizer", type=str, default="", help="Unused here; accepted for CLI symmetry.")
    p.add_argument(
        "--harness-json",
        type=str,
        default=HARNESS_JSON,
        help=f"Path to shared harness JSON (default {HARNESS_JSON}).",
    )
    p.add_argument(
        "--emit-input-only",
        action="store_true",
        help=f"Write {HARNESS_JSON} with default ids and exit (no model load).",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hj = Path(args.harness_json)

    if args.emit_input_only:
        ids = default_harness_ids(17)
        payload = {"input_ids": ids, "num_codebooks": 17}
        with hj.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"Wrote {hj}")
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA unavailable; falling back to cpu.", file=sys.stderr)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading LM from {args.ckpt} ...")
    model = get_moshi_lm(args.ckpt, device=device, dtype=dtype, copy_missing_weights=True)
    model.eval()
    num_cb = model.num_codebooks
    ids = load_or_create_harness(hj, num_cb)

    seq = torch.tensor(ids, dtype=torch.long, device=device).view(1, num_cb, 1)

    layers = model.transformer.layers
    layer0 = layers[0]

    # Warn if temporal projection changes width vs embedding (C++ path has no separate input_proj).
    tr = model.transformer
    if isinstance(tr, TemporalProjectedTransformer):
        print(
            "NOTE: TemporalProjectedTransformer — C++ embed_sum matches raw embed_codes; "
            "layer0 + layer15/layer31 residuals from transformer.layers are INNER width until output_proj. "
            "post_out_norm / final_logits are outer (LM) width. Align GGUF arch before comparing.",
            file=sys.stderr,
        )

    embed = model.embed_codes(seq)
    save_tensor_pt(embed, "embed_sum")

    cap: dict = {}
    attn = layer0.self_attn
    orig_fwd = attn.forward

    def fwd_hook(q, k, v):
        return streaming_mha_forward_capture(attn, q, k, v, cap)

    attn.forward = fwd_hook
    hooks = install_layer0_hooks(layer0, cap)
    hooks.extend(install_deep_hooks(model, layers, cap))

    try:
        with torch.no_grad():
            _transformer_out, _text_logits = model.forward_codes(seq)
    finally:
        attn.forward = orig_fwd
        for h in hooks:
            h.remove()

    # Persist captures (clone stored inside streaming_mha_forward_capture for q/k/v)
    save_tensor_pt(cap["layer0_post_norm1"], "layer0_post_norm1")
    save_tensor_pt(_flatten_qkv(cap["layer0_q_pre_rope"]), "layer0_q_pre_rope")
    save_tensor_pt(_flatten_qkv(cap["layer0_q_post_rope"]), "layer0_q_post_rope")
    save_tensor_pt(_flatten_qkv(cap["layer0_k_pre_rope"]), "layer0_k_pre_rope")
    save_tensor_pt(_flatten_qkv(cap["layer0_k_post_rope"]), "layer0_k_post_rope")
    save_tensor_pt(_flatten_qkv(cap["layer0_v"]), "layer0_v")

    save_tensor_pt(cap["layer0_post_attn"], "layer0_post_attn")
    save_tensor_pt(cap["layer0_post_norm2"], "layer0_post_norm2")
    save_tensor_pt(cap["layer0_post_ffn"], "layer0_post_ffn")
    save_tensor_pt(cap["layer0_residual_out"], "layer0_residual_out")

    for L in DEEP_LAYERS:
        save_tensor_pt(cap[f"layer{L}_x_in"], f"layer{L}_x_in")
        save_tensor_pt(cap[f"layer{L}_post_attn"], f"layer{L}_post_attn")
        pa_res = cap[f"layer{L}_x_in"] + cap[f"layer{L}_post_attn"]
        save_tensor_pt(pa_res, f"layer{L}_post_attn_residual")
        save_tensor_pt(cap[f"layer{L}_post_ffn"], f"layer{L}_post_ffn")
        save_tensor_pt(cap[f"layer{L}_residual_out"], f"layer{L}_residual_out")

    for name in DUMP_NAMES_DEEP_TAIL:
        save_tensor_pt(cap[name], name)

    all_names = (
        DUMP_NAMES
        + deep_layer_stage_names()
        + DUMP_NAMES_DEEP_TAIL
    )
    print("Saved:", ", ".join(f"pt_{n}.bin" for n in all_names))
    print("Sanity (shape / first 5 values):")
    for name in all_names:
        path = Path(f"pt_{name}.bin")
        a = np.fromfile(path, dtype=np.float32)
        print(f"  {name}: numel={a.size} head={a[:5].tolist()}")


if __name__ == "__main__":
    main()
