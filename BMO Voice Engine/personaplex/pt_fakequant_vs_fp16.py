#!/usr/bin/env python3
"""pt_fakequant_vs_fp16.py — Compare vanilla FP16 forward vs QAT MultiTierFakeQuantize forward.

Loads the same LM checkpoint twice: (A) dense FP16 matmuls, (B) identical weights with
`MultiTierFakeQuantize` parametrization on the same SEPTQ targets as QAT/export (in_proj_weight,
gating linear_in/out; out_proj excluded by default). Same harness ids as the C++/PT harness.

Requires SEPTQ tier masks + per-layer quant metadata. Dense QAT exports (`qat_septq_dense`)
typically omit masks; this script follows ``qat_meta.source_student_quant_meta`` (same rule as
``verify_septq_checkpoint.py --follow-source``), or use ``--septq-meta`` to point at the
multi-tier .pt that carries ``tier_masks_uint2`` and ``septq_meta``.

Assumptions (run on H100 host with checkpoint + moshi tree):
  - ``get_moshi_lm(qat_best.pt)`` loads with ``torch.float16`` on CUDA as in this harness.
  - ``qat_meta.train_layers`` / ``qat_meta.skip_modules_filters`` select fake-quant modules; if
    absent, defaults match ``qat_septq.py`` (layers 0–30, skip ``self_attn.out_proj``).

Outputs: stdout table (same column layout as ``compare_divergence.py``), plus
``pt_fp16_<stage>.bin`` and ``pt_fakequant_<stage>.bin`` (float32 row-major flatten).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

_HERE = Path(__file__).resolve().parent
_MOSHI_PKG = _HERE / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

from apply_septq import get_temporal_layers, parse_quantize_layers, parse_skip_module_filters  # noqa: E402
from compare_divergence import (  # noqa: E402
    COS_THRESHOLD,
    FINAL_LOGITS_TOPK,
    STAGES,
    TEXT_PAD_ID,
    logit_argmax,
    logit_rank_pad,
    logit_topk_indices,
    stats as divergence_stats,
)
from moshi.models.loaders import get_moshi_lm  # noqa: E402
from moshi.modules.transformer import KVCacheResult  # noqa: E402
from qat_septq import gather_qat_entries, register_fake_quant_for_entries  # noqa: E402


HARNESS_JSON_DEFAULT = "harness_input.json"
DEEP_LAYERS = [0, 1, 8, 15, 23, 31]
# Path B H3 Diagnostic: T2 taps on C++ align with these layer indices (bmo_h3_tap T2).
H3_T2_LAYERS = (0, 1, 4, 8, 16, 24, 31)
# T1–T6 hooks / stderr taps (C++ bmo_h3_tap non-T2) use this layer set.
H3_T1_T6_LAYERS = (0, 1, 4, 8)


def _resolve_path(base: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    cand = path.resolve()
    if cand.exists():
        return cand
    return (base / path).resolve()


def _tier_masks_from_ckpt(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw = ckpt.get("tier_masks_uint2")
    if not isinstance(raw, dict) or not raw:
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            raw = sm.get("tier_masks_uint2")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    return out


def _tier_masks_meta_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    raw = ckpt.get("tier_masks_meta")
    if isinstance(raw, dict):
        return raw
    sm = ckpt.get("septq_meta")
    if isinstance(sm, dict):
        raw = sm.get("tier_masks_meta")
    return raw if isinstance(raw, dict) else {}


def _septq_meta_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any] | None:
    sm = ckpt.get("septq_meta")
    return sm if isinstance(sm, dict) else None


def resolve_quant_sources(
    qat_ckpt: dict[str, Any],
    qat_path: Path,
    septq_meta_override: Path | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Return (quant_checkpoint_dict, tier_masks_uint2, tier_masks_meta) for register_fake_quant_for_entries."""
    if septq_meta_override is not None:
        src = torch.load(str(septq_meta_override), map_location="cpu")
        if not isinstance(src, dict):
            raise RuntimeError(f"--septq-meta must load to a dict, got {type(src)}")
        tier = _tier_masks_from_ckpt(src)
        meta = _tier_masks_meta_from_ckpt(src)
        if not tier:
            raise RuntimeError(f"{septq_meta_override}: missing tier_masks_uint2")
        sm = _septq_meta_from_ckpt(src)
        if not sm or not isinstance(sm.get("per_layer_stats"), list):
            raise RuntimeError(f"{septq_meta_override}: missing septq_meta.per_layer_stats")
        return src, tier, meta

    tier = _tier_masks_from_ckpt(qat_ckpt)
    tmeta = _tier_masks_meta_from_ckpt(qat_ckpt)
    sm = _septq_meta_from_ckpt(qat_ckpt)
    if tier and sm and isinstance(sm.get("per_layer_stats"), list):
        return qat_ckpt, tier, tmeta

    qm = qat_ckpt.get("qat_meta")
    rel = None
    if isinstance(qm, dict):
        rel = qm.get("source_student_quant_meta")
    if not isinstance(rel, str) or not rel.strip():
        raise RuntimeError(
            "Checkpoint has no tier_masks_uint2 / septq_meta and no qat_meta.source_student_quant_meta. "
            "Pass --septq-meta pointing at the SEPTQ multitier .pt (e.g. bmo_temporal_half_cushion_max.pt)."
        )
    src_path = _resolve_path(qat_path.parent, rel)
    if not src_path.is_file():
        raise RuntimeError(f"source_student_quant_meta not found: {src_path}")
    src = torch.load(str(src_path), map_location="cpu")
    if not isinstance(src, dict):
        raise RuntimeError(f"Source quant file must be a dict, got {type(src)}")
    tier = _tier_masks_from_ckpt(src)
    tmeta = _tier_masks_meta_from_ckpt(src)
    sm = _septq_meta_from_ckpt(src)
    if not tier:
        raise RuntimeError(f"{src_path}: missing tier_masks_uint2")
    if not sm or not isinstance(sm.get("per_layer_stats"), list):
        raise RuntimeError(f"{src_path}: missing septq_meta.per_layer_stats")
    return src, tier, tmeta


def load_harness_ids(path: Path, num_codebooks: int) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Harness JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    ids = payload.get("input_ids")
    if not isinstance(ids, list) or len(ids) != num_codebooks:
        raise ValueError(f"{path}: expected input_ids list of length {num_codebooks}, got {ids!r}")
    return [int(x) for x in ids]


def save_tensor_bin(t: torch.Tensor, prefix: str, name: str) -> None:
    arr = t.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    out = Path(f"{prefix}_{name}.bin")
    arr.tofile(out)


def _flatten_qkv(t: torch.Tensor) -> torch.Tensor:
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
        raise RuntimeError(f"Expected >=32 temporal layers, got {n_layers}")

    for L in DEEP_LAYERS:
        layer_mod = layers[L]

        def capture_x_in(li: int):
            def fn(_m, inp):
                cap[f"layer{li}_x_in"] = inp[0].detach().clone()

            return fn

        hooks.append(layer_mod.register_forward_pre_hook(capture_x_in(L)))
        hooks.append(layer_mod.self_attn.out_proj.register_forward_hook(grab(f"layer{L}_post_attn")))
        if layer_mod.gating is None:
            raise RuntimeError(f"Expected gating on layer {L}")
        if isinstance(layer_mod.gating, torch.nn.ModuleList):
            for gate in layer_mod.gating:
                hooks.append(gate.register_forward_hook(grab(f"layer{L}_post_ffn")))
        else:
            hooks.append(layer_mod.gating.register_forward_hook(grab(f"layer{L}_post_ffn")))
        hooks.append(layer_mod.register_forward_hook(grab(f"layer{L}_residual_out")))

    hooks.append(model.out_norm.register_forward_hook(grab("post_out_norm")))
    hooks.append(model.text_linear.register_forward_hook(grab("final_logits")))
    return hooks


def install_all_layer_residual_hooks_skip_layer0(
    model: torch.nn.Module, layers: torch.nn.Module, cap: dict
) -> list:
    """Capture ``layer{L}_residual_out`` for L=1..N-1 (layer 0 from ``install_layer0_hooks``)."""
    hooks: list = []

    def grab(name: str):
        def fn(_m, _inp, out):
            cap[name] = out.detach().clone()

        return fn

    n_layers = len(layers)
    for L in range(1, n_layers):
        hooks.append(layers[L].register_forward_hook(grab(f"layer{L}_residual_out")))
    hooks.append(model.out_norm.register_forward_hook(grab("post_out_norm")))
    hooks.append(model.text_linear.register_forward_hook(grab("final_logits")))
    return hooks


def run_forward_with_captures_all_residuals(model: torch.nn.Module, seq: torch.Tensor) -> dict[str, torch.Tensor]:
    """Like ``run_forward_with_captures`` but records ``layer{L}_residual_out`` for every layer L (0..N-1)."""
    layers = model.transformer.layers
    layer0 = layers[0]
    cap: dict[str, torch.Tensor] = {}
    attn = layer0.self_attn
    orig_fwd = attn.forward

    def fwd_hook(q, k, v):
        return streaming_mha_forward_capture(attn, q, k, v, cap)

    attn.forward = fwd_hook

    def capture_x0(_m, inp):
        cap["layer0_x_in"] = inp[0].detach().clone()

    hooks = [layer0.register_forward_pre_hook(capture_x0)]
    hooks.extend(install_layer0_hooks(layer0, cap))
    hooks.extend(install_all_layer_residual_hooks_skip_layer0(model, layers, cap))
    try:
        with torch.no_grad():
            _transformer_out, _text_logits = model.forward_codes(seq)
    finally:
        attn.forward = orig_fwd
        for h in hooks:
            h.remove()

    for L in range(len(layers)):
        if f"layer{L}_x_in" in cap and f"layer{L}_post_attn" in cap:
            cap[f"layer{L}_post_attn_residual"] = cap[f"layer{L}_x_in"] + cap[f"layer{L}_post_attn"]
    return cap


def _h3_pt_line(family: str, tag: str, layer: int, t: torch.Tensor) -> str:
    """family is ``PT_h3`` or ``PT_FQ_h3`` → ``[PT_h3_tap_T1]`` / ``[PT_FQ_h3_tap_T1]``."""
    t32 = t.detach().float().cpu().reshape(-1)
    ne = int(t32.numel())
    k = min(8, ne)
    l2 = float(torch.linalg.norm(t32).item())
    parts = ",".join(f"{float(t32[i]):.6f}" for i in range(k))
    return f"[{family}_tap_{tag}] L={layer} n={ne} l2={l2:.6f} first8={parts}"


def _h3_pt_line_head(family: str, tag: str, label: str, t: torch.Tensor) -> str:
    """Head-path taps (T7 post_out_norm, T8 final_logits)."""
    t32 = t.detach().float().cpu().reshape(-1)
    ne = int(t32.numel())
    k = min(8, ne)
    l2 = float(torch.linalg.norm(t32).item())
    parts = ",".join(f"{float(t32[i]):.6f}" for i in range(k))
    return f"[{family}_tap_{tag}] {label} n={ne} l2={l2:.6f} first8={parts}"


def format_pt_fq_h3_tap_t2_line(L: int, t: torch.Tensor) -> str:
    """One stderr line for PT fakequant T2 (post-layer residual); pair with C++ ``[h3_tap_T2]``."""
    t32 = t.detach().float().cpu().reshape(-1)
    ne = int(t32.numel())
    k = min(8, ne)
    l2 = float(torch.linalg.norm(t32).item())
    parts = ",".join(f"{float(t32[i]):.6f}" for i in range(k))
    return f"[PT_FQ_h3_tap_T2] L={L} n={ne} l2={l2:.6f} first8={parts}"


def write_milestone_residual_bin(path: Path, t: torch.Tensor, *, expect_n: int) -> None:
    """Write float32 row-major ``expect_n`` floats (full n_embd residual)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = t.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
    if int(arr.size) != int(expect_n):
        raise ValueError(f"{path.name}: expected {expect_n} elements, got {arr.size}")
    arr.tofile(path)


def run_forward_h3_pt_captures(model: torch.nn.Module, seq: torch.Tensor) -> dict[str, torch.Tensor]:
    """Second forward with hooks for Path B Diagnostic 2 (aligns with C++ ``BMO_H3_TAPS``)."""
    layers = model.transformer.layers
    cap: dict[str, torch.Tensor] = {}
    hooks: list = []

    def _first_inp(inp: torch.Tensor | tuple) -> torch.Tensor:
        return inp if isinstance(inp, torch.Tensor) else inp[0]

    def grab_pre_x(L: int):
        def fn(_m, inp, L=L):
            cap[f"layer{L}_x_in"] = _first_inp(inp).detach().clone()

        return fn

    def grab_norm1(L: int):
        def fn(_m, _inp, out, L=L):
            cap[f"layer{L}_post_norm1"] = out.detach().clone()

        return fn

    def grab_pre_outproj(L: int):
        def fn(_m, inp, L=L):
            cap[f"layer{L}_attn_merge_pre_outproj"] = _first_inp(inp).detach().clone()

        return fn

    def grab_post_attn(L: int):
        def fn(_m, _inp, out, L=L):
            cap[f"layer{L}_post_attn"] = out.detach().clone()

        return fn

    def grab_layer_out(L: int):
        def fn(_m, _inp, out, L=L):
            cap[f"layer{L}_residual_out"] = out.detach().clone()

        return fn

    def register_gating_hooks(L: int, gating_mod: torch.nn.Module) -> None:
        if isinstance(gating_mod, torch.nn.ModuleList):
            for gi, sub in enumerate(gating_mod):

                def make_hook(li: int, idx: int):
                    def fn(_m, _inp, out, li=li, idx=idx):
                        cap[f"layer{li}_post_ffn_g{idx}"] = out.detach().clone()

                    return fn

                hooks.append(sub.register_forward_hook(make_hook(L, gi)))
        else:

            def fn(_m, _inp, out, L=L):
                cap[f"layer{L}_post_ffn"] = out.detach().clone()

            hooks.append(gating_mod.register_forward_hook(fn))

    for L in H3_T1_T6_LAYERS:
        lay = layers[L]
        hooks.append(lay.register_forward_pre_hook(grab_pre_x(L)))
        hooks.append(lay.norm1.register_forward_hook(grab_norm1(L)))
        hooks.append(lay.self_attn.out_proj.register_forward_pre_hook(grab_pre_outproj(L)))
        hooks.append(lay.self_attn.out_proj.register_forward_hook(grab_post_attn(L)))
        if lay.gating is None:
            raise RuntimeError(f"layer {L}: expected gating module for h3 capture")
        register_gating_hooks(L, lay.gating)
        hooks.append(lay.register_forward_hook(grab_layer_out(L)))

    n_tl = len(layers)
    for L in H3_T2_LAYERS:
        if L in H3_T1_T6_LAYERS:
            continue
        if L >= n_tl:
            continue
        hooks.append(layers[L].register_forward_hook(grab_layer_out(L)))

    on = getattr(model, "out_norm", None)
    if on is not None:

        def grab_post_norm(_m, _inp, out):
            cap["h3_post_out_norm"] = out.detach().clone()

        hooks.append(on.register_forward_hook(grab_post_norm))

    tl = getattr(model, "text_linear", None)
    if tl is not None:

        def grab_text_logits(_m, _inp, out):
            cap["h3_final_logits"] = out.detach().clone()

        hooks.append(tl.register_forward_hook(grab_text_logits))

    try:
        with torch.no_grad():
            model.forward_codes(seq)
    finally:
        for h in hooks:
            h.remove()

    for L in H3_T1_T6_LAYERS:
        pn = cap.get(f"layer{L}_post_norm1")
        if pn is not None:
            w = layers[L].self_attn.in_proj_weight
            x2 = pn.reshape(-1, pn.shape[-1])
            cap[f"layer{L}_in_proj_replay"] = F.linear(x2, w)

    return cap


def format_h3_pt_report(cap: dict[str, torch.Tensor], *, family: str = "PT_h3") -> list[str]:
    """Lines matching C++ ``[h3_tap_T*]`` layout (first 8 + L2). ``family`` is ``PT_h3`` or ``PT_FQ_h3``."""
    title = "PT FP16 (dense)" if family == "PT_h3" else "PT fakequant"
    lines: list[str] = [
        "",
        f"=== {family} taps ({title}, second forward with hooks) ===",
        f"T1–T6: L in {H3_T1_T6_LAYERS}. T2-only rows: L in {tuple(x for x in H3_T2_LAYERS if x not in H3_T1_T6_LAYERS)}. T7/T8: head.",
        "NOTE: T1 uses x_in + post_attn (matches C++ when layer_scale_1 is Identity).",
    ]
    for L in H3_T1_T6_LAYERS:
        xi = cap.get(f"layer{L}_x_in")
        pa = cap.get(f"layer{L}_post_attn")
        if xi is not None and pa is not None:
            lines.append(_h3_pt_line(family, "T1", L, xi + pa))
        else:
            lines.append(f"[{family}_tap_T1] L={L} <MISSING>")
        for key, tag in (
            (f"layer{L}_residual_out", "T2"),
            (f"layer{L}_post_norm1", "T3"),
            (f"layer{L}_in_proj_replay", "T4"),
            (f"layer{L}_attn_merge_pre_outproj", "T5"),
        ):
            t = cap.get(key)
            if t is None:
                lines.append(f"[{family}_tap_{tag}] L={L} <MISSING>")
            else:
                lines.append(_h3_pt_line(family, tag, L, t))
        t6 = cap.get(f"layer{L}_post_ffn")
        if t6 is None:
            t6 = cap.get(f"layer{L}_post_ffn_g0")
        if t6 is None:
            lines.append(f"[{family}_tap_T6] L={L} <MISSING>")
        else:
            lines.append(_h3_pt_line(family, "T6", L, t6))

    for L in H3_T2_LAYERS:
        if L in H3_T1_T6_LAYERS:
            continue
        key = f"layer{L}_residual_out"
        t = cap.get(key)
        if t is None:
            lines.append(f"[{family}_tap_T2] L={L} <MISSING>")
        else:
            lines.append(_h3_pt_line(family, "T2", L, t))

    t7 = cap.get("h3_post_out_norm")
    if t7 is None:
        lines.append(f"[{family}_tap_T7] post_out_norm <MISSING>")
    else:
        lines.append(_h3_pt_line_head(family, "T7", "post_out_norm", t7))

    t8 = cap.get("h3_final_logits")
    if t8 is None:
        lines.append(f"[{family}_tap_T8] final_logits <MISSING>")
    else:
        lines.append(_h3_pt_line_head(family, "T8", "final_logits", t8))
    return lines


def _pt_fq_vec_for_tap(cap: dict[str, torch.Tensor], L: int, tap: str) -> torch.Tensor | None:
    if tap == "T1":
        xi = cap.get(f"layer{L}_x_in")
        pa = cap.get(f"layer{L}_post_attn")
        if xi is None or pa is None:
            return None
        return xi + pa
    if tap == "T2":
        return cap.get(f"layer{L}_residual_out")
    if tap == "T3":
        return cap.get(f"layer{L}_post_norm1")
    if tap == "T4":
        return cap.get(f"layer{L}_in_proj_replay")
    if tap == "T5":
        return cap.get(f"layer{L}_attn_merge_pre_outproj")
    if tap == "T6":
        t6 = cap.get(f"layer{L}_post_ffn")
        return t6 if t6 is not None else cap.get(f"layer{L}_post_ffn_g0")
    return None


def _load_cpp_h3_bin(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.fromfile(str(path), dtype=np.float32).astype(np.float64, copy=False)


def append_h3_cpp_vs_pt_fq_per_op_table(
    lines: list[str],
    *,
    bin_dir: Path,
    cap_fq: dict[str, torch.Tensor],
    cos_threshold: float = 0.95,
) -> None:
    """Full-vector cos(cpp, pt_fq) per tap; per-op gap = curr_cos - prev_cos along T1→T3→T4→T5→T6→T2."""
    tap_order = ("T1", "T3", "T4", "T5", "T6", "T2")
    lines.append("")
    lines.append(
        "=== H3 per-op: full-tensor cos(C++ bin, PT_FQ tap), first8 cos, per-op gap (curr-prev full cos) ==="
    )
    lines.append(f"bin_dir: {bin_dir}")
    lines.append(
        "| L | tap | full cos | first8 cos | per-op gap | below_0p95 | "
        "PT_FQ l2 | C++ l2 |"
    )
    lines.append("|---|-----|----------|------------|------------|------------|----------|---------|")
    for L in H3_T1_T6_LAYERS:
        prev_full: float | None = None
        for tap in tap_order:
            pt_t = _pt_fq_vec_for_tap(cap_fq, L, tap)
            if tap == "T2":
                cpp_path = bin_dir / f"cpp_residual_L{L}.bin"
            else:
                cpp_path = bin_dir / f"cpp_h3_{tap}_L{L}.bin"
            row_pt_l2 = ""
            row_cpp_l2 = ""
            flag = ""
            if pt_t is None:
                lines.append(f"| {L} | {tap} | MISSING_PT | nan | nan | — | — | — |")
                continue
            pt_np = pt_t.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
            row_pt_l2 = f"{float(np.linalg.norm(pt_np)):.4f}"
            cpp_np = _load_cpp_h3_bin(cpp_path)
            if cpp_np is None or cpp_np.size != pt_np.size:
                lines.append(
                    f"| {L} | {tap} | MISSING_CPP | nan | nan | — | — | {row_pt_l2:>8} | — |"
                )
                prev_full = None
                continue
            row_cpp_l2 = f"{float(np.linalg.norm(cpp_np)):.4f}"
            dot = float(np.dot(cpp_np, pt_np))
            nc = float(np.linalg.norm(cpp_np))
            nf = float(np.linalg.norm(pt_np))
            full_cos = dot / (nc * nf + 1e-30)
            a8 = cpp_np[:8]
            b8 = pt_np[:8]
            na8 = float(np.linalg.norm(a8))
            nb8 = float(np.linalg.norm(b8))
            fcos = float(np.dot(a8, b8) / (na8 * nb8 + 1e-30)) if na8 > 1e-30 and nb8 > 1e-30 else float("nan")
            gap_s = ""
            if prev_full is not None:
                gap = full_cos - prev_full
                gap_s = f"{gap:+.6f}"
            prev_full = full_cos
            if full_cos < cos_threshold:
                flag = "LOW"
            gap_disp = gap_s if gap_s else "—"
            lines.append(
                f"| {L} | {tap} | {full_cos:.6f} | {fcos:.6f} | {gap_disp:>10} | {flag:10} | {row_pt_l2:8} | {row_cpp_l2:7} |"
            )


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
        raise RuntimeError("Expected gating on layer 0")

    def layer_out_hook(_m, _inp, out):
        cap["layer0_residual_out"] = out.detach().clone()

    hooks.append(layer0.register_forward_hook(layer_out_hook))
    return hooks


def run_forward_with_captures(model: torch.nn.Module, seq: torch.Tensor) -> dict[str, torch.Tensor]:
    layers = model.transformer.layers
    layer0 = layers[0]
    cap: dict[str, torch.Tensor] = {}
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

    # post_attn_residual derived (same as dump_pt_tensors)
    for L in DEEP_LAYERS:
        cap[f"layer{L}_post_attn_residual"] = cap[f"layer{L}_x_in"] + cap[f"layer{L}_post_attn"]
    return cap


def tensor_for_stage(name: str, cap: dict, embed: torch.Tensor) -> torch.Tensor:
    if name == "embed_sum":
        return embed
    if name in (
        "layer0_q_pre_rope",
        "layer0_q_post_rope",
        "layer0_k_pre_rope",
        "layer0_k_post_rope",
        "layer0_v",
    ):
        return _flatten_qkv(cap[name])
    return cap[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FP16 vs MultiTierFakeQuantize activation cosines (PT harness stages).")
    p.add_argument("--ckpt", type=str, required=True, help="qat_best.pt (dense QAT export) or compatible .pt")
    p.add_argument(
        "--harness-input",
        type=str,
        default=HARNESS_JSON_DEFAULT,
        dest="harness_input",
        help=f"JSON with input_ids (default {HARNESS_JSON_DEFAULT})",
    )
    p.add_argument(
        "--septq-meta",
        type=str,
        default="",
        help="Override path to SEPTQ multitier checkpoint (tier_masks + septq_meta). "
        "If omitted, uses embedded data or qat_meta.source_student_quant_meta.",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu")
    p.add_argument(
        "--all-layer-residuals",
        action="store_true",
        help="Capture layer{L}_residual_out for every temporal layer (Path B triage / ceiling analysis).",
    )
    return p.parse_args()


def qat_selection_from_ckpt(qat_ckpt: dict[str, Any], n_temporal: int) -> tuple[list[int], list[str]]:
    qm = qat_ckpt.get("qat_meta")
    selected: list[int]
    if isinstance(qm, dict) and qm.get("train_layers") is not None:
        tl_raw = qm["train_layers"]
        if isinstance(tl_raw, str) and str(tl_raw).strip():
            selected = parse_quantize_layers(str(tl_raw), n_temporal)
        elif isinstance(tl_raw, list) and tl_raw:
            selected = sorted({int(x) for x in tl_raw if 0 <= int(x) < n_temporal})
        else:
            selected = list(range(n_temporal))
    else:
        selected = list(range(n_temporal))

    if isinstance(qm, dict) and isinstance(qm.get("skip_modules_filters"), list) and qm["skip_modules_filters"]:
        skip = [str(x) for x in qm["skip_modules_filters"]]
    else:
        skip = parse_skip_module_filters("self_attn.out_proj")
    return selected, skip


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    harness_path = Path(args.harness_input).resolve()
    septq_override = Path(args.septq_meta).resolve() if str(args.septq_meta).strip() else None

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA unavailable; falling back to cpu.", file=sys.stderr)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if str(ckpt_path).lower().endswith(".safetensors"):
        qat_payload: dict[str, Any] = {}
        if septq_override is None:
            raise SystemExit(
                "When --ckpt is a .safetensors weights file, pass --septq-meta to the multitier .pt "
                "(tier_masks_uint2 + septq_meta.per_layer_stats) used for fakequant registration."
            )
    else:
        qat_payload = torch.load(str(ckpt_path), map_location="cpu")
        if not isinstance(qat_payload, dict):
            raise SystemExit(f"Expected checkpoint dict, got {type(qat_payload)}")

    quant_ckpt, tier_masks_uint2, tier_masks_meta = resolve_quant_sources(
        qat_payload, ckpt_path, septq_override
    )

    print(f"Loading FP16 model from {ckpt_path} ...")
    model_fp16 = get_moshi_lm(str(ckpt_path), device=device, dtype=dtype, copy_missing_weights=True)
    model_fp16.eval()

    print(f"Loading second copy for fake-quant from {ckpt_path} ...")
    model_fq = get_moshi_lm(str(ckpt_path), device=device, dtype=dtype, copy_missing_weights=True)
    model_fq.eval()

    temporal_layers = get_temporal_layers(model_fq)
    if not temporal_layers:
        raise SystemExit("Could not resolve temporal layers on model")
    n_temp = len(temporal_layers)
    selected_layers, skip_filters = qat_selection_from_ckpt(qat_payload, n_temp)

    entries, _excluded = gather_qat_entries(
        student=model_fq,
        selected_layers=selected_layers,
        skip_module_filters=skip_filters,
    )
    if not entries:
        raise SystemExit("gather_qat_entries returned no modules; check train_layers / skip_modules")

    register_fake_quant_for_entries(
        entries=entries,
        quant_checkpoint=quant_ckpt,
        tier_masks_uint2=tier_masks_uint2,
        tier_masks_meta=tier_masks_meta,
    )
    print(f"Registered MultiTierFakeQuantize on {len(entries)} weight target(s).")

    num_cb = model_fp16.num_codebooks
    ids = load_harness_ids(harness_path, num_cb)
    seq = torch.tensor(ids, dtype=torch.long, device=device).view(1, num_cb, 1)

    embed_fp16 = model_fp16.embed_codes(seq)
    embed_fq = model_fq.embed_codes(seq)

    if args.all_layer_residuals:
        cap_fp16 = run_forward_with_captures_all_residuals(model_fp16, seq)
        cap_fq = run_forward_with_captures_all_residuals(model_fq, seq)
    else:
        cap_fp16 = run_forward_with_captures(model_fp16, seq)
        cap_fq = run_forward_with_captures(model_fq, seq)

    if not torch.equal(embed_fp16, embed_fq):
        print(
            "WARN: embed_sum differs between runs (unexpected - embeddings are not fake-quantized).",
            file=sys.stderr,
        )

    max_abs_all = 0.0
    for name in STAGES:
        if name == "embed_sum":
            continue
        try:
            ta = tensor_for_stage(name, cap_fp16, embed_fp16).float()
            tb = tensor_for_stage(name, cap_fq, embed_fq).float()
        except KeyError:
            continue
        if ta.shape == tb.shape:
            max_abs_all = max(max_abs_all, float((ta - tb).abs().max().item()))
    print(f"INFO: max |A-B| over hooked stages (excl. embed_sum) = {max_abs_all:.6g}", file=sys.stderr)

    diverged = False
    print(
        f"{'stage':<26} {'n':>10} {'cosine':>10} {'max|diff|':>12} {'rel_L2':>10} {'mean|A|':>10} {'mean|B|':>10}  notes"
    )
    print("-" * 120)

    for name in STAGES:
        try:
            t_a = tensor_for_stage(name, cap_fp16, embed_fp16)
            t_b = tensor_for_stage(name, cap_fq, embed_fq)
        except KeyError as exc:
            print(f"{name:<26} {'-':>10} {'-':>10} {'-':>12} {'-':>10} {'-':>10} {'-':>10}  MISSING capture ({exc})")
            diverged = True
            continue

        a = t_a.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        b = t_b.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        save_tensor_bin(torch.from_numpy(a), "pt_fp16", name)
        save_tensor_bin(torch.from_numpy(b), "pt_fakequant", name)

        if a.size != b.size:
            note = f"<-- DIVERGENCE (count A={a.size} B={b.size})"
            print(f"{name:<26} {a.size:>10} {'-':>10} {'-':>12} {'-':>10} {'-':>10} {'-':>10}  {note}")
            diverged = True
            continue

        cos, max_abs, rel_l2, m_a, m_b, n = divergence_stats(a, b)
        flag = ""

        if name == "final_logits":
            arg_a = logit_argmax(a)
            arg_b = logit_argmax(b)
            r_a = logit_rank_pad(a, TEXT_PAD_ID)
            r_b = logit_rank_pad(b, TEXT_PAD_ID)
            t5_a = logit_topk_indices(a, FINAL_LOGITS_TOPK)
            t5_b = logit_topk_indices(b, FINAL_LOGITS_TOPK)
            note_extra = (
                f"PAD_rank A={r_a} B={r_b}  argmax A={arg_a} B={arg_b}  "
                f"top{FINAL_LOGITS_TOPK}_A={t5_a}  top{FINAL_LOGITS_TOPK}_B={t5_b}"
            )
            if cos < COS_THRESHOLD or np.isnan(cos) or arg_a != arg_b:
                flag = "<-- DIVERGENCE"
                diverged = True
            print(
                f"{name:<26} {int(n):>10} {cos:>10.6f} {max_abs:>12.6g} {rel_l2:>10.6g} "
                f"{m_a:>10.6g} {m_b:>10.6g}  {flag} {note_extra}"
            )
            continue

        if cos < COS_THRESHOLD or np.isnan(cos):
            flag = "<-- DIVERGENCE"
            diverged = True
        print(
            f"{name:<26} {int(n):>10} {cos:>10.6f} {max_abs:>12.6g} {rel_l2:>10.6g} "
            f"{m_a:>10.6g} {m_b:>10.6g}  {flag}"
        )

    print("-" * 120)
    if args.all_layer_residuals:
        print("\n=== Per-layer residual cosine: FP16 vs fakequant (single harness step) ===")
        tl = get_temporal_layers(model_fp16)
        n_layers = len(tl) if tl else 0
        for L in range(n_layers):
            k = f"layer{L}_residual_out"
            if k not in cap_fp16 or k not in cap_fq:
                print(f"layer{L:2d}_residual_out  MISSING")
                continue
            z_a = cap_fp16[k].detach().float().reshape(-1)
            z_b = cap_fq[k].detach().float().reshape(-1)
            c = float(torch.sum(z_a * z_b) / (torch.norm(z_a) * torch.norm(z_b) + 1e-30))
            print(f"layer{L:2d}_residual_out  cos_fp16_vs_fq={c:.8f}")
        if "final_logits" in cap_fp16 and "final_logits" in cap_fq:
            fl_a = cap_fp16["final_logits"].detach().float().reshape(-1)
            fl_b = cap_fq["final_logits"].detach().float().reshape(-1)
            cfl = float(torch.sum(fl_a * fl_b) / (torch.norm(fl_a) * torch.norm(fl_b) + 1e-30))
            print(f"final_logits            cos_fp16_vs_fq={cfl:.8f}")
        print("=== end per-layer residual block ===\n")

    for diag_name in ("layer31_residual_out", "post_out_norm"):
        if diag_name in cap_fp16 and diag_name in cap_fq:
            z_a = cap_fp16[diag_name].detach().float().reshape(-1)
            z_b = cap_fq[diag_name].detach().float().reshape(-1)
            c = float(torch.sum(z_a * z_b) / (torch.norm(z_a) * torch.norm(z_b) + 1e-30))
            print(f"Diagnostic: cosine({diag_name}) FP16 vs fakequant = {c:.6f}")

    n_st = len(STAGES)
    print(f"INFO: wrote {n_st} x2 float32 bin files (pt_fp16_*.bin, pt_fakequant_*.bin).", file=sys.stderr)
    if diverged:
        print(
            "INFO: rows flagged with <-- DIVERGENCE use compare_divergence thresholds (cos < 0.999); "
            "low cosine here is often expected for FP16 vs fake-quant.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
