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
    cap_fp16 = run_forward_with_captures(model_fp16, seq)

    embed_fq = model_fq.embed_codes(seq)
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
