#!/usr/bin/env python3
"""pt_q4km_vs_bf16.py — Compare dense reference forward vs Q4_K_M fake-quant forward.

Same harness as ``pt_fakequant_vs_fp16.py`` (hook table, STAGES, harness JSON).
Target modules come from ``build_quantization_entries`` in ``apply_septq_multitier.py``,
filtered to the same multitier weight set as SEPTQ (``in_proj_weight``, ``gating.linear_in/out``;
``out_proj`` skipped via skip filters). No SEPTQ tier masks or per-module quant metadata.

``Q4KMFakeQuantize`` (``scripts/q4km_fakequant.py``) quantizes/dequantizes the last dimension
in blocks of 256; linear weights whose ``in_features`` is not divisible by 256 are skipped with a
warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from einops import rearrange

_HERE = Path(__file__).resolve().parent
_MOSHI_PKG = _HERE / "moshi"
_SCRIPTS = _HERE / "scripts"
for p in (_MOSHI_PKG, _SCRIPTS):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from apply_septq import get_temporal_layers, parse_quantize_layers, parse_skip_module_filters  # noqa: E402
from apply_septq_multitier import build_quantization_entries, get_module_name_map  # noqa: E402
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
from qat_septq import _is_multitier_target_entry  # noqa: E402

from q4km_fakequant import BLOCK as Q4_BLOCK, Q4KMFakeQuantize  # noqa: E402


HARNESS_JSON_DEFAULT = "harness_input.json"
DEEP_LAYERS = [0, 1, 8, 15, 23, 31]


def gather_q4km_entries(
    student: nn.Module,
    selected_layers: list[int],
    skip_module_filters: list[str],
) -> list[dict[str, Any]]:
    temporal_layers = get_temporal_layers(student)
    if not temporal_layers:
        raise RuntimeError("Could not resolve temporal transformer layers for student model")
    name_map = get_module_name_map(student)
    layer_plan = build_quantization_entries(
        temporal_layers=temporal_layers,
        selected_indices=selected_layers,
        name_map=name_map,
        skip_module_filters=skip_module_filters,
    )
    entries: list[dict[str, Any]] = []
    for idx in selected_layers:
        pack = layer_plan[idx]
        for entry in pack["entries"]:
            name = str(entry["name"])
            if _is_multitier_target_entry(name):
                entries.append(entry)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for e in entries:
        mod = e["module"]
        pname = "weight" if e["kind"] == "linear" else str(e["param_name"])
        key = (id(mod), pname)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def register_q4km_for_entries(entries: list[dict[str, Any]]) -> dict[str, tuple[nn.Module, str]]:
    state_key_to_param: dict[str, tuple[nn.Module, str]] = {}
    seen: set[tuple[int, str]] = set()
    n_skip = 0
    for e in entries:
        module = e["module"]
        param_name = "weight" if e["kind"] == "linear" else str(e["param_name"])
        state_key = str(e["name"])
        dedup_key = (id(module), param_name)
        if dedup_key in seen:
            state_key_to_param[state_key] = (module, param_name)
            continue
        seen.add(dedup_key)

        target = getattr(module, param_name, None)
        if not torch.is_tensor(target):
            raise RuntimeError(f"Cannot register Q4KM fake quant: missing tensor param {state_key}")
        if int(target.shape[-1]) % int(Q4_BLOCK) != 0:
            print(
                f"[register_q4km] skipping {state_key}: in_features={target.shape[-1]} not divisible by {Q4_BLOCK}",
                file=sys.stderr,
            )
            n_skip += 1
            continue

        if hasattr(module, "parametrizations") and param_name in getattr(module, "parametrizations", {}):
            print(f"[register_q4km] skipping {state_key}: already parametrized", file=sys.stderr)
            continue
        parametrize.register_parametrization(module, param_name, Q4KMFakeQuantize())
        original = module.parametrizations[param_name].original
        original.requires_grad = True
        state_key_to_param[state_key] = (module, param_name)
    if n_skip:
        print(f"[register_q4km] skipped {n_skip} target(s) incompatible with Q4_K block size.", file=sys.stderr)
    return state_key_to_param


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
    p = argparse.ArgumentParser(description="Reference vs Q4KMFakeQuantize activation cosines (PT harness stages).")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt or .safetensors)")
    p.add_argument(
        "--harness-input",
        type=str,
        default=HARNESS_JSON_DEFAULT,
        dest="harness_input",
        help=f"JSON with input_ids (default {HARNESS_JSON_DEFAULT})",
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


def _load_qat_meta_dict(ckpt_path: Path) -> dict[str, Any]:
    suf = ckpt_path.suffix.lower()
    if suf in {".safetensors", ".sft", ".sfts"}:
        return {}
    payload = torch.load(str(ckpt_path), map_location="cpu")
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    harness_path = Path(args.harness_input).resolve()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA unavailable; falling back to cpu.", file=sys.stderr)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    qat_payload = _load_qat_meta_dict(ckpt_path)

    print(f"Loading reference model from {ckpt_path} ...")
    model_ref = get_moshi_lm(str(ckpt_path), device=device, dtype=dtype, copy_missing_weights=True)
    model_ref.eval()

    print(f"Loading second copy for Q4_K_M fake-quant from {ckpt_path} ...")
    model_fq = get_moshi_lm(str(ckpt_path), device=device, dtype=dtype, copy_missing_weights=True)
    model_fq.eval()

    temporal_layers = get_temporal_layers(model_fq)
    if not temporal_layers:
        raise SystemExit("Could not resolve temporal layers on model")
    n_temp = len(temporal_layers)
    selected_layers, skip_filters = qat_selection_from_ckpt(qat_payload, n_temp)

    entries = gather_q4km_entries(
        student=model_fq,
        selected_layers=selected_layers,
        skip_module_filters=skip_filters,
    )
    if not entries:
        raise SystemExit("gather_q4km_entries returned no modules; check train_layers / skip_modules")

    register_q4km_for_entries(entries)
    print(f"Registered Q4KMFakeQuantize on compatible weight target(s) from {len(entries)} candidate(s).")

    num_cb = model_ref.num_codebooks
    ids = load_harness_ids(harness_path, num_cb)
    seq = torch.tensor(ids, dtype=torch.long, device=device).view(1, num_cb, 1)

    embed_ref = model_ref.embed_codes(seq)
    cap_ref = run_forward_with_captures(model_ref, seq)

    embed_fq = model_fq.embed_codes(seq)
    cap_fq = run_forward_with_captures(model_fq, seq)

    if not torch.equal(embed_ref, embed_fq):
        print(
            "WARN: embed_sum differs between runs (unexpected - embeddings are not fake-quantized).",
            file=sys.stderr,
        )

    max_abs_all = 0.0
    for name in STAGES:
        if name == "embed_sum":
            continue
        try:
            ta = tensor_for_stage(name, cap_ref, embed_ref).float()
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
            t_a = tensor_for_stage(name, cap_ref, embed_ref)
            t_b = tensor_for_stage(name, cap_fq, embed_fq)
        except KeyError as exc:
            print(f"{name:<26} {'-':>10} {'-':>10} {'-':>12} {'-':>10} {'-':>10} {'-':>10}  MISSING capture ({exc})")
            diverged = True
            continue

        a = t_a.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        b = t_b.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        save_tensor_bin(torch.from_numpy(a), "pt_bf16", name)
        save_tensor_bin(torch.from_numpy(b), "pt_q4km", name)

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
        if diag_name in cap_ref and diag_name in cap_fq:
            z_a = cap_ref[diag_name].detach().float().reshape(-1)
            z_b = cap_fq[diag_name].detach().float().reshape(-1)
            c = float(torch.sum(z_a * z_b) / (torch.norm(z_a) * torch.norm(z_b) + 1e-30))
            print(f"Diagnostic: cosine({diag_name}) reference vs Q4KM = {c:.6f}")

    n_st = len(STAGES)
    print(f"INFO: wrote {n_st} x2 float32 bin files (pt_bf16_*.bin, pt_q4km_*.bin).", file=sys.stderr)
    if diverged:
        print(
            "INFO: rows flagged with <-- DIVERGENCE use compare_divergence thresholds (cos < 0.999); "
            "low cosine here is often expected for dense vs fake-quant.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
