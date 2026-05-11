#!/usr/bin/env python3
"""Settle C++ T4 V-slice vs Python matvec: GGUF-unpacked W vs safetensors vs PTQ ckpt W.

Loads the same post-norm1 vector ``x`` (``cpp_h3_T3_L0.bin``, 4096 float32), builds
``T4 = W @ x`` three ways (GGUF unpack, fakequant(safetensors), fakequant(PTQ)),
compares V slices to ``cpp_h3_T4_L0.bin`` and ``pt_fq_T4_L0.bin`` when provided.

Reuses ``unpack_v5_dense`` from ``compare_v_row_quant_gguf_vs_ptq.py`` (importlib).

Example:
  python scripts/matvec_t4_v_slice_weight_source_audit.py \\
    --x-bin cpp_h3_T3_L0.bin \\
    --cpp-t4-v cpp_h3_T4_L0.bin \\
    --pt-t4-v pt_fq_T4_L0.bin \\
    --gguf bmo_septq_v5.gguf \\
    --ptq bmo_temporal_half_cushion_max.pt \\
    --safetensors v5_step1500_split.safetensors \\
    --layer 0
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_cvq():
    path = Path(__file__).resolve().parent / "compare_v_row_quant_gguf_vs_ptq.py"
    spec = importlib.util.spec_from_file_location("_cvq", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load compare_v_row_quant_gguf_vs_ptq")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_scalar_i32(tensors: dict[str, np.ndarray], key: str, fallback: int = 0) -> int:
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    return struct.unpack("<i", raw[:4])[0] if len(raw) >= 4 else fallback


def read_scalar_f32(tensors: dict[str, np.ndarray], key: str, fallback: float = 0.0) -> float:
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    return struct.unpack("<f", raw[:4])[0] if len(raw) >= 4 else fallback


def _get_module_meta(ckpt: dict[str, Any], target_name: str) -> dict[str, Any] | None:
    sm = ckpt.get("septq_meta")
    if not isinstance(sm, dict):
        return None
    pls = sm.get("per_layer_stats")
    if not isinstance(pls, list):
        return None
    for layer in pls:
        if not isinstance(layer, dict):
            continue
        mods = layer.get("modules")
        if not isinstance(mods, list):
            continue
        for m in mods:
            if isinstance(m, dict) and m.get("name") == target_name:
                return m
    return None


def unpack_tier_mask_uint2_flat(packed: np.ndarray, total: int) -> np.ndarray:
    flat = packed.astype(np.uint8).reshape(-1)
    expanded = np.zeros(flat.size * 4, dtype=np.uint8)
    for lane in range(4):
        expanded[lane::4] = (flat >> (lane * 2)) & 0b11
    return expanded[:total].astype(np.int32)


def pick_in_proj_weight(sd: dict[str, Any], layer: int) -> tuple[str, np.ndarray]:
    cands = [
        f"transformer.layers.{layer}.self_attn.in_proj.weight",
        f"transformer.layers.{layer}.self_attn.in_proj_weight",
    ]
    for k in cands:
        if k in sd:
            return k, np.asarray(torch.as_tensor(sd[k], dtype=torch.float32).cpu())
    for k, v in sd.items():
        if f"layers.{layer}.self_attn.in_proj" in k and k.endswith("weight"):
            return k, np.asarray(torch.as_tensor(v, dtype=torch.float32).cpu())
    raise SystemExit("Could not find in_proj weight in state_dict / safetensors")


def pick_tier_mask(masks: dict[str, Any], mod_key: str) -> np.ndarray:
    t = masks.get(mod_key)
    if t is None:
        for k, v in masks.items():
            if "in_proj" in str(k) and str(k).endswith("in_proj_weight"):
                t = v
                break
    if t is None:
        raise SystemExit(f"No tier_masks_uint2 entry for {mod_key!r}")
    return np.asarray(torch.as_tensor(t, dtype=torch.uint8).cpu()).reshape(-1)


def fakequant_dense_numpy(
    w: np.ndarray,
    tier: np.ndarray,
    *,
    scale_int8: float,
    zp_int8: float,
    scale_int4: float,
    zp_int4: float,
    scale_low: float,
    zp_low: float,
) -> np.ndarray:
    """Vectorized ``MultiTierFakeQuantize`` forward (tier 0 = identity)."""
    out = w.astype(np.float64).copy()
    s8, z8 = max(float(scale_int8), 1e-12), float(zp_int8)
    s4, z4 = max(float(scale_int4), 1e-12), float(zp_int4)
    s2, z2 = max(float(scale_low), 1e-12), float(zp_low)
    m1 = tier == 1
    if np.any(m1):
        q = np.clip(np.round(out[m1] / s8 + z8), 0.0, 255.0)
        out[m1] = s8 * (q - z8)
    m2 = tier == 2
    if np.any(m2):
        q = np.clip(np.round(out[m2] / s4 + z4), 0.0, 15.0)
        out[m2] = s4 * (q - z4)
    m3 = tier >= 3
    if np.any(m3):
        q = np.clip(np.round(out[m3] / s2 + z2), 0.0, 3.0)
        out[m3] = s2 * (q - z2)
    return out.astype(np.float32)


def cos_l2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return c, float(np.linalg.norm(a)), float(np.linalg.norm(b))


def v_slice(t4: np.ndarray, n_embd: int = 4096) -> np.ndarray:
    return np.asarray(t4, dtype=np.float64).ravel()[2 * n_embd : 3 * n_embd]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-bin", type=Path, required=True, help="cpp_h3_T3_L0.bin (4096 float32)")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--ptq", type=Path, required=True, help="bmo_temporal_half_cushion_max.pt (masks + meta + W)")
    ap.add_argument("--safetensors", type=Path, required=True)
    ap.add_argument("--cpp-t4-v", type=Path, default=None, help="cpp_h3_T4_L0.bin full 12288 (optional)")
    ap.add_argument("--pt-t4-v", type=Path, default=None, help="pt_fq_T4_L0.bin full 12288 (optional)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-embd", type=int, default=4096)
    args = ap.parse_args()

    cvq = _load_cvq()
    unpack_v5_dense = cvq.unpack_v5_dense
    load_gguf_tensors = cvq.load_gguf_tensors

    L = int(args.layer)
    n_embd = int(args.n_embd)
    mod_key = f"transformer.layers.{L}.self_attn.in_proj_weight"

    x = np.fromfile(str(args.x_bin.resolve()), dtype=np.float32).astype(np.float64)
    if x.size != n_embd:
        raise SystemExit(f"--x-bin length {x.size} != --n-embd {n_embd}")

    tensors = load_gguf_tensors(args.gguf.resolve())
    base = f"transformer_layers_{L}_self_attn_in_proj_weight"
    pw = np.asarray(tensors[base + ".packed_weights"]).view(np.uint8).ravel()
    pm = np.asarray(tensors[base + ".packed_mask"]).view(np.uint8).ravel()
    fv = np.asarray(tensors[base + ".fp16_values"])
    rows = read_scalar_i32(tensors, base + ".rows")
    cols = read_scalar_i32(tensors, base + ".cols")
    n2 = read_scalar_i32(tensors, base + ".n_2bit_bytes")
    n4 = read_scalar_i32(tensors, base + ".n_4bit_bytes")
    n8 = read_scalar_i32(tensors, base + ".n_8bit_bytes")
    sl = read_scalar_f32(tensors, base + ".scale_low")
    s4 = read_scalar_f32(tensors, base + ".scale_int4")
    s8 = read_scalar_f32(tensors, base + ".scale_int8")
    zl = read_scalar_f32(tensors, base + ".zp_low")
    z4 = read_scalar_f32(tensors, base + ".zp_int4")
    z8 = read_scalar_f32(tensors, base + ".zp_int8")

    W_gguf = unpack_v5_dense(
        pw, pm, rows=rows, cols=cols, n2=n2, n4=n4, n8=n8, scale_low=sl, scale_int4=s4, scale_int8=s8, zp_low=zl, zp_int4=z4, zp_int8=z8, fp16_values=fv
    )
    if W_gguf.shape[1] != x.size:
        raise SystemExit(f"W cols {W_gguf.shape[1]} != x dim {x.size}")
    T4_gguf = (W_gguf.astype(np.float64) @ x).astype(np.float64)
    V_gguf = v_slice(T4_gguf, n_embd)

    ckpt = torch.load(str(args.ptq.resolve()), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("ptq must be a dict checkpoint")
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(sd, dict):
        raise SystemExit("no state_dict in ptq")

    masks = ckpt.get("tier_masks_uint2")
    if not isinstance(masks, dict):
        sm = ckpt.get("septq_meta")
        masks = sm.get("tier_masks_uint2") if isinstance(sm, dict) else None
    if not isinstance(masks, dict):
        raise SystemExit("tier_masks_uint2 missing in ptq")

    total = rows * cols
    packed = pick_tier_mask(masks, mod_key)
    need = (total + 3) // 4
    tier_flat = unpack_tier_mask_uint2_flat(packed[:need], total)
    tier = tier_flat.reshape(rows, cols)

    meta = _get_module_meta(ckpt, mod_key)
    if not isinstance(meta, dict):
        raise SystemExit("missing septq_meta module entry for in_proj")
    scale_low = float(meta.get("quant_scale_low") or meta.get("quant_scale") or 0.0)
    zp_low = float(meta.get("quant_zero_point_low") or meta.get("quant_zero_point") or 0.0)
    scale_int4 = float(meta["quant_scale_int4"])
    zp_int4 = float(meta["quant_zero_point_int4"])
    scale_int8 = float(meta["quant_scale_int8"])
    zp_int8 = float(meta["quant_zero_point_int8"])

    _, W_pt = pick_in_proj_weight(sd, L)
    if W_pt.shape != (rows, cols):
        raise SystemExit(f"PTQ W shape {W_pt.shape} vs GGUF ({rows},{cols})")

    from safetensors import safe_open

    with safe_open(str(args.safetensors.resolve()), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        pref = f"transformer.layers.{L}.self_attn.in_proj"
        sk = next((k for k in keys if k.startswith(pref) and k.endswith("weight")), None)
        if sk is None:
            sk = next((k for k in keys if "in_proj" in k and "weight" in k), keys[0])
        W_st = f.get_tensor(sk).detach().to(torch.float32).cpu().numpy()
    if W_st.shape != (rows, cols):
        raise SystemExit(f"safetensors W shape {W_st.shape} vs ({rows},{cols})")

    W_fq_st = fakequant_dense_numpy(
        W_st, tier, scale_int8=scale_int8, zp_int8=zp_int8, scale_int4=scale_int4, zp_int4=zp_int4, scale_low=scale_low, zp_low=zp_low
    )
    W_fq_pt = fakequant_dense_numpy(
        W_pt, tier, scale_int8=scale_int8, zp_int8=zp_int8, scale_int4=scale_int4, zp_int4=zp_int4, scale_low=scale_low, zp_low=zp_low
    )

    T4_st = (W_fq_st.astype(np.float64) @ x).astype(np.float64)
    T4_pt = (W_fq_pt.astype(np.float64) @ x).astype(np.float64)
    V_st = v_slice(T4_st, n_embd)
    V_pt = v_slice(T4_pt, n_embd)

    cpp_v: np.ndarray | None = None
    pt_v: np.ndarray | None = None
    if args.cpp_t4_v is not None:
        t4c = np.fromfile(str(args.cpp_t4_v.resolve()), dtype=np.float32).astype(np.float64)
        if t4c.size != 3 * n_embd:
            raise SystemExit(f"cpp T4 length {t4c.size} != {3 * n_embd}")
        cpp_v = v_slice(t4c, n_embd)
    if args.pt_t4_v is not None:
        t4p = np.fromfile(str(args.pt_t4_v.resolve()), dtype=np.float32).astype(np.float64)
        if t4p.size != 3 * n_embd:
            raise SystemExit(f"pt T4 length {t4p.size} != {3 * n_embd}")
        pt_v = v_slice(t4p, n_embd)

    def report_pair(name: str, a: np.ndarray, b: np.ndarray) -> None:
        c, la, lb = cos_l2(a, b)
        print(f"  {name:48s} cos={c:.10f}  L2_a={la:.6f}  L2_b={lb:.6f}")

    print("\n=== Step 4: V-slice (4096) pairwise cosines + L2 ===\n")
    if cpp_v is not None:
        report_pair("cos(C++_V, V_gguf_unpack_matvec)", cpp_v, V_gguf)
        report_pair("cos(C++_V, V_safetensors_fakequant_matvec)", cpp_v, V_st)
        report_pair("cos(C++_V, V_ptq_fakequant_matvec)", cpp_v, V_pt)
    else:
        print("  (skip C++ bins: pass --cpp-t4-v)")

    if pt_v is not None:
        report_pair("cos(PT_FQ_V, V_gguf_unpack_matvec)", pt_v, V_gguf)
        report_pair("cos(PT_FQ_V, V_safetensors_fakequant_matvec)", pt_v, V_st)
        report_pair("cos(PT_FQ_V, V_ptq_fakequant_matvec)", pt_v, V_pt)
    else:
        print("  (skip PT_FQ bins: pass --pt-t4-v)")

    report_pair("cos(V_safetensors_fakequant, V_ptq_fakequant)", V_st, V_pt)

    print("\n=== L2 norms of the six conceptual V slices (where available) ===")
    for lab, vec in (
        ("L2(V_gguf_unpack_matvec)", V_gguf),
        ("L2(V_safetensors_fakequant)", V_st),
        ("L2(V_ptq_fakequant)", V_pt),
    ):
        v = np.asarray(vec, dtype=np.float64).ravel()
        print(f"  {lab}: {float(np.linalg.norm(v)):.6f}")
    if cpp_v is not None:
        print(f"  L2(C++_V): {float(np.linalg.norm(cpp_v)):.6f}")
    if pt_v is not None:
        print(f"  L2(PT_FQ_V): {float(np.linalg.norm(pt_v)):.6f}")

    # --- Step 5 verdict ---
    thr = 0.999
    c_st_pt, _, _ = cos_l2(V_st, V_pt)
    print("\n=== Step 5: Verdict (threshold cos >= {:.3f} for 'match') ===".format(thr))

    def ok(c: float) -> bool:
        return c >= thr

    if cpp_v is None:
        print("  Incomplete: add --cpp-t4-v for A/C classification vs GGUF matvec.")
        print("  Provisional: cos(V_safetensors_fq, V_ptq_fq) = {:.6f}".format(c_st_pt))
        if ok(c_st_pt):
            print("  -> D-like: fakequant V slices nearly identical for this x (same mask/scales).")
        else:
            print("  -> Weight source matters for this x (safetensors vs PTQ W under same mask).")
        return

    c_cpp_gguf, _, _ = cos_l2(cpp_v, V_gguf)
    c_cpp_st, _, _ = cos_l2(cpp_v, V_st)
    c_cpp_pt, _, _ = cos_l2(cpp_v, V_pt)

    c_ptfq_st = c_ptfq_pt = float("nan")
    if pt_v is not None:
        c_ptfq_st, _, _ = cos_l2(pt_v, V_st)
        c_ptfq_pt, _, _ = cos_l2(pt_v, V_pt)

    verdict = "?"
    note = "See cosines above."

    if not ok(c_cpp_gguf):
        verdict = "C"
        note = "C++ does not match GGUF-unpack matvec — kernel vs numpy reference."
    elif (
        pt_v is not None
        and ok(c_ptfq_st)
        and ok(c_cpp_pt)
        and not ok(c_cpp_st)
    ):
        verdict = "B"
        note = "PT_FQ tracks safetensors-fakequant matvec; C++ tracks PTQ-fakequant — weight-source split."
    elif ok(c_cpp_gguf) and not ok(c_cpp_st) and ok(c_cpp_pt):
        verdict = "A"
        note = "C++ matches GGUF + PTQ-fakequant matvec, not safetensors-fakequant (add --pt-t4-v to try for B)."
    elif ok(c_st_pt):
        verdict = "D"
        note = "V_safetensors_fq ≈ V_ptq_fq for this x; same fakequant pipeline output from both W sources."
    elif ok(c_cpp_gguf) and ok(c_cpp_st) and ok(c_cpp_pt):
        verdict = "?"
        note = "C++ matches all three paths (unexpected if W differ materially)."

    print(f"  Verdict: {verdict}")
    print(f"  {note}")
    if verdict == "A":
        print("  Action: Point pt_fakequant_vs_fp16 (and any golden) at PTQ checkpoint W, not safetensors-only weights.")
    elif verdict == "B":
        print("  Action: Same as A — align reference pipeline weights with PTQ-derived GGUF/runtime.")
    elif verdict == "C":
        print("  Action: Debug fused matvec vs numpy W@x with GGUF-unpacked W (kernel indexing / layout).")
    elif verdict == "D":
        print("  Action: Re-check safetensors vs PTQ diff on full W; bug may be outside weight-source (x capture, layout).")
    else:
        print("  Action: Inspect printed cosines and tighten x/T4 bin provenance.")


if __name__ == "__main__":
    main()
