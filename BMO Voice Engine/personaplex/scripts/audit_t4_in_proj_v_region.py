#!/usr/bin/env python3
"""Audit layer-L self_attn in_proj T4: Q/K/V regions, per-head V cos, GGUF vs PTQ tiers/scales.

Requires: ``gguf`` (pip), ``numpy``, ``torch``. Run from personaplex repo root.

Example:
  python scripts/audit_t4_in_proj_v_region.py \\
    --cpp-t4 build/cpp_h3_bins/cpp_h3_T4_L0.bin \\
    --pt-t4  pt_fq_t4_l0.bin \\
    --gguf   bmo_septq_v5.gguf \\
    --ptq    bmo_temporal_half_cushion_max.pt \\
    --n-embd 4096 --n-heads 32

If you do not have ``pt_fq_t4_l0.bin``, dump the 12288-float in_proj output (Q||K||V) from PT_FQ
in the same layout as C++ ``qkv`` and save with ``astype(np.float32).tofile(...)``.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _get_module_meta(ckpt: dict[str, Any], target_name: str) -> dict[str, Any] | None:
    """Same lookup as ``qat_septq.get_module_meta`` without importing that module (heavy deps)."""
    septq_meta = ckpt.get("septq_meta")
    if not isinstance(septq_meta, dict):
        return None
    per_layer_stats = septq_meta.get("per_layer_stats")
    if not isinstance(per_layer_stats, list):
        return None
    for layer in per_layer_stats:
        if not isinstance(layer, dict):
            continue
        modules = layer.get("modules")
        if not isinstance(modules, list):
            continue
        for mod in modules:
            if isinstance(mod, dict) and mod.get("name") == target_name:
                return mod
    return None


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64).ravel()))


def unpack_u2_le(byte_val: int, lane: int) -> int:
    return (int(byte_val) >> (lane * 2)) & 0x3


def tier_name(t: int) -> str:
    if t <= 0:
        return "FP16"
    if t == 1:
        return "INT8"
    if t == 2:
        return "INT4"
    return "INT2"


def load_gguf_tensors(path: Path) -> dict[str, np.ndarray]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    tensors: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        data = tensor.data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        tensors[tensor.name] = data
    return tensors


def read_scalar_i32(tensors: dict[str, np.ndarray], key: str, fallback: int = 0) -> int:
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    if len(raw) < 4:
        return fallback
    return struct.unpack("<i", raw[:4])[0]


def read_scalar_f32(tensors: dict[str, np.ndarray], key: str, fallback: float = 0.0) -> float:
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    if len(raw) < 4:
        return fallback
    return struct.unpack("<f", raw[:4])[0]


def gguf_tiers_for_in_proj(
    tensors: dict[str, np.ndarray], *, layer: int, n_embd: int
) -> tuple[int, int, np.ndarray, dict[str, dict[str, int]]]:
    """Return (rows, cols, tier_per_pos_flat, counts_by_region)."""
    base = f"transformer_layers_{layer}_self_attn_in_proj_weight"
    pm = tensors.get(base + ".packed_mask")
    if pm is None:
        raise SystemExit(f"GGUF missing {base}.packed_mask")
    rows = read_scalar_i32(tensors, base + ".rows")
    cols = read_scalar_i32(tensors, base + ".cols")
    if rows != 3 * n_embd:
        print(f"[warn] GGUF rows={rows} != 3*n_embd={3 * n_embd}; using rows as given", file=sys.stderr)
    total = rows * cols
    packed = pm.view(np.uint8).ravel()
    need = (total + 3) // 4
    if packed.size < need:
        raise SystemExit(f"packed_mask bytes {packed.size} < required {need} for total={total}")
    tier_flat = np.zeros(total, dtype=np.int32)
    for pos in range(total):
        mbyte = int(packed[pos // 4])
        tier_flat[pos] = unpack_u2_le(mbyte, pos % 4)
    q_end, k_end, v_end = n_embd, 2 * n_embd, 3 * n_embd
    regions = {
        "Q": (0, q_end),
        "K": (q_end, k_end),
        "V": (k_end, min(v_end, rows)),
    }
    counts: dict[str, dict[str, int]] = {}
    for rname, (r0, r1) in regions.items():
        # Row-major flatten: row r covers [r*cols, (r+1)*cols).
        idx0, idx1 = r0 * cols, r1 * cols
        sub = tier_flat[idx0:idx1]
        c = {"FP16": 0, "INT8": 0, "INT4": 0, "INT2": 0}
        for t in sub:
            c[tier_name(int(t))] += 1
        counts[rname] = c
    return rows, cols, tier_flat, counts


def ptq_tiers_from_mask(
    ckpt: dict[str, Any], *, module_key: str, rows: int, cols: int
) -> np.ndarray | None:
    masks = ckpt.get("tier_masks_uint2")
    if not isinstance(masks, dict):
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            masks = sm.get("tier_masks_uint2")
    if not isinstance(masks, dict):
        return None
    import torch

    t = masks.get(module_key)
    if t is None:
        for k, v in masks.items():
            if str(k).endswith(module_key.split(".")[-1]) and "in_proj" in str(k):
                t = v
                break
    if t is None:
        return None
    flat = torch.as_tensor(t, dtype=torch.uint8).reshape(-1)
    expanded = torch.zeros(flat.numel() * 4, dtype=torch.uint8)
    for lane in range(4):
        expanded[lane::4] = (flat >> (lane * 2)) & 0b11
    need = rows * cols
    arr = expanded[:need].numpy().astype(np.int32)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpp-t4", type=Path, required=True)
    ap.add_argument("--pt-t4", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--ptq", type=Path, default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-embd", type=int, default=4096)
    ap.add_argument("--n-heads", type=int, default=32)
    args = ap.parse_args()

    n_embd = int(args.n_embd)
    n_heads = int(args.n_heads)
    head_dim = n_embd // n_heads
    if head_dim * n_heads != n_embd:
        raise SystemExit("n_embd must be divisible by n_heads")

    cpp = np.fromfile(str(args.cpp_t4.resolve()), dtype=np.float32).astype(np.float64)
    pt = np.fromfile(str(args.pt_t4.resolve()), dtype=np.float32).astype(np.float64)
    if cpp.size != pt.size:
        raise SystemExit(f"length mismatch cpp={cpp.size} pt={pt.size}")
    if cpp.size != 3 * n_embd:
        raise SystemExit(f"expected 3*n_embd={3 * n_embd} floats, got {cpp.size}")

    print("=== Step 1: per-region cos(C++, PT_FQ) + L2 ===")
    for name, sl in (("Q", slice(0, n_embd)), ("K", slice(n_embd, 2 * n_embd)), ("V", slice(2 * n_embd, 3 * n_embd))):
        a, b = cpp[sl], pt[sl]
        print(
            f"  {name}: cos={_cos(a, b):.8f}  L2_cpp={_l2(a):.6f}  L2_pt={_l2(b):.6f}  "
            f"first8_cpp={np.array2string(a[:8], precision=4)}"
        )

    print("\n=== Step 2: per-head cos within V (32 heads x head_dim) ===")
    v_cpp = cpp[2 * n_embd : 3 * n_embd]
    v_pt = pt[2 * n_embd : 3 * n_embd]
    head_c = []
    for h in range(n_heads):
        s = h * head_dim
        e = (h + 1) * head_dim
        head_c.append(_cos(v_cpp[s:e], v_pt[s:e]))
    hc = np.array(head_c, dtype=np.float64)
    print(f"  min={hc.min():.8f}  max={hc.max():.8f}  mean={hc.mean():.8f}")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    hist_labels = ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-0.95", ">0.95"]
    for (lo, hi), lab in zip(bins, hist_labels):
        n = int(np.sum((hc >= lo) & (hc < hi)))
        print(f"  hist {lab:12s}: {n}")
    worst = np.argsort(hc)[:5]
    best = np.argsort(-hc)[:5]
    print("  worst 5 heads (idx, cos):", [(int(i), float(hc[i])) for i in worst])
    print("  best 5 heads  (idx, cos):", [(int(i), float(hc[i])) for i in best])

    if args.gguf is None or args.ptq is None:
        print("\n=== Steps 3-4: skipped (pass --gguf and --ptq) ===")
        print("\n=== Step 5 (manual) ===")
        print("  Hypothesis: use Step 1-2 numbers; add GGUF/PTQ tier+scale when paths provided.")
        return

    print("\n=== Step 3: GGUF tier element counts by Q/K/V row bands ===")
    tensors = load_gguf_tensors(args.gguf.resolve())
    rows, cols, tier_gguf, gguf_counts = gguf_tiers_for_in_proj(tensors, layer=int(args.layer), n_embd=n_embd)
    print(f"  GGUF rows={rows} cols={cols} total_elems={rows * cols}")
    for rname, c in gguf_counts.items():
        tot = sum(c.values())
        print(f"  GGUF {rname}: total_elems={tot}  " + " ".join(f"{k}={v}" for k, v in c.items()))

    ckpt = torch.load(str(args.ptq.resolve()), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("--ptq must be a dict checkpoint")
    mod_key = f"transformer.layers.{int(args.layer)}.self_attn.in_proj_weight"
    pt_tier = ptq_tiers_from_mask(ckpt, module_key=mod_key, rows=rows, cols=cols)
    if pt_tier is None:
        print("  [warn] Could not resolve tier_masks_uint2 for in_proj; skip PT mask region compare")
    else:
        q_end, k_end = n_embd, 2 * n_embd
        idx_q = slice(0, q_end * cols)
        idx_k = slice(q_end * cols, k_end * cols)
        idx_v = slice(k_end * cols, min(3 * n_embd, rows) * cols)
        def cnt_region(sl: slice, name: str) -> dict[str, int]:
            sub = pt_tier[sl]
            c = {"FP16": 0, "INT8": 0, "INT4": 0, "INT2": 0}
            for t in sub:
                c[tier_name(int(t))] += 1
            return c

        print("  PTQ tier_mask counts by region (same linear layout as GGUF):")
        for nm, sl in (("Q", idx_q), ("K", idx_k), ("V", idx_v)):
            c = cnt_region(sl, nm)
            tot = sum(c.values())
            print(f"  PTQ {nm}: total_elems={tot}  " + " ".join(f"{k}={v}" for k, v in c.items()))
        v0, v1 = k_end * cols, min(3 * n_embd, rows) * cols
        g_seg, p_seg = tier_gguf[v0:v1], pt_tier[v0:v1]
        n_cmp = min(g_seg.size, p_seg.size)
        if g_seg.size != p_seg.size:
            print(
                f"  [warn] V-region length mismatch GGUF={g_seg.size} PTQ={p_seg.size}; comparing first {n_cmp}",
                file=sys.stderr,
            )
        n_mismatch = int(np.sum(g_seg[:n_cmp] != p_seg[:n_cmp]))
        print(f"  V-region tier mismatches (GGUF vs PTQ mask): {n_mismatch} / {n_cmp}")

    print("\n=== Step 4: GGUF vs PTQ global scales (in_proj is one scale set per tensor) ===")
    base = f"transformer_layers_{int(args.layer)}_self_attn_in_proj_weight"
    gguf_scales = {
        "scale_low": read_scalar_f32(tensors, base + ".scale_low"),
        "scale_int4": read_scalar_f32(tensors, base + ".scale_int4"),
        "scale_int8": read_scalar_f32(tensors, base + ".scale_int8"),
        "zp_low": read_scalar_f32(tensors, base + ".zp_low"),
        "zp_int4": read_scalar_f32(tensors, base + ".zp_int4"),
        "zp_int8": read_scalar_f32(tensors, base + ".zp_int8"),
    }
    meta = _get_module_meta(ckpt, mod_key)
    if not isinstance(meta, dict):
        print(f"  [warn] get_module_meta missing for {mod_key}")
        ptq_scales = {}
    else:
        ptq_scales = {
            "scale_low": float(meta.get("quant_scale_low") or meta.get("quant_scale") or 0.0),
            "scale_int4": float(meta.get("quant_scale_int4") or 0.0),
            "scale_int8": float(meta.get("quant_scale_int8") or 0.0),
            "zp_low": float(meta.get("quant_zero_point_low") or 0.0),
            "zp_int4": float(meta.get("quant_zero_point_int4") or 0.0),
            "zp_int8": float(meta.get("quant_zero_point_int8") or 0.0),
        }
    for k in sorted(gguf_scales.keys()):
        gv, pv = gguf_scales[k], ptq_scales.get(k, float("nan"))
        delta = gv - pv if pv == pv else float("nan")
        print(f"  {k:12s}  GGUF={gv:.8g}  PTQ={pv:.8g}  delta={delta:.8g}")

    sample_rows = [8200, 8500, 9000, 10000, 12000]
    print(f"\n  Note: scales are global per tensor (not per output row). Sample V row indices {sample_rows} share these scalars.")

    print("\n=== Step 5: template verdict (fill after reading numbers above) ===")
    print("  1) If V region cos << Q/K: V matvec path / V-weight tiers dominate.")
    print("  2) If few heads drive low cos: localized head-subspace issue (mask row blocks).")
    print("  3) If GGUF vs PTQ tier counts in V differ: export / mask serialization bug.")
    print("  4) If scales differ: Bug B class (re-verify export path).")
    print("  5) If tiers+scales match but V cos bad: kernel read/unpack for V rows (D1 said matvec ok — recheck row range in proto).")


if __name__ == "__main__":
    main()
