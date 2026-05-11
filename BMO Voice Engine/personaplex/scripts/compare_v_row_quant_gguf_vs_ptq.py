#!/usr/bin/env python3
"""Distinguish (A) GGUF vs PTQ quantized payload mismatch vs (B) dequant order mismatch.

Reads GGUF packed streams + PT checkpoint dense ``in_proj`` weight + ``tier_masks_uint2``
and ``septq_meta`` scales. For V rows 8192, 10000, 12287 (default), counts per-tier
quantized-value mismatches (GGUF decode vs PT re-round from same ``W``).

Also compares two dequant orderings in float32 (``(q-zp)*scale`` vs ``scale*(q-zp)``)
and ``MultiTierFakeQuantize``-style row reconstruction vs GGUF v5 linear unpack.

No writes to GGUF/PT/kernels. Requires: numpy, torch, gguf.

Example:
  python scripts/compare_v_row_quant_gguf_vs_ptq.py \\
    --gguf bmo_septq_v5.gguf --ptq bmo_temporal_half_cushion_max.pt --layer 0
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


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


def load_gguf_tensors(path: Path) -> dict[str, np.ndarray]:
    import gguf

    reader = gguf.GGUFReader(str(path))
    out: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        d = tensor.data
        out[tensor.name] = d if isinstance(d, np.ndarray) else np.asarray(d)
    return out


def unpack_u2_le(byte_val: int, lane: int) -> int:
    return (int(byte_val) >> (lane * 2)) & 0x3


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


def pick_in_proj_weight(sd: dict[str, torch.Tensor], layer: int) -> tuple[str, torch.Tensor]:
    cands = [
        f"transformer.layers.{layer}.self_attn.in_proj.weight",
        f"transformer.layers.{layer}.self_attn.in_proj_weight",
    ]
    for k in cands:
        if k in sd:
            return k, sd[k]
    for k, v in sd.items():
        if f"layers.{layer}.self_attn.in_proj" in k and k.endswith("weight") and isinstance(v, torch.Tensor):
            return k, v
    raise SystemExit("Could not find in_proj weight in state_dict")


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


def fakequant_row_numpy(
    w_row: np.ndarray,
    tier_row: np.ndarray,
    *,
    scale_int8: float,
    zp_int8: float,
    scale_int4: float,
    zp_int4: float,
    scale_low: float,
    zp_low: float,
) -> np.ndarray:
    """Mirror ``MultiTierFakeQuantize._fake_quant_affine`` per element (float32)."""
    out = w_row.astype(np.float64).copy()
    s8, z8 = max(scale_int8, 1e-12), zp_int8
    s4, z4 = max(scale_int4, 1e-12), zp_int4
    s2, z2 = max(scale_low, 1e-12), zp_low
    m1 = tier_row == 1
    if np.any(m1):
        q = np.clip(np.round(out[m1] / s8 + z8), 0.0, 255.0)
        out[m1] = s8 * (q - z8)
    m2 = tier_row == 2
    if np.any(m2):
        q = np.clip(np.round(out[m2] / s4 + z4), 0.0, 15.0)
        out[m2] = s4 * (q - z4)
    m3 = tier_row >= 3
    if np.any(m3):
        q = np.clip(np.round(out[m3] / s2 + z2), 0.0, 3.0)
        out[m3] = s2 * (q - z2)
    # tier 0: identity (full float row segment)
    return out.astype(np.float32)


def unpack_v5_dense(
    pw: np.ndarray,
    pm: np.ndarray,
    *,
    rows: int,
    cols: int,
    n2: int,
    n4: int,
    n8: int,
    scale_low: float,
    scale_int4: float,
    scale_int8: float,
    zp_low: float,
    zp_int4: float,
    zp_int8: float,
    fp16_values: np.ndarray,
) -> np.ndarray:
    """``bmo_compute.cpp`` ``unpack_layer_to_f32_blockwise`` for ``packing_version >= 5``."""
    total = rows * cols
    tier = np.zeros(total, dtype=np.int32)
    for pos in range(total):
        tier[pos] = unpack_u2_le(int(pm[pos // 4]), pos % 4)
    out = np.zeros(total, dtype=np.float32)
    stream2 = pw[:n2].view(np.uint8).ravel()
    stream4 = pw[n2 : n2 + n4].view(np.uint8).ravel()
    stream8 = pw[n2 + n4 : n2 + n4 + n8].view(np.uint8).ravel()
    fv = fp16_values.view(np.float16).ravel()
    c2 = c4 = c8 = c16 = 0
    for pos in range(total):
        t = int(tier[pos])
        if t == 0:
            out[pos] = float(fv[c16])
            c16 += 1
        elif t == 1:
            q = int(stream8[c8])
            c8 += 1
            out[pos] = (float(q) - zp_int8) * scale_int8
        elif t == 2:
            idx = c4
            bb = int(stream4[idx // 2])
            q = (bb & 0x0F) if (idx % 2 == 0) else ((bb >> 4) & 0x0F)
            c4 += 1
            out[pos] = (float(q) - zp_int4) * scale_int4
        else:
            idx = c2
            bb = int(stream2[idx // 4])
            q = unpack_u2_le(bb, idx % 4)
            c2 += 1
            out[pos] = (float(q) - zp_low) * scale_low
    return out.reshape(rows, cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--ptq", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=str, default="8192,10000,12287", help="Comma-separated output row indices")
    ap.add_argument("--safetensors", type=Path, default=None, help="Optional FP16 reference weights (same key)")
    args = ap.parse_args()

    L = int(args.layer)
    want_rows = [int(x.strip()) for x in str(args.rows).split(",") if x.strip()]

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

    ckpt = torch.load(str(args.ptq.resolve()), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("checkpoint must be a dict")
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(sd, dict):
        raise SystemExit("no state_dict in checkpoint")
    _, w_t = pick_in_proj_weight(sd, L)
    w = torch.as_tensor(w_t, dtype=torch.float32).cpu().numpy()
    if w.ndim != 2 or w.shape[0] != rows or w.shape[1] != cols:
        raise SystemExit(f"weight shape {w.shape} vs GGUF rows={rows} cols={cols}")

    masks = ckpt.get("tier_masks_uint2")
    if not isinstance(masks, dict):
        sm = ckpt.get("septq_meta")
        masks = sm.get("tier_masks_uint2") if isinstance(sm, dict) else None
    if not isinstance(masks, dict):
        raise SystemExit("tier_masks_uint2 missing")
    mod_key = f"transformer.layers.{L}.self_attn.in_proj_weight"
    packed_mask_pt = pick_tier_mask(masks, mod_key)
    total = rows * cols
    if packed_mask_pt.size != (total + 3) // 4 and packed_mask_pt.size != total // 4 + (total % 4 != 0):
        # allow exact packed bytes match
        need = (total + 3) // 4
        if packed_mask_pt.size < need:
            raise SystemExit(f"tier mask packed bytes {packed_mask_pt.size} need {need}")
    tier_flat = unpack_tier_mask_uint2_flat(packed_mask_pt[: (total + 3) // 4], total)
    if tier_flat.shape[0] != total:
        raise SystemExit("tier_flat size mismatch")

    meta = _get_module_meta(ckpt, mod_key)
    if not isinstance(meta, dict):
        raise SystemExit("missing per_layer_stats module meta for in_proj")
    scale_low = float(meta.get("quant_scale_low") or meta.get("quant_scale") or 0.0)
    zp_low = float(meta.get("quant_zero_point_low") or meta.get("quant_zero_point") or 0.0)
    scale_int4 = float(meta["quant_scale_int4"])
    zp_int4 = float(meta["quant_zero_point_int4"])
    scale_int8 = float(meta["quant_scale_int8"])
    zp_int8 = float(meta["quant_zero_point_int8"])

    print(f"GGUF base={base} rows={rows} cols={cols}")
    print(f"Scales GGUF vs PTQ meta (should match audit): low {sl:g} vs {scale_low:g}, ...")

    # --- Step 1: per-row tier bucket mismatch counts (quant integers + FP16 bits)
    tier = tier_flat.reshape(rows, cols)
    wmat = w.astype(np.float32)
    cum1 = np.cumsum((tier_flat == 1).astype(np.int64))
    cum2 = np.cumsum((tier_flat == 2).astype(np.int64))
    cum3 = np.cumsum((tier_flat >= 3).astype(np.int64))
    cum0 = np.cumsum((tier_flat == 0).astype(np.int64))
    z = np.zeros(1, dtype=np.int64)
    before1 = np.concatenate([z, cum1[:-1]])
    before2 = np.concatenate([z, cum2[:-1]])
    before3 = np.concatenate([z, cum3[:-1]])
    before0 = np.concatenate([z, cum0[:-1]])

    pw_u8 = pw
    strm2 = pw_u8[:n2]
    s4b = pw_u8[n2 : n2 + n4]
    s8b = pw_u8[n2 + n4 : n2 + n4 + n8]
    fv_f = fv.view(np.float16).ravel()

    for r in want_rows:
        if r < 0 or r >= rows:
            print(f"[skip] row {r} out of range [0,{rows})")
            continue
        tr = tier[r, :]
        wr = wmat[r, :].astype(np.float64)
        pos0 = r * cols + np.arange(cols, dtype=np.int64)
        miss = {"FP16": 0, "INT8": 0, "INT4": 0, "INT2": 0}
        for j in range(cols):
            pos = int(pos0[j])
            t = int(tr[j])
            if t == 0:
                bi = int(before0[pos])
                gg = int(np.asarray(fv_f[bi : bi + 1], dtype=np.float16).view(np.uint16)[0])
                pt = int(np.asarray([float(wr[j])], dtype=np.float16).view(np.uint16)[0])
                if gg != pt:
                    miss["FP16"] += 1
            elif t == 1:
                bi = int(before1[pos])
                qg = int(s8b[bi])
                qpt = int(np.clip(np.round(wr[j] / max(scale_int8, 1e-12) + zp_int8), 0.0, 255.0))
                if qg != qpt:
                    miss["INT8"] += 1
            elif t == 2:
                bi = int(before2[pos])
                bb = int(s4b[bi // 2])
                qg = (bb & 0x0F) if (bi % 2 == 0) else ((bb >> 4) & 0x0F)
                qpt = int(np.clip(np.round(wr[j] / max(scale_int4, 1e-12) + zp_int4), 0.0, 15.0))
                if qg != qpt:
                    miss["INT4"] += 1
            else:
                bi = int(before3[pos])
                bb = int(strm2[bi // 4])
                qg = unpack_u2_le(bb, bi % 4)
                qpt = int(np.clip(np.round(wr[j] / max(scale_low, 1e-12) + zp_low), 0.0, 3.0))
                if qg != qpt:
                    miss["INT2"] += 1
        print(f"\nRow {r} quantized mismatches (GGUF decode vs PT re-round from same W):")
        for k, v in miss.items():
            print(f"  {k}: {v} / {cols}")

    # --- Step 2: dequant order + fakequant vs unpack for row 8192
    r0 = 8192 if 8192 < rows else min(want_rows)
    if r0 >= rows:
        r0 = rows - 1
    dense_unpack = unpack_v5_dense(pw_u8, pm, rows=rows, cols=cols, n2=n2, n4=n4, n8=n8, scale_low=sl, scale_int4=s4, scale_int8=s8, zp_low=zl, zp_int4=z4, zp_int8=z8, fp16_values=fv)
    row_u = dense_unpack[r0].astype(np.float64)
    row_fq = fakequant_row_numpy(wmat[r0], tier[r0], scale_int8=s8, zp_int8=z8, scale_int4=s4, zp_int4=z4, scale_low=sl, zp_low=zl)
    row_fq = row_fq.astype(np.float64)
    cos = float(np.dot(row_u, row_fq) / (np.linalg.norm(row_u) * np.linalg.norm(row_fq) + 1e-30))
    mad = float(np.max(np.abs(row_u - row_fq)))
    print(f"\nRow {r0}: cos(unpack_v5_dense, fakequant_row)={cos:.10f}  max_abs_diff={mad:.6g}")

    # Per-tier contribution to diff (same row)
    for tname, m in ("FP16", tier[r0] == 0), ("INT8", tier[r0] == 1), ("INT4", tier[r0] == 2), ("INT2", tier[r0] >= 3):
        if not np.any(m):
            continue
        d = np.abs(row_u[m] - row_fq[m])
        print(f"  tier {tname}: max_abs_diff={float(d.max()):.6g}  n={int(m.sum())}")

    if args.safetensors is not None:
        from safetensors import safe_open

        with safe_open(str(args.safetensors.resolve()), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            pref = f"transformer.layers.{L}.self_attn.in_proj"
            ref_key = next((k for k in keys if k.startswith(pref) and k.endswith("weight")), keys[0])
            wref = np.asarray(f.get_tensor(ref_key), dtype=np.float32)
        if wref.shape != wmat.shape:
            print(f"\n[safetensors] skip shape {wref.shape} != {wmat.shape}", file=sys.stderr)
        else:
            wr = wref[r0].astype(np.float64)
            print(
                f"\nOptional ref row {r0}: cos(unpack, ref)="
                f"{float(np.dot(row_u, wr) / (np.linalg.norm(row_u) * np.linalg.norm(wr) + 1e-30)):.6f} "
                f"cos(fakequant, ref)="
                f"{float(np.dot(row_fq, wr) / (np.linalg.norm(row_fq) * np.linalg.norm(wr) + 1e-30)):.6f}"
            )

    # Toy: INT dequant order
    qtest = np.array([3.0, 15.0, 200.0], dtype=np.float64)
    zps = np.array([zl, z4, z8], dtype=np.float64)
    sss = np.array([sl, s4, s8], dtype=np.float64)
    a = (qtest - zps) * sss
    b = sss * (qtest - zps)
    print(f"\nDequant order sanity (float64): max|a-b|={np.max(np.abs(a-b)):.3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
