#!/usr/bin/env python3
"""Byte-for-byte compare PTQ ``tier_masks_uint2`` vs GGUF ``*.packed_mask`` (v5 per-element layout)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
except Exception as e:  # pragma: no cover
    print("ERROR: gguf import failed:", e, file=sys.stderr)
    sys.exit(2)


def _canonical_gguf_base(module_name: str) -> str:
    """Mirror ``export_bmo_gguf.canonical_transformer_multitier_gguf_base``."""
    import re

    u = module_name.replace(".", "_")
    return re.sub(r"^transformer_inner_layers_(\d+)_", r"transformer_layers_\1_", u)


def _read_gguf_tensor_bytes(gguf_path: Path, tensor_name: str) -> np.ndarray:
    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.name != tensor_name:
            continue
        data = getattr(tensor, "data", None)
        if data is None:
            raise RuntimeError(f"GGUF tensor {tensor_name!r} has no .data payload")
        return np.ascontiguousarray(np.frombuffer(data, dtype=np.uint8))
    raise KeyError(f"GGUF tensor not found: {tensor_name!r} in {gguf_path}")


def _read_gguf_scalar_i32(gguf_path: Path, tensor_name: str) -> int:
    arr = _read_gguf_tensor_bytes(gguf_path, tensor_name)
    if arr.size < 4:
        raise ValueError(f"scalar tensor {tensor_name!r} too small: {arr.size} bytes")
    return int(np.frombuffer(arr[:4], dtype="<i4")[0])


def dump_self_consistency(gguf_path: Path, base: str, *, max_rows: int) -> None:
    rows = _read_gguf_scalar_i32(gguf_path, f"{base}.rows")
    cols = _read_gguf_scalar_i32(gguf_path, f"{base}.cols")
    pm = _read_gguf_tensor_bytes(gguf_path, f"{base}.packed_mask")
    need_b = (rows * cols + 3) // 4
    if int(pm.size) != int(need_b):
        print(f"[dump] {base}: packed_mask bytes {pm.size} != expected {need_b}", file=sys.stderr)
        return

    n2 = n4 = n8 = n16 = 0
    print(f"[dump] {base}: rows={rows} cols={cols} (tier0=int(fp16) elems)")
    for r in range(rows):
        c2 = c4 = c8 = c16 = 0
        for c in range(cols):
            ei = r * cols + c
            tier = int((pm[ei // 4] >> ((ei % 4) * 2)) & 3)
            if tier == 0:
                c16 += 1
            elif tier == 1:
                c8 += 1
            elif tier == 2:
                c4 += 1
            else:
                c2 += 1
        n2 += c2
        n4 += c4
        n8 += c8
        n16 += c16
        if r < max_rows:
            s = c2 + c4 + c8 + c16
            ok = "OK" if s == cols else f"BAD(sum={s})"
            print(f"  row{r:02d}: c2={c2:5d} c4={c4:5d} c8={c8:5d} c16={c16:5d} sum={s:5d} {ok}")
    print(f"[dump] totals: tier3(int2)={n2} tier2(int4)={n4} tier1(int8)={n8} tier0(fp16)={n16} all={n2+n4+n8+n16}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--septq-ckpt", required=True, help="PTQ checkpoint with tier_masks_uint2")
    p.add_argument("--gguf", required=True, help="Exported v5 GGUF path")
    p.add_argument("--max-modules", type=int, default=0, help="If >0, only check first N modules (debug)")
    p.add_argument(
        "--dump-gguf-base",
        default=None,
        help="Optional GGUF tensor base (e.g. transformer_layers_0_self_attn_in_proj_weight) to print per-row tier counts",
    )
    p.add_argument("--dump-max-rows", type=int, default=12)
    args = p.parse_args()

    ckpt_path = Path(args.septq_ckpt)
    gguf_path = Path(args.gguf)
    if not ckpt_path.is_file():
        print("ERROR: septq ckpt not found:", ckpt_path, file=sys.stderr)
        return 2
    if not gguf_path.is_file():
        print("ERROR: gguf not found:", gguf_path, file=sys.stderr)
        return 2

    ck = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ck, dict):
        print("ERROR: checkpoint must be a dict", file=sys.stderr)
        return 2

    tier_masks = ck.get("tier_masks_uint2")
    if not isinstance(tier_masks, dict) or not tier_masks:
        sm = ck.get("septq_meta")
        if isinstance(sm, dict):
            tier_masks = sm.get("tier_masks_uint2")
    if not isinstance(tier_masks, dict) or not tier_masks:
        print("ERROR: tier_masks_uint2 missing/empty", file=sys.stderr)
        return 2

    modules = list(tier_masks.keys())
    if args.max_modules and args.max_modules > 0:
        modules = modules[: int(args.max_modules)]

    if args.dump_gguf_base:
        dump_self_consistency(gguf_path, str(args.dump_gguf_base), max_rows=int(args.dump_max_rows))

    ok = 0
    bad = 0
    for mod in modules:
        t = tier_masks.get(mod)
        if not torch.is_tensor(t):
            print(f"MISMATCH {mod}: ckpt mask is not a tensor ({type(t)})")
            bad += 1
            continue
        ck_bytes = t.detach().cpu().contiguous().view(-1)
        if ck_bytes.dtype != torch.uint8:
            ck_bytes = ck_bytes.to(torch.uint8)
        ck_np = np.ascontiguousarray(ck_bytes.numpy().astype(np.uint8, copy=False))

        base = _canonical_gguf_base(mod)
        key = f"{base}.packed_mask"
        try:
            gg_np = _read_gguf_tensor_bytes(gguf_path, key)
        except KeyError as e:
            print(f"MISMATCH {mod}: {e}")
            bad += 1
            continue

        if gg_np.shape != ck_np.shape or not np.array_equal(gg_np, ck_np):
            print(f"MISMATCH {mod}: bytes differ (ckpt={ck_np.nbytes} gguf={gg_np.nbytes})")
            n = min(32, ck_np.size, gg_np.size)
            print("  ckpt[:n]=", ck_np[:n].tobytes().hex())
            print("  gguf[:n]=", gg_np[:n].tobytes().hex())
            bad += 1
            continue

        print(f"MATCH {mod}")
        ok += 1

    print(f"\nSUMMARY: MATCH={ok} MISMATCH={bad} TOTAL={ok + bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
