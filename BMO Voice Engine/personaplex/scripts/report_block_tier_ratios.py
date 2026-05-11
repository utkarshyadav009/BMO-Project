#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch


def _shape_from_meta(meta: Dict[str, Any], key: str, n_blocks_per_row: int) -> tuple[int, int]:
    item = meta.get(key, {})
    shape = item.get("shape") if isinstance(item, dict) else None
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    # Fallback if metadata is missing: assume full blocks.
    return 0, int(n_blocks_per_row * 32)


def main() -> None:
    p = argparse.ArgumentParser(description="Report per-tensor block-tier fractions from block_tier_map.")
    p.add_argument("ckpt", help="Path to SEPTQ checkpoint with top-level block_tier_map")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    block_tier_map = ckpt.get("block_tier_map")
    if not isinstance(block_tier_map, dict) or not block_tier_map:
        raise RuntimeError("Checkpoint missing top-level non-empty 'block_tier_map'")

    tier_masks_meta = ckpt.get("tier_masks_meta")
    if not isinstance(tier_masks_meta, dict):
        tier_masks_meta = {}

    rows: List[Dict[str, Any]] = []
    total_fp16 = 0
    total_int8 = 0
    total_int4 = 0
    total_int2 = 0
    total_elements = 0
    total_blocks = 0
    total_packed_bytes_est = 0.0

    for name, map_tensor in block_tier_map.items():
        t = map_tensor.detach().to(dtype=torch.uint8, device="cpu").contiguous()
        if t.ndim != 2:
            raise ValueError(f"{name}: expected block_tier_map shape (rows, n_blocks_per_row), got {tuple(t.shape)}")
        r, n_blocks_per_row = int(t.shape[0]), int(t.shape[1])
        rows_dim, cols_dim = _shape_from_meta(tier_masks_meta, name, n_blocks_per_row)
        if rows_dim > 0 and rows_dim != r:
            raise ValueError(f"{name}: row mismatch block_tier_map={r} tier_masks_meta={rows_dim}")

        fp16 = int8 = int4 = int2 = 0
        for row in range(r):
            for blk in range(n_blocks_per_row):
                start = blk * 32
                end = min(cols_dim, start + 32)
                n = max(0, end - start)
                tier = int(t[row, blk].item())
                if tier == 3:
                    fp16 += n
                elif tier == 2:
                    int8 += n
                elif tier == 1:
                    int4 += n
                elif tier == 0:
                    int2 += n
                else:
                    raise ValueError(f"{name}: invalid tier value {tier} (expected 0..3)")

        elems = fp16 + int8 + int4 + int2
        frac_fp16 = float(fp16 / max(1, elems))
        frac_int8 = float(int8 / max(1, elems))
        frac_int4 = float(int4 / max(1, elems))
        frac_int2 = float(int2 / max(1, elems))
        eff_bits = 16.0 * frac_fp16 + 8.0 * frac_int8 + 4.0 * frac_int4 + 2.0 * frac_int2
        packed_bytes_est = (eff_bits * elems) / 8.0

        rows.append(
            {
                "module_name": str(name),
                "rows": r,
                "n_blocks": int(r * n_blocks_per_row),
                "elements": int(elems),
                "frac_fp16": frac_fp16,
                "frac_int8": frac_int8,
                "frac_int4": frac_int4,
                "frac_int2": frac_int2,
                "eff_bits": eff_bits,
                "packed_mb_est": packed_bytes_est / (1024.0 * 1024.0),
                "flag": frac_fp16 > 0.30,
            }
        )
        total_fp16 += fp16
        total_int8 += int8
        total_int4 += int4
        total_int2 += int2
        total_elements += elems
        total_blocks += int(r * n_blocks_per_row)
        total_packed_bytes_est += packed_bytes_est

    rows.sort(key=lambda x: x["frac_fp16"], reverse=True)

    print(
        f"{'module_name':80s} {'n_blocks':>10s} {'frac_FP16':>10s} {'frac_INT8':>10s} "
        f"{'frac_INT4':>10s} {'frac_INT2':>10s} {'eff_bpw':>10s} {'packed_MB':>12s} {'flag':>6s}"
    )
    for r in rows:
        flag = "YES" if r["flag"] else "-"
        print(
            f"{r['module_name'][:80]:80s} {r['n_blocks']:10d} {r['frac_fp16']:10.4f} {r['frac_int8']:10.4f} "
            f"{r['frac_int4']:10.4f} {r['frac_int2']:10.4f} {r['eff_bits']:10.4f} {r['packed_mb_est']:12.3f} {flag:>6s}"
        )

    frac_fp16 = total_fp16 / max(1, total_elements)
    frac_int8 = total_int8 / max(1, total_elements)
    frac_int4 = total_int4 / max(1, total_elements)
    frac_int2 = total_int2 / max(1, total_elements)
    eff_bits_total = 16.0 * frac_fp16 + 8.0 * frac_int8 + 4.0 * frac_int4 + 2.0 * frac_int2
    print("-" * 170)
    print(
        f"{'MODEL_TOTAL':80s} {total_blocks:10d} {frac_fp16:10.4f} {frac_int8:10.4f} "
        f"{frac_int4:10.4f} {frac_int2:10.4f} {eff_bits_total:10.4f} {total_packed_bytes_est / (1024.0 * 1024.0):12.3f} {'-':>6s}"
    )
    print(
        f"\nModel totals: elements={total_elements} blocks={total_blocks} "
        f"estimated_size_MB={total_packed_bytes_est / (1024.0 * 1024.0):.3f}"
    )


if __name__ == "__main__":
    main()
