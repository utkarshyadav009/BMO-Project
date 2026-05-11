"""Inspect per-block tier-purity statistics on an existing SEPTQ multitier .pt.

Used as a 10-minute spike to predict how much information loss block-aligning
the per-element tier mask will introduce.

Reasoning:
- If most blocks are already (near-)pure (one dominant tier covers most of the
  block), block-aligning is nearly free. Path 3 lands z_s >= 0.93 with high
  probability.
- If blocks are highly mixed (entropy near uniform), QAT genuinely used
  per-element granularity to protect specific weights inside blocks. Block-
  aligning will hurt; expect z_s < 0.90 from PTQ-only.

What it computes per quantized tensor (and aggregated across tensors):
- Per-block dominant-tier purity:
    purity = max_count / block_size  (1.0 means all 32 elements share a tier)
- Per-block tier entropy in bits.
- Distribution of "blocks with N unique tiers" (N=1..4).
- Importance-score variance within blocks vs across blocks (ratio); only
  available if per_layer_stats carries per-element scores or a usable proxy.
- A predicted lower bound on "fraction of elements that would be re-tiered"
  if we collapse each block to its majority tier.

Run on the H100 box. No GPU needed. ~10 minutes for the full model.

Example:
    python scripts/inspect_mask_tier_purity.py \
        --septq-ckpt bmo_temporal_half_cushion_max.pt \
        --block-size 32 \
        --tensor transformer.layers.0.self_attn.in_proj_weight
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Iterable

import numpy as np
import torch

TIER_NAMES = {0: "FP16", 1: "INT8", 2: "INT4", 3: "INT2"}


def unpack_tier_mask_uint2(packed: torch.Tensor, total: int) -> torch.Tensor:
    """Mirror of qat_septq.unpack_tier_mask_uint2 for a flat layout."""
    flat = packed.detach().to(dtype=torch.uint8).reshape(-1)
    expanded = torch.zeros(flat.numel() * 4, dtype=torch.uint8)
    for lane in range(4):
        expanded[lane::4] = (flat >> (lane * 2)) & 0b11
    return expanded[:total]


def tier_counts_per_block(tier_flat: np.ndarray, block_size: int) -> np.ndarray:
    """Returns array of shape (n_blocks, 4) giving count per tier per block."""
    n = tier_flat.size
    n_blocks = (n + block_size - 1) // block_size

    pad = n_blocks * block_size - n
    if pad > 0:
        tier_flat = np.concatenate([tier_flat, np.full(pad, 255, dtype=tier_flat.dtype)])

    grid = tier_flat.reshape(n_blocks, block_size)
    counts = np.zeros((n_blocks, 4), dtype=np.int32)
    for t in range(4):
        counts[:, t] = (grid == t).sum(axis=1)
    return counts


def block_stats(counts: np.ndarray, block_size: int) -> dict:
    """Per-block summary metrics. counts: (n_blocks, 4)."""
    n_blocks = counts.shape[0]

    n_unique = (counts > 0).sum(axis=1)
    dominant_count = counts.max(axis=1)
    purity = dominant_count.astype(np.float64) / block_size

    p = counts.astype(np.float64) / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
        entropy = -(p * log_p).sum(axis=1)

    n_collapsed = block_size - dominant_count
    frac_retiered = n_collapsed.sum() / float(n_blocks * block_size)

    promoted_int8 = (counts[:, 0] > 0) & (counts[:, 0] < block_size)
    promoted_int4 = ((counts[:, 0] + counts[:, 1]) > 0) & (counts[:, 0] + counts[:, 1] < block_size)

    return {
        "n_blocks": int(n_blocks),
        "purity_mean": float(purity.mean()),
        "purity_median": float(np.median(purity)),
        "purity_p10": float(np.quantile(purity, 0.10)),
        "purity_p90": float(np.quantile(purity, 0.90)),
        "entropy_mean": float(entropy.mean()),
        "entropy_p90": float(np.quantile(entropy, 0.90)),
        "n_unique_dist": {k: int((n_unique == k).sum()) for k in (1, 2, 3, 4)},
        "frac_retiered_if_collapsed_to_majority": float(frac_retiered),
        "frac_blocks_with_fp16_element": float(promoted_int8.mean()),
        "frac_blocks_with_int8_or_better_element": float(promoted_int4.mean()),
        "tier_element_share": {
            TIER_NAMES[t]: float(counts[:, t].sum() / counts.sum())
            for t in range(4)
        },
    }


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def print_tensor_report(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"  n_blocks                                : {stats['n_blocks']:>10d}")
    print(f"  per-block dominant-tier purity (mean)   : {fmt_pct(stats['purity_mean'])}")
    print(f"  per-block dominant-tier purity (median) : {fmt_pct(stats['purity_median'])}")
    print(f"  per-block purity p10/p90                : {fmt_pct(stats['purity_p10'])} / {fmt_pct(stats['purity_p90'])}")
    print(f"  per-block tier entropy (bits, mean)     : {stats['entropy_mean']:6.3f}")
    print(f"  per-block tier entropy (bits, p90)      : {stats['entropy_p90']:6.3f}")
    print(f"  unique-tier distribution                : "
          + " ".join(f"{k}t={stats['n_unique_dist'][k]:>7d}" for k in (1, 2, 3, 4)))
    print(f"  fraction of elements re-tiered if collapsed to majority : {fmt_pct(stats['frac_retiered_if_collapsed_to_majority'])}")
    print(f"  blocks containing >=1 FP16 element      : {fmt_pct(stats['frac_blocks_with_fp16_element'])}")
    print(f"  blocks containing >=1 INT8-or-better el.: {fmt_pct(stats['frac_blocks_with_int8_or_better_element'])}")
    print(f"  tier element share                      : "
          + "  ".join(f"{n}={fmt_pct(s)}" for n, s in stats["tier_element_share"].items()))


def aggregate_global(per_tensor: list[tuple[str, dict, int]]) -> dict:
    if not per_tensor:
        return {}

    total_blocks = sum(s["n_blocks"] for _, s, _ in per_tensor)
    total_elements = sum(n_elem for _, _, n_elem in per_tensor)

    weighted_purity = sum(s["purity_mean"] * s["n_blocks"] for _, s, _ in per_tensor) / total_blocks
    weighted_entropy = sum(s["entropy_mean"] * s["n_blocks"] for _, s, _ in per_tensor) / total_blocks
    weighted_retiered = sum(s["frac_retiered_if_collapsed_to_majority"] * n_elem
                            for _, s, n_elem in per_tensor) / total_elements
    weighted_with_fp16 = sum(s["frac_blocks_with_fp16_element"] * s["n_blocks"] for _, s, _ in per_tensor) / total_blocks

    return {
        "n_tensors": len(per_tensor),
        "total_blocks": total_blocks,
        "total_elements": total_elements,
        "weighted_purity_mean": weighted_purity,
        "weighted_entropy_mean": weighted_entropy,
        "weighted_frac_retiered_if_collapsed_to_majority": weighted_retiered,
        "weighted_frac_blocks_with_fp16_element": weighted_with_fp16,
    }


def predict_path3_outcome(global_stats: dict) -> str:
    p = global_stats["weighted_purity_mean"]
    e = global_stats["weighted_entropy_mean"]
    r = global_stats["weighted_frac_retiered_if_collapsed_to_majority"]
    if p >= 0.85 and e <= 0.8 and r <= 0.15:
        return "STRONG GREEN. Block-aligning preserves >=85% of per-element tier choices on average. Expect PTQ-only z_s>=0.93."
    if p >= 0.65 and e <= 1.3 and r <= 0.30:
        return "YELLOW. Block-aligning re-tiers 15-30% of elements. PTQ-only z_s likely 0.85-0.93; QAT recovery probable."
    if p >= 0.50 and r <= 0.50:
        return "ORANGE. >=30% of elements forced off their per-element tier. PTQ-only z_s likely 0.80-0.88; QAT may help only partially."
    return "RED. Per-element tier choices are highly diverse within blocks. Path 3 will lose substantial accuracy. Consider Path 2 or smaller block_size."


def select_target_tensors(meta_lookup: dict, only_one: str | None) -> Iterable[str]:
    if only_one:
        if only_one in meta_lookup:
            return [only_one]
        sys.stderr.write(f"[warn] tensor '{only_one}' not in tier_masks_uint2; available examples:\n")
        for k in list(meta_lookup.keys())[:8]:
            sys.stderr.write(f"  - {k}\n")
        return []
    return list(meta_lookup.keys())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--septq-ckpt", default="bmo_temporal_half_cushion_max.pt",
                   help="Path to multitier SEPTQ .pt with tier_masks_uint2.")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--tensor", default=None,
                   help="If given, restrict analysis to a single tensor by full module name.")
    p.add_argument("--per-tensor", action="store_true",
                   help="Print per-tensor report (otherwise only global aggregates and a few samples).")
    p.add_argument("--top-mixed", type=int, default=8,
                   help="Print the top-N most-mixed tensors at the end.")
    args = p.parse_args()

    if not os.path.exists(args.septq_ckpt):
        sys.stderr.write(f"[ERROR] {args.septq_ckpt} not found\n")
        return 2

    print(f"[load] {args.septq_ckpt}")
    ckpt = torch.load(args.septq_ckpt, map_location="cpu", weights_only=False)
    masks = ckpt.get("tier_masks_uint2")
    if not isinstance(masks, dict):
        sys.stderr.write("[ERROR] tier_masks_uint2 missing or not a dict\n")
        return 2

    septq_meta = ckpt.get("septq_meta", {})
    pls = septq_meta.get("per_layer_stats", []) if isinstance(septq_meta, dict) else []
    shapes_by_name = {}
    for L in pls:
        for m in L.get("modules", []) if isinstance(L, dict) else []:
            n = m.get("name") or m.get("module_name") or m.get("module")
            te = m.get("total_elements")
            if n and isinstance(te, int):
                shapes_by_name[n] = te

    targets = select_target_tensors(masks, args.tensor)
    if not targets:
        return 2

    per_tensor: list[tuple[str, dict, int]] = []
    print(f"[analyze] {len(targets)} tensor(s), block_size={args.block_size}")

    for name in targets:
        packed = masks[name]
        if shapes_by_name.get(name):
            total = int(shapes_by_name[name])
        else:
            total = int(packed.numel()) * 4
            if total < 4:
                continue

        flat = unpack_tier_mask_uint2(packed, total).numpy()
        counts = tier_counts_per_block(flat, args.block_size)
        stats = block_stats(counts, args.block_size)
        per_tensor.append((name, stats, total))

        if args.per_tensor:
            print_tensor_report(name, stats)

    if not per_tensor:
        sys.stderr.write("[ERROR] no tensors analyzed\n")
        return 2

    if not args.per_tensor:
        sample = sorted(per_tensor, key=lambda x: x[1]["purity_mean"])
        print("\n=== sample tensors (least pure first) ===")
        for name, stats, _ in sample[:5]:
            print_tensor_report(name, stats)
        if len(sample) > 5:
            print("\n=== sample tensors (most pure last) ===")
            for name, stats, _ in sample[-3:]:
                print_tensor_report(name, stats)

    g = aggregate_global(per_tensor)
    print("\n" + "=" * 70)
    print("=== GLOBAL AGGREGATES (block-weighted across all targeted tensors) ===")
    print("=" * 70)
    print(f"  n_tensors              : {g['n_tensors']:>10d}")
    print(f"  total blocks           : {g['total_blocks']:>10d}")
    print(f"  total quantized elems  : {g['total_elements']:>10d}")
    print(f"  weighted purity (mean) : {fmt_pct(g['weighted_purity_mean'])}")
    print(f"  weighted entropy (bits): {g['weighted_entropy_mean']:6.3f}")
    print(f"  weighted frac re-tiered if collapsed to majority : {fmt_pct(g['weighted_frac_retiered_if_collapsed_to_majority'])}")
    print(f"  weighted frac blocks containing >=1 FP16 element : {fmt_pct(g['weighted_frac_blocks_with_fp16_element'])}")

    print("\n=== PATH-3 PREDICTION (heuristic, not a substitute for the spike) ===")
    print(f"  {predict_path3_outcome(g)}")

    worst = sorted(per_tensor, key=lambda x: x[1]["frac_retiered_if_collapsed_to_majority"], reverse=True)
    print(f"\n=== top {min(args.top_mixed, len(worst))} most-mixed tensors (highest re-tier fraction) ===")
    print(f"{'tensor':70s} | {'purity':>7s} | {'entropy':>7s} | {'re-tier':>8s}")
    print("-" * 100)
    for name, stats, _ in worst[: args.top_mixed]:
        print(f"{name[:70]:70s} | {fmt_pct(stats['purity_mean'])} | "
              f"{stats['entropy_mean']:7.3f} | {fmt_pct(stats['frac_retiered_if_collapsed_to_majority'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
