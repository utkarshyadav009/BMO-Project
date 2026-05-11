#!/usr/bin/env python3
"""Compare per-module cos/mse between two SEPTQ multitier checkpoints (mask modes)."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected dict checkpoint, got {type(obj)}: {path}")
    return obj


def _module_index(ckpt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sm = ckpt.get("septq_meta")
    if not isinstance(sm, dict):
        raise SystemExit("Checkpoint missing septq_meta dict")
    layers = sm.get("per_layer_stats")
    if not isinstance(layers, list):
        raise SystemExit("septq_meta.per_layer_stats must be a non-empty list")

    out: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_idx = int(layer.get("layer_idx", -1))
        modules = layer.get("modules")
        if not isinstance(modules, list):
            continue
        for m in modules:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name", ""))
            if not name:
                continue
            out[name] = {
                "layer_idx": layer_idx,
                "cos": float(m.get("cos", 0.0)),
                "mse": float(m.get("mse", 0.0)),
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare cos/mse per module between per-element and block-aligned SEPTQ runs."
    )
    p.add_argument(
        "--ckpt-pe",
        type=str,
        default="bmo_temporal_half_cushion_max.pt",
        help="Baseline PTQ checkpoint (per-element tier mask).",
    )
    p.add_argument(
        "--ckpt-ba",
        type=str,
        default="bmo_temporal_blockaligned.pt",
        help="Block-aligned tier mask PTQ checkpoint.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    pe_path = Path(args.ckpt_pe)
    ba_path = Path(args.ckpt_ba)
    if not pe_path.is_absolute():
        pe_path = (root / pe_path).resolve()
    if not ba_path.is_absolute():
        ba_path = (root / ba_path).resolve()

    pe = _load(pe_path)
    ba = _load(ba_path)

    idx_pe = _module_index(pe)
    idx_ba = _module_index(ba)

    only_pe = sorted(set(idx_pe) - set(idx_ba))
    only_ba = sorted(set(idx_ba) - set(idx_pe))
    if only_pe:
        print(f"[WARN] modules only in --ckpt-pe ({len(only_pe)}): {only_pe[:3]}...", file=sys.stderr)
    if only_ba:
        print(f"[WARN] modules only in --ckpt-ba ({len(only_ba)}): {only_ba[:3]}...", file=sys.stderr)

    common = sorted(
        set(idx_pe) & set(idx_ba),
        key=lambda n: (idx_pe[n]["layer_idx"], n),
    )
    if not common:
        print("[ERROR] no overlapping module names between checkpoints", file=sys.stderr)
        sys.exit(2)

    rows: list[tuple[int, str, float, float, float, float, float, float]] = []
    dcos_list: list[float] = []
    for name in common:
        layer_idx = int(idx_pe[name]["layer_idx"])
        cos_pe = float(idx_pe[name]["cos"])
        cos_ba = float(idx_ba[name]["cos"])
        mse_pe = float(idx_pe[name]["mse"])
        mse_ba = float(idx_ba[name]["mse"])
        dcos = cos_ba - cos_pe
        if mse_pe > 0.0:
            dmse_ratio = (mse_ba - mse_pe) / mse_pe
        else:
            dmse_ratio = float("inf") if mse_ba > 0.0 else 0.0
        rows.append((layer_idx, name, cos_pe, cos_ba, dcos, mse_pe, mse_ba, dmse_ratio))
        dcos_list.append(dcos)

    print(
        f"{'layer':>5}  {'module':<72}  {'cos_PE':>9}  {'cos_BA':>9}  "
        f"{'Δcos':>9}  {'mse_PE':>11}  {'mse_BA':>11}  {'Δmse_ratio':>12}"
    )
    print("-" * 160)
    for layer_idx, name, cos_pe, cos_ba, dcos, mse_pe, mse_ba, dmse_ratio in rows:
        short = name if len(name) <= 72 else name[:69] + "..."
        dr = f"{dmse_ratio:.4f}" if dmse_ratio != float("inf") else "inf"
        print(
            f"{layer_idx:5d}  {short:<72}  {cos_pe:9.6f}  {cos_ba:9.6f}  "
            f"{dcos:9.6f}  {mse_pe:11.4e}  {mse_ba:11.4e}  {dr:>12}"
        )

    mean_dcos = float(sum(dcos_list) / len(dcos_list))
    median_dcos = float(statistics.median(dcos_list))
    max_cos_drop = float(min(dcos_list))
    concerning = sum(1 for d in dcos_list if d < -0.005)

    print("-" * 160)
    print(
        f"Summary (n={len(dcos_list)} paired modules): "
        f"mean Δcos={mean_dcos:.6f}  median Δcos={median_dcos:.6f}  "
        f"max cos drop (min Δcos)={max_cos_drop:.6f}  "
        f"count(Δcos < -0.005)={concerning}"
    )

    gate_ok = median_dcos > -0.005
    if gate_ok:
        print("[RESULT] GATE median_Δcos > -0.005  ->  PASS (proceed toward QAT retrain)", file=sys.stderr)
        sys.exit(0)
    print(
        "[RESULT] GATE median_Δcos <= -0.005  ->  FAIL (block-aligned regression too large)",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
