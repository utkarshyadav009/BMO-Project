#!/usr/bin/env python3
"""
Audit a SEPTQ/QAT checkpoint without running the model.

This answers three concrete questions:
  1. Did the checkpoint actually contain SEPTQ tier masks and metadata?
  2. Which temporal matrices were covered or missed?
  3. What memory footprint should the packed GGUF representation have?
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


DEFAULT_CKPT = Path("qat_septq_final_run/qat_best.pt")

EXPECTED_LAYERS = 32
EXPECTED_KINDS = {
    "self_attn.in_proj": "self_attn.in_proj_weight",
    "self_attn.out_proj": "self_attn.out_proj.weight",
    "gating.linear_in": "gating.linear_in.weight",
    "gating.linear_out": "gating.linear_out.weight",
}
EXPECTED_PACKED_KINDS = {
    "self_attn.in_proj": "self_attn.in_proj_weight",
    "gating.linear_in": "gating.linear_in.weight",
    "gating.linear_out": "gating.linear_out.weight",
}


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def fmt_gib(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 3):.2f} GiB"


def fmt_mib(nbytes: int) -> str:
    return f"{nbytes / (1024 ** 2):.2f} MiB"


def load_checkpoint(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}")
    return obj


def resolve_maybe_relative(path_text: str, base_dir: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    cwd_candidate = p.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / p).resolve()


def print_payload_summary(ckpt: dict[str, Any], path: Path) -> Path | None:
    keys = sorted(str(k) for k in ckpt.keys())
    print(f"[verify_septq] payload keys: {keys}")
    print(f"[verify_septq] model_mode: {ckpt.get('model_mode')}")
    print(f"[verify_septq] force_dense: {ckpt.get('force_dense')}")

    qat_meta = ckpt.get("qat_meta")
    source_quant_path = None
    if isinstance(qat_meta, dict):
        source = qat_meta.get("source_student_quant_meta")
        print(f"[verify_septq] qat_meta.quant_scheme: {qat_meta.get('quant_scheme')}")
        print(f"[verify_septq] qat_meta.source_student_quant_meta: {source}")
        if isinstance(source, str) and source:
            source_quant_path = resolve_maybe_relative(source, path.parent)

    septq_meta = ckpt.get("septq_meta")
    if isinstance(septq_meta, dict):
        for key in (
            "quant_scheme",
            "selection_mode",
            "quantize_layers_arg",
            "selected_temporal_layers",
            "skip_first_n_temporal",
            "skip_last_n_temporal",
            "skip_modules_filters",
            "excluded_modules",
            "quantized_elements_total",
            "estimated_weight_gib",
            "tier_mask_total_gib",
        ):
            if key in septq_meta:
                print(f"[verify_septq] septq_meta.{key}: {septq_meta.get(key)}")

    if ckpt.get("model_mode") == "qat_septq_dense":
        print(
            "[verify_septq] NOTE: this checkpoint is a dense QAT export. "
            "It is expected to contain no tier_masks_uint2/septq_meta payload."
        )
    elif isinstance(septq_meta, dict) and ckpt.get("force_dense") is True:
        print(
            "[verify_septq] NOTE: force_dense=True means the state_dict remains loadable as dense weights; "
            "this SEPTQ payload can still carry tier masks separately."
        )
    return source_quant_path


def get_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    sd = ckpt.get("state_dict")
    if isinstance(sd, dict):
        return {str(k): v for k, v in sd.items() if torch.is_tensor(v)}
    return {str(k): v for k, v in ckpt.items() if torch.is_tensor(v)}


def get_tier_masks(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw = ckpt.get("tier_masks_uint2")
    if not isinstance(raw, dict) or not raw:
        septq_meta = ckpt.get("septq_meta")
        raw = septq_meta.get("tier_masks_uint2") if isinstance(septq_meta, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(k): v.detach().cpu().to(torch.uint8).reshape(-1) for k, v in raw.items() if torch.is_tensor(v)}


def get_tier_masks_meta(ckpt: dict[str, Any]) -> dict[str, Any]:
    raw = ckpt.get("tier_masks_meta")
    if isinstance(raw, dict):
        return raw
    septq_meta = ckpt.get("septq_meta")
    raw = septq_meta.get("tier_masks_meta") if isinstance(septq_meta, dict) else None
    return raw if isinstance(raw, dict) else {}


def get_module_meta(ckpt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    septq_meta = ckpt.get("septq_meta")
    if not isinstance(septq_meta, dict):
        return out
    per_layer_stats = septq_meta.get("per_layer_stats")
    if not isinstance(per_layer_stats, list):
        return out
    for layer in per_layer_stats:
        if not isinstance(layer, dict):
            continue
        modules = layer.get("modules")
        if not isinstance(modules, list):
            continue
        for mod in modules:
            if isinstance(mod, dict) and isinstance(mod.get("name"), str):
                out[str(mod["name"])] = mod
    return out


def normalize_weight_name(name: str) -> str:
    text = str(name)
    for suffix in (".weight", ".q_weight", ".dense_weight", ".orig_weight"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def candidates_for_weight_key(mask_key: str) -> list[str]:
    cands = [mask_key]
    if not mask_key.endswith(".weight"):
        cands.append(mask_key + ".weight")
    cands.extend([
        mask_key + ".q_weight",
        mask_key + ".dense_weight",
        mask_key + ".orig_weight",
    ])
    return list(dict.fromkeys(cands))


def find_weight_tensor(mask_key: str, state_dict: dict[str, torch.Tensor]) -> tuple[str | None, torch.Tensor | None]:
    for cand in candidates_for_weight_key(mask_key):
        t = state_dict.get(cand)
        if t is not None and t.ndim >= 2:
            return cand, t
    base = normalize_weight_name(mask_key)
    for key, t in state_dict.items():
        if normalize_weight_name(key) == base and t.ndim >= 2:
            return key, t
    return None, None


def unpack_uint2_counts(packed: torch.Tensor, numel: int) -> list[int]:
    vals = []
    for shift in (0, 2, 4, 6):
        vals.append(((packed >> shift) & 0x3).reshape(-1))
    expanded = torch.stack(vals, dim=1).reshape(-1)[:numel]
    counts = torch.bincount(expanded.to(torch.long), minlength=4)
    return [int(x) for x in counts.tolist()[:4]]


def expected_names(layer: int, kinds: dict[str, str]) -> list[str]:
    return [f"transformer.layers.{layer}.{suffix}" for suffix in kinds.values()]


def is_expected_temporal_key(key: str) -> bool:
    return bool(re.match(r"^transformer\.layers\.\d+\.", key))


def estimate_packed_bytes(numel: int, counts: list[int], include_mask: bool = True) -> dict[str, int]:
    fp16_count, int8_count, int4_count, int2_count = counts
    q2_bytes = (int2_count + 3) // 4
    q4_bytes = (int4_count + 1) // 2
    q8_bytes = int8_count
    fp16_indices_bytes = fp16_count * 4
    fp16_values_bytes = fp16_count * 2
    mask_bytes = (numel + 3) // 4 if include_mask else 0
    total = q2_bytes + q4_bytes + q8_bytes + fp16_indices_bytes + fp16_values_bytes + mask_bytes
    return {
        "total": total,
        "packed_weights": q2_bytes + q4_bytes + q8_bytes,
        "packed_mask": mask_bytes,
        "fp16_indices": fp16_indices_bytes,
        "fp16_values": fp16_values_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify whether a SEPTQ checkpoint really contains valid multi-tier masks.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--layers", type=int, default=EXPECTED_LAYERS)
    parser.add_argument("--sample", type=int, default=12, help="number of largest packed modules to print")
    parser.add_argument(
        "--follow-source",
        action="store_true",
        help="if this is a dense QAT export, audit qat_meta.source_student_quant_meta instead",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"[verify_septq] checkpoint not found: {args.checkpoint}")

    print(f"[verify_septq] loading {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint)
    source_quant_path = print_payload_summary(ckpt, args.checkpoint)
    if args.follow_source and not get_tier_masks(ckpt) and source_quant_path is not None:
        if source_quant_path.exists():
            print(f"[verify_septq] following source_student_quant_meta -> {source_quant_path}")
            ckpt = load_checkpoint(source_quant_path)
            source_quant_path = print_payload_summary(ckpt, source_quant_path)
        else:
            print(f"[verify_septq] source_student_quant_meta not found on disk: {source_quant_path}")

    state_dict = get_state_dict(ckpt)
    masks = get_tier_masks(ckpt)
    mask_meta = get_tier_masks_meta(ckpt)
    module_meta = get_module_meta(ckpt)

    print(f"[verify_septq] state_dict tensors: {len(state_dict)}")
    print(f"[verify_septq] tier_masks_uint2 entries: {len(masks)}")
    print(f"[verify_septq] module metadata entries: {len(module_meta)}")

    missing_pack_expected = []
    missing_out_proj = []
    for layer in range(args.layers):
        for name in expected_names(layer, EXPECTED_PACKED_KINDS):
            if name not in masks:
                missing_pack_expected.append(name)
        for name in expected_names(layer, {"self_attn.out_proj": EXPECTED_KINDS["self_attn.out_proj"]}):
            if name not in masks:
                missing_out_proj.append(name)

    if missing_pack_expected:
        print(f"[verify_septq] MISSING expected packed temporal masks: {len(missing_pack_expected)}")
        for name in missing_pack_expected[:80]:
            print(f"  missing {name}")
    else:
        print("[verify_septq] expected in_proj/gating coverage: COMPLETE")

    if missing_out_proj:
        print(f"[verify_septq] out_proj masks absent: {len(missing_out_proj)} / {args.layers}")
        print("  note: this matches the current QAT code path, which excludes self_attn.out_proj")
    else:
        print("[verify_septq] out_proj coverage: COMPLETE")

    total_dense_f16 = 0
    total_packed = 0
    total_components: dict[str, int] = defaultdict(int)
    total_counts = [0, 0, 0, 0]
    invalid = []
    audited = []

    for mask_key, packed_mask in sorted(masks.items()):
        weight_key, weight = find_weight_tensor(mask_key, state_dict)
        raw_shape = None
        if isinstance(mask_meta.get(mask_key), dict):
            raw_shape = mask_meta[mask_key].get("shape")
        if raw_shape is not None and len(raw_shape) >= 2:
            rows, cols = int(raw_shape[0]), int(raw_shape[1])
            numel = rows * cols
        elif weight is not None:
            numel = int(weight.numel())
            rows = int(weight.shape[0])
            cols = int(weight.numel() // max(1, rows))
        else:
            invalid.append((mask_key, "missing matching weight tensor and shape metadata"))
            continue

        expected_mask_bytes = (numel + 3) // 4
        if int(packed_mask.numel()) < expected_mask_bytes:
            invalid.append((mask_key, f"mask too small: have={packed_mask.numel()} need={expected_mask_bytes}"))
            continue

        counts = unpack_uint2_counts(packed_mask[:expected_mask_bytes], numel)
        if sum(counts) != numel:
            invalid.append((mask_key, f"tier counts sum mismatch: counts={counts} numel={numel}"))
            continue

        packed_est = estimate_packed_bytes(numel, counts)
        dense_f16 = numel * 2
        total_dense_f16 += dense_f16
        total_packed += packed_est["total"]
        for k, v in packed_est.items():
            if k != "total":
                total_components[k] += v
        for i, c in enumerate(counts):
            total_counts[i] += c

        audited.append({
            "mask_key": mask_key,
            "weight_key": weight_key,
            "numel": numel,
            "shape": (rows, cols),
            "counts": counts,
            "dense_f16": dense_f16,
            "packed": packed_est["total"],
            "compression": dense_f16 / max(1, packed_est["total"]),
        })

    print(f"[verify_septq] audited packed modules: {len(audited)}")
    if invalid:
        print(f"[verify_septq] INVALID packed entries: {len(invalid)}")
        for name, reason in invalid[:40]:
            print(f"  invalid {name}: {reason}")
    else:
        print("[verify_septq] mask integrity: PASS")

    print("[verify_septq] tier counts across audited masks:")
    total_elems = max(1, sum(total_counts))
    labels = ["fp16/outlier", "int8", "int4", "int2/low"]
    for label, count in zip(labels, total_counts):
        print(f"  {label:<13} {count:14d}  {100.0 * count / total_elems:7.3f}%")

    print("[verify_septq] theoretical GGUF packed footprint from checkpoint masks:")
    print(f"  dense f16 equivalent: {fmt_gib(total_dense_f16)}")
    print(f"  packed estimate:      {fmt_gib(total_packed)}")
    if total_packed > 0:
        print(f"  compression:          {total_dense_f16 / total_packed:.2f}x")
    for k, v in sorted(total_components.items(), key=lambda item: item[1], reverse=True):
        print(f"  {k:<15} {fmt_gib(v):>9}")

    print(f"[verify_septq] largest {args.sample} packed modules by estimated packed bytes:")
    for item in sorted(audited, key=lambda x: x["packed"], reverse=True)[: args.sample]:
        counts = item["counts"]
        print(
            f"  {fmt_mib(item['packed']):>11}  comp={item['compression']:5.2f}x  "
            f"shape={item['shape'][0]}x{item['shape'][1]}  "
            f"tiers={counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}  "
            f"{item['mask_key']}"
        )

    temporal_mask_count = sum(1 for key in masks if is_expected_temporal_key(key))
    print("[verify_septq] verdict:")
    if len(audited) > 0 and not invalid:
        print("  YES: checkpoint contains valid uint2 SEPTQ masks and tier metadata.")
    else:
        print("  NO: checkpoint masks/metadata are missing or invalid.")
    if missing_pack_expected:
        print("  BUT: expected temporal coverage is incomplete.")
    if missing_out_proj:
        print("  ALSO: self_attn.out_proj is not currently in the packed target set.")
    print(f"  temporal mask entries seen: {temporal_mask_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
