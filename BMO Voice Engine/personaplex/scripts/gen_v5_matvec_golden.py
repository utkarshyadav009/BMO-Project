#!/usr/bin/env python3
"""Emit binary golden vectors for bmo_v5_runtime_test (Deliverable 2).

Loads ``bmo_temporal_half_cushion_max.pt`` (or --quant-ckpt) for tier masks +
per-layer quant params, dense FP16 weights from the same checkpoint state_dict,
applies MultiTierFakeQuantize (same as pt_fakequant_vs_fp16), then for each
tensor: x ~ N(0,1) with fixed RNG seed, y = W_fakequant @ x.

Usage (from personaplex repo root on server):
  python scripts/gen_v5_matvec_golden.py \\
    --pt bmo_temporal_half_cushion_max.pt \\
    --out build/v5_matvec_golden.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qat_septq import MultiTierFakeQuantize, resolve_multitier_quant_params  # noqa: E402

MAGIC = b"BMV5"
VERSION = 1

DEFAULT_MODULES: list[str] = [
    "transformer.layers.0.self_attn.in_proj_weight",
    "transformer.layers.15.gating.linear_in.weight",
    "transformer.layers.30.gating.linear_out.weight",
]


def _gguf_base_for_state_dict_key(module_key: str) -> str:
    """
    Map a PyTorch LM state_dict key to the corresponding GGUF tensor base name
    used inside `bmo_prepare_device_packed_tensors` (ctx.packed_registry).
    """
    # Expected formats:
    #   transformer.layers.{L}.self_attn.in_proj_weight
    #   transformer.layers.{L}.gating.linear_in.weight
    #   transformer.layers.{L}.gating.linear_out.weight
    parts = module_key.split(".")
    if len(parts) < 5:
        raise SystemExit(f"Unrecognized module key format: {module_key!r}")

    # Find the layer index right after "layers".
    try:
        layers_i = parts.index("layers")
        layer_idx = int(parts[layers_i + 1])
    except Exception as e:
        raise SystemExit(f"Could not parse layer index from {module_key!r}: {e}") from e

    if parts[-3:] == ["self_attn", "in_proj_weight", "weight"]:
        raise SystemExit(f"Unexpected key suffix duplication: {module_key!r}")

    # Determine module kind by matching known tail patterns.
    if module_key.endswith("self_attn.in_proj_weight"):
        return f"transformer_layers_{layer_idx}_self_attn_in_proj_weight"
    if module_key.endswith("gating.linear_in.weight"):
        return f"transformer_layers_{layer_idx}_gating_linear_in_weight"
    if module_key.endswith("gating.linear_out.weight"):
        return f"transformer_layers_{layer_idx}_gating_linear_out_weight"

    raise SystemExit(f"Unsupported module for matvec golden generation: {module_key!r}")


def _unwrap_sd(ckpt: dict) -> dict:
    sd = ckpt.get("state_dict")
    return sd if isinstance(sd, dict) else ckpt


def _tier_masks(ckpt: dict) -> dict[str, torch.Tensor]:
    raw = ckpt.get("tier_masks_uint2")
    if not isinstance(raw, dict) or not raw:
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            raw = sm.get("tier_masks_uint2")
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if torch.is_tensor(v)}


def _tier_masks_meta(ckpt: dict) -> dict[str, dict]:
    raw = ckpt.get("tier_masks_meta")
    if isinstance(raw, dict):
        return raw
    sm = ckpt.get("septq_meta")
    if isinstance(sm, dict):
        raw = sm.get("tier_masks_meta")
    return raw if isinstance(raw, dict) else {}


def _resolve_quant_ckpt(pt_path: Path, override: Path | None) -> dict:
    if override is not None:
        src = torch.load(str(override), map_location="cpu")
        if not isinstance(src, dict):
            raise SystemExit(f"--quant-ckpt must be a dict, got {type(src)}")
        return src
    payload = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"--pt must load to dict, got {type(payload)}")
    if _tier_masks(payload) and isinstance(payload.get("septq_meta"), dict):
        return payload
    qm = payload.get("qat_meta")
    rel = qm.get("source_student_quant_meta") if isinstance(qm, dict) else None
    if not isinstance(rel, str) or not rel.strip():
        raise SystemExit("Checkpoint missing tier_masks/septq_meta; pass --quant-ckpt")
    cand = (pt_path.parent / rel).resolve()
    if not cand.is_file():
        cand = Path(rel).resolve()
    if not cand.is_file():
        raise SystemExit(f"source_student_quant_meta not found: {rel}")
    src = torch.load(str(cand), map_location="cpu")
    if not isinstance(src, dict):
        raise SystemExit(f"quant ckpt must be dict: {cand}")
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path, required=True, help="Dense .pt (state_dict + optional qat_meta)")
    ap.add_argument("--quant-ckpt", type=Path, default=None, help="Override SEPTQ multitier dict (default: pt or qat_meta.source)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--modules",
        nargs="*",
        default=None,
        help=(
            "Optional list of PyTorch state_dict keys to test. "
            "Example: --modules transformer.layers.0.self_attn.in_proj_weight "
            "transformer.layers.1.gating.linear_in.weight ..."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pt_path = args.pt.resolve()
    quant_ckpt = _resolve_quant_ckpt(pt_path, args.quant_ckpt)

    tier_masks_uint2 = _tier_masks(quant_ckpt)
    tier_masks_meta = _tier_masks_meta(quant_ckpt)
    if not tier_masks_uint2:
        raise SystemExit("No tier_masks_uint2 in quant checkpoint")

    payload = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit("--pt must be a dict checkpoint")
    sd = _unwrap_sd(payload)

    rng = np.random.default_rng(int(args.seed))

    requested_modules: list[str] = args.modules if args.modules is not None and len(args.modules) > 0 else DEFAULT_MODULES
    tens: list[tuple[str, str]] = [(  # (gguf_base, state_dict_key)
        _gguf_base_for_state_dict_key(m),
        m,
    ) for m in requested_modules]

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(tens)))

        for gguf_base, module_name in tens:
            if module_name not in sd:
                raise SystemExit(f"Missing weight in --pt state_dict: {module_name!r}")
            if module_name not in tier_masks_uint2:
                raise SystemExit(f"Missing tier mask for {module_name!r}")

            w_dense = sd[module_name].detach().to(device="cpu", dtype=torch.float32)
            if w_dense.dim() != 2:
                raise SystemExit(f"{module_name}: expected 2D weight, got {tuple(w_dense.shape)}")

            packed_mask_ckpt = tier_masks_uint2[module_name].detach().cpu()
            raw_shape = tier_masks_meta.get(module_name, {}).get("shape", list(w_dense.shape))
            if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
                raise SystemExit(f"Bad tier_masks_meta.shape for {module_name}: {raw_shape!r}")
            target_shape = (int(raw_shape[0]), int(raw_shape[1]))
            if tuple(target_shape) != tuple(w_dense.shape):
                raise SystemExit(f"Shape mismatch mask {target_shape} vs W {tuple(w_dense.shape)}")

            needed_packed = (target_shape[0] * target_shape[1] + 3) // 4
            mask_flat = packed_mask_ckpt.to(dtype=torch.uint8).reshape(-1)
            if int(mask_flat.numel()) < int(needed_packed):
                raise SystemExit(f"packed mask too small for {module_name}")

            qp = resolve_multitier_quant_params(module_name, ckpt=quant_ckpt)
            fq = MultiTierFakeQuantize(
                tier_mask_packed=mask_flat,
                tier_mask_shape=target_shape,
                scale_int8=qp["scale_int8"],
                zero_point_int8=qp["zero_point_int8"],
                scale_int4=qp["scale_int4"],
                zero_point_int4=qp["zero_point_int4"],
                scale_int2=qp["scale_int2"],
                zero_point_int2=qp["zero_point_int2"],
            )
            w_fq = fq(w_dense).detach().to(dtype=torch.float32).numpy()
            rows, cols = int(w_fq.shape[0]), int(w_fq.shape[1])
            x = rng.standard_normal(cols, dtype=np.float32)
            y = w_fq @ x

            name_b = gguf_base.encode("utf-8")
            f.write(struct.pack("<I", len(name_b)))
            f.write(name_b)
            f.write(struct.pack("<ii", rows, cols))
            f.write(struct.pack("<I", cols))
            f.write(x.tobytes(order="C"))
            f.write(y.tobytes(order="C"))

    print(f"[gen_v5_matvec_golden] wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
