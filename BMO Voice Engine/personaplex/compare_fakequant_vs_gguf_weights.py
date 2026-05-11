#!/usr/bin/env python3
"""compare_fakequant_vs_gguf_weights.py

Compare dequantized weights: PyTorch MultiTierFakeQuantize (same wiring as
pt_fakequant_vs_fp16.py) vs GGUF block unpack (vectorized mirror of
bmo_compute.cpp::unpack_layer_to_f32_blockwise, via convert_septq_to_fp16.py).

C++ vs Python unpack: as of this script, convert_septq_to_fp16.unpack_layer_to_f32_blockwise
matches bmo_compute.cpp (unpack_u2_le, mask byte = block_idx/4 lane block_idx%4,
tier 0=FP16 block, 1=int8, 2=int4 low nibble first, 3=2-bit 4-per-byte LE,
dequant (q - zp) * scale). No divergence noted.

Notable (export path): export_bmo_gguf.create_packed_layer accepts packed_mask_tensor
but does not use it — GGUF block tiers are derived from per-block max-abs + thresholds
or ratio metadata. tier_masks_uint2 in the student .pt is per *element* for QAT fake
quant. If export ignored a student block mask, w_pt_fq (element mask) can disagree
with w_gguf_dq (GGUF block mask) even when both unpack implementations are correct.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qat_septq import MultiTierFakeQuantize, resolve_multitier_quant_params

# Mirror pt_fakequant_vs_fp16.resolve_quant_sources (helpers not exported from that module).
from convert_septq_to_fp16 import (  # noqa: E402
    dequant_one_base,
    _get_numpy,
    _tensor_dict,
    pick_tensor,
    read_scalar_i32,
    read_scalar_f32,
)

_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_GGUF_PY = _REPO_ROOT / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

import gguf  # type: ignore  # noqa: E402


def _err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def _unwrap_state_dict(ckpt: dict[str, Any]) -> dict[str, Any]:
    sd = ckpt.get("state_dict")
    if isinstance(sd, dict):
        return sd
    return ckpt


def _resolve_path(base: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    cand = path.resolve()
    if cand.exists():
        return cand
    return (base / path).resolve()


def _tier_masks_from_ckpt(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw = ckpt.get("tier_masks_uint2")
    if not isinstance(raw, dict) or not raw:
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            raw = sm.get("tier_masks_uint2")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    return out


def _tier_masks_meta_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any]:
    raw = ckpt.get("tier_masks_meta")
    if isinstance(raw, dict):
        return raw
    sm = ckpt.get("septq_meta")
    if isinstance(sm, dict):
        raw = sm.get("tier_masks_meta")
    return raw if isinstance(raw, dict) else {}


def _septq_meta_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any] | None:
    sm = ckpt.get("septq_meta")
    return sm if isinstance(sm, dict) else None


def resolve_quant_checkpoint(
    qat_ckpt: dict[str, Any],
    qat_path: Path,
    septq_ckpt_path: Path | None,
) -> dict[str, Any]:
    """Quant metadata source: same rules as pt_fakequant_vs_fp16.resolve_quant_sources."""
    if septq_ckpt_path is not None:
        src = torch.load(str(septq_ckpt_path), map_location="cpu")
        if not isinstance(src, dict):
            raise ValueError(f"SEPTQ checkpoint must be a dict, got {type(src)}")
        tier = _tier_masks_from_ckpt(src)
        if not tier:
            raise ValueError(f"{septq_ckpt_path}: missing tier_masks_uint2")
        sm = _septq_meta_from_ckpt(src)
        if not sm or not isinstance(sm.get("per_layer_stats"), list):
            raise ValueError(f"{septq_ckpt_path}: missing septq_meta.per_layer_stats")
        return src

    tier = _tier_masks_from_ckpt(qat_ckpt)
    sm = _septq_meta_from_ckpt(qat_ckpt)
    if tier and sm and isinstance(sm.get("per_layer_stats"), list):
        return qat_ckpt

    qm = qat_ckpt.get("qat_meta")
    rel = None
    if isinstance(qm, dict):
        rel = qm.get("source_student_quant_meta")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(
            "qat_meta.source_student_quant_meta is missing or not a non-empty string; "
            "pass --septq-ckpt explicitly."
        )
    src_path = _resolve_path(qat_path.parent, rel)
    if not src_path.is_file():
        raise ValueError(f"source_student_quant_meta path not found: {src_path}")
    src = torch.load(str(src_path), map_location="cpu")
    if not isinstance(src, dict):
        raise ValueError(f"Source quant file must be a dict, got {type(src)}")
    tier = _tier_masks_from_ckpt(src)
    if not tier:
        raise ValueError(f"{src_path}: missing tier_masks_uint2")
    sm = _septq_meta_from_ckpt(src)
    if not sm or not isinstance(sm.get("per_layer_stats"), list):
        raise ValueError(f"{src_path}: missing septq_meta.per_layer_stats")
    return src


def _parse_qkv(s: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("--qkv-split must be three comma-separated integers")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _tier_per_block_numpy(packed_mask: np.ndarray, n_blocks: int) -> np.ndarray:
    pm = np.asarray(packed_mask).ravel().view(np.uint8)
    n_mask_bytes = (n_blocks + 3) // 4
    pm_b = pm[:n_mask_bytes]
    pm_rep = np.repeat(pm_b, 4)[:n_blocks]
    shifts = np.tile(np.array([0, 2, 4, 6], dtype=np.uint8), n_mask_bytes)[:n_blocks]
    return ((pm_rep >> shifts) & np.uint8(0x3)).astype(np.int64)


def _cosine_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    na = float(torch.linalg.norm(a))
    nb = float(torch.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def _stats_pair_with_bs(
    w_pt: torch.Tensor,
    w_gg: torch.Tensor,
    elem_mask_1d: torch.Tensor,
    block_size: int,
) -> dict[str, float | int]:
    m = elem_mask_1d.to(dtype=torch.bool)
    a = w_pt.flatten()[m]
    b = w_gg.flatten()[m]
    n = int(a.numel())
    if n == 0:
        return {
            "cosine": float("nan"),
            "max_abs_err": 0.0,
            "mean_abs_err": 0.0,
            "rmse": 0.0,
            "n_elem": 0,
            "n_blocks": 0,
        }
    diff = (a - b).abs()
    rmse = float(torch.sqrt(torch.mean((a - b) ** 2)).item())
    total = int(w_pt.numel())
    flat_idx = torch.arange(total, dtype=torch.int64)[m]
    bid = flat_idx // int(block_size)
    n_blocks = int(torch.unique(bid).numel())
    return {
        "cosine": _cosine_torch(a, b),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "rmse": rmse,
        "n_elem": n,
        "n_blocks": n_blocks,
    }


def _row_mask(w: torch.Tensor, cols: int, row_lo: int, row_hi: int) -> torch.Tensor:
    rows = int(w.shape[0])
    dev = w.device
    r = torch.arange(rows * cols, device=dev, dtype=torch.int64)
    rr = r // cols
    return (rr >= row_lo) & (rr < row_hi)


def _fmt_row(name: str, st: dict[str, float | int], w: int = 28) -> str:
    cos = st["cosine"]
    cos_s = f"{cos:.6f}" if math.isfinite(float(cos)) else "nan"
    return (
        f"{name:<{w}} | {cos_s:>8} | {st['max_abs_err']:.6e} | {st['mean_abs_err']:.6e} | "
        f"{st['rmse']:.6e} | {int(st['n_elem']):>8} | {int(st['n_blocks']):>6}"
    )


def _first_mismatch_lines(
    w_pt: torch.Tensor,
    w_gg: torch.Tensor,
    *,
    tier_pb: np.ndarray,
    block_size: int,
    rows: int,
    cols: int,
    threshold: float,
    packed_weights: np.ndarray,
    packed_mask: np.ndarray,
    n_2bit_bytes: int,
    n_4bit_bytes: int,
    n_8bit_bytes: int,
    scale_low: float,
    scale_int4: float,
    scale_int8: float,
    fp16_values: np.ndarray,
) -> list[str]:
    total = rows * cols
    n_blocks = int(tier_pb.shape[0])
    pw = np.asarray(packed_weights).ravel().view(np.uint8)
    stream2 = pw[:n_2bit_bytes]
    stream4 = pw[n_2bit_bytes : n_2bit_bytes + n_4bit_bytes]
    stream8 = pw[n_2bit_bytes + n_4bit_bytes : n_2bit_bytes + n_4bit_bytes + n_8bit_bytes]

    is0 = tier_pb == 0
    is1 = tier_pb == 1
    is2 = tier_pb == 2
    is3 = tier_pb == 3

    def _excl_cumsum(mask: np.ndarray) -> np.ndarray:
        cs = np.cumsum(mask.astype(np.int64))
        return cs - mask.astype(np.int64)

    c16_blocks = _excl_cumsum(is0)
    c8_blocks = _excl_cumsum(is1)
    c4_blocks = _excl_cumsum(is2)
    c2_blocks = _excl_cumsum(is3)

    w_pt_np = w_pt.detach().cpu().numpy().reshape(-1)
    w_gg_np = w_gg.detach().cpu().numpy().reshape(-1)

    for block_idx in range(n_blocks):
        lo = block_idx * block_size
        hi = min(lo + block_size, total)
        sl_pt = w_pt_np[lo:hi]
        sl_gg = w_gg_np[lo:hi]
        mad = float(np.max(np.abs(sl_pt - sl_gg))) if hi > lo else 0.0
        if mad > threshold:
            tier = int(tier_pb[block_idx])
            mbyte = int(np.asarray(packed_mask).ravel().view(np.uint8)[block_idx // 4])
            scale_used = {0: "n/a_fp16", 1: scale_int8, 2: scale_int4, 3: scale_low}[tier]
            # Byte preview for this block's tier payload
            if tier == 0:
                off = int(c16_blocks[block_idx]) * block_size
                fv = np.asarray(fp16_values).ravel()
                if fv.dtype == np.uint16:
                    fv = fv.view(np.float16)
                elif fv.dtype != np.float16:
                    fv = fv.astype(np.float16, copy=False)
                n_h = min(8, max(0, fv.size - off))  # 8 fp16 = 16 bytes
                prev = fv[off : off + n_h].view(np.uint8) if n_h else np.zeros(0, dtype=np.uint8)
            elif tier == 1:
                elem_off = int(c8_blocks[block_idx]) * block_size
                prev = stream8[elem_off : elem_off + 16]
            elif tier == 2:
                elem_off = int(c4_blocks[block_idx]) * block_size
                byte_lo = elem_off // 2
                prev = stream4[byte_lo : byte_lo + 16]
            else:
                elem_off = int(c2_blocks[block_idx]) * block_size
                byte_lo = elem_off // 4
                prev = stream2[byte_lo : byte_lo + 16]

            hx = prev.tobytes()[:16].hex() if prev.size else ""

            r_first, c_first = divmod(lo, cols)
            r_last, c_last = divmod(hi - 1, cols)
            return [
                "",
                "=== First mismatch (row-major blocks) ===",
                f"block_idx          : {block_idx}",
                f"flat index range   : [{lo}, {hi})  (exclusive hi)",
                (
                    f"row/col corners    : first=({r_first},{c_first}) last=({r_last},{c_last}) "
                    "(row-major flatten)"
                ),
                f"tier               : {tier}  (0=FP16,1=INT8,2=INT4,3=INT2)",
                f"scale_used         : {scale_used}",
                f"mask_byte[block//4]: 0x{mbyte:02x}  (lane = block_idx % 4)",
                f"payload_bytes_hex[:16]: {hx}",
                f"w_pt_fq[:8]        : {np.array2string(sl_pt[:8], precision=6)}",
                f"w_gguf_dq[:8]      : {np.array2string(sl_gg[:8], precision=6)}",
                f"|diff|[:8]        : {np.array2string(np.abs(sl_pt - sl_gg)[:8], precision=6)}",
            ]
    return [
        "",
        "=== First mismatch (row-major blocks) ===",
        f"No block exceeded max_abs_err > {threshold}",
    ]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Compare MultiTierFakeQuantize dequant vs GGUF block unpack (C++ mirror)."
    )
    ap.add_argument("--qat-ckpt", type=Path, default=here / "qat_septq_final_run" / "qat_best.pt")
    ap.add_argument(
        "--septq-ckpt",
        type=Path,
        default=None,
        help="SEPTQ multitier .pt (tier_masks + septq_meta). Default: qat_meta.source_student_quant_meta",
    )
    ap.add_argument("--gguf", type=Path, default=here / "bmo_septq_v3.gguf")
    ap.add_argument(
        "--module-name",
        type=str,
        default="transformer.layers.0.self_attn.in_proj_weight",
    )
    ap.add_argument(
        "--gguf-name",
        type=str,
        default="transformer_layers_0_self_attn_in_proj_weight",
    )
    ap.add_argument("--qkv-split", type=str, default="4096,4096,4096")
    ap.add_argument("--mismatch-threshold", type=float, default=1e-5)
    args = ap.parse_args()

    qat_path = args.qat_ckpt.resolve()
    gguf_path = args.gguf.resolve()

    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)

    if not qat_path.is_file():
        _err(f"QAT checkpoint not found: {qat_path}")
        sys.exit(2)

    try:
        qat_payload = torch.load(str(qat_path), map_location="cpu")
        if not isinstance(qat_payload, dict):
            raise ValueError(f"QAT checkpoint must be a dict, got {type(qat_payload)}")
        quant_ckpt = resolve_quant_checkpoint(qat_payload, qat_path, args.septq_ckpt)
    except Exception as exc:
        _err(str(exc))
        sys.exit(2)

    module_name = str(args.module_name)
    tier_masks_uint2 = _tier_masks_from_ckpt(quant_ckpt)
    tier_masks_meta = _tier_masks_meta_from_ckpt(quant_ckpt)

    if module_name not in tier_masks_uint2:
        _err(f"tier_masks_uint2 has no entry for module_name={module_name!r}")
        sys.exit(2)

    try:
        qp = resolve_multitier_quant_params(module_name, ckpt=quant_ckpt)
    except Exception as exc:
        _err(f"per_layer_stats / resolve_multitier_quant_params: {exc}")
        sys.exit(2)

    sd = _unwrap_state_dict(qat_payload)
    if module_name not in sd:
        _err(f"Weight key missing in QAT checkpoint state_dict: {module_name!r}")
        sys.exit(2)

    w_dense = sd[module_name].detach().to(device="cpu", dtype=torch.float32)
    if w_dense.dim() != 2:
        _err(f"Expected 2D weight, got shape {tuple(w_dense.shape)}")
        sys.exit(2)

    packed_mask_ckpt = tier_masks_uint2[module_name].detach().cpu()
    raw_shape = tier_masks_meta.get(module_name, {}).get("shape", list(w_dense.shape))
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
        _err(f"Invalid tier_masks_meta.shape for {module_name}: {raw_shape!r}")
        sys.exit(2)
    target_shape = (int(raw_shape[0]), int(raw_shape[1]))
    if tuple(target_shape) != tuple(w_dense.shape):
        _err(f"Tier mask shape {target_shape} != weight shape {tuple(w_dense.shape)}")
        sys.exit(2)

    needed_packed = (target_shape[0] * target_shape[1] + 3) // 4
    mask_flat = packed_mask_ckpt.to(dtype=torch.uint8).reshape(-1)
    if int(mask_flat.numel()) < int(needed_packed):
        _err(f"Packed tier mask too small: have={mask_flat.numel()} need={needed_packed}")
        sys.exit(2)

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
    w_pt_fq = fq(w_dense).detach().to(dtype=torch.float32)

    if not gguf_path.is_file():
        _err(f"GGUF not found: {gguf_path}")
        sys.exit(2)

    try:
        reader = gguf.GGUFReader(str(gguf_path))
        by_name = _tensor_dict(reader)
        base = str(args.gguf_name)
        w_gguf_np = dequant_one_base(by_name, base)
    except Exception as exc:
        _err(f"GGUF unpack failed for base={base!r}: {exc}")
        sys.exit(2)

    w_gguf_dq = torch.from_numpy(np.asarray(w_gguf_np, dtype=np.float32))

    if w_gguf_dq.shape != w_pt_fq.shape:
        _err(f"Shape mismatch: fake-quant {tuple(w_pt_fq.shape)} vs GGUF {tuple(w_gguf_dq.shape)}")
        sys.exit(2)

    rows, cols = int(w_pt_fq.shape[0]), int(w_pt_fq.shape[1])
    try:
        n_q, n_k, n_v = _parse_qkv(args.qkv_split)
    except Exception as exc:
        _err(str(exc))
        sys.exit(2)
    if n_q + n_k + n_v != rows:
        _err(f"--qkv-split sum {n_q}+{n_k}+{n_v} != rows={rows}")
        sys.exit(2)

    # GGUF block metadata for tier masks + mismatch report
    pw_t = pick_tensor(by_name, base, "packed_weights")
    pm_t = pick_tensor(by_name, base, "packed_mask")
    if pw_t is None or pm_t is None:
        _err("Internal: missing packed_weights or packed_mask after dequant_one_base")
        sys.exit(2)

    pw_np = _get_numpy(pw_t).view(np.uint8).ravel()
    pm_np = _get_numpy(pm_t).view(np.uint8).ravel()
    bs = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "block_size")), 32)
    block_size = int(bs) if int(bs) > 0 else 32
    total = rows * cols
    n_blocks = (total + block_size - 1) // block_size
    tier_pb = _tier_per_block_numpy(pm_np, n_blocks)

    n2 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_2bit_bytes")))
    n4 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_4bit_bytes")))
    n8 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_8bit_bytes")))
    scale_low = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_low")), 1.0)
    scale_int4 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_int4")), 1.0)
    scale_int8 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_int8")), 1.0)

    fv_t = pick_tensor(by_name, base, "fp16_values")
    fv_np = _get_numpy(fv_t) if fv_t is not None else np.array([], dtype=np.uint8)

    dev = w_pt_fq.device
    full_m = torch.ones(rows * cols, dtype=torch.bool, device=dev)

    tier_names = ("FP16", "INT8", "INT4", "INT2")
    tier_pb_t = torch.from_numpy(tier_pb.astype(np.int64)).to(device=dev)

    def elem_tier_mask(tcode: int) -> torch.Tensor:
        bid = torch.arange(total, device=dev, dtype=torch.int64) // block_size
        return tier_pb_t[bid] == tcode

    # --- build report ---
    w = 28
    hdr = (
        f"{'subset':<{w}} | {'cosine':>8} | {'max_abs_err':>12} | {'mean_abs_err':>12} | "
        f"{'rmse':>12} | {'n_elem':>8} | {'n_blk':>6}"
    )
    log("")
    log("=== compare_fakequant_vs_gguf_weights ===")
    log(f"qat_ckpt      : {qat_path}")
    log(f"quant_ckpt    : (tier_masks + per_layer_stats source)")
    log(f"gguf          : {gguf_path}")
    log(f"module_name   : {module_name}")
    log(f"gguf base     : {base}")
    log(f"shape         : {tuple(w_pt_fq.shape)}  block_size={block_size}")
    log("")
    log("=== Diagnostics (subset | cosine | max_abs_err | mean_abs_err | rmse | n_elem | n_blk) ===")
    log(hdr)
    log("-" * len(hdr))

    st_all = _stats_pair_with_bs(w_pt_fq, w_gguf_dq, full_m, block_size)
    log(_fmt_row("whole_tensor", st_all, w))

    q_lo, q_hi = 0, n_q
    k_lo, k_hi = n_q, n_q + n_k
    v_lo, v_hi = n_q + n_k, rows

    for label, lo, hi in (("Q_rows", q_lo, q_hi), ("K_rows", k_lo, k_hi), ("V_rows", v_lo, v_hi)):
        rm = _row_mask(w_pt_fq, cols, lo, hi)
        st = _stats_pair_with_bs(w_pt_fq, w_gguf_dq, rm, block_size)
        log(_fmt_row(label, st, w))

    for ti, name in enumerate(tier_names):
        st = _stats_pair_with_bs(w_pt_fq, w_gguf_dq, elem_tier_mask(ti), block_size)
        log(_fmt_row(f"tier_{name}", st, w))

    log("")
    log("=== Q/K/V × tier (12 subsets; same columns as main table) ===")
    log(hdr)
    log("-" * len(hdr))
    for band, lo, hi in (("Q", q_lo, q_hi), ("K", k_lo, k_hi), ("V", v_lo, v_hi)):
        row_m = _row_mask(w_pt_fq, cols, lo, hi)
        for ti, tn in enumerate(tier_names):
            st = _stats_pair_with_bs(w_pt_fq, w_gguf_dq, row_m & elem_tier_mask(ti), block_size)
            log(_fmt_row(f"{band}×{tn}", st, w))

    lines.extend(
        _first_mismatch_lines(
            w_pt_fq,
            w_gguf_dq,
            tier_pb=tier_pb,
            block_size=block_size,
            rows=rows,
            cols=cols,
            threshold=float(args.mismatch_threshold),
            packed_weights=pw_np,
            packed_mask=pm_np,
            n_2bit_bytes=n2,
            n_4bit_bytes=n4,
            n_8bit_bytes=n8,
            scale_low=scale_low,
            scale_int4=scale_int4,
            scale_int8=scale_int8,
            fp16_values=fv_np,
        )
    )

    log("")
    log(
        "Conventions: mask uint2 LE (lane i = bits 2*i..2*i+1); block tier byte = "
        "packed_mask[block_idx//4], lane block_idx%4; tier 0=FP16 block, 1=int8 stream8, "
        "2=int4 low nibble first per byte, 3=int2 4-per-byte LE; dequant (q-zp)*scale. "
        "Fake-quant: mask per *element* (qat unpack), tier 0 passthrough, 1/2/3→int8/int4/int2 "
        "with round(clamp) same as MultiTierFakeQuantize."
    )

    # Single print block
    print("\n".join(lines))


if __name__ == "__main__":
    main()
