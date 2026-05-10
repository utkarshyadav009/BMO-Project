#!/usr/bin/env python3
"""Compare PyTorch fused in_proj_weight vs dequantized SEPTQ QKV in a BMO GGUF.

Mirrors `unpack_layer_to_f32_blockwise` in `bmo_compute.cpp` (tier 0=FP16, 1=i8, 2=i4, 3=i2).

PyTorch (`StreamingMultiheadAttention`): fused Linear weight [3*d_model, d_model], bias=False;
  projected = linear(x, in_proj_weight); layout along output channels is Q block, then K, then V
  (`einops rearrange` with p=3, h=num_heads).

GGUF stores **one fused** packed tensor per layer (not separate linear_q/k/v). Canonical tensor prefix:
  `transformer_layers_{L}_self_attn_in_proj_weight` — multitier exports may emit
  `transformer_inner_layers_{L}_self_attn_in_proj_weight` when checkpoints use
  `TemporalProjectedTransformer` module paths.

Usage:
  PYTHONPATH=./moshi:./llama.cpp/gguf-py python probe_qkv_split.py \\
    --ckpt /path/to/qat_best.pt \\
    --gguf /path/to/bmo_septq_v3.gguf \\
    --out-dir ./probe_qkv_out

Depends: numpy, torch; gguf from vendored `llama.cpp/gguf-py`.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np


def _setup_paths(script_dir: Path) -> None:
    for rel in ("llama.cpp/gguf-py", "moshi"):
        p = script_dir / rel
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


def _unpack_u2_le(byte_val: int, idx: int) -> int:
    return (int(byte_val) >> (idx * 2)) & 3


def _fp16_to_f32_storage(elem) -> float:
    """Interpret GGUF fp16 tensor elements whether stored as float16 or raw uint16."""
    x = np.asarray(elem)
    if x.dtype == np.uint16:
        return float(np.frombuffer(struct.pack("H", int(x.item())), dtype=np.float16)[0])
    return float(np.asarray(x, dtype=np.float32))


def dequant_blockwise_septq_v3(
    packed_weights: np.ndarray,
    packed_mask: np.ndarray,
    *,
    rows: int,
    cols: int,
    block_size: int,
    n_2bit_bytes: int,
    n_4bit_bytes: int,
    n_8bit_bytes: int,
    scale_low: float,
    scale_int4: float,
    scale_int8: float,
    zp_low: float,
    zp_int4: float,
    zp_int8: float,
    fp16_values: np.ndarray,
) -> np.ndarray:
    """Byte-identical layout intention vs `unpack_layer_to_f32_blockwise` (bmo_compute.cpp)."""
    total = rows * cols
    out_w = np.zeros(total, dtype=np.float32)
    n_blocks = (total + block_size - 1) // block_size
    pw = np.asarray(packed_weights).ravel().astype(np.uint8)
    pm = np.asarray(packed_mask).ravel().astype(np.uint8)
    fv16 = np.asarray(fp16_values)

    stream2 = pw[:n_2bit_bytes]
    stream4 = pw[n_2bit_bytes : n_2bit_bytes + n_4bit_bytes]
    stream8 = pw[n_2bit_bytes + n_4bit_bytes : n_2bit_bytes + n_4bit_bytes + n_8bit_bytes]

    c2 = c4 = c8 = c16 = 0

    for block_idx in range(n_blocks):
        mbyte = int(pm[block_idx // 4])
        tier = _unpack_u2_le(mbyte, block_idx % 4)
        off = 0
        if tier == 0:
            off = c16
            c16 += block_size
        elif tier == 1:
            off = c8
            c8 += block_size
        elif tier == 2:
            off = c4
            c4 += block_size
        else:
            off = c2
            c2 += block_size

        for ib in range(block_size):
            pos = block_idx * block_size + ib
            if pos >= total:
                break
            if tier == 0:
                v = _fp16_to_f32_storage(fv16[off + ib])
            elif tier == 1:
                q = int(stream8[off + ib])
                v = (float(q) - zp_int8) * scale_int8
            elif tier == 2:
                idx = off + ib
                b = int(stream4[idx // 2])
                qq = (b & 0x0F) if (idx % 2 == 0) else ((b >> 4) & 0x0F)
                v = (float(qq) - zp_int4) * scale_int4
            else:
                idx = off + ib
                b = int(stream2[idx // 4])
                qq = _unpack_u2_le(b, idx % 4)
                v = (float(qq) - zp_low) * scale_low
            out_w[pos] = np.float32(v)

    return out_w.reshape(rows, cols)


def _load_qkv_gguf_bundle(reader, layer_idx: int) -> tuple[str, dict]:
    wanted_suffixes = (
        "packed_weights",
        "packed_mask",
        "block_size",
        "n_2bit_bytes",
        "n_4bit_bytes",
        "n_8bit_bytes",
        "scale_low",
        "scale_int4",
        "scale_int8",
        "zp_low",
        "zp_int4",
        "zp_int8",
        "fp16_values",
        "rows",
        "cols",
    )
    by_name = {t.name: t for t in reader.tensors}
    for candidate_prefix in (
        f"transformer_layers_{layer_idx}_self_attn_in_proj_weight",
        f"transformer_inner_layers_{layer_idx}_self_attn_in_proj_weight",
    ):
        keys = [f"{candidate_prefix}.{s}" for s in wanted_suffixes]
        if all(k in by_name for k in keys):
            return candidate_prefix, {s: by_name[f"{candidate_prefix}.{s}"] for s in wanted_suffixes}

    pat = re.compile(
        rf"^(transformer_(?:inner_)?layers_{layer_idx}_self_attn_in_proj_weight)\.packed_weights$"
    )
    for tname in by_name:
        if pat.match(tname):
            pref = tname.rsplit(".", 1)[0]
            keys = [f"{pref}.{s}" for s in wanted_suffixes]
            if all(k in by_name for k in keys):
                return pref, {s: by_name[f"{pref}.{s}"] for s in wanted_suffixes}

    avail = sorted({t.name for t in reader.tensors if "in_proj" in t.name and "self_attn" in t.name})[:48]
    msg = "\n  ".join(avail) if avail else "(none)"
    raise SystemExit(
        f"No fused SEPTQ bundle found for temporal layer {layer_idx}. Related GGUF names (sample):\n  {msg}"
    )


def _unwrap_ckpt_state(path: Path) -> dict:
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return dict(obj["state_dict"])
    if isinstance(obj, dict):
        tens = {k: v for k, v in obj.items() if hasattr(v, "shape")}
        if tens:
            return tens
    raise SystemExit(f"Unrecognized checkpoint layout in {path}")


def _find_layer_in_proj_key(sd: dict, layer_idx: int) -> str:
    needle = f".layers.{layer_idx}.self_attn.in_proj_weight"
    candidates = [k for k in sd if k.endswith(needle) and "depformer" not in k]
    if not candidates:
        raise SystemExit(f"No state_dict key ending with {needle!r}")
    for pref in (
        f"transformer.layers.{layer_idx}.self_attn.in_proj_weight",
        f"transformer.inner.layers.{layer_idx}.self_attn.in_proj_weight",
    ):
        if pref in sd:
            return pref
    candidates.sort(key=len)
    return candidates[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _report_pair(tag: str, pt_a: np.ndarray, gguf_b: np.ndarray) -> None:
    ca = pt_a.astype(np.float32).ravel()
    cb = gguf_b.astype(np.float32).ravel()
    if ca.size != cb.size:
        print(f"  {tag}: SIZE_MISMATCH pt={ca.size} gguf={cb.size}")
        return
    diff = np.abs(ca - cb)
    print(
        f"  {tag}: cosine={_cosine(ca, cb):.8f} "
        f"max_abs={float(diff.max()):.6e} mean_abs={float(diff.mean()):.6e}"
    )


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    _setup_paths(script_dir)

    try:
        from gguf import GGUFReader
        import torch
    except ImportError as e:
        raise SystemExit(f"Import failed (need torch + gguf): {e}") from e

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("probe_qkv_out"))
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = _unwrap_ckpt_state(args.ckpt)
    ckpt_key = _find_layer_in_proj_key(sd, args.layer)
    w_pt = sd[ckpt_key].detach().cpu().float().numpy()
    if w_pt.ndim != 2:
        raise SystemExit(f"{ckpt_key}: expected matrix, got {w_pt.shape}")
    rows, cols = map(int, w_pt.shape)
    if rows % 3 != 0:
        raise SystemExit(f"in_proj rows {rows} not divisible by 3")
    d_model = rows // 3

    np.save(out_dir / "pt_in_proj_weight_layer0.npy", w_pt.astype(np.float32))
    pt_q = w_pt[0:d_model, :]
    pt_k = w_pt[d_model : 2 * d_model, :]
    pt_v = w_pt[2 * d_model :, :]
    np.save(out_dir / "pt_layer0_qw.npy", pt_q.astype(np.float32))
    np.save(out_dir / "pt_layer0_kw.npy", pt_k.astype(np.float32))
    np.save(out_dir / "pt_layer0_vw.npy", pt_v.astype(np.float32))

    reader = GGUFReader(str(args.gguf))
    gguf_pref, blobs = _load_qkv_gguf_bundle(reader, args.layer)

    rows_g = int(np.asarray(blobs["rows"].data).reshape(-1)[0])
    cols_g = int(np.asarray(blobs["cols"].data).reshape(-1)[0])
    block_size = int(np.asarray(blobs["block_size"].data).reshape(-1)[0])
    n2 = int(np.asarray(blobs["n_2bit_bytes"].data).reshape(-1)[0])
    n4 = int(np.asarray(blobs["n_4bit_bytes"].data).reshape(-1)[0])
    n8 = int(np.asarray(blobs["n_8bit_bytes"].data).reshape(-1)[0])
    sl = float(np.asarray(blobs["scale_low"].data).reshape(-1)[0])
    s4 = float(np.asarray(blobs["scale_int4"].data).reshape(-1)[0])
    s8 = float(np.asarray(blobs["scale_int8"].data).reshape(-1)[0])
    zl = float(np.asarray(blobs["zp_low"].data).reshape(-1)[0])
    z4 = float(np.asarray(blobs["zp_int4"].data).reshape(-1)[0])
    z8 = float(np.asarray(blobs["zp_int8"].data).reshape(-1)[0])

    gguf_w = dequant_blockwise_septq_v3(
        np.asarray(blobs["packed_weights"].data),
        np.asarray(blobs["packed_mask"].data),
        rows=rows_g,
        cols=cols_g,
        block_size=block_size,
        n_2bit_bytes=n2,
        n_4bit_bytes=n4,
        n_8bit_bytes=n8,
        scale_low=sl,
        scale_int4=s4,
        scale_int8=s8,
        zp_low=zl,
        zp_int4=z4,
        zp_int8=z8,
        fp16_values=np.asarray(blobs["fp16_values"].data),
    )

    np.save(out_dir / "gguf_dequant_in_proj_layer0.npy", gguf_w.astype(np.float32))

    print("=== probe_qkv_split: inputs ===")
    print(f"  ckpt key (layer {args.layer}): {ckpt_key}")
    print(f"  PT fused shape: {w_pt.shape} (expect [{3 * d_model}, {d_model}] )")
    print(f"  GGUF bundle prefix: {gguf_pref}")
    print(f"  GGUF meta rows/cols: {(rows_g, cols_g)}  block_size={block_size}")

    print("\n=== Full fused matrix vs PT ===")
    if w_pt.shape == gguf_w.shape:
        _report_pair("fused [same shape]", w_pt, gguf_w)
        _report_pair("fused gguf vs PT.T (wrong-axis sanity)", w_pt, gguf_w.T)
    else:
        print(f"  SHAPE_MISMATCH PT={w_pt.shape} GGUF_dequant={gguf_w.shape}")
        print("  (cannot compare slices meaningfully until shapes agree — export/metadata bug?)")

    g_d = rows_g // 3
    g_q = gguf_w[0:g_d, :]
    g_k = gguf_w[g_d : 2 * g_d, :]
    g_v = gguf_w[2 * g_d :, :]

    labels_pt = ("pt_Q", "pt_K", "pt_V")
    mats_pt = (pt_q, pt_k, pt_v)
    labels_g = ("gguf_Q", "gguf_K", "gguf_V")
    mats_g = (g_q, g_k, g_v)

    print("\n=== 3×3 block cosine matrix (flattened rows): PT slice vs GGUF slice ===")
    best = (-1.0, "", "")
    for lp, mp in zip(labels_pt, mats_pt):
        for lg, mg in zip(labels_g, mats_g):
            c = _cosine(mp, mg)
            if np.isfinite(c) and c > best[0]:
                best = (c, lp, lg)
            print(f"  {lp} vs {lg}: cosine={c:.8f}")
    print(f"\n  BEST_PAIR cosine={best[0]:.8f}  ({best[1]} vs {best[2]})")

    print("\n=== Transpose tests per slice (PT slice vs GGUF slice.T) ===")
    for lp, mp in zip(labels_pt, mats_pt):
        for lg, mg in zip(labels_g, mats_g):
            print(f"  {lp} vs {lg}.T: cosine={_cosine(mp, mg.T):.8f}")

    print("\n=== Alternative PT layouts (transpose / swapped axes) ===")
    print(f"  PT fused vs GGUF fused (both .T): cosine={_cosine(w_pt.T, gguf_w.T):.8f}")
    if gguf_w.shape == (cols, rows):
        print(
            f"  GGUF matches swapped shape {gguf_w.shape} vs PT {w_pt.shape}; "
            f"cosine PT vs gguf.T: {_cosine(w_pt, gguf_w.T):.8f}"
        )
    if cols == d_model and rows == 3 * d_model:
        wt_alt = w_pt.reshape(d_model, 3 * d_model)
        print(
            "  PT wrongly viewed as [d_model, 3*d_model] — compare column-blocks to gguf Q/K/V slices:"
        )
        for blk_idx, tag in enumerate(("col_blk0→Q?", "col_blk1→K?", "col_blk2→V?")):
            sl = wt_alt[:, blk_idx * d_model : (blk_idx + 1) * d_model]
            for lg, mg in zip(labels_g, mats_g):
                print(f"    {tag} vs {lg}: cosine={_cosine(sl, mg):.8f}")

    print("\n=== Diagnosis summary (interpret AFTER inspecting cosines above) ===")
    print(
        "  • GGUF fused matches PT fused at cosine≈1 → SEPTQ+layout OK vs checkpoint;"
        " if runtime diverges, bug is likely C++/GEMM layout (not export)."
    )
    print(
        "  • GGUF fused mismatches PT, but one BEST_PAIR aligns PT-Q with GGUF-V → QKV chunk permutation bug."
    )
    print(
        "  • Full fused matches after swapping transpose vs PT → probable row-major vs col-major export mishandle."
    )
    print(
        "  • All combinations weak (~0) with mismatched shapes / garbage norms → wrong tensor aliasing "
        "or incompatible GGUF (e.g. wrong prefix)."
    )

    print(f"\nSaved NumPy artifacts under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
