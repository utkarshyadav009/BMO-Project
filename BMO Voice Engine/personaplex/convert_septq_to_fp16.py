#!/usr/bin/env python3
# flake8: noqa
"""convert_septq_to_fp16.py

Strip SEPTQ packed blobs from a BMO GGUF and emit dense FP16 weights so the C++
runtime uses `dense_weight` / `ggml_mul_mat` (see `resolve_linear` and
`apply_linear_with_transient_unpack` in bmo_compute.cpp).

Verification (run on the machine that holds the GGUF):
    python -c "import gguf; r = gguf.GGUFReader('bmo_septq_v3.gguf'); print(len(r.tensors), 'tensors'); [print(t.name, t.shape, t.tensor_type) for t in r.tensors[:30]]"

Conversion:
    python convert_septq_to_fp16.py --in bmo_septq_v3.gguf --out bmo_fp16_test.gguf

Sources of truth (do not drift from these without checking C++):
    • Block-wise unpack: `unpack_layer_to_f32_blockwise` in bmo_compute.cpp (tier 0=FP16 block,
      1=i8, 2=i4, 3=2-bit / ``scale_low`` path).
    • Legacy per-element mask unpack: `unpack_layer_to_f32` when ``block_size`` is 0.
    • Export / quantization inverse: `create_packed_layer` + tensor naming in export_bmo_gguf.py
      (`BLOCK_SIZE = 32`, mask packing `pack_uint2_mask_le`, single concatenated ``packed_weights``).

Assumptions to sanity-check on your GGUF:
    • Tensor names follow `{base}.packed_weights` (canonical for repo exports). If your artifact uses
      only underscore-separated variants (`{base}_packed_*`), extend `_gather_packed_bundle` after
      listing tensors once with the snippet above.
    • Rows/cols scalars match PyTorch Linear layout `[out_features, in_features]` stored row-major;
      dense FP16 output keeps shape `(rows, cols)` like `export_dense_tensor(..., preserve_half=True)`.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Prefer vendored gguf-py (same as diet_gguf.py)
_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_GGUF_PY = _REPO_ROOT / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

import gguf  # type: ignore  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

# Tensor stems omitted when replacing packed artifacts with one dense `{base}` tensor.
PACKED_ARTIFACT_SUFFIXES: tuple[str, ...] = (
    "packed_weights",
    "packed_mask",
    "packed_values_low",
    "packed_values_int4",
    "packed_values_int8",
    "n_2bit_bytes",
    "n_4bit_bytes",
    "n_8bit_bytes",
    "block_size",
    "n_blocks",
    "scale_low",
    "scale_int4",
    "scale_int8",
    "zp_low",
    "zp_int4",
    "zp_int8",
    "threshold_8bit",
    "threshold_4bit",
    "threshold_2bit",
    "fp16_values",
    "fp16_indices",
    "packing_version",
    "rows",
    "cols",
    "idx2_start",
    "idx4_start",
    "idx8_start",
)


def unpack_u2_le(byte_val: int, lane: int) -> int:
    return (int(byte_val) >> (lane * 2)) & 0x3


def read_scalar_i32(raw: np.ndarray | Any, fallback: int = 0) -> int:
    if raw is None:
        return fallback
    buf = np.asarray(raw).view(np.uint8).tobytes()
    if len(buf) < 4:
        return fallback
    return struct.unpack("<i", buf[:4])[0]


def read_scalar_f32(raw: np.ndarray | Any, fallback: float = 0.0) -> float:
    if raw is None:
        return fallback
    buf = np.asarray(raw).view(np.uint8).tobytes()
    if len(buf) < 4:
        return fallback
    return struct.unpack("<f", buf[:4])[0]


def fp16_elem_to_f32(x: Any) -> float:
    v = np.asarray(x)
    if v.dtype == np.float16:
        return float(v.flat[0])
    if v.dtype == np.float32:
        return float(v.flat[0])
    if v.dtype == np.uint16:
        return float(np.frombuffer(struct.pack("H", int(v.flat[0])), dtype=np.float16)[0])
    return float(v.astype(np.float32).flat[0])


def unpack_layer_to_f32_blockwise(
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
    """Vectorized mirror of ``unpack_layer_to_f32_blockwise`` (bmo_compute.cpp).

    Tier convention (matches C++ + worker report):
        0 = FP16 block (block_size FP16 values from `fp16_values`)
        1 = int8     (`stream8`, dequant `(q - zp_int8) * scale_int8`)
        2 = int4     (`stream4`, low nibble first within byte; `(q - zp_int4) * scale_int4`)
        3 = 2-bit    (`stream2`, lanes LE within byte; `(q - zp_low) * scale_low`)
    """
    total = int(rows) * int(cols)
    n_blocks = (total + block_size - 1) // block_size

    pw = np.asarray(packed_weights).ravel().view(np.uint8)
    pm = np.asarray(packed_mask).ravel().view(np.uint8)

    fv = np.asarray(fp16_values).ravel()
    if fv.dtype == np.uint16:
        fv = fv.view(np.float16)
    if fv.dtype not in (np.float16, np.float32):
        fv = fv.astype(np.float16, copy=False)

    stream2 = pw[:n_2bit_bytes]
    stream4 = pw[n_2bit_bytes : n_2bit_bytes + n_4bit_bytes]
    stream8 = pw[n_2bit_bytes + n_4bit_bytes : n_2bit_bytes + n_4bit_bytes + n_8bit_bytes]

    # --- decode tier for each of n_blocks blocks (vectorized) -----------------
    n_mask_bytes = (n_blocks + 3) // 4
    pm_b = pm[:n_mask_bytes]
    # Each mask byte holds 4 lanes (2 bits each, little-endian).
    pm_rep = np.repeat(pm_b, 4)[:n_blocks]
    shifts = np.tile(np.array([0, 2, 4, 6], dtype=np.uint8), n_mask_bytes)[:n_blocks]
    tier_per_block = ((pm_rep >> shifts) & np.uint8(0x3)).astype(np.int8)

    is0 = tier_per_block == 0  # fp16
    is1 = tier_per_block == 1  # int8
    is2 = tier_per_block == 2  # int4
    is3 = tier_per_block == 3  # 2-bit

    # Per-tier exclusive cumulative block-count → element offsets into each stream.
    def _excl_cumsum(mask: np.ndarray) -> np.ndarray:
        cs = np.cumsum(mask.astype(np.int64))
        return cs - mask.astype(np.int64)

    c16_blocks = _excl_cumsum(is0)
    c8_blocks = _excl_cumsum(is1)
    c4_blocks = _excl_cumsum(is2)
    c2_blocks = _excl_cumsum(is3)

    out_w = np.empty(total, dtype=np.float32)
    block_starts = np.arange(n_blocks, dtype=np.int64) * block_size
    inblock = np.arange(block_size, dtype=np.int64)

    def _scatter(block_indices: np.ndarray, deq_block: np.ndarray) -> None:
        if block_indices.size == 0:
            return
        out_pos = block_starts[block_indices, None] + inblock[None, :]  # [B, BS]
        valid = out_pos < total
        out_w[out_pos[valid]] = deq_block[valid].astype(np.float32, copy=False)

    if is1.any():
        b1 = np.flatnonzero(is1)
        elem_off = c8_blocks[b1] * block_size
        src = elem_off[:, None] + inblock[None, :]
        q = stream8[src].astype(np.float32)
        deq = (q - np.float32(zp_int8)) * np.float32(scale_int8)
        _scatter(b1, deq)

    if is0.any():
        b0 = np.flatnonzero(is0)
        elem_off = c16_blocks[b0] * block_size
        src = elem_off[:, None] + inblock[None, :]
        deq = fv[src]
        _scatter(b0, deq)

    if is2.any():
        b2 = np.flatnonzero(is2)
        elem_off = c4_blocks[b2] * block_size
        idx = elem_off[:, None] + inblock[None, :]
        byte_idx = idx >> 1
        lane = (idx & 1).astype(np.uint8)
        bytes_ = stream4[byte_idx]
        nibble = np.where(
            lane == 0,
            bytes_ & np.uint8(0x0F),
            (bytes_ >> 4) & np.uint8(0x0F),
        ).astype(np.float32)
        deq = (nibble - np.float32(zp_int4)) * np.float32(scale_int4)
        _scatter(b2, deq)

    if is3.any():
        b3 = np.flatnonzero(is3)
        elem_off = c2_blocks[b3] * block_size
        idx = elem_off[:, None] + inblock[None, :]
        byte_idx = idx >> 2
        lane = (idx & 0x3).astype(np.uint8)
        bytes_ = stream2[byte_idx]
        q = ((bytes_ >> (lane * np.uint8(2))) & np.uint8(0x3)).astype(np.float32)
        deq = (q - np.float32(zp_low)) * np.float32(scale_low)
        _scatter(b3, deq)

    return out_w.reshape(rows, cols)


def unpack_layer_to_f32_legacy(
    packed_weights: np.ndarray,
    packed_mask: np.ndarray,
    *,
    rows: int,
    cols: int,
    n_2bit_bytes: int,
    n_4bit_bytes: int,
    n_8bit_bytes: int,
    scale_low: float,
    scale_int4: float,
    scale_int8: float,
    zp_low: float,
    zp_int4: float,
    zp_int8: float,
    fp16_indices: np.ndarray,
    fp16_values: np.ndarray,
) -> np.ndarray:
    """Mirror ``unpack_layer_to_f32`` (bmo_compute.cpp): mask bits index **elements**, tier 0 filled from fp16."""
    total = rows * cols
    out_w = np.zeros(total, dtype=np.float32)
    pw = np.asarray(packed_weights).ravel().astype(np.uint8, copy=False)
    pm = np.asarray(packed_mask).ravel().astype(np.uint8, copy=False)

    stream2 = pw[:n_2bit_bytes]
    stream4 = pw[n_2bit_bytes : n_2bit_bytes + n_4bit_bytes]
    stream8 = pw[n_2bit_bytes + n_4bit_bytes : n_2bit_bytes + n_4bit_bytes + n_8bit_bytes]

    idx2 = idx4 = idx8 = 0
    for pos in range(total):
        mbyte = int(pm[pos // 4])
        tier = unpack_u2_le(mbyte, pos % 4)
        v = 0.0
        if tier >= 3:
            b = int(stream2[idx2 // 4])
            q = unpack_u2_le(b, idx2 % 4)
            idx2 += 1
            v = (float(q) - zp_low) * scale_low
        elif tier == 2:
            b = int(stream4[idx4 // 2])
            q = (b & 0x0F) if (idx4 % 2 == 0) else ((b >> 4) & 0x0F)
            idx4 += 1
            v = (float(q) - zp_int4) * scale_int4
        elif tier == 1:
            q = int(stream8[idx8])
            idx8 += 1
            v = (float(q) - zp_int8) * scale_int8
        out_w[pos] = np.float32(v)

    fi = np.asarray(fp16_indices).reshape(-1).astype(np.int32, copy=False)
    fv = np.asarray(fp16_values)
    for i in range(fi.shape[0]):
        pos = int(fi[i])
        if 0 <= pos < total:
            out_w[pos] = fp16_elem_to_f32(fv[i])

    return out_w.reshape(rows, cols)


def _tensor_dict(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    return {t.name: t for t in reader.tensors}


def _get_numpy(reader_tensor: gguf.ReaderTensor) -> np.ndarray:
    data = reader_tensor.data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return data


def discover_packed_bases(names: set[str]) -> list[str]:
    bases: set[str] = set()
    for n in names:
        if n.endswith(".packed_weights"):
            bases.add(n[: -len(".packed_weights")])
        elif n.endswith("_packed_weights"):
            bases.add(n[: -len("_packed_weights")])
    # Also pick up bases that only expose `.packed_mask` + split streams (no single blob).
    for n in names:
        if n.endswith(".packed_mask"):
            bases.add(n[: -len(".packed_mask")])
        elif n.endswith("_packed_mask"):
            bases.add(n[: -len("_packed_mask")])
    return sorted(bases)


def build_packed_stream(by_name: dict[str, gguf.ReaderTensor], base: str) -> tuple[np.ndarray | None, str]:
    """Return (packed_weights_uint8, mode) where mode is ``single`` or ``split``."""
    dot = base + ".packed_weights"
    if dot in by_name:
        return _get_numpy(by_name[dot]).view(np.uint8).ravel(), "single"

    # Hyphenated split streams (not in current C++ loader — kept for forward-looking GGUFs).
    parts_low = (
        base + ".packed_values_low",
        base + "_packed_values_low",
    )
    p4 = (base + ".packed_values_int4", base + "_packed_values_int4")
    p8 = (base + ".packed_values_int8", base + "_packed_values_int8")
    low = next((by_name[k] for k in parts_low if k in by_name), None)
    mid = next((by_name[k] for k in p4 if k in by_name), None)
    hi = next((by_name[k] for k in p8 if k in by_name), None)
    if low is not None and mid is not None and hi is not None:
        return (
            np.concatenate(
                [
                    _get_numpy(low).view(np.uint8).ravel(),
                    _get_numpy(mid).view(np.uint8).ravel(),
                    _get_numpy(hi).view(np.uint8).ravel(),
                ]
            ),
            "split",
        )
    return None, ""


def pick_tensor(by_name: dict[str, gguf.ReaderTensor], base: str, suffix: str) -> gguf.ReaderTensor | None:
    for key in (f"{base}.{suffix}", f"{base}_{suffix}"):
        if key in by_name:
            return by_name[key]
    return None


def dequant_one_base(by_name: dict[str, gguf.ReaderTensor], base: str) -> np.ndarray:
    pw, _stream_mode = build_packed_stream(by_name, base)
    if pw is None:
        raise KeyError(f"No packed byte stream for {base} (need `.packed_weights` or split `packed_values_*`).")

    pm_t = pick_tensor(by_name, base, "packed_mask")
    if pm_t is None:
        raise KeyError(f"Missing packed_mask for {base}")

    rows = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "rows")))
    cols = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "cols")))
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid rows/cols for {base}: {rows}x{cols}")

    n2 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_2bit_bytes")))
    n4 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_4bit_bytes")))
    n8 = read_scalar_i32(_get_numpy(pick_tensor(by_name, base, "n_8bit_bytes")))
    scale_low = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_low")), 1.0)
    scale_int4 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_int4")), 1.0)
    scale_int8 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "scale_int8")), 1.0)
    zp_low = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "zp_low")), 1.5)
    zp_int4 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "zp_int4")), 7.5)
    zp_int8 = read_scalar_f32(_get_numpy(pick_tensor(by_name, base, "zp_int8")), 127.5)

    bs_t = pick_tensor(by_name, base, "block_size")
    block_size_meta = read_scalar_i32(_get_numpy(bs_t)) if bs_t is not None else 0
    block_size = block_size_meta if block_size_meta > 0 else 32

    fv_t = pick_tensor(by_name, base, "fp16_values")
    if fv_t is None:
        raise KeyError(f"Missing fp16_values for {base}")
    fv_np = _get_numpy(fv_t)
    if fv_t.tensor_type == GGMLQuantizationType.F16:
        fp16_arr = fv_np.reshape(-1).view(np.float16)
    elif fv_t.tensor_type == GGMLQuantizationType.F32:
        fp16_arr = fv_np.reshape(-1).astype(np.float32)
    else:
        fp16_arr = fv_np.reshape(-1)

    fi_t = pick_tensor(by_name, base, "fp16_indices")
    # Legacy path only when explicit per-element fp16_indices exist *and* block_size is absent/zero.
    use_legacy = fi_t is not None and block_size_meta <= 0

    if use_legacy:
        w_f32 = unpack_layer_to_f32_legacy(
            pw,
            _get_numpy(pm_t).view(np.uint8),
            rows=rows,
            cols=cols,
            n_2bit_bytes=n2,
            n_4bit_bytes=n4,
            n_8bit_bytes=n8,
            scale_low=scale_low,
            scale_int4=scale_int4,
            scale_int8=scale_int8,
            zp_low=zp_low,
            zp_int4=zp_int4,
            zp_int8=zp_int8,
            fp16_indices=_get_numpy(fi_t).reshape(-1),
            fp16_values=fp16_arr,
        )
    else:
        w_f32 = unpack_layer_to_f32_blockwise(
            pw,
            _get_numpy(pm_t).view(np.uint8),
            rows=rows,
            cols=cols,
            block_size=block_size,
            n_2bit_bytes=n2,
            n_4bit_bytes=n4,
            n_8bit_bytes=n8,
            scale_low=scale_low,
            scale_int4=scale_int4,
            scale_int8=scale_int8,
            zp_low=zp_low,
            zp_int4=zp_int4,
            zp_int8=zp_int8,
            fp16_values=fp16_arr,
        )

    return w_f32.astype(np.float32)


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == gguf.Keys.General.ALIGNMENT:
            writer.add_custom_alignment(int(value))
        else:
            writer.add_key_value(field.name, value, val_type, sub_type=sub_type)


def should_skip_tensor(name: str, bases_converted: set[str]) -> bool:
    if name.endswith(".bias"):
        maybe_base = name[: -len(".bias")]
        if maybe_base in bases_converted:
            return False
    for base in bases_converted:
        if name == base:
            return True
        if name.startswith(base + "."):
            suffix = name[len(base) + 1 :]
            if suffix in PACKED_ARTIFACT_SUFFIXES:
                return True
        for suf in PACKED_ARTIFACT_SUFFIXES:
            if name == f"{base}_{suf}":
                return True
    return False


def emit_trigger_name(by_name: dict[str, gguf.ReaderTensor], base: str) -> str | None:
    """First tensor in reader order we intercept to splice in dense `{base}` — prefer `.packed_weights`."""
    if base + ".packed_weights" in by_name:
        return base + ".packed_weights"
    if base + "_packed_weights" in by_name:
        return base + "_packed_weights"
    if base + ".packed_mask" in by_name:
        return base + ".packed_mask"
    if base + "_packed_mask" in by_name:
        return base + "_packed_mask"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert SEPTQ packed BMO GGUF to dense FP16 weights.")
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="Input GGUF path.")
    ap.add_argument("--out", dest="out", type=Path, default=Path("bmo_fp16_test.gguf"))
    args = ap.parse_args()

    inp = args.inp.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not inp.is_file():
        print(f"[convert] Input not found: {inp}", file=sys.stderr)
        return 2

    reader = gguf.GGUFReader(str(inp))
    by_name = _tensor_dict(reader)
    all_names = set(by_name)

    bases_try = discover_packed_bases(all_names)
    print(f"[convert] discovered {len(bases_try)} packed bases", flush=True)
    bases_ok: list[str] = []
    dense_arrays: dict[str, np.ndarray] = {}

    import time as _time

    _t0 = _time.time()
    for i, base in enumerate(bases_try):
        try:
            arr_f32 = dequant_one_base(by_name, base)
            dense_arrays[base] = arr_f32.astype(np.float16, copy=False)
            bases_ok.append(base)
            elapsed = _time.time() - _t0
            print(
                f"[convert] [{i+1:3d}/{len(bases_try)}] dequantized {base} shape={arr_f32.shape} "
                f"({elapsed:6.1f}s elapsed)",
                flush=True,
            )
        except Exception as e:
            print(f"[convert] WARN skip base={base!r}: {e}", file=sys.stderr, flush=True)

    bases_set = set(bases_ok)
    trigger_map: dict[str, str] = {}
    for b in bases_ok:
        trig = emit_trigger_name(by_name, b)
        if trig is None:
            print(f"[convert] WARN no emit trigger tensor for base={b!r}; dense weight not inserted.", file=sys.stderr)
        else:
            trigger_map[trig] = b

    # Preserve reader order; splice dense tensor at the trigger tensor for each base.
    out_plan: list[tuple[str, np.ndarray, GGMLQuantizationType]] = []
    emitted: set[str] = set()

    for tensor in reader.tensors:
        if tensor.name in trigger_map:
            b = trigger_map[tensor.name]
            if b not in emitted:
                # dense_arrays[b] is already FP16 (we converted at dequant time to halve RAM).
                arr16 = dense_arrays[b]
                if arr16.dtype != np.float16:
                    arr16 = arr16.astype(np.float16, copy=False)
                out_plan.append((b, np.ascontiguousarray(arr16), GGMLQuantizationType.F16))
                emitted.add(b)
            continue
        if should_skip_tensor(tensor.name, bases_set):
            continue
        raw = _get_numpy(tensor)
        out_plan.append((tensor.name, raw, tensor.tensor_type))

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else "bmo"

    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer)

    largest: list[tuple[int, str, tuple[int, ...]]] = []

    for name, arr, raw_dtype in out_plan:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, raw_dtype=raw_dtype)
        largest.append((arr.nbytes, name, tuple(int(x) for x in arr.shape)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for name, arr, _rd in out_plan:
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)

    writer.close()

    out_size = out.stat().st_size
    largest.sort(reverse=True)
    top3 = largest[:3]

    print(f"[convert] Packed bases dequantized: {len(bases_ok)}")
    for b in bases_ok:
        sh = dense_arrays[b].shape
        print(f"           • {b} -> FP16 dense shape={sh}")

    print(f"[convert] Output GGUF path: {out}")
    print(f"[convert] Output file size: {out_size / (1024 ** 3):.4f} GiB ({out_size} bytes)")
    print("[convert] Top 3 largest tensors (output):")
    for nb, nm, sh in top3:
        print(f"           • {nm}: {nb / (1024 ** 2):.2f} MiB  shape={sh}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
