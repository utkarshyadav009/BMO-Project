#!/usr/bin/env python3
# flake8: noqa
"""convert_septq_to_fp16_tieronly.py

Rewrite SEPTQ-packed tensors to FP16-only tier (block mask tier 0 everywhere) while keeping the
packed GGUF layout expected by ``bmo_prepare_device_packed_tensors`` (``.packed_weights`` present,
``fp16_values`` block-major, zero-length quantized streams).

Usage:
    python convert_septq_to_fp16_tieronly.py --in bmo_septq_v3.gguf --out bmo_fp16_tieronly.gguf

Sanity listing:
    python -c "import gguf; r=gguf.GGUFReader('bmo_septq_v3.gguf'); ..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_GGUF_PY = _REPO_ROOT / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

import gguf  # type: ignore  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

from convert_septq_to_fp16 import (  # noqa: E402
    PACKED_ARTIFACT_SUFFIXES,
    copy_metadata,
    dequant_one_base,
    discover_packed_bases,
    pick_tensor,
    _get_numpy,
    _tensor_dict,
)

BLOCK_SIZE = 32

# Split-stream layouts (reader-side only; output always uses a single packed_weights blob).
_SPLIT_SUFFIXES: tuple[str, ...] = (
    "packed_values_low",
    "packed_values_int4",
    "packed_values_int8",
)

_LEGACY_ROWIDX_SUFFIXES: tuple[str, ...] = (
    "idx2_start",
    "idx4_start",
    "idx8_start",
)

_OPTIONAL_SUFFIXES: tuple[str, ...] = (
    "threshold_8bit",
    "threshold_4bit",
    "threshold_2bit",
    "packing_version",
)

_CANONICAL_ARTIFACT_ORDER: tuple[str, ...] = (
    "packed_mask",
    "packed_weights",
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
    *(_OPTIONAL_SUFFIXES),
    "fp16_values",
    "rows",
    "cols",
)


def pack_uint2_mask_le(mask_unpacked: np.ndarray) -> np.ndarray:
    """Same packing as ``export_bmo_gguf.pack_uint2_mask_le`` (avoids importing torch)."""
    flat = mask_unpacked.ravel().astype(np.uint8)
    rem = (-flat.size) % 4
    if rem:
        flat = np.concatenate([flat, np.zeros(rem, dtype=np.uint8)])
    flat4 = flat.reshape(-1, 4)
    packed = (
        flat4[:, 0] | (flat4[:, 1] << 2) | (flat4[:, 2] << 4) | (flat4[:, 3] << 6)
    ).astype(np.uint8)
    return packed


def tensor_key(base: str, suffix: str, sep: str) -> str:
    return f"{base}.{suffix}" if sep == "." else f"{base}_{suffix}"


def parse_suffix(base: str, name: str, sep: str) -> str | None:
    if sep == ".":
        prefix = base + "."
        if name.startswith(prefix):
            return name[len(prefix) :]
        return None
    prefix = base + "_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return None


def sep_for_base(by_name: dict[str, gguf.ReaderTensor], base: str) -> str:
    if tensor_key(base, "packed_weights", ".") in by_name:
        return "."
    if tensor_key(base, "packed_weights", "_") in by_name:
        return "_"
    if tensor_key(base, "packed_mask", ".") in by_name:
        return "."
    if tensor_key(base, "packed_mask", "_") in by_name:
        return "_"
    low_dot = tensor_key(base, "packed_values_low", ".")
    low_us = tensor_key(base, "packed_values_low", "_")
    if low_dot in by_name:
        return "."
    if low_us in by_name:
        return "_"
    return "."


def artifact_suffixes_present(by_name: dict[str, gguf.ReaderTensor], base: str, sep: str) -> set[str]:
    out: set[str] = set()
    for name in by_name:
        suf = parse_suffix(base, name, sep)
        if suf is None:
            continue
        if (
            suf in PACKED_ARTIFACT_SUFFIXES
            or suf in _SPLIT_SUFFIXES
            or suf in _LEGACY_ROWIDX_SUFFIXES
            or suf == "fp16_indices"
        ):
            out.add(suf)
    return out


def preferred_trigger_tensor(reader: gguf.GGUFReader, base: str, sep: str) -> str | None:
    """Prefer `.packed_weights` then `.packed_mask` / split streams so bundles splice onto stable keys."""
    by_name = _tensor_dict(reader)
    idx_map = {t.name: i for i, t in enumerate(reader.tensors)}
    prefs = (
        "packed_weights",
        "packed_mask",
        "packed_values_low",
        "packed_values_int4",
        "packed_values_int8",
    )
    candidates: list[tuple[int, int, str]] = []
    for pi, suf in enumerate(prefs):
        key = tensor_key(base, suf, sep)
        if key in by_name:
            candidates.append((pi, idx_map[key], key))
    if candidates:
        candidates.sort()
        return candidates[0][2]

    skip_trigger = {"fp16_indices"}
    present = artifact_suffixes_present(by_name, base, sep)
    for t in reader.tensors:
        suf = parse_suffix(base, t.name, sep)
        if suf is None or suf in skip_trigger:
            continue
        if suf in present:
            return t.name
    return None


def replicate_scalar_like(val: float | int, tmpl: gguf.ReaderTensor | None, *, as_float: bool) -> np.ndarray:
    """Fill a GGUF scalar tensor buffer matching ``tmpl`` shape/dtype (fallback: one-element I32/F32)."""
    if tmpl is None:
        dt = np.float32 if as_float else np.int32
        return np.array([val], dtype=dt)
    raw = _get_numpy(tmpl)
    dt = raw.dtype
    fill = np.array(val, dtype=dt).reshape(())
    return np.full(raw.shape, fill, dtype=dt)


def ggml_type_for_suffix(
    by_name: dict[str, gguf.ReaderTensor],
    base: str,
    sep: str,
    suffix: str,
    *,
    fallback: GGMLQuantizationType,
) -> GGMLQuantizationType:
    rt = pick_tensor(by_name, base, suffix)
    if rt is not None:
        return rt.tensor_type
    return fallback


def infer_fallback_types(by_name: dict[str, gguf.ReaderTensor]) -> dict[str, GGMLQuantizationType]:
    """First-seen GGML types keyed by artifact suffix (``.suffix`` or ``_suffix``)."""
    buckets: dict[str, GGMLQuantizationType] = {}
    wanted = set(PACKED_ARTIFACT_SUFFIXES) | set(_OPTIONAL_SUFFIXES)
    for suf in sorted(wanted):
        for name, t in by_name.items():
            if name.endswith("." + suf) or name.endswith("_" + suf):
                buckets[suf] = t.tensor_type
                break
    return buckets


def encode_fp16_tieronly_bundle(
    by_name: dict[str, gguf.ReaderTensor],
    base: str,
    sep: str,
    w_f32: np.ndarray,
    fallback_types: dict[str, GGMLQuantizationType],
) -> dict[str, tuple[np.ndarray, GGMLQuantizationType]]:
    rows, cols = int(w_f32.shape[0]), int(w_f32.shape[1])
    total = rows * cols
    n_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    tiers = np.zeros(n_blocks, dtype=np.uint8)
    packed_mask = pack_uint2_mask_le(tiers)
    pm_slots = int(packed_mask.size * 4)

    fp16_buf = np.zeros(pm_slots * BLOCK_SIZE, dtype=np.float16)
    w_flat = w_f32.reshape(-1).astype(np.float16, copy=False)
    fp16_buf[:total] = w_flat

    # Placeholder bytes never read by the kernel (all blocks are FP16 tier so 2/4/8-bit streams are
    # untouched). BUT bmo.cpp's loader splits tensors into a "scalar pool" (size <= SCALAR_MAX=4096)
    # and a "big pool" (> SCALAR_MAX). Only the big pool is `cudaHostRegister(...,cudaHostRegisterMapped)`,
    # so `cudaHostGetDevicePointer` on a tiny scalar-pool tensor returns "invalid argument" later in
    # `bmo_prepare_device_packed_tensors`. Pad above SCALAR_MAX so packed_weights lands in the big pool
    # and gets host-mapped. 8 KiB of zeros × 128 bases = 1 MiB total — negligible vs the 15.6 GiB GGUF.
    PLACEHOLDER_BYTES = 8192
    pw_placeholder = np.zeros(PLACEHOLDER_BYTES, dtype=np.uint8)

    def _ftype(suffix: str, fb: GGMLQuantizationType) -> GGMLQuantizationType:
        return ggml_type_for_suffix(by_name, base, sep, suffix, fallback=fallback_types.get(suffix, fb))

    def _scalar_int(name: str, val: int) -> tuple[np.ndarray, GGMLQuantizationType]:
        tmpl = pick_tensor(by_name, base, name)
        arr = replicate_scalar_like(val, tmpl, as_float=False)
        gt = _ftype(name, GGMLQuantizationType.I32)
        return arr, gt

    def _scalar_f32(name: str, val: float) -> tuple[np.ndarray, GGMLQuantizationType]:
        tmpl = pick_tensor(by_name, base, name)
        arr = replicate_scalar_like(val, tmpl, as_float=True)
        gt = _ftype(name, GGMLQuantizationType.F32)
        return arr, gt

    out: dict[str, tuple[np.ndarray, GGMLQuantizationType]] = {}

    out["packed_mask"] = (np.ascontiguousarray(packed_mask), _ftype("packed_mask", GGMLQuantizationType.I8))
    out["packed_weights"] = (np.ascontiguousarray(pw_placeholder), _ftype("packed_weights", GGMLQuantizationType.I8))

    out["n_2bit_bytes"] = _scalar_int("n_2bit_bytes", 0)
    out["n_4bit_bytes"] = _scalar_int("n_4bit_bytes", 0)
    out["n_8bit_bytes"] = _scalar_int("n_8bit_bytes", 0)
    out["block_size"] = _scalar_int("block_size", BLOCK_SIZE)
    out["n_blocks"] = _scalar_int("n_blocks", n_blocks)

    out["scale_low"] = _scalar_f32("scale_low", 1.0)
    out["scale_int4"] = _scalar_f32("scale_int4", 1.0)
    out["scale_int8"] = _scalar_f32("scale_int8", 1.0)
    out["zp_low"] = _scalar_f32("zp_low", 0.0)
    out["zp_int4"] = _scalar_f32("zp_int4", 0.0)
    out["zp_int8"] = _scalar_f32("zp_int8", 0.0)

    for opt in _OPTIONAL_SUFFIXES:
        tmpl_t = pick_tensor(by_name, base, opt)
        if tmpl_t is None:
            continue
        raw = _get_numpy(tmpl_t)
        out[opt] = (np.ascontiguousarray(raw.copy()), tmpl_t.tensor_type)

    fv_gt = _ftype("fp16_values", GGMLQuantizationType.F16)
    out["fp16_values"] = (np.ascontiguousarray(fp16_buf), fv_gt)

    out["rows"] = _scalar_int("rows", rows)
    out["cols"] = _scalar_int("cols", cols)

    return out


def should_skip_encoded_tensor(name: str, encoded_bases: set[str], sep_by_base: dict[str, str]) -> tuple[bool, str | None]:
    """Return (skip_verbatim, base_if_encoded)."""
    for base in encoded_bases:
        sep = sep_by_base[base]
        suf = parse_suffix(base, name, sep)
        if suf is None:
            continue
        if suf == "fp16_indices":
            return True, base
        if suf in PACKED_ARTIFACT_SUFFIXES or suf in _SPLIT_SUFFIXES or suf in _LEGACY_ROWIDX_SUFFIXES:
            return True, base
    return False, None


def print_sample_tensor_types(by_name: dict[str, gguf.ReaderTensor]) -> None:
    for prefer_dot in (True, False):
        suf = "packed_weights"
        for n in by_name:
            ends = f".{suf}" if prefer_dot else f"_{suf}"
            if n.endswith(ends):
                t = by_name[n]
                print(f"[inspect] sample {suf}: name={n!r} shape={t.shape} tensor_type={t.tensor_type}")
                break
    for prefer_dot in (True, False):
        suf = "fp16_values"
        for n in by_name:
            ends = f".{suf}" if prefer_dot else f"_{suf}"
            if n.endswith(ends):
                t = by_name[n]
                print(f"[inspect] sample {suf}: name={n!r} shape={t.shape} tensor_type={t.tensor_type}")
                break
    for scalar in ("n_2bit_bytes", "block_size", "scale_low"):
        for n in by_name:
            if n.endswith("." + scalar) or n.endswith("_" + scalar):
                t = by_name[n]
                print(f"[inspect] sample {scalar}: name={n!r} shape={t.shape} tensor_type={t.tensor_type}")
                break


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert SEPTQ GGUF to FP16-tier-only packed tensors (C++ packed path compatible)."
    )
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", type=Path, default=Path("bmo_fp16_tieronly.gguf"))
    args = ap.parse_args()

    inp = args.inp.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if not inp.is_file():
        print(f"[tieronly] Input not found: {inp}", file=sys.stderr)
        return 2

    reader = gguf.GGUFReader(str(inp))
    by_name = _tensor_dict(reader)

    print("[tieronly] Sample tensor types from input (first matches):", flush=True)
    print_sample_tensor_types(by_name)

    sep_summary: dict[str, str] = {}
    all_names = set(by_name)
    bases_try = discover_packed_bases(all_names)
    for b in bases_try:
        sep_summary[b] = sep_for_base(by_name, b)
    seps_used = sorted(set(sep_summary.values()))
    print(f"[tieronly] Separator style per base: unique={seps_used}", flush=True)
    if len(seps_used) == 1:
        print(f"[tieronly] Global separator: {seps_used[0]!r}", flush=True)
    else:
        print("[tieronly] Mixed separators across bases; per-base naming preserved.", flush=True)
    if "_" in seps_used:
        print(
            "[tieronly] WARN at least one base uses '_' artifact names; "
            "`bmo_prepare_device_packed_tensors` looks up `base + \".packed_weights\"` only — "
            "underscore GGUFs may not register packed tensors at runtime.",
            flush=True,
        )

    fallback_types = infer_fallback_types(by_name)

    bundles: dict[str, dict[str, tuple[np.ndarray, GGMLQuantizationType]]] = {}
    bases_ok: list[str] = []

    import time as _time

    _t0 = _time.time()
    for i, base in enumerate(bases_try):
        sep = sep_summary[base]
        try:
            w_f32 = dequant_one_base(by_name, base)
            bundles[base] = encode_fp16_tieronly_bundle(by_name, base, sep, w_f32, fallback_types)
            bases_ok.append(base)
            elapsed = _time.time() - _t0
            print(
                f"[tieronly] [{i+1:3d}/{len(bases_try)}] encoded {base} shape={w_f32.shape} sep={sep!r} "
                f"({elapsed:6.1f}s elapsed)",
                flush=True,
            )
        except Exception as e:
            print(f"[tieronly] WARN skip base={base!r}: {e}", file=sys.stderr, flush=True)

    encoded_set = set(bases_ok)
    sep_by_base = {b: sep_summary[b] for b in bases_ok}

    triggers: dict[str, str] = {}
    for b in bases_ok:
        trig = preferred_trigger_tensor(reader, b, sep_summary[b])
        if trig is None:
            print(f"[tieronly] WARN no artifact tensors to trigger splice for base={b!r}", file=sys.stderr)
        else:
            triggers[b] = trig

    emitted: set[str] = set()

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else "bmo"

    writer = gguf.GGUFWriter(out, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer)

    plan: list[tuple[str, np.ndarray, GGMLQuantizationType]] = []

    had_optional = {b: artifact_suffixes_present(by_name, b, sep_summary[b]) & set(_OPTIONAL_SUFFIXES) for b in bases_ok}

    def ordered_suffixes(base: str) -> Iterable[str]:
        present_opt = had_optional.get(base, set())
        for suf in _CANONICAL_ARTIFACT_ORDER:
            if suf in _OPTIONAL_SUFFIXES and suf not in present_opt:
                continue
            yield suf

    for tensor in reader.tensors:
        skip, enc_base = should_skip_encoded_tensor(tensor.name, encoded_set, sep_by_base)
        if skip and enc_base is not None and enc_base in triggers:
            if tensor.name == triggers[enc_base] and enc_base not in emitted:
                bundle = bundles[enc_base]
                sep = sep_summary[enc_base]
                for suf in ordered_suffixes(enc_base):
                    arr, gt = bundle[suf]
                    plan.append((tensor_key(enc_base, suf, sep), arr, gt))
                emitted.add(enc_base)
            continue

        raw = _get_numpy(tensor)
        plan.append((tensor.name, raw, tensor.tensor_type))

    for b in bases_ok:
        if b not in emitted:
            print(
                f"[tieronly] ERROR never emitted bundle for base={b!r} (trigger={triggers.get(b)})",
                file=sys.stderr,
            )
            return 3

    largest: list[tuple[int, str, tuple[int, ...]]] = []
    for name, arr, raw_dtype in plan:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, raw_dtype=raw_dtype)
        largest.append((arr.nbytes, name, tuple(int(x) for x in arr.shape)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for name, arr, _rd in plan:
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)

    writer.close()

    out_size = out.stat().st_size
    largest.sort(reverse=True)
    top3 = largest[:3]

    print(f"[tieronly] Re-encoded packed bases: {len(bases_ok)}")
    print(f"[tieronly] Output: {out}")
    print(f"[tieronly] Size: {out_size / (1024 ** 3):.4f} GiB ({out_size} bytes)")
    print("[tieronly] Top 3 largest tensors:")
    for nb, nm, sh in top3:
        print(f"           • {nm}: {nb / (1024 ** 2):.2f} MiB  shape={sh}")

    # Types used for standard artifacts (from first successful base).
    if bases_ok:
        b0 = bases_ok[0]
        print(f"[tieronly] GGML types used for artifacts (sample base {b0!r}):")
        for suf in _CANONICAL_ARTIFACT_ORDER:
            if suf not in bundles[b0]:
                continue
            gt = bundles[b0][suf][1]
            print(f"           • {suf}: {gt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
