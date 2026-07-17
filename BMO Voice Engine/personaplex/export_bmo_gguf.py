"""
export_bmo_gguf.py

Export a QAT-trained multi-tier Moshi model checkpoint (bmo_jetson_ready.pt)
to a packed GGUF file. Uses the official `gguf` Python package when available,
and falls back to writing a `.npz` archive if the installed gguf API is
incompatible.

Packing conventions (documented in the GGUF header/tensors):
- Little-endian packing for 2-bit tier masks: value[0] -> bits 0-1,
  value[1] -> bits 2-3, value[2] -> bits 4-5, value[3] -> bits 6-7.

This script follows the MultiTierLinearInference conventions in
`profile_jetson.py` and reconstructs per-weight quantization then stores
compact packed blobs per-layer plus metadata needed for dequantization.

Usage: pip install gguf torch numpy
       python export_bmo_gguf.py bmo_jetson_ready.pt out.gguf
"""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Try to import gguf; search in llama.cpp/gguf-py locally first.
_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_GGUF_PY = _REPO_ROOT / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

try:
    import gguf
except Exception:
    gguf = None

BLOCK_SIZE = 32
DEFAULT_RATIO_FP16 = 0.02
DEFAULT_RATIO_INT8 = 0.12
DEFAULT_RATIO_INT4 = 0.36


def canonical_transformer_multitier_gguf_base(underscore_base: str) -> str:
    """SEPTQ multitier stores paths like ``transformer.inner.layers.N.*``.

    Those become ``transformer_inner_layers_N_*`` when dots are replaced with underscores,
    but ``bmo.cpp`` only loads the canonical ``transformer_layers_N_*`` tensor names.
    Rewrite the prefix so packed tensors remain discoverable on-device.
    """
    return re.sub(r"^transformer_inner_layers_(\d+)_", r"transformer_layers_\1_", underscore_base)


def pick_transformer_param_key(state_dict: Dict[str, Any], layer_idx: int, dotted_suffix: str) -> str | None:
    """Prefer ``transformer.layers`` checkpoints; fall back to ``transformer.inner.layers``."""
    for pref in ("transformer.layers", "transformer.inner.layers"):
        k = f"{pref}.{layer_idx}.{dotted_suffix}"
        if k in state_dict:
            return k
    return None


def unpack_tier_mask_uint2(packed: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    total = int(target_shape[0]) * int(target_shape[1])
    expanded = torch.zeros(total, dtype=torch.uint8, device=packed.device)
    for i in range(4):
        expanded[i::4] = (packed >> (i * 2)) & 0b11
    return expanded[:total].reshape(target_shape)


def pack_uint2_mask_le(mask_unpacked: np.ndarray) -> np.ndarray:
    """Pack unpacked uint2 mask values (0..3) into bytes, 4 values per byte,
    little-endian bit ordering (lowest two bits first).
    """
    flat = mask_unpacked.ravel().astype(np.uint8)
    # pad to multiple of 4
    rem = (-flat.size) % 4
    if rem:
        flat = np.concatenate([flat, np.zeros(rem, dtype=np.uint8)])
    flat4 = flat.reshape(-1, 4)
    packed = (flat4[:, 0] | (flat4[:, 1] << 2) | (flat4[:, 2] << 4) | (flat4[:, 3] << 6)).astype(np.uint8)
    return packed


def pack_2bit_values_le(values: np.ndarray) -> np.ndarray:
    """Pack 2-bit values (0..3) 4-per-byte little-endian.
    """
    flat = values.ravel().astype(np.uint8)
    rem = (-flat.size) % 4
    if rem:
        flat = np.concatenate([flat, np.zeros(rem, dtype=np.uint8)])
    flat4 = flat.reshape(-1, 4)
    packed = (flat4[:, 0] | (flat4[:, 1] << 2) | (flat4[:, 2] << 4) | (flat4[:, 3] << 6)).astype(np.uint8)
    return packed


def pack_4bit_values_le(values: np.ndarray) -> np.ndarray:
    """Pack 4-bit values (0..15) 2-per-byte little-endian.
    """
    flat = values.ravel().astype(np.uint8)
    rem = (-flat.size) % 2
    if rem:
        flat = np.concatenate([flat, np.zeros(rem, dtype=np.uint8)])
    flat2 = flat.reshape(-1, 2)
    packed = (flat2[:, 0] | (flat2[:, 1] << 4)).astype(np.uint8)
    return packed


def _affine_params_for_values(values: np.ndarray, qmax: int, fallback_zp: float) -> tuple[float, float]:
    if values.size == 0:
        return 1.0, fallback_zp
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return 1.0, fallback_zp
    scale = max((vmax - vmin) / float(qmax), 1e-12)
    zp = float(np.clip(np.round(-vmin / scale), 0, qmax))
    return float(scale), zp


def _resolve_path_for_ckpt(base_dir: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    cand = path.resolve()
    if cand.exists():
        return cand
    return (base_dir / path).resolve()


def _tier_masks_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    raw = ckpt.get("tier_masks_uint2")
    if not isinstance(raw, dict) or not raw:
        sm = ckpt.get("septq_meta")
        if isinstance(sm, dict):
            raw = sm.get("tier_masks_uint2")
    return raw if isinstance(raw, dict) else {}


def _merged_tier_masks_uint2(qat_ckpt: Dict[str, Any], stats_ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """PTQ-output checkpoints usually carry ``tier_masks_uint2``; QAT checkpoints (e.g. ``qat_best.pt``) often do not.

    Merge stats-source (PTQ) masks first, then apply any masks present on the QAT checkpoint (QAT wins on key
    collision). Used by the legacy ``pack_temporal_tensor`` path so ``create_packed_layer`` receives per-element
    bytes without requiring the multi-tier ``candidate_layers`` discovery loop to fire.
    """
    a = _tier_masks_from_ckpt(stats_ckpt)
    b = _tier_masks_from_ckpt(qat_ckpt)
    out: Dict[str, Any] = {}
    if isinstance(a, dict):
        out.update(a)
    if isinstance(b, dict):
        out.update(b)
    return out


def _lookup_tier_mask_uint2_for_weight_key(tier_masks: Dict[str, Any], weight_state_key: str) -> torch.Tensor | None:
    """Resolve ``tier_masks_uint2[module]`` from a ``state_dict`` weight key like ``....in_proj.weight``."""
    if not tier_masks or not isinstance(weight_state_key, str) or not weight_state_key.strip():
        return None
    candidates: list[str] = [weight_state_key]
    for suf in (".weight", ".q_weight"):
        if weight_state_key.endswith(suf):
            base = weight_state_key[: -len(suf)]
            if base and base not in candidates:
                candidates.append(base)
            break
    for cand in candidates:
        for name in _canonical_module_name_aliases(cand):
            t = tier_masks.get(name)
            if torch.is_tensor(t):
                return t
    return None


def _septq_meta_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, Any] | None:
    sm = ckpt.get("septq_meta")
    return sm if isinstance(sm, dict) else None


def resolve_stats_source_ckpt(
    qat_ckpt: Dict[str, Any],
    qat_path: Path,
    septq_ckpt_path: str | Path | None,
) -> Dict[str, Any]:
    """Checkpoint whose ``septq_meta.per_layer_stats`` holds PTQ-era quant scales (QAT may omit them).

    Same resolution rules as ``compare_fakequant_vs_gguf_weights.resolve_quant_checkpoint``.
    """
    if septq_ckpt_path is not None and str(septq_ckpt_path).strip():
        src_path = _resolve_path_for_ckpt(qat_path.parent, septq_ckpt_path)
        if not src_path.is_file():
            raise FileNotFoundError(f"--septq-ckpt not found: {src_path}")
        src = torch.load(str(src_path), map_location="cpu")
        if not isinstance(src, dict):
            raise ValueError(f"SEPTQ checkpoint must be a dict, got {type(src)}")
        sm = _septq_meta_from_ckpt(src)
        if not sm or not isinstance(sm.get("per_layer_stats"), list):
            raise ValueError(f"{src_path}: missing septq_meta.per_layer_stats")
        return src

    tier = _tier_masks_from_ckpt(qat_ckpt)
    sm = _septq_meta_from_ckpt(qat_ckpt)
    if tier and sm and isinstance(sm.get("per_layer_stats"), list) and len(sm["per_layer_stats"]) > 0:
        return qat_ckpt

    qm = qat_ckpt.get("qat_meta")
    rel = None
    if isinstance(qm, dict):
        rel = qm.get("source_student_quant_meta")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(
            "qat_meta.source_student_quant_meta is missing or empty; pass --septq-ckpt to the multitier PTQ .pt "
            "that contains septq_meta.per_layer_stats."
        )
    src_path = _resolve_path_for_ckpt(qat_path.parent, rel)
    if not src_path.is_file():
        raise FileNotFoundError(f"source_student_quant_meta path not found: {src_path}")
    src = torch.load(str(src_path), map_location="cpu")
    if not isinstance(src, dict):
        raise ValueError(f"Source quant file must be a dict, got {type(src)}")
    sm2 = _septq_meta_from_ckpt(src)
    if not sm2 or not isinstance(sm2.get("per_layer_stats"), list):
        raise ValueError(f"{src_path}: missing septq_meta.per_layer_stats")
    return src


def _module_stat_name(mod: Dict[str, Any]) -> str | None:
    name = mod.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    for k in ("module_name", "module"):
        v = mod.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _canonical_module_name_aliases(full_name: str) -> list[str]:
    """State dict may use ``transformer.layers`` while stats use ``transformer.inner.layers`` (or vice versa)."""
    names = [full_name]
    if "transformer.inner.layers." in full_name:
        names.append(full_name.replace("transformer.inner.layers.", "transformer.layers.", 1))
    elif full_name.startswith("transformer.layers."):
        names.append(full_name.replace("transformer.layers.", "transformer.inner.layers.", 1))
    return list(dict.fromkeys(names))


def build_module_meta_lookup(stats_ckpt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map full module / state_dict weight names -> per-module PTQ stats dict (from ``per_layer_stats``)."""
    septq_meta = stats_ckpt.get("septq_meta", {})
    if not isinstance(septq_meta, dict):
        return {}
    per_layer_stats = septq_meta.get("per_layer_stats", [])
    if not isinstance(per_layer_stats, list):
        return {}

    module_meta_lookup: Dict[str, Dict[str, Any]] = {}
    for layer in per_layer_stats:
        if not isinstance(layer, dict):
            continue
        modules = layer.get("modules")
        if not isinstance(modules, list):
            continue
        for mod in modules:
            if not isinstance(mod, dict):
                continue
            name = _module_stat_name(mod)
            if not name:
                continue
            for alias in _canonical_module_name_aliases(name):
                module_meta_lookup[alias] = mod
    return module_meta_lookup


def _ratio_counts(n_blocks: int, ratio_fp16: float, ratio_int8: float, ratio_int4: float) -> tuple[int, int, int]:
    n_fp16 = int(round(float(ratio_fp16) * n_blocks))
    n_int8 = int(round(float(ratio_int8) * n_blocks))
    n_int4 = int(round(float(ratio_int4) * n_blocks))
    n_fp16 = max(0, min(n_blocks, n_fp16))
    n_int8 = max(0, min(n_blocks - n_fp16, n_int8))
    n_int4 = max(0, min(n_blocks - n_fp16 - n_int8, n_int4))
    return n_fp16, n_int8, n_int4


# Conservative hard limit for a single GGUF tensor payload (many writers use uint32 sizes).
_GGUF_SINGLE_TENSOR_MAX_BYTES = (1 << 31) - 1


def _assert_single_tensor_payload_fits(tensor_label: str, nbytes: int) -> None:
    if nbytes < 0:
        raise ValueError(f"{tensor_label}: invalid negative nbytes={nbytes}")
    if nbytes > _GGUF_SINGLE_TENSOR_MAX_BYTES:
        raise RuntimeError(
            f"{tensor_label}: tensor payload is {nbytes} bytes, which exceeds the exporter's "
            f"hard limit ({_GGUF_SINGLE_TENSOR_MAX_BYTES} bytes). Refusing to write a likely-truncated GGUF."
        )


def create_packed_layer(
    layer_name: str,
    dense_weight: torch.Tensor,
    packed_mask_tensor: torch.Tensor | None,
    module_meta: Dict[str, Any],
    bias: torch.Tensor | None = None,
    *,
    block_size: int = BLOCK_SIZE,
    ratio_fp16: float = DEFAULT_RATIO_FP16,
    ratio_int8: float = DEFAULT_RATIO_INT8,
    ratio_int4: float = DEFAULT_RATIO_INT4,
    block_tier_map_tensor: torch.Tensor | None = None,
    export_gguf_base: str | None = None,
    require_ptq_scales: bool = False,
) -> Dict[str, Any]:
    """Return SEPTQ packed artifacts for this layer.

    Mask layouts:
      - v4 (``packing_version=3``): ``packed_mask`` stores 2 bits per **32-element block**
        (4 blocks per uint8), matching legacy CUDA v2 expectations.
      - v5 (``packing_version=5``): ``packed_mask`` stores 2 bits per **element** over the
        padded weight grid (``rows * padded_cols`` elems, 4 elems/byte). When
        ``packed_mask_tensor`` matches the per-element byte length, the exporter **serializes
        those bytes verbatim** (no max(abs), no majority vote, no re-tiering).

    The returned dict contains at minimum:
      - packed_mask: uint8 bytes (uint2 tiers packed 4-per-byte, little-endian lanes)
      - packed_weights: concatenated [2bit_bytes | 4bit_bytes | 8bit_bytes]
      - n_2bit_bytes, n_4bit_bytes, n_8bit_bytes: int sizes
      - scale_low, scale_int4, scale_int8 (float32)
      - fp16_values (float16)
      - bias (float32) if provided
    """
    if block_size != 32:
        raise RuntimeError("SEPTQ v3 CUDA path currently expects block_size=32")

    w = dense_weight.detach().cpu().float()
    rows, cols = int(w.shape[0]), int(w.shape[1])
    n_blocks_per_row = (cols + block_size - 1) // block_size
    n_blocks = rows * n_blocks_per_row
    padded_cols = n_blocks_per_row * block_size
    total = rows * cols
    if padded_cols != cols:
        w_padded = torch.zeros((rows, padded_cols), dtype=torch.float32)
        w_padded[:, :cols] = w
    else:
        w_padded = w
    blocks = (
        w_padded.reshape(rows, n_blocks_per_row, block_size)
        .numpy()
        .astype(np.float32, copy=False)
        .reshape(n_blocks, block_size)
    )
    block_max = np.max(np.abs(blocks), axis=1)

    per_element_v5 = False
    packed_mask_bytes: np.ndarray | None = None
    tier_flat: np.ndarray | None = None
    if packed_mask_tensor is not None:
        pm_flat = packed_mask_tensor.detach().cpu().contiguous().view(-1)
        if pm_flat.dtype != torch.uint8:
            pm_flat = pm_flat.to(torch.uint8)
        need_elem_b = (rows * padded_cols + 3) // 4
        need_block_b = (n_blocks + 3) // 4
        n_pm = int(pm_flat.numel())
        if n_pm == need_elem_b:
            per_element_v5 = True
            _assert_single_tensor_payload_fits(f"{layer_name}.packed_mask", int(need_elem_b))
            mask_np = pm_flat.numpy().astype(np.uint8, copy=False)
            packed_mask_bytes = np.ascontiguousarray(mask_np)
            idx = np.arange(rows * padded_cols, dtype=np.int64)
            byte_ix = idx // 4
            shift = (idx % 4) * 2
            tier_flat = ((packed_mask_bytes[byte_ix] >> shift) & 3).astype(np.uint8, copy=False)
            w_flat_chk = w_padded.contiguous().reshape(-1).numpy().astype(np.float32, copy=False)
            if int(tier_flat.shape[0]) != int(w_flat_chk.shape[0]):
                raise RuntimeError(
                    f"{layer_name}: internal shape mismatch tiers={tier_flat.shape} weights={w_flat_chk.shape}"
                )
        elif n_pm != need_block_b:
            raise ValueError(
                f"{layer_name}: packed_mask has {n_pm} bytes; expected {need_elem_b} (per-element v5) "
                f"or {need_block_b} (per-block v4) for rows={rows} cols={cols} padded_cols={padded_cols} "
                f"n_blocks={n_blocks}"
            )

    def _maybe_meta_float(*keys: str) -> float | None:
        for key in keys:
            v = module_meta.get(key, None)
            if v is None:
                continue
            if torch.is_tensor(v):
                return float(v.detach().item())
            return float(v)
        return None

    threshold_8bit = _maybe_meta_float("threshold_8bit", "septq_threshold_8bit")
    threshold_4bit = _maybe_meta_float("threshold_4bit", "septq_threshold_4bit")
    threshold_2bit = _maybe_meta_float("threshold_2bit", "septq_threshold_2bit")
    used_block_tier_map = False
    if per_element_v5:
        if block_tier_map_tensor is not None:
            print(
                f"[export] {export_gguf_base or layer_name}: ignoring block_tier_map for {layer_name}: "
                "per-element packed_mask bytes are authoritative (v5).",
                file=sys.stderr,
            )
        used_block_tier_map = False
        threshold_8bit = float("nan")
        threshold_4bit = float("nan")
        threshold_2bit = float("nan")
        assert tier_flat is not None
        block_tiers = np.zeros(0, dtype=np.uint8)  # unused in v5 path
    elif block_tier_map_tensor is not None:
        block_tier_map = block_tier_map_tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        expected_shape = (rows, n_blocks_per_row)
        if tuple(block_tier_map.shape) != expected_shape:
            raise ValueError(
                f"{layer_name}: block_tier_map shape mismatch: got {tuple(block_tier_map.shape)} expected {expected_shape}"
            )
        block_tier_map_np = block_tier_map.numpy()
        if np.any(block_tier_map_np > 3):
            raise ValueError(f"{layer_name}: block_tier_map contains values outside [0, 3]")

        if packed_mask_tensor is not None:
            unpacked = unpack_tier_mask_uint2(
                packed_mask_tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
                (rows, cols),
            ).numpy()
            for r in range(rows):
                for b in range(n_blocks_per_row):
                    start = b * block_size
                    end = min(cols, start + block_size)
                    if end <= start:
                        continue
                    vals = unpacked[r, start:end]
                    unique_vals = np.unique(vals)
                    if unique_vals.size != 1:
                        raise ValueError(
                            f"{layer_name}: non-uniform tier in row={r} block={b}; values={unique_vals.tolist()}"
                        )
                    canonical_from_mask = np.uint8(3 - int(unique_vals[0]))
                    if canonical_from_mask != block_tier_map_np[r, b]:
                        raise ValueError(
                            f"{layer_name}: block_tier_map mismatch at row={r} block={b}: "
                            f"mask={int(canonical_from_mask)} map={int(block_tier_map_np[r, b])}"
                        )

        # Internal encoding remains 0=FP16,1=INT8,2=INT4,3=INT2.
        block_tiers = (3 - block_tier_map_np).reshape(-1).astype(np.uint8, copy=False)
        used_block_tier_map = True
        threshold_8bit = float("nan")
        threshold_4bit = float("nan")
        threshold_2bit = float("nan")
    elif not per_element_v5:
        sys.stderr.write(
            f"[WARN] {export_gguf_base or layer_name}: block_tier_map missing in checkpoint; "
            "falling back to max(abs)-based block tier selection.\n"
        )
        block_tiers = np.full(n_blocks, 3, dtype=np.uint8)
        if threshold_8bit is not None and threshold_4bit is not None and threshold_2bit is not None:
            block_tiers[block_max > threshold_2bit] = 2
            block_tiers[block_max > threshold_4bit] = 1
            block_tiers[block_max > threshold_8bit] = 0
        else:
            # The existing SEPTQ v1 checkpoint stores ratios rather than explicit
            # thresholds. Derive block thresholds by ranking max-abs block scores.
            order = np.argsort(-block_max, kind="stable")
            n_fp16_blocks, n_int8_blocks, n_int4_blocks = _ratio_counts(
                n_blocks,
                float(module_meta.get("fp16_ratio_real", ratio_fp16)),
                float(module_meta.get("int8_ratio_real", ratio_int8)),
                float(module_meta.get("int4_ratio_real", ratio_int4)),
            )
            block_tiers[order[:n_fp16_blocks]] = 0
            block_tiers[order[n_fp16_blocks:n_fp16_blocks + n_int8_blocks]] = 1
            block_tiers[order[n_fp16_blocks + n_int8_blocks:n_fp16_blocks + n_int8_blocks + n_int4_blocks]] = 2

            def _boundary(count: int) -> float:
                if count <= 0:
                    return float("inf")
                if count >= n_blocks:
                    return float("-inf")
                return float(np.nextafter(block_max[order[count - 1]], -np.inf))

            threshold_8bit = _boundary(n_fp16_blocks)
            threshold_4bit = _boundary(n_fp16_blocks + n_int8_blocks)
            threshold_2bit = _boundary(n_fp16_blocks + n_int8_blocks + n_int4_blocks)

    if per_element_v5:
        assert tier_flat is not None
        w_flat_f32 = w_padded.contiguous().reshape(-1).numpy().astype(np.float32, copy=False)
        values_int2 = w_flat_f32[tier_flat == 3]
        values_int4 = w_flat_f32[tier_flat == 2]
        values_int8 = w_flat_f32[tier_flat == 1]
    else:
        values_int2 = blocks[block_tiers == 3].reshape(-1)
        values_int4 = blocks[block_tiers == 2].reshape(-1)
        values_int8 = blocks[block_tiers == 1].reshape(-1)

    fallback_low = _affine_params_for_values(values_int2, 3, 1.5)
    fallback_int4 = _affine_params_for_values(values_int4, 15, 7.5)
    fallback_int8 = _affine_params_for_values(values_int8, 255, 127.5)

    def _meta_float(key: str) -> float | None:
        v = module_meta.get(key)
        if v is None:
            return None
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)

    low_s = _meta_float("quant_scale_low")
    if low_s is None:
        low_s = _meta_float("quant_scale")
    low_z = _meta_float("quant_zero_point_low")
    if low_z is None:
        low_z = _meta_float("quant_zero_point")
    if low_s is not None and low_z is not None:
        scale_low, zp_low = low_s, low_z
        low_ptq = True
    else:
        scale_low, zp_low = float(fallback_low[0]), float(fallback_low[1])
        low_ptq = False

    s4, z4 = _meta_float("quant_scale_int4"), _meta_float("quant_zero_point_int4")
    if s4 is not None and z4 is not None:
        scale_int4, zp_int4 = s4, z4
        int4_ptq = True
    else:
        scale_int4, zp_int4 = float(fallback_int4[0]), float(fallback_int4[1])
        int4_ptq = False

    s8, z8 = _meta_float("quant_scale_int8"), _meta_float("quant_zero_point_int8")
    if s8 is not None and z8 is not None:
        scale_int8, zp_int8 = s8, z8
        int8_ptq = True
    else:
        scale_int8, zp_int8 = float(fallback_int8[0]), float(fallback_int8[1])
        int8_ptq = False

    full_ptq = low_ptq and int4_ptq and int8_ptq
    if require_ptq_scales and not full_ptq:
        log_name = export_gguf_base or layer_name
        raise ValueError(
            f"--require-ptq-scales: expected full PTQ quant_scale_* / quant_zero_point_* in per_layer_stats for "
            f"{layer_name!r} (GGUF base {log_name!r}); got low_ptq={low_ptq} int4_ptq={int4_ptq} int8_ptq={int8_ptq}. "
            f"Ensure export resolves the student PTQ checkpoint (qat_meta.source_student_quant_meta or --septq-ckpt)."
        )

    for tag, sc in (("scale_low", scale_low), ("scale_int4", scale_int4), ("scale_int8", scale_int8)):
        if not math.isfinite(sc) or sc <= 0.0:
            raise ValueError(f"Invalid {tag}={sc!r} for {layer_name!r} (must be finite and > 0)")
    if not (0.0 <= zp_low <= 3.0):
        raise ValueError(f"Invalid zp_low={zp_low!r} for {layer_name!r} (expected in [0, 3])")
    if not (0.0 <= zp_int4 <= 15.0):
        raise ValueError(f"Invalid zp_int4={zp_int4!r} for {layer_name!r} (expected in [0, 15])")
    if not (0.0 <= zp_int8 <= 255.0):
        raise ValueError(f"Invalid zp_int8={zp_int8!r} for {layer_name!r} (expected in [0, 255])")

    log_label = export_gguf_base or layer_name
    tag = "PTQ" if full_ptq else "FALLBACK"
    print(
        f"[export] {log_label}: scales ({tag}): low={scale_low:g} int4={scale_int4:g} int8={scale_int8:g}  "
        f"zp: low={zp_low:g} int4={zp_int4:g} int8={zp_int8:g}",
        file=sys.stderr,
    )

    q3_vals = np.clip(np.round(values_int2 / scale_low + zp_low), 0, 3).astype(np.uint8)
    q2_vals = np.clip(np.round(values_int4 / scale_int4 + zp_int4), 0, 15).astype(np.uint8)
    q1_vals = np.clip(np.round(values_int8 / scale_int8 + zp_int8), 0, 255).astype(np.uint8)
    if per_element_v5:
        assert tier_flat is not None
        fp16_values = w_flat_f32[tier_flat == 0].astype(np.float16)
    else:
        fp16_values = blocks[block_tiers == 0].reshape(-1).astype(np.float16)

    packed_2 = pack_2bit_values_le(q3_vals) if q3_vals.size else np.zeros(0, dtype=np.uint8)
    packed_4 = pack_4bit_values_le(q2_vals) if q2_vals.size else np.zeros(0, dtype=np.uint8)
    packed_8 = q1_vals.view(np.uint8) if q1_vals.size else np.zeros(0, dtype=np.uint8)
    packed_weights = np.concatenate([packed_2, packed_4, packed_8])

    out: Dict[str, Any] = {}
    if per_element_v5:
        assert packed_mask_bytes is not None
        out["packed_mask"] = packed_mask_bytes
        if int(np.sum(tier_flat == 0)) != int(fp16_values.size):
            raise ValueError(
                f"{layer_name}: fp16_values length {int(fp16_values.size)} does not match tier==0 count "
                f"{int(np.sum(tier_flat == 0))} in per-element mask"
            )
    else:
        out["packed_mask"] = pack_uint2_mask_le(block_tiers)
    out["packed_weights"] = packed_weights
    out["n_2bit_bytes"] = np.int32(packed_2.size)
    out["n_4bit_bytes"] = np.int32(packed_4.size)
    out["n_8bit_bytes"] = np.int32(packed_8.size)
    out["block_size"] = np.int32(block_size)
    out["n_blocks"] = np.int32(n_blocks)
    out["scale_low"] = np.float32(scale_low)
    out["scale_int4"] = np.float32(scale_int4)
    out["scale_int8"] = np.float32(scale_int8)
    out["zp_low"] = np.float32(zp_low)
    out["zp_int4"] = np.float32(zp_int4)
    out["zp_int8"] = np.float32(zp_int8)
    out["threshold_8bit"] = np.float32(threshold_8bit)
    out["threshold_4bit"] = np.float32(threshold_4bit)
    out["threshold_2bit"] = np.float32(threshold_2bit)
    out["fp16_values"] = fp16_values
    out["packing_version"] = np.int32(5 if per_element_v5 else 3)
    if bias is not None:
        out["bias"] = bias.detach().cpu().numpy().astype(np.float32)

    # Also include original dense shape for reference
    out["rows"] = np.int32(rows)
    out["cols"] = np.int32(cols)

    out["_scale_source"] = "ptq" if full_ptq else "fallback"
    if per_element_v5:
        assert tier_flat is not None
        n_elems = int(rows * padded_cols)
        n_fp16_blocks = int(np.sum(tier_flat == 0))
        n_int8_blocks = int(np.sum(tier_flat == 1))
        n_int4_blocks = int(np.sum(tier_flat == 2))
        n_int2_blocks = int(np.sum(tier_flat == 3))
        frac_fp16 = float(n_fp16_blocks / max(1, n_elems))
        frac_int8 = float(n_int8_blocks / max(1, n_elems))
        frac_int4 = float(n_int4_blocks / max(1, n_elems))
        frac_int2 = float(n_int2_blocks / max(1, n_elems))
    else:
        n_fp16_blocks = int(np.sum(block_tiers == 0))
        n_int8_blocks = int(np.sum(block_tiers == 1))
        n_int4_blocks = int(np.sum(block_tiers == 2))
        n_int2_blocks = int(np.sum(block_tiers == 3))
        frac_fp16 = float(n_fp16_blocks / max(1, n_blocks))
        frac_int8 = float(n_int8_blocks / max(1, n_blocks))
        frac_int4 = float(n_int4_blocks / max(1, n_blocks))
        frac_int2 = float(n_int2_blocks / max(1, n_blocks))
    effective_bits = (
        16.0 * frac_fp16
        + 8.0 * frac_int8
        + 4.0 * frac_int4
        + 2.0 * frac_int2
    )
    packed_bytes = int(
        packed_2.size + packed_4.size + packed_8.size + out["packed_mask"].nbytes + fp16_values.nbytes
    )
    out["_tier_summary"] = {
        "module_name": str(layer_name),
        "n_blocks": int(n_blocks),
        "frac_fp16": float(frac_fp16),
        "frac_int8": float(frac_int8),
        "frac_int4": float(frac_int4),
        "frac_int2": float(frac_int2),
        "effective_bits_per_weight": float(effective_bits),
        "packed_mb": float(packed_bytes / (1024.0 * 1024.0)),
        "used_block_tier_map": bool(used_block_tier_map),
        "per_element_mask": bool(per_element_v5),
    }
    return out


def write_with_gguf(out_path: str, blobs: Dict[str, Any]) -> None:
    """Attempt to write blobs dict to a GGUF file using the gguf package.

    This function tries common writer APIs in the wild and will raise an
    informative error if none are found.
    """
    if gguf is None:
        raise RuntimeError("gguf package not found; please `pip install gguf`")

    # Find a writer class or factory
    Writer = None
    for cand in ("GGUFWriter", "GgufWriter", "GGUF", "Writer"):
        if hasattr(gguf, cand):
            Writer = getattr(gguf, cand)
            break

    if Writer is None:
        # Some distributions provide a flat 'save' function that accepts dict
        if hasattr(gguf, "save"):
            try:
                gguf.save(out_path, blobs)
                return
            except Exception as e:
                raise RuntimeError("gguf.save failed: " + str(e))
        raise RuntimeError("Could not find a GGUF writer API in the installed gguf package")

    # Try to construct writer and add tensors. Support both add and add_tensor names.
    writer = None
    try:
        writer = Writer(out_path, "bmo")
    except Exception:
        try:
            writer = Writer(out_path)
        except Exception:
            # maybe it's used as context manager/constructor signature differs
            try:
                writer = Writer(filepath=out_path)
            except Exception as e:
                raise RuntimeError("Unable to instantiate gguf writer: " + str(e))

    # common add method names we will probe
    add_methods = [m for m in ("add", "add_tensor", "add_array", "add_numpy") if hasattr(writer, m)]
    if not add_methods:
        # Some writers use attribute-like assignment; try 'set_tensor' or similar
        add_methods = [m for m in ("set_tensor",) if hasattr(writer, m)]
    if not add_methods:
        raise RuntimeError("gguf writer does not expose a known add() API")

    # use numpy arrays for writing
    for name, val in blobs.items():
        is_qtype = False
        qtype = None
        
        if isinstance(val, dict) and "qtype" in val:
            arr = val["qdata"]
            qtype = val["qtype"]
            is_qtype = True
        else:
            arr = val
            if torch.is_tensor(arr):
                arr = arr.detach().cpu().numpy()
            if np.isscalar(arr) or isinstance(arr, np.generic):
                arr = np.array([arr])
            if not isinstance(arr, np.ndarray):
                try:
                    arr = np.array(arr)
                except Exception:
                    continue
            if arr.ndim == 0:
                arr = arr.reshape(1)
                
        if isinstance(arr, np.ndarray) and arr.size > 0:
            _assert_single_tensor_payload_fits(str(name), int(arr.nbytes))
            
        if not is_qtype and hasattr(arr, 'dtype') and arr.dtype.kind == 'u':
            mapping = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
            arr = arr.view(mapping.get(arr.dtype.itemsize, np.int8))

        added = False
        for m in add_methods:
            try:
                if is_qtype:
                    getattr(writer, m)(name, arr, raw_dtype=qtype)
                else:
                    getattr(writer, m)(name, arr)
                added = True
                break
            except TypeError:
                # try alternative call signatures
                try:
                    getattr(writer, m)(name, arr, dtype=arr.dtype)
                    added = True
                    break
                except Exception:
                    continue
            except Exception as e:
                raise
        if not added:
            raise RuntimeError(f"Failed to add tensor {name} to gguf writer")

    # try close/write
    if hasattr(writer, "write_header_to_file"):
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    elif hasattr(writer, "write"):
        writer.write()
    elif hasattr(writer, "close"):
        writer.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to bmo_jetson_ready.pt checkpoint")
    p.add_argument("out", help="Output path (gguf recommended, .npz fallback)")
    p.add_argument(
        "--septq-ckpt",
        default=None,
        help="Optional multitier PTQ .pt with septq_meta.per_layer_stats (default: resolve via qat_meta.source_student_quant_meta).",
    )
    p.add_argument(
        "--require-ptq-scales",
        action="store_true",
        help="Fail export if any packed tensor lacks full PTQ quant_scale_* / quant_zero_point_* in per_layer_stats.",
    )
    args = p.parse_args()

    ckpt_path = args.ckpt
    out_path = args.out
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found:", ckpt_path)
        sys.exit(2)

    print("[EXPORT] Loading checkpoint (mmap=False to allow full CPU access)...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt
    septq_meta = ckpt.get("septq_meta", {})
    if not isinstance(septq_meta, dict):
        septq_meta = {}

    qat_path = Path(ckpt_path).resolve()
    stats_source = resolve_stats_source_ckpt(ckpt, qat_path, args.septq_ckpt)
    module_meta_lookup = build_module_meta_lookup(stats_source)
    print(f"[EXPORT] module_meta lookup: {len(module_meta_lookup)} entries from PTQ per_layer_stats (resolved source).")

    tier_masks_resolved = _merged_tier_masks_uint2(ckpt, stats_source)
    if tier_masks_resolved:
        print(
            f"[EXPORT] tier_masks_uint2 resolved: {len(tier_masks_resolved)} module(s) "
            f"(merged QAT checkpoint + PTQ stats source for legacy block-pack path).",
            file=sys.stderr,
        )

    # Collect blobs to write
    blobs: Dict[str, Any] = {}

    total_orig_bytes = 0
    total_packed_bytes = 0
    packed_scale_stats = {"ptq": 0, "fallback": 0}
    tier_summaries: list[Dict[str, Any]] = []

    def _count_scale_source(blobs_for_layer: Dict[str, Any]) -> None:
        src = str(blobs_for_layer.pop("_scale_source", "fallback"))
        if src == "ptq":
            packed_scale_stats["ptq"] += 1
        else:
            packed_scale_stats["fallback"] += 1
        summary = blobs_for_layer.pop("_tier_summary", None)
        if isinstance(summary, dict):
            tier_summaries.append(summary)
    ratio_fp16 = float(septq_meta.get("ratio_fp16", DEFAULT_RATIO_FP16)) if isinstance(septq_meta, dict) else DEFAULT_RATIO_FP16
    ratio_int8 = float(septq_meta.get("ratio_int8", DEFAULT_RATIO_INT8)) if isinstance(septq_meta, dict) else DEFAULT_RATIO_INT8
    ratio_int4 = float(septq_meta.get("ratio_int4", DEFAULT_RATIO_INT4)) if isinstance(septq_meta, dict) else DEFAULT_RATIO_INT4

    def add_packed_blobs(base: str, blobs_for_layer: Dict[str, Any]) -> int:
        nonlocal total_packed_bytes
        for suffix in (
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
            "threshold_8bit",
            "threshold_4bit",
            "threshold_2bit",
            "fp16_values",
            "packing_version",
            "rows",
            "cols",
        ):
            if suffix in blobs_for_layer:
                blobs[f"{base}.{suffix}"] = blobs_for_layer[suffix]
        if "bias" in blobs_for_layer:
            blobs[f"{base}.bias"] = blobs_for_layer["bias"]

        packed_bytes = (
            int(blobs_for_layer["n_2bit_bytes"])
            + int(blobs_for_layer["n_4bit_bytes"])
            + int(blobs_for_layer["n_8bit_bytes"])
            + blobs_for_layer["packed_mask"].nbytes
            + blobs_for_layer["fp16_values"].nbytes
        )
        total_packed_bytes += packed_bytes
        return packed_bytes

    def pack_tile_region_ffn(src_key: str, dst_key: str, bias_key: str | None = None) -> bool:
        nonlocal total_orig_bytes, total_packed_bytes
        actual_key = src_key
        if actual_key not in state_dict:
            if f"{actual_key}.weight" in state_dict:
                actual_key = f"{actual_key}.weight"
            else:
                return False
                
        tile_region_meta = ckpt.get("tile_region_metadata", {})
        if not tile_region_meta or "tiles" not in tile_region_meta or actual_key not in tile_region_meta["tiles"]:
            return False
            
        tile_info = tile_region_meta["tiles"][actual_key]
        tile_tiers = tile_info["tile_tiers"]
        tile_grid = tile_info["tile_grid"]
        tile_shape = tile_info["tile_shape"]
        n_tiles_total = len(tile_tiers)
        
        outlier_meta = ckpt.get("outlier_metadata", {})
        outlier_info = outlier_meta.get(actual_key, None)
        
        stats = module_meta_lookup.get(actual_key, {})
        scale_int8 = float(stats.get("quant_scale_int8", 1.0))
        zp_int8 = float(stats.get("quant_zero_point_int8", 0.0))
        scale_int4 = float(stats.get("quant_scale_int4", 1.0))
        zp_int4 = float(stats.get("quant_zero_point_int4", 0.0))
        scale_low = float(stats.get("quant_scale_low", 1.0))
        zp_low = float(stats.get("quant_zero_point_low", 0.0))
        
        w_orig = state_dict[actual_key].float()
        rows, cols = w_orig.shape
        
        w_bulk = w_orig.clone()
        if outlier_info is not None:
            indices = outlier_info["indices"]
            w_bulk.reshape(-1)[indices] = 0.0
            
        tile_rows, tile_cols = tile_shape
        n_tiles_col = tile_grid[1]
        
        fp16_tiles = []
        int8_tiles = []
        int4_tiles = []
        int2_tiles = []
        
        for t_idx in range(n_tiles_total):
            tile_r = t_idx // n_tiles_col
            tile_c = t_idx % n_tiles_col
            row_start = tile_r * tile_rows
            col_start = tile_c * tile_cols
            tile = w_bulk[row_start:row_start+tile_rows, col_start:col_start+tile_cols]
            
            tier = int(tile_tiers[t_idx])
            if tier == 0:
                fp16_tiles.append(tile.to(torch.float16).numpy())
            elif tier == 1:
                q = torch.round(tile / scale_int8 + zp_int8).clamp(0, 255).to(torch.uint8).numpy()
                int8_tiles.append(q)
            elif tier == 2:
                q = torch.round(tile / scale_int4 + zp_int4).clamp(0, 15).to(torch.uint8).numpy()
                int4_tiles.append(q)
            elif tier == 3:
                q = torch.round(tile / scale_low + zp_low).clamp(0, 3).to(torch.uint8).numpy()
                int2_tiles.append(q)
                
        fp16_bytes = np.stack(fp16_tiles).tobytes() if fp16_tiles else b""
        int8_bytes = np.stack(int8_tiles).tobytes() if int8_tiles else b""
        
        if int4_tiles:
            int4_flat = np.stack(int4_tiles).ravel()
            int4_packed = (int4_flat[0::2] | (int4_flat[1::2] << 4)).astype(np.uint8)
            int4_bytes = int4_packed.tobytes()
        else:
            int4_bytes = b""
            
        if int2_tiles:
            int2_flat = np.stack(int2_tiles).ravel()
            int2_packed = (
                int2_flat[0::4]
                | (int2_flat[1::4] << 2)
                | (int2_flat[2::4] << 4)
                | (int2_flat[3::4] << 6)
            ).astype(np.uint8)
            int2_bytes = int2_packed.tobytes()
        else:
            int2_bytes = b""
            
        packed_weights = np.frombuffer(fp16_bytes + int8_bytes + int4_bytes + int2_bytes, dtype=np.uint8)
        
        n_fp16 = len(fp16_tiles)
        n_int8 = len(int8_tiles)
        n_int4 = len(int4_tiles)
        n_int2 = len(int2_tiles)
        n_tiles = np.array([n_fp16, n_int8, n_int4, n_int2], dtype=np.int32)
        
        off0 = 0
        off1 = len(fp16_bytes)
        off2 = off1 + len(int8_bytes)
        off3 = off2 + len(int4_bytes)
        off4 = off3 + len(int2_bytes)
        tier_offsets = np.array([off0, off1, off2, off3, off4], dtype=np.int32)
        
        blobs[f"{dst_key}.packed_weights"] = packed_weights
        blobs[f"{dst_key}.tile_tiers"] = tile_tiers.numpy().astype(np.uint8)
        blobs[f"{dst_key}.n_tiles"] = n_tiles
        blobs[f"{dst_key}.tier_offsets"] = tier_offsets
        blobs[f"{dst_key}.rows"] = np.int32(rows)
        blobs[f"{dst_key}.cols"] = np.int32(cols)
        blobs[f"{dst_key}.packing_version"] = np.int32(6)
        
        if outlier_info is not None:
            blobs[f"{dst_key}.outlier_indices"] = outlier_info["indices"].numpy().astype(np.int32)
            blobs[f"{dst_key}.outlier_values"] = outlier_info["values"].to(torch.float16).numpy()
            blobs[f"{dst_key}.n_outliers"] = np.int32(len(outlier_info["indices"]))
        else:
            blobs[f"{dst_key}.n_outliers"] = np.int32(0)
            
        blobs[f"{dst_key}.scale_int8"] = np.float32(scale_int8)
        blobs[f"{dst_key}.zp_int8"] = np.float32(zp_int8)
        blobs[f"{dst_key}.scale_int4"] = np.float32(scale_int4)
        blobs[f"{dst_key}.zp_int4"] = np.float32(zp_int4)
        blobs[f"{dst_key}.scale_low"] = np.float32(scale_low)
        blobs[f"{dst_key}.zp_low"] = np.float32(zp_low)
        
        resolved_bias_key = bias_key
        if resolved_bias_key and resolved_bias_key not in state_dict:
            if f"{resolved_bias_key}.bias" in state_dict:
                resolved_bias_key = f"{resolved_bias_key}.bias"
        if resolved_bias_key and resolved_bias_key in state_dict:
            blobs[f"{dst_key}.bias"] = state_dict[resolved_bias_key].float().numpy()
            
        orig_bytes = w_orig.numel() * w_orig.element_size()
        packed_bytes_count = len(packed_weights) + len(tile_tiers) + n_tiles.nbytes + tier_offsets.nbytes
        if outlier_info is not None:
            packed_bytes_count += outlier_info["indices"].numel() * 4 + outlier_info["values"].numel() * 2
            
        total_orig_bytes += orig_bytes
        total_packed_bytes += packed_bytes_count
        
        print(f"[EXPORT]   tile-region ffn {actual_key} -> {dst_key} orig={orig_bytes/1e9:.4f} GB packed={packed_bytes_count/1e9:.4f} GB")
        return True

    def pack_group_int4_attn(src_key: str, dst_key: str) -> bool:
        nonlocal total_orig_bytes, total_packed_bytes
        actual_key = src_key
        if actual_key not in state_dict:
            if f"{actual_key}.weight" in state_dict:
                actual_key = f"{actual_key}.weight"
            else:
                return False
                
        attn_meta = stats_source.get("attn_int4_meta", {})
        if not attn_meta or actual_key not in attn_meta:
            return False
            
        meta = attn_meta[actual_key]
        group_size = int(meta["group_size"])
        n_groups = int(meta["n_groups"])
        scale = meta["scale"]
        zero = meta["zero"]
        
        w_orig = state_dict[actual_key].float()
        rows, cols = w_orig.shape
        
        w_grouped = w_orig.view(rows, n_groups, group_size)
        scale_expanded = scale.unsqueeze(-1)
        zero_expanded = zero.unsqueeze(-1)
        q = torch.clamp(torch.round((w_grouped - zero_expanded) / scale_expanded), 0, 15).to(torch.uint8)
        
        q_flat = q.view(-1).numpy()
        packed_weights = (q_flat[0::2] | (q_flat[1::2] << 4)).astype(np.uint8)
        
        blobs[f"{dst_key}.packed_weights"] = packed_weights
        blobs[f"{dst_key}.scales"] = scale.float().numpy()
        blobs[f"{dst_key}.zeros"] = zero.float().numpy()
        blobs[f"{dst_key}.group_size"] = np.int32(group_size)
        blobs[f"{dst_key}.n_groups"] = np.int32(n_groups)
        blobs[f"{dst_key}.rows"] = np.int32(rows)
        blobs[f"{dst_key}.cols"] = np.int32(cols)
        blobs[f"{dst_key}.packing_version"] = np.int32(10)
        
        orig_bytes = w_orig.numel() * w_orig.element_size()
        packed_bytes_count = packed_weights.nbytes + scale.numel() * 4 + zero.numel() * 4
        
        total_orig_bytes += orig_bytes
        total_packed_bytes += packed_bytes_count
        
        print(f"[EXPORT]   uniform-group-int4 {actual_key} -> {dst_key} orig={orig_bytes/1e9:.4f} GB packed={packed_bytes_count/1e9:.4f} GB")
        return True

    def pack_temporal_tensor(src_key: str, dst_key: str, bias_key: str | None = None) -> bool:
        nonlocal total_orig_bytes
        if src_key not in state_dict or f"{dst_key}.packed_weights" in blobs:
            return False
            
        if pack_tile_region_ffn(src_key, dst_key, bias_key=bias_key):
            return True
        if pack_group_int4_attn(src_key, dst_key):
            return True
            
        # Any layer that doesn't match the new FFN/Attention quant formats is skipped (remains dense float16).
        return False


    # Iterate through state_dict and find layers that match MultiTier naming
    # Use the module_meta_lookup keys as canonical layer names when present.
    tier_masks_uint2 = ckpt.get("tier_masks_uint2", {})
    if not tier_masks_uint2:
        tier_masks_uint2 = septq_meta.get("tier_masks_uint2", {})
    block_tier_map = ckpt.get("block_tier_map", {})
    if not isinstance(block_tier_map, dict):
        block_tier_map = {}

    candidate_layers = []
    if tier_masks_uint2:
        candidate_layers = list(tier_masks_uint2.keys())
    else:
        # find all keys ending with 'tier_mask_packed'
        for key in list(state_dict.keys()):
            if key.endswith("tier_mask_packed"):
                module_name = key[: -len(".tier_mask_packed")]
                candidate_layers.append(module_name)

    # Deduplicate while preserving discovery order.
    seen_layers = set()
    candidate_layers = [name for name in candidate_layers if not (name in seen_layers or seen_layers.add(name))]

    print(f"[EXPORT] Found {len(candidate_layers)} candidate multi-tier layers")

    for layer_name in candidate_layers:
        print(f"[EXPORT] Processing layer {layer_name} ...")
        weight_key = f"{layer_name}.q_weight"
        dense_key_alt = f"{layer_name}.weight"
        dense_tensor = None
        if tier_masks_uint2 and layer_name in tier_masks_uint2:
            packed_mask = tier_masks_uint2[layer_name]
        else:
            mask_key = f"{layer_name}.tier_mask_packed"
            if mask_key not in state_dict:
                raise RuntimeError(f"Missing tier mask for multi-tier layer {layer_name}: {mask_key}")
            packed_mask = state_dict[mask_key]

        # Prefer an explicit dense tensor if present; otherwise the checkpoint may store precomputed q_weight.
        if dense_key_alt in state_dict:
            dense_tensor = state_dict[dense_key_alt]
        else:
            # try to locate 'dense' or original weight via heuristics
            for cand in (layer_name, f"{layer_name}.dense_weight", f"{layer_name}.orig_weight", f"{layer_name}.q_weight"):
                if cand in state_dict:
                    dense_tensor = state_dict[cand]
                    break

        if dense_tensor is None:
            raise RuntimeError(f"Missing dense weight for multi-tier layer {layer_name}")

        module_meta = module_meta_lookup.get(layer_name, {})
        bias = state_dict.get(f"{layer_name}.bias", None)

        base = canonical_transformer_multitier_gguf_base(layer_name.replace('.', '_'))
        if pack_tile_region_ffn(layer_name, base, bias_key=f"{layer_name}.bias"):
            continue

        blobs_for_layer = create_packed_layer(
            layer_name,
            dense_tensor,
            packed_mask,
            module_meta,
            bias=bias,
            ratio_fp16=ratio_fp16,
            ratio_int8=ratio_int8,
            ratio_int4=ratio_int4,
            block_tier_map_tensor=block_tier_map.get(layer_name),
            export_gguf_base=base,
            require_ptq_scales=bool(args.require_ptq_scales),
        )
        _count_scale_source(blobs_for_layer)

        # Store into blobs under user-specified naming
        orig_bytes = dense_tensor.numel() * dense_tensor.element_size()
        packed_bytes = add_packed_blobs(base, blobs_for_layer)
        total_orig_bytes += orig_bytes

        print(f"[EXPORT]   layer orig={orig_bytes/1e9:.4f} GB packed={packed_bytes/1e9:.4f} GB")

    print("[EXPORT] Processing unquantized LayerNorms...")
    norm_count = 0

    # Moshi uses RMSNorm with a learned `.alpha` parameter (shape [1,1,dim]).
    # Export as `transformer_layers_{i}_norm1_weight` / `norm2_weight` so the
    # C++ loader finds them via ggml_get_tensor(ctx, prefix + "_norm1_weight").
    for i in range(32):
        for norm_idx in [1, 2]:
            alpha_key = pick_transformer_param_key(state_dict, i, f"norm{norm_idx}.alpha")
            if alpha_key is not None:
                val = state_dict[alpha_key]
                # alpha is [1,1,dim] — flatten to [dim] for ggml broadcast
                arr = val.detach().cpu().float().numpy().reshape(-1)
                out_key = f"transformer_layers_{i}_norm{norm_idx}_weight"
                blobs[out_key] = arr
                bytes_sz = arr.nbytes
                total_orig_bytes += bytes_sz
                total_packed_bytes += bytes_sz
                norm_count += 1
                if i == 0:
                    print(f"[EXPORT]   {alpha_key} -> {out_key}  shape={arr.shape}  first5={arr[:5]}")

    # Fallback: also catch any .weight/.bias norms (e.g. LayerNorm if present)
    for key, val in state_dict.items():
        if (".norm" in key or "norm." in key) and (key.endswith(".weight") or key.endswith(".bias")):
            new_key = key.replace('.', '_')
            if new_key not in blobs:  # don't double-export
                arr = val.detach().cpu().numpy().astype(np.float32)
                blobs[new_key] = arr
                bytes_sz = arr.nbytes
                total_orig_bytes += bytes_sz
                total_packed_bytes += bytes_sz
                norm_count += 1
    print(f"[EXPORT]   Found and exported {norm_count} norm tensors.")

    print("[EXPORT] Processing dense attention/output/embedding tensors...")
    dense_export_count = 0

    def is_q4_0_tensor(name: str) -> bool:
        if name in ("text_emb.weight", "text_linear.weight", "depformer_text_emb.weight"):
            return True
        # any emb/depformer_emb
        if name.startswith("emb.") and name.endswith(".weight"):
            return True
        if name.startswith("depformer_in.") and name.endswith(".weight"):
            return True
        if name.startswith("depformer_emb.") and name.endswith(".weight"):
            return True
        # Any depth layer weights if depth_int4_meta exists
        if ("depformer_layers_" in name or "depformer.layers." in name) and (name.endswith("_weight") or name.endswith(".weight")) and "norm" not in name and "bias" not in name:
            if ckpt.get("depth_int4_meta", {}):
                return True
        return False

    def export_dense_tensor(
        src_key: str,
        dst_key: str | None = None,
        flatten: bool = False,
        preserve_half: bool = False,
    ) -> None:
        """Export a dense tensor from the state_dict to the GGUF blobs.

        Args:
            src_key: Key in state_dict.
            dst_key: Key in output blobs (defaults to src_key).
            flatten: If True, reshape to 1-D (used for norms with (1,1,dim) shape).
            preserve_half: If True, store as float16 instead of float32.
                           Used for depth weights that are bf16 in the checkpoint
                           to avoid doubling their on-disk size.
        """
        nonlocal dense_export_count, total_orig_bytes, total_packed_bytes
        if src_key not in state_dict:
            return
            
        key_name = dst_key or src_key
        if is_q4_0_tensor(key_name):
            tensor = state_dict[src_key].detach().cpu()
            w_orig = tensor.float().numpy()
            import gguf
            qdata = gguf.quantize(w_orig, gguf.GGMLQuantizationType.Q4_0)
            blobs[key_name] = {
                "qdata": qdata,
                "shape": w_orig.shape,
                "qtype": gguf.GGMLQuantizationType.Q4_0
            }
            orig_bytes = w_orig.nbytes
            packed_bytes = qdata.nbytes
            total_orig_bytes += orig_bytes
            total_packed_bytes += packed_bytes
            dense_export_count += 1
            print(f"[EXPORT]   Q4_0 quantize {src_key} -> {key_name} orig={orig_bytes/1e9:.4f} GB packed={packed_bytes/1e9:.4f} GB")
            return

        tensor = state_dict[src_key].detach().cpu()
        if preserve_half:
            # bf16 -> fp16 (numpy doesn't support bf16 natively)
            arr = tensor.to(torch.float16).numpy()
        else:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                tensor = tensor.float()
            arr = tensor.numpy().astype(np.float32, copy=False)
        if flatten:
            arr = arr.reshape(-1)
        blobs[dst_key or src_key] = arr
        bytes_sz = arr.nbytes
        total_orig_bytes += bytes_sz
        total_packed_bytes += bytes_sz
        dense_export_count += 1

    # Temporal attention output projection.
    # SEPTQ v3 block-packs out_proj as well; v1 skipped it for quality.
    for i in range(32):
        weight_key = pick_transformer_param_key(state_dict, i, "self_attn.out_proj.weight")
        bias_key = pick_transformer_param_key(state_dict, i, "self_attn.out_proj.bias")
        dst_key = f"transformer_layers_{i}_self_attn_out_proj_weight"
        if weight_key is None:
            continue
        if pack_temporal_tensor(weight_key, dst_key, bias_key=bias_key):
            if bias_key is not None and f"{dst_key}.bias" not in blobs:
                export_dense_tensor(bias_key, f"transformer_layers_{i}_self_attn_out_proj_bias")
        else:
            export_dense_tensor(weight_key, dst_key, preserve_half=True)
            if bias_key is not None:
                export_dense_tensor(bias_key, f"transformer_layers_{i}_self_attn_out_proj_bias")

    # Generalized fallback for quantizable temporal tensors; this intentionally
    # includes layer 31 even when the source SEPTQ v1 run skipped it.
    for i in range(32):
        in_proj_key = pick_transformer_param_key(state_dict, i, "self_attn.in_proj_weight")
        gating_in_key = pick_transformer_param_key(state_dict, i, "gating.linear_in.weight")
        gating_out_key = pick_transformer_param_key(state_dict, i, "gating.linear_out.weight")

        if in_proj_key is not None:
            dst_key = f"transformer_layers_{i}_self_attn_in_proj_weight"
            if f"{dst_key}.packed_weights" not in blobs and dst_key not in blobs:
                if not pack_temporal_tensor(in_proj_key, dst_key):
                    export_dense_tensor(in_proj_key, dst_key, preserve_half=True)

        if gating_in_key is not None:
            dst_key = f"transformer_layers_{i}_gating_linear_in_weight"
            if f"{dst_key}.packed_weights" not in blobs and dst_key not in blobs:
                if not pack_temporal_tensor(gating_in_key, dst_key):
                    export_dense_tensor(gating_in_key, dst_key, preserve_half=True)

        if gating_out_key is not None:
            dst_key = f"transformer_layers_{i}_gating_linear_out_weight"
            if f"{dst_key}.packed_weights" not in blobs and dst_key not in blobs:
                if not pack_temporal_tensor(gating_out_key, dst_key):
                    export_dense_tensor(gating_out_key, dst_key, preserve_half=True)

    def export_depth_tensor(src_key: str, dst_key: str) -> None:
        nonlocal total_orig_bytes, total_packed_bytes, dense_export_count
        if src_key not in state_dict:
            return
            
        meta_key = src_key
        if meta_key.endswith(".weight"):
            meta_key = meta_key[:-7]
        elif meta_key.endswith("_weight"):
            meta_key = meta_key[:-7]
            
        depth_int8_meta = ckpt.get("depth_int8_meta", {})
        depth_int4_meta = ckpt.get("depth_int4_meta", {})
        
        has_int8_meta = depth_int8_meta and meta_key in depth_int8_meta
        has_int4_meta = depth_int4_meta and meta_key in depth_int4_meta
        is_depformer_in_proj = "depformer" in src_key and "self_attn.in_proj_weight" in src_key
        
        if has_int4_meta:
            export_dense_tensor(src_key, dst_key, preserve_half=True)
        elif has_int8_meta or (is_depformer_in_proj and not depth_int4_meta):
            w_orig = state_dict[src_key].float()
            if has_int8_meta:
                meta = depth_int8_meta[meta_key]
                scale = meta["scale"]
            else:
                # Calculate per-channel scales dynamically
                absmax = w_orig.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
                scale = (absmax / 127.0).squeeze(-1)
                
            scale_expanded = scale.unsqueeze(-1)
            q = torch.clamp(torch.round(w_orig / scale_expanded), -128, 127).to(torch.int8)
            
            blobs[f"{dst_key}.packed_weights"] = q.numpy()
            blobs[f"{dst_key}.scales"] = scale.float().numpy()
            blobs[f"{dst_key}.rows"] = np.int32(w_orig.shape[0])
            blobs[f"{dst_key}.cols"] = np.int32(w_orig.shape[1])
            blobs[f"{dst_key}.packing_version"] = np.int32(20)
            
            orig_bytes = w_orig.numel() * w_orig.element_size()
            packed_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size()
            total_orig_bytes += orig_bytes
            total_packed_bytes += packed_bytes
            dense_export_count += 1
            print(f"[EXPORT]   depth-channel-int8 {src_key} -> {dst_key} (dynamic={not has_int8_meta}) orig={orig_bytes/1e9:.4f} GB packed={packed_bytes/1e9:.4f} GB")
        else:
            export_dense_tensor(src_key, dst_key, preserve_half=True)

    # Depth stack: explicit probe-based exports.
    # Depth norms are (1,1,1024) and must be flattened to (1024,) for C++ RMSNorm.
    # Depth weights are kept as fp16 to avoid the bf16->float32 size doubling.
    for i in range(6):
        export_dense_tensor(f"depformer.layers.{i}.norm1.alpha", f"depformer_layers_{i}_norm1_weight", flatten=True)
        export_dense_tensor(f"depformer.layers.{i}.norm2.alpha", f"depformer_layers_{i}_norm2_weight", flatten=True)
        export_depth_tensor(f"depformer.layers.{i}.self_attn.in_proj_weight", f"depformer_layers_{i}_self_attn_in_proj_weight")
        export_depth_tensor(f"depformer.layers.{i}.self_attn.out_proj.weight", f"depformer_layers_{i}_self_attn_out_proj_weight")
        for step in range(16):
            export_depth_tensor(
                f"depformer.layers.{i}.gating.{step}.linear_in.weight",
                f"depformer_layers_{i}_gating_{step}_linear_in_weight",
            )
            export_depth_tensor(
                f"depformer.layers.{i}.gating.{step}.linear_out.weight",
                f"depformer_layers_{i}_gating_{step}_linear_out_weight",
            )

    # Depth attention/output tensors: check for per-step split variants.
    # The main stacked tensors were already exported above. This loop handles
    # any bias tensors and alternative per-step split key layouts (in_projs.{step}).
    for i in range(6):
        export_dense_tensor(f"depformer.layers.{i}.self_attn.out_proj.bias", f"depformer_layers_{i}_self_attn_out_proj_bias")
        for step in range(16):
            export_depth_tensor(
                f"depformer.layers.{i}.self_attn.in_projs.{step}.weight",
                f"depformer_layers_{i}_self_attn_in_projs_{step}_weight",
            )
            export_dense_tensor(
                f"depformer.layers.{i}.self_attn.in_projs.{step}.bias",
                f"depformer_layers_{i}_self_attn_in_projs_{step}_bias",
            )

    # Depth embeddings / text path / heads.
    for idx in range(16):
        export_dense_tensor(f"emb.{idx}.weight", preserve_half=True)  # temporal codebook emb: fp16 saves ~0.25 GB
        export_dense_tensor(f"depformer_in.{idx}.weight", preserve_half=True)
        export_dense_tensor(f"depformer_emb.{idx}.weight", preserve_half=True)
        export_dense_tensor(f"linears.{idx}.weight", preserve_half=True)  # depth output heads
    export_dense_tensor("depformer_text_emb.weight", preserve_half=True)
    export_dense_tensor("text_emb.weight", preserve_half=True)   # temporal text emb: fp16 saves ~0.25 GB
    export_dense_tensor("text_linear.weight", preserve_half=True) # temporal text linear: fp16 saves ~0.25 GB
    export_dense_tensor("text_linear.bias")  # bias stays float32 (tiny)
    export_dense_tensor("token_embedding", preserve_half=True)
    export_dense_tensor("output_head", preserve_half=True)

    # Final temporal RMSNorm (out_norm) — flatten from (1,1,4096) to (4096,).
    export_dense_tensor("out_norm.alpha", "out_norm_weight", flatten=True)

    print(f"[EXPORT]   Found and exported {dense_export_count} dense tensors.")

    # ========== COMPLETENESS CHECK ==========
    # Verify that all expected transformer layer tensors have been exported.
    # This prevents silent gaps like the L31 exclusion from recurring.
    print("[EXPORT] Running completeness check on temporal transformer tensors...")
    expected_tensor_patterns = [
        "norm1.alpha",
        "norm2.alpha",
        "self_attn.in_proj_weight",
        "self_attn.out_proj.weight",
        "gating.linear_in.weight",
        "gating.linear_out.weight",
    ]
    missing_tensors = []
    for layer_idx in range(32):
        for pattern in expected_tensor_patterns:
            src_key = pick_transformer_param_key(state_dict, layer_idx, pattern)
            if src_key is None:
                continue  # Tensor doesn't exist in checkpoint; skip it
            # Map to expected GGUF naming convention. Norm keys use `_weight` in GGUF.
            if pattern == "norm1.alpha":
                gguf_key = f"transformer_layers_{layer_idx}_norm1_weight"
            elif pattern == "norm2.alpha":
                gguf_key = f"transformer_layers_{layer_idx}_norm2_weight"
            else:
                gguf_key = f"transformer_layers_{layer_idx}_" + pattern.replace(".", "_")
            packed_marker = f"{gguf_key}.packed_weights"
            if gguf_key not in blobs and packed_marker not in blobs:
                missing_tensors.append((layer_idx, pattern, src_key, gguf_key))
    
    if missing_tensors:
        error_msg = "Completeness check failed: the following transformer tensors were not exported to GGUF:\n"
        for layer_idx, pattern, src_key, gguf_key in missing_tensors:
            error_msg += f"  Layer {layer_idx}: {pattern} (expected as '{gguf_key}')\n"
        raise RuntimeError(error_msg)
    print(f"[EXPORT] Completeness check passed: all {32 * len(expected_tensor_patterns)} expected tensors are present.")

    print("[EXPORT] Running completeness check on depth stack tensors...")
    depth_missing = []
    for layer_idx in range(6):
        depth_expected = [
            (f"depformer.layers.{layer_idx}.norm1.alpha", f"depformer_layers_{layer_idx}_norm1_weight"),
            (f"depformer.layers.{layer_idx}.norm2.alpha", f"depformer_layers_{layer_idx}_norm2_weight"),
            (f"depformer.layers.{layer_idx}.self_attn.in_proj_weight", f"depformer_layers_{layer_idx}_self_attn_in_proj_weight"),
            (f"depformer.layers.{layer_idx}.self_attn.out_proj.weight", f"depformer_layers_{layer_idx}_self_attn_out_proj_weight"),
        ]
        for step in range(16):
            depth_expected.append((
                f"depformer.layers.{layer_idx}.gating.{step}.linear_in.weight",
                f"depformer_layers_{layer_idx}_gating_{step}_linear_in_weight",
            ))
            depth_expected.append((
                f"depformer.layers.{layer_idx}.gating.{step}.linear_out.weight",
                f"depformer_layers_{layer_idx}_gating_{step}_linear_out_weight",
            ))

        for src_key, gguf_key in depth_expected:
            packed_marker = f"{gguf_key}.packed_weights"
            if src_key in state_dict and gguf_key not in blobs and packed_marker not in blobs:
                depth_missing.append((layer_idx, src_key, gguf_key))

    if depth_missing:
        error_msg = "Completeness check failed: the following depth tensors were not exported to GGUF:\n"
        for layer_idx, src_key, gguf_key in depth_missing:
            error_msg += f"  Depth layer {layer_idx}: {src_key} (expected as '{gguf_key}')\n"
        raise RuntimeError(error_msg)
    print("[EXPORT] Completeness check passed: all expected depth stack tensors are present.")

    # Additional completeness: linears (depth output heads), out_norm, embeddings
    print("[EXPORT] Running completeness check on output heads and embeddings...")
    extra_missing = []
    for idx in range(16):
        for src_key, gguf_key in [
            (f"linears.{idx}.weight", f"linears.{idx}.weight"),
            (f"depformer_emb.{idx}.weight", f"depformer_emb.{idx}.weight"),
            (f"depformer_in.{idx}.weight", f"depformer_in.{idx}.weight"),
        ]:
            packed_marker = f"{gguf_key}.packed_weights"
            if src_key in state_dict and gguf_key not in blobs and packed_marker not in blobs:
                extra_missing.append((src_key, gguf_key))
    for src_key, gguf_key in [
        ("out_norm.alpha", "out_norm_weight"),
        ("depformer_text_emb.weight", "depformer_text_emb.weight"),
        ("text_emb.weight", "text_emb.weight"),
        ("text_linear.weight", "text_linear.weight"),
    ]:
        packed_marker = f"{gguf_key}.packed_weights"
        if src_key in state_dict and gguf_key not in blobs and packed_marker not in blobs:
            extra_missing.append((src_key, gguf_key))
    if extra_missing:
        error_msg = "Completeness check failed: the following tensors were not exported to GGUF:\n"
        for src_key, gguf_key in extra_missing:
            error_msg += f"  {src_key} (expected as '{gguf_key}')\n"
        raise RuntimeError(error_msg)
    print("[EXPORT] Completeness check passed: all output heads and embeddings are present.")

    print("[EXPORT] Writing output...")
    try:
        if out_path.endswith(".npz") or gguf is None:
            # fallback save
            print("[EXPORT] Writing .npz fallback (gguf not available or user requested .npz)")
            np.savez_compressed(out_path, **blobs)
        else:
            write_with_gguf(out_path, blobs)
    except Exception as e:
        # fall back to npz if gguf writing fails
        fallback = out_path + ".npz"
        print(f"[EXPORT] gguf write failed: {e}; falling back to {fallback}")
        np.savez_compressed(fallback, **blobs)

    print("[EXPORT] Done.")
    print(f"[EXPORT] Total original size: {total_orig_bytes/1e9:.4f} GB")
    print(f"[EXPORT] Total packed size:   {total_packed_bytes/1e9:.4f} GB")
    print(
        f"[export] tensors with PTQ scales:  {packed_scale_stats['ptq']}\n"
        f"[export] tensors with fallback scales: {packed_scale_stats['fallback']}",
        file=sys.stderr,
    )
    if tier_summaries:
        print("[EXPORT] Per-tensor block tier mix:")
        print(
            f"{'module_name':80s} {'n_blocks':>10s} {'frac_FP16':>10s} {'frac_INT8':>10s} "
            f"{'frac_INT4':>10s} {'frac_INT2':>10s} {'eff_bpw':>10s} {'packed_MB':>10s}"
        )
        for s in tier_summaries:
            print(
                f"{str(s['module_name'])[:80]:80s} {int(s['n_blocks']):10d} "
                f"{float(s['frac_fp16']):10.4f} {float(s['frac_int8']):10.4f} "
                f"{float(s['frac_int4']):10.4f} {float(s['frac_int2']):10.4f} "
                f"{float(s['effective_bits_per_weight']):10.4f} {float(s['packed_mb']):10.3f}"
            )


if __name__ == "__main__":
    main()
