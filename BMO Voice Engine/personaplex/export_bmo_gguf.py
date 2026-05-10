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
import os
import re
import struct
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch

# Try to import gguf; we'll detect a usable writer API at runtime.
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


def _ratio_counts(n_blocks: int, ratio_fp16: float, ratio_int8: float, ratio_int4: float) -> tuple[int, int, int]:
    n_fp16 = int(round(float(ratio_fp16) * n_blocks))
    n_int8 = int(round(float(ratio_int8) * n_blocks))
    n_int4 = int(round(float(ratio_int4) * n_blocks))
    n_fp16 = max(0, min(n_blocks, n_fp16))
    n_int8 = max(0, min(n_blocks - n_fp16, n_int8))
    n_int4 = max(0, min(n_blocks - n_fp16 - n_int8, n_int4))
    return n_fp16, n_int8, n_int4


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
) -> Dict[str, Any]:
    """Return block-wise SEPTQ v3 packed artifacts for this layer.

    The returned dict contains at minimum:
      - packed_mask: uint8 bytes (4 uint2 block tags per byte)
      - packed_weights: concatenated [2bit_bytes | 4bit_bytes | 8bit_bytes]
      - n_2bit_bytes, n_4bit_bytes, n_8bit_bytes: int sizes
      - scale_low, scale_int4, scale_int8 (float32)
      - fp16_values (float16), with full tier-0 blocks stored sequentially
      - bias (float32) if provided
    """
    if block_size != 32:
        raise RuntimeError("SEPTQ v3 CUDA path currently expects block_size=32")

    w = dense_weight.detach().cpu().float()
    rows, cols = int(w.shape[0]), int(w.shape[1])
    flat_w = w.reshape(-1)
    total = int(flat_w.numel())
    n_blocks = (total + block_size - 1) // block_size
    padded_total = n_blocks * block_size
    fw = flat_w.numpy().astype(np.float32, copy=False)
    if padded_total != total:
        fw_padded = np.zeros(padded_total, dtype=np.float32)
        fw_padded[:total] = fw
    else:
        fw_padded = fw
    blocks = fw_padded.reshape(n_blocks, block_size)

    # Extract scales and zero points (similar to profile_jetson._module_meta_float)
    def _get_meta(key: str, fallback: str | None = None, default: float | None = None) -> float:
        v = module_meta.get(key, None)
        if v is None and fallback is not None:
            v = module_meta.get(fallback, None)
        if v is None:
            if default is not None:
                return float(default)
            raise RuntimeError(f"Missing quant metadata '{key}' for {layer_name}")
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)

    block_max = np.max(np.abs(blocks), axis=1)

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

    values_int2 = blocks[block_tiers == 3].reshape(-1)
    values_int4 = blocks[block_tiers == 2].reshape(-1)
    values_int8 = blocks[block_tiers == 1].reshape(-1)

    fallback_low = _affine_params_for_values(values_int2, 3, 1.5)
    fallback_int4 = _affine_params_for_values(values_int4, 15, 7.5)
    fallback_int8 = _affine_params_for_values(values_int8, 255, 127.5)

    scale_low = _get_meta("quant_scale_low", "quant_scale", fallback_low[0])
    zp_low = _get_meta("quant_zero_point_low", "quant_zero_point", fallback_low[1])
    scale_int4 = _get_meta("quant_scale_int4", default=fallback_int4[0])
    zp_int4 = _get_meta("quant_zero_point_int4", default=fallback_int4[1])
    scale_int8 = _get_meta("quant_scale_int8", default=fallback_int8[0])
    zp_int8 = _get_meta("quant_zero_point_int8", default=fallback_int8[1])

    q3_vals = np.clip(np.round(values_int2 / scale_low + zp_low), 0, 3).astype(np.uint8)
    q2_vals = np.clip(np.round(values_int4 / scale_int4 + zp_int4), 0, 15).astype(np.uint8)
    q1_vals = np.clip(np.round(values_int8 / scale_int8 + zp_int8), 0, 255).astype(np.uint8)
    fp16_values = blocks[block_tiers == 0].reshape(-1).astype(np.float16)

    packed_2 = pack_2bit_values_le(q3_vals) if q3_vals.size else np.zeros(0, dtype=np.uint8)
    packed_4 = pack_4bit_values_le(q2_vals) if q2_vals.size else np.zeros(0, dtype=np.uint8)
    packed_8 = q1_vals.view(np.uint8) if q1_vals.size else np.zeros(0, dtype=np.uint8)
    packed_weights = np.concatenate([packed_2, packed_4, packed_8])

    out: Dict[str, Any] = {}
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
    out["packing_version"] = np.int32(3)
    if bias is not None:
        out["bias"] = bias.detach().cpu().numpy().astype(np.float32)

    # Also include original dense shape for reference
    out["rows"] = np.int32(rows)
    out["cols"] = np.int32(cols)

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
        if hasattr(arr, 'dtype') and arr.dtype.kind == 'u':
            mapping = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
            arr = arr.view(mapping.get(arr.dtype.itemsize, np.int8))

        added = False
        for m in add_methods:
            try:
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
    per_layer_stats = septq_meta.get("per_layer_stats", [])

    # Build module meta lookup similar to profile_jetson._build_module_meta_lookup
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
            name = mod.get("name")
            if isinstance(name, str) and name:
                module_meta_lookup[name] = mod

    # Collect blobs to write
    blobs: Dict[str, Any] = {}

    total_orig_bytes = 0
    total_packed_bytes = 0
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

    def pack_temporal_tensor(src_key: str, dst_key: str, bias_key: str | None = None) -> bool:
        nonlocal total_orig_bytes
        if src_key not in state_dict or f"{dst_key}.packed_weights" in blobs:
            return False
        dense_tensor = state_dict[src_key]
        if not torch.is_tensor(dense_tensor) or dense_tensor.ndim != 2:
            return False
        module_meta = module_meta_lookup.get(src_key, {})
        bias = state_dict.get(bias_key) if bias_key else None
        blobs_for_layer = create_packed_layer(
            src_key,
            dense_tensor,
            None,
            module_meta,
            bias=bias,
            ratio_fp16=ratio_fp16,
            ratio_int8=ratio_int8,
            ratio_int4=ratio_int4,
        )
        orig_bytes = dense_tensor.numel() * dense_tensor.element_size()
        total_orig_bytes += orig_bytes
        packed_bytes = add_packed_blobs(dst_key, blobs_for_layer)
        print(f"[EXPORT]   block-pack {src_key} -> {dst_key} orig={orig_bytes/1e9:.4f} GB packed={packed_bytes/1e9:.4f} GB")
        return True

    # Iterate through state_dict and find layers that match MultiTier naming
    # Use the module_meta_lookup keys as canonical layer names when present.
    tier_masks_uint2 = ckpt.get("tier_masks_uint2", {})
    if not tier_masks_uint2:
        tier_masks_uint2 = septq_meta.get("tier_masks_uint2", {})

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

        blobs_for_layer = create_packed_layer(
            layer_name,
            dense_tensor,
            packed_mask,
            module_meta,
            bias=bias,
            ratio_fp16=ratio_fp16,
            ratio_int8=ratio_int8,
            ratio_int4=ratio_int4,
        )

        # Store into blobs under user-specified naming
        base = canonical_transformer_multitier_gguf_base(layer_name.replace('.', '_'))
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
        if not pack_temporal_tensor(weight_key, dst_key, bias_key=bias_key):
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
            if f"{dst_key}.packed_weights" not in blobs:
                pack_temporal_tensor(in_proj_key, dst_key)

        if gating_in_key is not None:
            dst_key = f"transformer_layers_{i}_gating_linear_in_weight"
            if f"{dst_key}.packed_weights" not in blobs:
                pack_temporal_tensor(gating_in_key, dst_key)

        if gating_out_key is not None:
            dst_key = f"transformer_layers_{i}_gating_linear_out_weight"
            if f"{dst_key}.packed_weights" not in blobs:
                pack_temporal_tensor(gating_out_key, dst_key)

    # Depth stack: explicit probe-based exports.
    # Depth norms are (1,1,1024) and must be flattened to (1024,) for C++ RMSNorm.
    # Depth weights are kept as fp16 to avoid the bf16->float32 size doubling.
    for i in range(6):
        export_dense_tensor(f"depformer.layers.{i}.norm1.alpha", f"depformer_layers_{i}_norm1_weight", flatten=True)
        export_dense_tensor(f"depformer.layers.{i}.norm2.alpha", f"depformer_layers_{i}_norm2_weight", flatten=True)
        export_dense_tensor(f"depformer.layers.{i}.self_attn.in_proj_weight", f"depformer_layers_{i}_self_attn_in_proj_weight", preserve_half=True)
        export_dense_tensor(f"depformer.layers.{i}.self_attn.out_proj.weight", f"depformer_layers_{i}_self_attn_out_proj_weight", preserve_half=True)
        for step in range(16):
            export_dense_tensor(
                f"depformer.layers.{i}.gating.{step}.linear_in.weight",
                f"depformer_layers_{i}_gating_{step}_linear_in_weight",
                preserve_half=True,
            )
            export_dense_tensor(
                f"depformer.layers.{i}.gating.{step}.linear_out.weight",
                f"depformer_layers_{i}_gating_{step}_linear_out_weight",
                preserve_half=True,
            )

    # Depth attention/output tensors: check for per-step split variants.
    # The main stacked tensors were already exported above. This loop handles
    # any bias tensors and alternative per-step split key layouts (in_projs.{step}).
    for i in range(6):
        export_dense_tensor(f"depformer.layers.{i}.self_attn.out_proj.bias", f"depformer_layers_{i}_self_attn_out_proj_bias")
        for step in range(16):
            export_dense_tensor(
                f"depformer.layers.{i}.self_attn.in_projs.{step}.weight",
                f"depformer_layers_{i}_self_attn_in_projs_{step}_weight",
                preserve_half=True,
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
            if src_key in state_dict and gguf_key not in blobs:
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
            if src_key in state_dict and gguf_key not in blobs:
                extra_missing.append((src_key, gguf_key))
    for src_key, gguf_key in [
        ("out_norm.alpha", "out_norm_weight"),
        ("depformer_text_emb.weight", "depformer_text_emb.weight"),
        ("text_emb.weight", "text_emb.weight"),
        ("text_linear.weight", "text_linear.weight"),
    ]:
        if src_key in state_dict and gguf_key not in blobs:
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


if __name__ == "__main__":
    main()
