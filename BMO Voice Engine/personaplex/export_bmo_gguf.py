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


def create_packed_layer(
    layer_name: str,
    dense_weight: torch.Tensor,
    packed_mask_tensor: torch.Tensor,
    module_meta: Dict[str, Any],
    bias: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """Return a dict of numpy arrays representing packed artifacts for this layer.

    The returned dict contains at minimum:
      - packed_mask: uint8 bytes (4 uint2 per byte)
      - packed_weights: concatenated [2bit_bytes | 4bit_bytes | 8bit_bytes]
      - n_2bit_bytes, n_4bit_bytes, n_8bit_bytes: int sizes
      - scale_low, scale_int4, scale_int8 (float32)
      - fp16_indices (int32) and fp16_values (float16)
      - bias (float32) if provided
    """
    w = dense_weight.detach().cpu().float()
    rows, cols = int(w.shape[0]), int(w.shape[1])
    flat_w = w.reshape(-1)

    # Extract scales and zero points (similar to profile_jetson._module_meta_float)
    def _get_meta(key: str, fallback: str | None = None) -> float:
        v = module_meta.get(key, None)
        if v is None and fallback is not None:
            v = module_meta.get(fallback, None)
        if v is None:
            raise RuntimeError(f"Missing quant metadata '{key}' for {layer_name}")
        if torch.is_tensor(v):
            return float(v.detach().item())
        return float(v)

    scale_low = _get_meta("quant_scale_low", "quant_scale")
    zp_low = _get_meta("quant_zero_point_low", "quant_zero_point")
    scale_int4 = _get_meta("quant_scale_int4")
    zp_int4 = _get_meta("quant_zero_point_int4")
    scale_int8 = _get_meta("quant_scale_int8")
    zp_int8 = _get_meta("quant_zero_point_int8")

    # Unpack mask to per-weight values on CPU
    unpacked_mask = unpack_tier_mask_uint2(packed_mask_tensor.cpu(), (rows, cols)).reshape(-1).to(torch.uint8)
    unpack_np = unpacked_mask.numpy().astype(np.uint8)

    # Compute per-position quantized integers into flat_q (uint32 to hold up to 255)
    flat_q = np.zeros(flat_w.numel(), dtype=np.uint32)
    fw = flat_w.numpy()

    # Tier 3 -> use 2-bit (0..3)
    t3 = unpack_np >= 3
    if t3.any():
        q3 = np.round(fw[t3] / scale_low + zp_low).astype(np.int64)
        q3 = np.clip(q3, 0, 3).astype(np.uint8)
        flat_q[t3] = q3

    # Tier 2 -> 4-bit (0..15)
    t2 = unpack_np == 2
    if t2.any():
        q2 = np.round(fw[t2] / scale_int4 + zp_int4).astype(np.int64)
        q2 = np.clip(q2, 0, 15).astype(np.uint8)
        flat_q[t2] = q2

    # Tier 1 -> 8-bit (0..255)
    t1 = unpack_np == 1
    if t1.any():
        q1 = np.round(fw[t1] / scale_int8 + zp_int8).astype(np.int64)
        q1 = np.clip(q1, 0, 255).astype(np.uint8)
        flat_q[t1] = q1

    # Tier 0 -> fp16 passthrough; record indices and values
    t0 = unpack_np == 0
    fp16_indices = np.nonzero(t0)[0].astype(np.int32)
    fp16_values = fw[t0].astype(np.float16)

    # Build packed arrays for each bucket in index order (dense compact arrays)
    q3_vals = flat_q[t3].astype(np.uint8)
    q2_vals = flat_q[t2].astype(np.uint8)
    q1_vals = flat_q[t1].astype(np.uint8)

    packed_2 = pack_2bit_values_le(q3_vals) if q3_vals.size else np.zeros(0, dtype=np.uint8)
    packed_4 = pack_4bit_values_le(q2_vals) if q2_vals.size else np.zeros(0, dtype=np.uint8)
    packed_8 = q1_vals.view(np.uint8) if q1_vals.size else np.zeros(0, dtype=np.uint8)

    # Concatenate into a single packed_weights blob (order: 2bit | 4bit | 8bit)
    packed_weights = np.concatenate([packed_2, packed_4, packed_8])

    # For correctness: reconstruct flat_q from packed representation and assert equality
    # Reconstruct step-by-step
    recon_flat_q = np.zeros_like(flat_q, dtype=np.uint8)

    # Recreate sequential consumers
    p = 0
    # 2-bit
    n2_bytes = packed_2.size
    if n2_bytes:
        raw2 = packed_weights[p : p + n2_bytes]
        p += n2_bytes
        raw2 = raw2.astype(np.uint8)
        # expand 4 values per byte
        vals2 = np.empty(raw2.size * 4, dtype=np.uint8)
        vals2[0::4] = raw2 & 0b11
        vals2[1::4] = (raw2 >> 2) & 0b11
        vals2[2::4] = (raw2 >> 4) & 0b11
        vals2[3::4] = (raw2 >> 6) & 0b11
        vals2 = vals2[: q3_vals.size]
        recon_flat_q[np.nonzero(t3)[0]] = vals2

    # 4-bit
    n4_bytes = packed_4.size
    if n4_bytes:
        raw4 = packed_weights[p : p + n4_bytes]
        p += n4_bytes
        raw4 = raw4.astype(np.uint8)
        vals4 = np.empty(raw4.size * 2, dtype=np.uint8)
        vals4[0::2] = raw4 & 0x0F
        vals4[1::2] = (raw4 >> 4) & 0x0F
        vals4 = vals4[: q2_vals.size]
        recon_flat_q[np.nonzero(t2)[0]] = vals4

    # 8-bit
    n8_bytes = packed_8.size
    if n8_bytes:
        raw8 = packed_weights[p : p + n8_bytes]
        p += n8_bytes
        recon_flat_q[np.nonzero(t1)[0]] = raw8

    # fp16 indices/values are not part of packed_weights; they are stored separately

    # Compare reconstructed quantized flat array (for quant tiers) against original
    expected_flat_q = flat_q.astype(np.uint8)
    # For fp16 positions, expected_flat_q entries were zero; ensure recon has zeros
    if not np.array_equal(recon_flat_q, expected_flat_q & 0xFF):
        raise AssertionError(f"Packing round-trip failed for layer {layer_name}")

    out: Dict[str, Any] = {}
    out["packed_mask"] = pack_uint2_mask_le(unpack_np := unpacked_mask.cpu().numpy().astype(np.uint8))
    out["packed_weights"] = packed_weights
    out["n_2bit_bytes"] = np.int32(packed_2.size)
    out["n_4bit_bytes"] = np.int32(packed_4.size)
    out["n_8bit_bytes"] = np.int32(packed_8.size)
    out["scale_low"] = np.float32(scale_low)
    out["scale_int4"] = np.float32(scale_int4)
    out["scale_int8"] = np.float32(scale_int8)
    out["fp16_indices"] = fp16_indices
    out["fp16_values"] = fp16_values
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
        if isinstance(val, (np.ndarray, np.generic)):
            arr = val
        elif isinstance(val, (list, tuple)):
            arr = np.array(val)
        else:
            # try to coerce tensors
            try:
                if torch.is_tensor(val):
                    arr = val.detach().cpu().numpy()
                else:
                    arr = np.array(val)
            except Exception:
                # skip unknown types
                continue

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
    if hasattr(writer, "write"):
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

    # Iterate through state_dict and find layers that match MultiTier naming
    # We'll look for keys like '<module>.tier_mask_packed' or '<module>.weight'
    # Use the module_meta_lookup keys as canonical layer names when present.
    candidate_layers = []
    # find all keys ending with 'tier_mask_packed'
    for key in list(state_dict.keys()):
        if key.endswith("tier_mask_packed"):
            module_name = key[: -len(".tier_mask_packed")]
            candidate_layers.append(module_name)

    print(f"[EXPORT] Found {len(candidate_layers)} candidate multi-tier layers")

    for layer_name in candidate_layers:
        print(f"[EXPORT] Processing layer {layer_name} ...")
        mask_key = f"{layer_name}.tier_mask_packed"
        weight_key = f"{layer_name}.q_weight"  # note: checkpoint may have original dense weight under different key
        # The profile_jetson loader keeps the original dense weight in the checkpoint under '<module>.dense_weight' or similar
        dense_key_alt = f"{layer_name}.weight"
        dense_tensor = None
        if mask_key not in state_dict:
            print(f"[EXPORT]   Warning: mask key {mask_key} missing; skipping")
            continue
        packed_mask = state_dict[mask_key]

        # Prefer an explicit dense tensor if present; otherwise the checkpoint may store precomputed q_weight.
        if dense_key_alt in state_dict:
            dense_tensor = state_dict[dense_key_alt]
        else:
            # try to locate 'dense' or original weight via heuristics
            for cand in (f"{layer_name}.dense_weight", f"{layer_name}.orig_weight", f"{layer_name}.q_weight"):
                if cand in state_dict:
                    dense_tensor = state_dict[cand]
                    break

        if dense_tensor is None:
            print(f"[EXPORT]   Could not find dense weight for {layer_name}; skipping")
            continue

        module_meta = module_meta_lookup.get(layer_name, {})
        bias = state_dict.get(f"{layer_name}.bias", None)

        blobs_for_layer = create_packed_layer(layer_name, dense_tensor, packed_mask, module_meta, bias=bias)

        # Store into blobs under user-specified naming
        base = layer_name.replace('.', '_')
        blobs[f"{base}.packed_mask"] = blobs_for_layer["packed_mask"]
        blobs[f"{base}.packed_weights"] = blobs_for_layer["packed_weights"]
        blobs[f"{base}.n_2bit_bytes"] = np.int32(int(blobs_for_layer["n_2bit_bytes"]))
        blobs[f"{base}.n_4bit_bytes"] = np.int32(int(blobs_for_layer["n_4bit_bytes"]))
        blobs[f"{base}.n_8bit_bytes"] = np.int32(int(blobs_for_layer["n_8bit_bytes"]))
        blobs[f"{base}.scale_low"] = np.float32(blobs_for_layer["scale_low"])
        blobs[f"{base}.scale_int4"] = np.float32(blobs_for_layer["scale_int4"])
        blobs[f"{base}.scale_int8"] = np.float32(blobs_for_layer["scale_int8"])
        blobs[f"{base}.fp16_indices"] = blobs_for_layer["fp16_indices"]
        blobs[f"{base}.fp16_values"] = blobs_for_layer["fp16_values"]
        if "bias" in blobs_for_layer:
            blobs[f"{base}.bias"] = blobs_for_layer["bias"]
        blobs[f"{base}.rows"] = blobs_for_layer["rows"]
        blobs[f"{base}.cols"] = blobs_for_layer["cols"]

        # sizes
        orig_bytes = dense_tensor.numel() * dense_tensor.element_size()
        packed_bytes = (
            int(blobs_for_layer["n_2bit_bytes"]) + int(blobs_for_layer["n_4bit_bytes"]) + int(blobs_for_layer["n_8bit_bytes"])
            + blobs_for_layer["fp16_indices"].nbytes
            + blobs_for_layer["fp16_values"].nbytes
        )
        total_orig_bytes += orig_bytes
        total_packed_bytes += packed_bytes

        print(f"[EXPORT]   layer orig={orig_bytes/1e9:.4f} GB packed={packed_bytes/1e9:.4f} GB")

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
