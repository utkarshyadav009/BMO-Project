#!/usr/bin/env python3
"""verify_unpack.py — Simulate the C++ multi-tier unpacker in Python and compare
against the ground-truth dense weight from the PyTorch checkpoint.

Target layer: transformer.layers.0.gating.linear_in.weight
GGUF key:     transformer_layers_0_gating_linear_in_weight

This script:
  1. Loads the GGUF and extracts all packed components for the target layer.
  2. Loads the PyTorch checkpoint and extracts the dense weight.
  3. Replicates the exact C++ unpack_layer_to_f32() logic element-by-element.
  4. Compares the Python-unpacked result against the PyTorch ground truth.
"""

import sys
import struct
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# GGUF Reader (minimal, works with the gguf pip package)
# ──────────────────────────────────────────────────────────────────────
def load_gguf_tensors(path: str) -> dict:
    """Load all tensors from a GGUF file into a dict of numpy arrays."""
    import gguf
    reader = gguf.GGUFReader(path)
    tensors = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = tensor.data
        # data may be a memoryview or numpy array — normalise
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        tensors[name] = data
    return tensors


def read_scalar_i32(tensors: dict, key: str, fallback: int = 0) -> int:
    """Read a scalar int32 stored as a 1-element int8 tensor (GGUF coercion)."""
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    if len(raw) < 4:
        return fallback
    return struct.unpack('<i', raw[:4])[0]


def read_scalar_f32(tensors: dict, key: str, fallback: float = 0.0) -> float:
    """Read a scalar float32 stored as a 1-element int8 tensor (GGUF coercion)."""
    t = tensors.get(key)
    if t is None:
        return fallback
    raw = t.view(np.uint8).tobytes()
    if len(raw) < 4:
        return fallback
    return struct.unpack('<f', raw[:4])[0]


# ──────────────────────────────────────────────────────────────────────
# C++ Unpacker Replica
# ──────────────────────────────────────────────────────────────────────
def unpack_u2_le(byte_val: int, lane: int) -> int:
    """Exact replica of C++ unpack_u2_le(byte, lane)."""
    return (byte_val >> (lane * 2)) & 0x3


def unpack_layer_to_f32(
    packed_weights: np.ndarray,  # raw uint8 bytes
    packed_mask: np.ndarray,     # raw uint8 bytes
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
    fp16_indices: np.ndarray,    # int32
    fp16_values: np.ndarray,     # float16
) -> np.ndarray:
    """Exact Python replica of the C++ unpack_layer_to_f32() function."""
    total = rows * cols
    out_w = np.zeros(total, dtype=np.float32)

    # Split the packed_weights stream exactly as C++ does
    stream2 = packed_weights[:n_2bit_bytes]
    stream4 = packed_weights[n_2bit_bytes : n_2bit_bytes + n_4bit_bytes]
    stream8 = packed_weights[n_2bit_bytes + n_4bit_bytes : n_2bit_bytes + n_4bit_bytes + n_8bit_bytes]

    idx2 = 0
    idx4 = 0
    idx8 = 0

    for pos in range(total):
        mbyte = int(packed_mask[pos // 4])
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
        # tier == 0 -> v stays 0.0 (will be overwritten by fp16)

        out_w[pos] = v

    # Overwrite fp16 positions
    for i in range(len(fp16_indices)):
        pos = int(fp16_indices[i])
        if 0 <= pos < total:
            out_w[pos] = float(fp16_values[i])

    # Stream consumption check (same as C++)
    used2 = (idx2 + 3) // 4
    used4 = (idx4 + 1) // 2
    used8 = idx8
    if used2 != n_2bit_bytes or used4 != n_4bit_bytes or used8 != n_8bit_bytes:
        print(f"[WARNING] Stream padding mismatch!")
        print(f"  used2={used2} n2={n_2bit_bytes} delta={used2-n_2bit_bytes}")
        print(f"  used4={used4} n4={n_4bit_bytes} delta={used4-n_4bit_bytes}")
        print(f"  used8={used8} n8={n_8bit_bytes} delta={used8-n_8bit_bytes}")

    return out_w.reshape(rows, cols)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    gguf_path = "bmo_weights_v3.gguf"
    pt_path = "bmo_jetson_ready.pt"
    if len(sys.argv) > 1:
        gguf_path = sys.argv[1]
    if len(sys.argv) > 2:
        pt_path = sys.argv[2]

    layer_gguf_base = "transformer_layers_0_gating_linear_in_weight"
    layer_pt_key = "transformer.layers.0.gating.linear_in.weight"

    # ── Load GGUF ──
    print(f"[verify_unpack] Loading GGUF: {gguf_path}")
    tensors = load_gguf_tensors(gguf_path)
    print(f"[verify_unpack] Loaded {len(tensors)} tensors from GGUF")

    # Extract components
    base = layer_gguf_base
    packed_weights = tensors[base + ".packed_weights"].view(np.uint8)
    packed_mask    = tensors[base + ".packed_mask"].view(np.uint8)
    fp16_indices   = tensors[base + ".fp16_indices"]
    fp16_values    = tensors[base + ".fp16_values"]

    # Read scalars (stored as coerced int8 blobs in GGUF)
    rows         = read_scalar_i32(tensors, base + ".rows")
    cols         = read_scalar_i32(tensors, base + ".cols")
    n_2bit_bytes = read_scalar_i32(tensors, base + ".n_2bit_bytes")
    n_4bit_bytes = read_scalar_i32(tensors, base + ".n_4bit_bytes")
    n_8bit_bytes = read_scalar_i32(tensors, base + ".n_8bit_bytes")
    scale_low    = read_scalar_f32(tensors, base + ".scale_low", 1.0)
    scale_int4   = read_scalar_f32(tensors, base + ".scale_int4", 1.0)
    scale_int8   = read_scalar_f32(tensors, base + ".scale_int8", 1.0)
    zp_low       = read_scalar_f32(tensors, base + ".zp_low", 0.0)
    zp_int4      = read_scalar_f32(tensors, base + ".zp_int4", 0.0)
    zp_int8      = read_scalar_f32(tensors, base + ".zp_int8", 0.0)

    # Interpret fp16_values correctly
    raw_fp16 = tensors[base + ".fp16_values"].view(np.uint8)
    fp16_vals = np.frombuffer(raw_fp16.tobytes(), dtype=np.float16)

    # Interpret fp16_indices correctly
    raw_idx = tensors[base + ".fp16_indices"].view(np.uint8)
    fp16_idx = np.frombuffer(raw_idx.tobytes(), dtype=np.int32)

    print(f"\n[verify_unpack] Layer: {base}")
    print(f"  rows={rows}  cols={cols}  total={rows*cols}")
    print(f"  n_2bit_bytes={n_2bit_bytes}  n_4bit_bytes={n_4bit_bytes}  n_8bit_bytes={n_8bit_bytes}")
    print(f"  scale_low={scale_low:.6f}  scale_int4={scale_int4:.6f}  scale_int8={scale_int8:.6f}")
    print(f"  zp_low={zp_low:.6f}  zp_int4={zp_int4:.6f}  zp_int8={zp_int8:.6f}")
    print(f"  packed_weights bytes={packed_weights.size}")
    print(f"  packed_mask bytes={packed_mask.size}")
    print(f"  fp16_indices count={fp16_idx.size}")
    print(f"  fp16_values count={fp16_vals.size}")

    # ── Unpack (C++ replica) ──
    print(f"\n[verify_unpack] Running C++ unpacker replica...")
    unpacked = unpack_layer_to_f32(
        packed_weights, packed_mask,
        rows, cols,
        n_2bit_bytes, n_4bit_bytes, n_8bit_bytes,
        scale_low, scale_int4, scale_int8,
        zp_low, zp_int4, zp_int8,
        fp16_idx, fp16_vals,
    )
    print(f"  Unpacked shape: {unpacked.shape}")
    print(f"  Unpacked stats: mean={unpacked.mean():.6f}  std={unpacked.std():.6f}  "
          f"min={unpacked.min():.6f}  max={unpacked.max():.6f}")

    # ── Load PyTorch ground truth ──
    print(f"\n[verify_unpack] Loading PyTorch checkpoint: {pt_path}")
    import torch
    ckpt = torch.load(pt_path, map_location="cpu")
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    gt = state_dict[layer_pt_key].detach().cpu().float().numpy()
    print(f"  Ground truth shape: {gt.shape}")
    print(f"  Ground truth stats: mean={gt.mean():.6f}  std={gt.std():.6f}  "
          f"min={gt.min():.6f}  max={gt.max():.6f}")

    # ── Compare ──
    print(f"\n[verify_unpack] === COMPARISON ===")

    # First 10 values side by side
    flat_u = unpacked.ravel()
    flat_g = gt.ravel()
    n = min(10, flat_u.size, flat_g.size)
    print(f"\n  {'idx':>5}  {'C++ Unpack':>14}  {'PyTorch GT':>14}  {'Diff':>14}")
    print(f"  {'---':>5}  {'----------':>14}  {'----------':>14}  {'----':>14}")
    for i in range(n):
        diff = flat_u[i] - flat_g[i]
        print(f"  {i:5d}  {flat_u[i]:14.8f}  {flat_g[i]:14.8f}  {diff:14.8f}")

    # Cosine similarity
    dot = np.sum(flat_u * flat_g)
    norm_u = np.sqrt(np.sum(flat_u ** 2))
    norm_g = np.sqrt(np.sum(flat_g ** 2))
    cosine_sim = dot / (norm_u * norm_g + 1e-12)

    # MAE
    mae = np.mean(np.abs(flat_u - flat_g))

    # MSE
    mse = np.mean((flat_u - flat_g) ** 2)

    # Max absolute error
    max_err = np.max(np.abs(flat_u - flat_g))

    print(f"\n  Cosine Similarity: {cosine_sim:.8f}")
    print(f"  MAE:               {mae:.8f}")
    print(f"  MSE:               {mse:.8f}")
    print(f"  Max Abs Error:     {max_err:.8f}")
    print(f"  Ratio (mean_unpack / mean_gt): {flat_u.mean() / (flat_g.mean() + 1e-12):.6f}")

    # Tier distribution
    total = rows * cols
    tier_counts = [0, 0, 0, 0]
    for pos in range(min(total, 100000)):  # sample first 100k for speed
        mbyte = int(packed_mask[pos // 4])
        tier = unpack_u2_le(mbyte, pos % 4)
        tier_counts[tier] += 1
    sampled = sum(tier_counts)
    print(f"\n  Tier distribution (first {sampled} elements):")
    for t in range(4):
        print(f"    tier {t}: {tier_counts[t]:8d}  ({100*tier_counts[t]/sampled:.1f}%)")


if __name__ == "__main__":
    main()
