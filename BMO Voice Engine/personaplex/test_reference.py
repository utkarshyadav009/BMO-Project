"""
Minimal Python ground-truth harness for one packed multi-tier layer.

Loads layer artifacts from bmo_weights.gguf.npz (export_bmo_gguf.py fallback output),
reconstructs the transient F32 weight matrix using the same little-endian unpacking
conventions, performs y = x @ W^T (implemented as F.linear), and prints first 20 values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_uint2_le_stream(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpack little-endian 2-bit stream: 4 values per byte."""
    packed = packed.astype(np.uint8).reshape(-1)
    out = np.empty(packed.size * 4, dtype=np.uint8)
    out[0::4] = packed & 0b11
    out[1::4] = (packed >> 2) & 0b11
    out[2::4] = (packed >> 4) & 0b11
    out[3::4] = (packed >> 6) & 0b11
    return out[:count]


@dataclass
class LayerBlob:
    rows: int
    cols: int
    packed_mask: np.ndarray
    packed_weights: np.ndarray
    n_2bit_bytes: int
    n_4bit_bytes: int
    n_8bit_bytes: int
    scale_low: float
    scale_int4: float
    scale_int8: float
    fp16_indices: np.ndarray
    fp16_values: np.ndarray
    zp_low: float = 0.0
    zp_int4: float = 0.0
    zp_int8: float = 0.0
    bias: np.ndarray | None = None


def load_layer_blob(npz_path: str, layer_base: str) -> LayerBlob:
    z = np.load(npz_path)

    def get(name: str):
        key = f"{layer_base}.{name}"
        if key not in z:
            raise KeyError(f"Missing key in npz: {key}")
        return z[key]

    # zero-points are optional in current exporter; default to 0 if absent
    def get_optional_scalar(name: str, default: float) -> float:
        key = f"{layer_base}.{name}"
        if key in z:
            return float(np.array(z[key]).reshape(-1)[0])
        return default

    bias_key = f"{layer_base}.bias"
    bias = z[bias_key] if bias_key in z else None

    return LayerBlob(
        rows=int(np.array(get("rows")).reshape(-1)[0]),
        cols=int(np.array(get("cols")).reshape(-1)[0]),
        packed_mask=get("packed_mask").astype(np.uint8),
        packed_weights=get("packed_weights").astype(np.uint8),
        n_2bit_bytes=int(np.array(get("n_2bit_bytes")).reshape(-1)[0]),
        n_4bit_bytes=int(np.array(get("n_4bit_bytes")).reshape(-1)[0]),
        n_8bit_bytes=int(np.array(get("n_8bit_bytes")).reshape(-1)[0]),
        scale_low=float(np.array(get("scale_low")).reshape(-1)[0]),
        scale_int4=float(np.array(get("scale_int4")).reshape(-1)[0]),
        scale_int8=float(np.array(get("scale_int8")).reshape(-1)[0]),
        fp16_indices=get("fp16_indices").astype(np.int32),
        fp16_values=get("fp16_values").astype(np.float16),
        zp_low=get_optional_scalar("zp_low", 0.0),
        zp_int4=get_optional_scalar("zp_int4", 0.0),
        zp_int8=get_optional_scalar("zp_int8", 0.0),
        bias=None if bias is None else bias.astype(np.float32),
    )


class MultiTierLinearInference(nn.Module):
    """Single-layer module mirroring profile_jetson MultiTier dequant semantics."""

    def __init__(self, blob: LayerBlob):
        super().__init__()
        self.rows = blob.rows
        self.cols = blob.cols

        self.register_buffer("packed_mask", torch.from_numpy(blob.packed_mask))
        self.register_buffer("packed_weights", torch.from_numpy(blob.packed_weights))

        self.n_2bit_bytes = blob.n_2bit_bytes
        self.n_4bit_bytes = blob.n_4bit_bytes
        self.n_8bit_bytes = blob.n_8bit_bytes

        self.scale_low = float(blob.scale_low)
        self.scale_int4 = float(blob.scale_int4)
        self.scale_int8 = float(blob.scale_int8)
        self.zp_low = float(blob.zp_low)
        self.zp_int4 = float(blob.zp_int4)
        self.zp_int8 = float(blob.zp_int8)

        self.register_buffer("fp16_indices", torch.from_numpy(blob.fp16_indices))
        self.register_buffer("fp16_values", torch.from_numpy(blob.fp16_values.astype(np.float32)))

        if blob.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.from_numpy(blob.bias), requires_grad=False)

    def _dequantize_weight_f32(self) -> torch.Tensor:
        total = self.rows * self.cols

        # 1) Unpack tier mask (uint2 LE)
        mask = torch.empty(total, dtype=torch.uint8)
        pm = self.packed_mask
        mask[0::4] = pm & 0b11
        mask[1::4] = (pm >> 2) & 0b11
        mask[2::4] = (pm >> 4) & 0b11
        mask[3::4] = (pm >> 6) & 0b11
        mask = mask[:total]

        # 2) Decode packed weight streams
        pw = self.packed_weights
        p2 = pw[: self.n_2bit_bytes]
        p4 = pw[self.n_2bit_bytes : self.n_2bit_bytes + self.n_4bit_bytes]
        p8 = pw[self.n_2bit_bytes + self.n_4bit_bytes : self.n_2bit_bytes + self.n_4bit_bytes + self.n_8bit_bytes]

        q = torch.zeros(total, dtype=torch.float32)

        idx_t3 = torch.nonzero(mask >= 3, as_tuple=False).squeeze(-1)
        if idx_t3.numel() > 0:
            vals = torch.empty(p2.numel() * 4, dtype=torch.uint8)
            vals[0::4] = p2 & 0b11
            vals[1::4] = (p2 >> 2) & 0b11
            vals[2::4] = (p2 >> 4) & 0b11
            vals[3::4] = (p2 >> 6) & 0b11
            vals = vals[: idx_t3.numel()].to(torch.float32)
            q[idx_t3] = (vals - self.zp_low) * self.scale_low

        idx_t2 = torch.nonzero(mask == 2, as_tuple=False).squeeze(-1)
        if idx_t2.numel() > 0:
            vals = torch.empty(p4.numel() * 2, dtype=torch.uint8)
            vals[0::2] = p4 & 0x0F
            vals[1::2] = (p4 >> 4) & 0x0F
            vals = vals[: idx_t2.numel()].to(torch.float32)
            q[idx_t2] = (vals - self.zp_int4) * self.scale_int4

        idx_t1 = torch.nonzero(mask == 1, as_tuple=False).squeeze(-1)
        if idx_t1.numel() > 0:
            vals = p8[: idx_t1.numel()].to(torch.float32)
            q[idx_t1] = (vals - self.zp_int8) * self.scale_int8

        # Tier 0 exact values
        if self.fp16_indices.numel() > 0:
            q[self.fp16_indices.long()] = self.fp16_values

        return q.view(self.rows, self.cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize_weight_f32()
        return F.linear(x, w, self.bias)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="bmo_weights.gguf.npz")
    ap.add_argument("--layer", default="transformer_layers_0_gating_linear_in")
    args = ap.parse_args()

    blob = load_layer_blob(args.npz, args.layer)
    layer = MultiTierLinearInference(blob)

    x = torch.ones((1, 1, blob.cols), dtype=torch.float32)
    y = layer(x)

    first20 = y.reshape(-1)[:20].detach().cpu().numpy()
    print("Ground Truth first 20 values:")
    print(np.array2string(first20, precision=7, separator=", "))


if __name__ == "__main__":
    main()
