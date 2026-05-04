#!/usr/bin/env python3
"""verify_all_layers.py — 32-layer cascade dump harness in PyTorch.

Runs the full transformer stack on a dummy all-ones tensor and dumps the
output after each layer to pt_out_layer_{i}.bin.
"""

import sys

import torch

sys.path.insert(0, "moshi")

from moshi.models.loaders import get_moshi_lm


def main():
    checkpoint = "bmo_jetson_ready.pt"
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    print(f"[verify_all_layers] Loading checkpoint: {checkpoint}")
    model = get_moshi_lm(checkpoint, device="cpu", dtype=torch.bfloat16)
    model.eval()

    transformer = model.transformer
    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "inner"):
        layers = transformer.inner.layers
    else:
        raise RuntimeError("Cannot find transformer layers in model")

    if len(layers) == 0:
        raise RuntimeError("No transformer layers found")

    print(f"[verify_all_layers] Found {len(layers)} transformer layers")
    dummy = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    print("[verify_all_layers] Running 32-layer cascade")
    print("[verify_all_layers] Input: torch.ones(1, 1, 4096, dtype=bfloat16)")

    x = dummy
    with torch.no_grad():
        for i, layer in enumerate(layers):
            x = layer(x)
            out_f32 = x.detach().float().contiguous().cpu()
            path = f"pt_out_layer_{i}.bin"
            with open(path, "wb") as f:
                f.write(out_f32.numpy().tobytes())
            print(
                f"[verify_all_layers] dumped {path}: "
                f"shape={tuple(out_f32.shape)} dtype={out_f32.dtype}"
            )


if __name__ == "__main__":
    main()