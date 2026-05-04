#!/usr/bin/env python3
"""verify_all_layers.py — Layer-0 sub-layer variance audit in PyTorch.

Runs only transformer layer 0 on a dummy all-ones tensor and dumps the outputs
of norm1, self_attn, norm2, and gating (FFN) to binary files.
"""

import sys

import torch

sys.path.insert(0, "moshi")

from moshi.models.loaders import get_moshi_lm


def _extract_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if torch.is_tensor(item):
                return item
    raise RuntimeError(f"Hook output is not a tensor-like object: {type(output)}")


def _dump_hook_output(path):
    def _hook(_module, _inputs, output):
        tensor = _extract_tensor(output).detach().float().contiguous().cpu()
        with open(path, "wb") as f:
            f.write(tensor.numpy().tobytes())
        print(
            f"[verify_all_layers] dumped {path}: "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )

    return _hook


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
    layer0 = layers[0]

    for attr in ("norm1", "self_attn", "norm2", "gating"):
        if not hasattr(layer0, attr):
            raise RuntimeError(f"Layer 0 missing expected submodule: {attr}")

    hooks = [
        layer0.norm1.register_forward_hook(_dump_hook_output("pt_l0_norm1.bin")),
        layer0.self_attn.register_forward_hook(_dump_hook_output("pt_l0_attn.bin")),
        layer0.norm2.register_forward_hook(_dump_hook_output("pt_l0_norm2.bin")),
        layer0.gating.register_forward_hook(_dump_hook_output("pt_l0_ffn.bin")),
    ]

    dummy = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    print("[verify_all_layers] Running Layer 0 sub-layer audit")
    print("[verify_all_layers] Input: torch.ones(1, 1, 4096, dtype=bfloat16)")

    try:
        with torch.no_grad():
            output = layer0(dummy)
    finally:
        for hook in hooks:
            hook.remove()

    out_f32 = output.detach().float().contiguous().cpu()
    with open("pt_l0_out.bin", "wb") as f:
        f.write(out_f32.numpy().tobytes())
    print(
        "[verify_all_layers] dumped pt_l0_out.bin: "
        f"shape={tuple(out_f32.shape)} dtype={out_f32.dtype}"
    )


if __name__ == "__main__":
    main()