#!/usr/bin/env python3
"""Verify depth-stack step 0 math against a PyTorch reference.

This reproduces the dummy C++ validation inputs and writes the final output
tensor to pt_depth_out.bin for comparison with cpp_depth_out.bin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from moshi.modules.rope import apply_rope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch depth-step 0 validator")
    parser.add_argument("ckpt", nargs="?", default="bmo_jetson_ready.pt")
    parser.add_argument("out", nargs="?", default="pt_depth_out.bin")
    return parser.parse_args()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight.view(1, 1, -1)


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    out_path = Path(args.out).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt

    temporal_out = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    text_tokens = torch.tensor([[0]], dtype=torch.long)
    audio_tokens = torch.tensor([[0]], dtype=torch.long)

    z_s = F.linear(temporal_out.float(), state_dict["depformer_in.0.weight"].float())
    # Step 0 uses ONLY depformer_text_emb (text token embedding).
    # depformer_emb[k-1] is used for steps k>=1 (audio codebook embeddings).
    # See moshi/models/lm.py: depformer_text_emb for cb_index==0, depformer_emb[cb_index-1] otherwise.
    x = z_s + state_dict["depformer_text_emb.weight"][0].float().view(1, 1, -1)

    num_heads = 16
    hidden_dim = 1024
    head_dim = hidden_dim // num_heads
    offset = torch.zeros(1, dtype=torch.long)

    for i in range(6):
        prefix = f"depformer.layers.{i}"

        x_norm = rms_norm(x, state_dict[f"{prefix}.norm1.alpha"].float())
        w_in = state_dict[f"{prefix}.self_attn.in_proj_weight"].float()[:3072]
        projected = F.linear(x_norm, w_in)
        q, k, v = projected.chunk(3, dim=-1)

        q = q.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        q, k = apply_rope(q, k, offset, max_period=10000, time_before_heads=False)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(1, 1, hidden_dim)

        w_out = state_dict[f"{prefix}.self_attn.out_proj.weight"].float()[:1024]
        attn = F.linear(attn, w_out)
        x = x + attn

        x_norm = rms_norm(x, state_dict[f"{prefix}.norm2.alpha"].float())
        ff_in_w = state_dict[f"{prefix}.gating.0.linear_in.weight"].float()
        ff_out_w = state_dict[f"{prefix}.gating.0.linear_out.weight"].float()

        ff_in = F.linear(x_norm, ff_in_w)
        gate, up = ff_in.chunk(2, dim=-1)
        ff_act = F.silu(gate) * up
        ff_out = F.linear(ff_act, ff_out_w)
        x = x + ff_out

    out = x.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    out_path.write_bytes(out.tobytes())
    print(f"[verify_depth] wrote {out_path} ({out.size} float32 values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())