#!/usr/bin/env python3
"""Verify depth-stack step 0 math with explicit debug output.

This version prints more diagnostics to help debug the divergence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from moshi.modules.rope import apply_rope


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight.view(1, 1, -1)


def main() -> int:
    ckpt_path = Path("bmo_jetson_ready.pt").resolve()
    out_path = Path("pt_depth_out.bin").resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("[verify_depth_debug] Loading checkpoint...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt

    temporal_out = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    text_tokens = torch.tensor([[0]], dtype=torch.long)

    print("[verify_depth_debug] Computing z_s (depformer_in projection)...")
    z_s = F.linear(temporal_out.float(), state_dict["depformer_in.0.weight"].float())
    print(f"  z_s.shape = {z_s.shape}")
    print(f"  z_s min/max = {z_s.min():.6f} / {z_s.max():.6f}")
    
    # Debug: dump z_s
    z_s_flat = z_s.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    z_s_flat.tofile("pt_debug_z_s.bin")
    print(f"  Wrote pt_debug_z_s.bin ({z_s_flat.nbytes} bytes)")

    print("[verify_depth_debug] Getting text embedding (depformer_text_emb.weight[0])...")
    text_emb = state_dict["depformer_text_emb.weight"][0].float()
    print(f"  text_emb.shape = {text_emb.shape}")
    print(f"  text_emb min/max = {text_emb.min():.6f} / {text_emb.max():.6f}")
    
    # Debug: dump text_emb
    text_emb_flat = text_emb.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    text_emb_flat.tofile("pt_debug_text_emb.bin")
    print(f"  Wrote pt_debug_text_emb.bin ({text_emb_flat.nbytes} bytes)")

    print("[verify_depth_debug] Adding embedding to z_s...")
    x = z_s + text_emb.view(1, 1, -1)
    print(f"  x.shape = {x.shape}")
    print(f"  x min/max = {x.min():.6f} / {x.max():.6f}")
    
    # Debug: dump x_init
    x_init_flat = x.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    x_init_flat.tofile("pt_debug_x_init.bin")
    print(f"  Wrote pt_debug_x_init.bin ({x_init_flat.nbytes} bytes)")

    num_heads = 16
    hidden_dim = 1024
    head_dim = hidden_dim // num_heads
    offset = torch.zeros(1, dtype=torch.long)

    print("[verify_depth_debug] Running 6 depth transformer layers...")
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
        
        print(f"  Layer {i}: x min/max = {x.min():.6f} / {x.max():.6f}")

    print("[verify_depth_debug] Writing final output...")
    out = x.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    out_path.write_bytes(out.tobytes())
    print(f"  Wrote {out_path} ({out.size} float32 values)")
    print(f"  Final output min/max = {out.min():.6f} / {out.max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
