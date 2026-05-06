#!/usr/bin/env python3
"""
Deep debug: dump PyTorch values after each depth layer.

This will show us exactly which layer causes the divergence.
"""

import torch
import torch.nn.functional as F
import numpy as np
from moshi.modules.rope import apply_rope


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight.view(1, 1, -1)


ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
state_dict = ckpt.get("state_dict") or ckpt

temporal_out = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
z_s = F.linear(temporal_out.float(), state_dict["depformer_in.0.weight"].float())
x = z_s + state_dict["depformer_text_emb.weight"][0].float().view(1, 1, -1)

print(f"[DEBUG] x_init: min={x.min():.6f} max={x.max():.6f}")

num_heads = 16
hidden_dim = 1024
head_dim = hidden_dim // num_heads
offset = torch.zeros(1, dtype=torch.long)

for i in range(6):
    prefix = f"depformer.layers.{i}"
    
    # Check weights exist
    w_in_key = f"{prefix}.self_attn.in_proj_weight"
    w_out_key = f"{prefix}.self_attn.out_proj.weight"
    
    if w_in_key not in state_dict:
        print(f"[ERROR] {w_in_key} not found!")
        break
    
    w_in_full = state_dict[w_in_key].float()
    w_out_full = state_dict[w_out_key].float()
    
    print(f"\n[LAYER {i}] Weights:")
    print(f"  w_in_full shape: {w_in_full.shape} (should be (49152, 1024) = 16*3072 x 1024)")
    print(f"  w_out_full shape: {w_out_full.shape} (should be (16384, 1024) = 16*1024 x 1024)")
    
    # Step 0 slicing
    w_in = w_in_full[:3072]
    w_out = w_out_full[:1024]
    
    print(f"  w_in (step 0 slice) shape: {w_in.shape} (should be (3072, 1024))")
    print(f"  w_out (step 0 slice) shape: {w_out.shape} (should be (1024, 1024))")
    
    x_norm = rms_norm(x, state_dict[f"{prefix}.norm1.alpha"].float())
    projected = F.linear(x_norm, w_in)
    q, k, v = projected.chunk(3, dim=-1)

    q = q.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = k.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v = v.view(1, 1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    q, k = apply_rope(q, k, offset, max_period=10000, time_before_heads=False)
    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    attn = attn.permute(0, 2, 1, 3).contiguous().view(1, 1, hidden_dim)

    attn = F.linear(attn, w_out)
    x = x + attn

    print(f"  After attention: x min={x.min():.6f} max={x.max():.6f}")

    x_norm = rms_norm(x, state_dict[f"{prefix}.norm2.alpha"].float())
    ff_in_w = state_dict[f"{prefix}.gating.0.linear_in.weight"].float()
    ff_out_w = state_dict[f"{prefix}.gating.0.linear_out.weight"].float()

    ff_in = F.linear(x_norm, ff_in_w)
    gate, up = ff_in.chunk(2, dim=-1)
    ff_act = F.silu(gate) * up
    ff_out = F.linear(ff_act, ff_out_w)
    x = x + ff_out
    
    print(f"  After FFN: x min={x.min():.6f} max={x.max():.6f}")
    
    # Save layer output
    x_flat = x.reshape(-1).contiguous().cpu().float().numpy().astype(np.float32)
    x_flat.tofile(f"pt_debug_layer_{i}.bin")

print(f"\n[DEBUG] Final: min={x.min():.6f} max={x.max():.6f}")
