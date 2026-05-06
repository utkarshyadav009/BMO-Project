#!/usr/bin/env python3
"""Step-by-step depth computation with detailed dumps for debugging.

Run this AFTER running verify_depth.py to compare intermediate tensors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * denom * weight.view(1, 1, -1)


def main():
    ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt

    temporal_out = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    text_tokens = torch.tensor([[0]], dtype=torch.long)

    # Step 1: depformer_in projection
    z_s = F.linear(temporal_out.float(), state_dict["depformer_in.0.weight"].float())
    print(f"[DEBUG] z_s.shape = {z_s.shape}")
    print(f"[DEBUG] z_s.min/max = {z_s.min():.6f} / {z_s.max():.6f}")
    
    # Step 2: text embedding lookup
    text_emb_full = state_dict["depformer_text_emb.weight"]
    print(f"[DEBUG] text_emb_full.shape = {text_emb_full.shape}")
    text_emb_row_0 = text_emb_full[0].float()
    print(f"[DEBUG] text_emb_row_0.shape = {text_emb_row_0.shape}")
    print(f"[DEBUG] text_emb_row_0.min/max = {text_emb_row_0.min():.6f} / {text_emb_row_0.max():.6f}")
    
    # Step 3: add embedding to z_s
    x = z_s + text_emb_row_0.view(1, 1, -1)
    print(f"[DEBUG] x (after embedding add).shape = {x.shape}")
    print(f"[DEBUG] x.min/max = {x.min():.6f} / {x.max():.6f}")
    
    # Step 4: first transformer layer
    prefix = "depformer.layers.0"
    
    x_norm = rms_norm(x, state_dict[f"{prefix}.norm1.alpha"].float())
    print(f"[DEBUG] x_norm.min/max = {x_norm.min():.6f} / {x_norm.max():.6f}")
    
    # Get the full in_proj_weight and slice for step 0
    w_in_full = state_dict[f"{prefix}.self_attn.in_proj_weight"].float()
    print(f"[DEBUG] w_in_full.shape = {w_in_full.shape}")
    
    # Step 0 uses slice [0:3072]
    w_in = w_in_full[:3072]
    print(f"[DEBUG] w_in (step 0 slice).shape = {w_in.shape}")
    print(f"[DEBUG] w_in.min/max = {w_in.min():.6f} / {w_in.max():.6f}")
    
    projected = F.linear(x_norm, w_in)
    q, k, v = projected.chunk(3, dim=-1)
    
    print(f"[DEBUG] After attn in_proj: q.min/max = {q.min():.6f} / {q.max():.6f}")
    print(f"[DEBUG] After attn in_proj: k.min/max = {k.min():.6f} / {k.max():.6f}")
    print(f"[DEBUG] After attn in_proj: v.min/max = {v.min():.6f} / {v.max():.6f}")
    
    # Now check: what if C++ is using a different embedding?
    print("\n[CHECK] What if C++ accidentally used depformer_emb.0?")
    if "depformer_emb.0.weight" in state_dict:
        alt_emb = state_dict["depformer_emb.0.weight"][0].float()
        print(f"[DEBUG] depformer_emb.0.weight[0].shape = {alt_emb.shape}")
        print(f"[DEBUG] depformer_emb.0.weight[0].min/max = {alt_emb.min():.6f} / {alt_emb.max():.6f}")
        x_alt = z_s + alt_emb.view(1, 1, -1)
        print(f"[DEBUG] x_alt (wrong embedding).min/max = {x_alt.min():.6f} / {x_alt.max():.6f}")
    
    print("\n[CONCLUSION] Correct embedding for step 0 is depformer_text_emb.weight[0]")


if __name__ == "__main__":
    main()
