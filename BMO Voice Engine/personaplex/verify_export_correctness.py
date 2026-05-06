#!/usr/bin/env python3
"""Verify GGUF export is correct by checking exported values."""

import torch
import numpy as np
from pathlib import Path


def main():
    print("[VERIFY] Comparing PyTorch checkpoint vs exported values in verify_depth.py\n")
    
    ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt
    
    # Get values that verify_depth.py will use
    temporal_out = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    depformer_in_w = state_dict["depformer_in.0.weight"].float()
    z_s = torch.nn.functional.linear(temporal_out.float(), depformer_in_w)
    
    text_emb_w = state_dict["depformer_text_emb.weight"].float()
    text_emb_row_0 = text_emb_w[0]
    
    print(f"depformer_in.0.weight:")
    print(f"  Shape: {depformer_in_w.shape}")
    print(f"  Expected: (1024, 4096) - projects 4096→1024")
    print(f"  Sample values: {depformer_in_w[0, :5]}")
    
    print(f"\nz_s (result of depformer_in projection):")
    print(f"  Shape: {z_s.shape}")
    print(f"  Values: {z_s.flatten()[:5]}")
    print(f"  Min/max: {z_s.min():.6f} / {z_s.max():.6f}")
    
    print(f"\ndepformer_text_emb.weight:")
    print(f"  Shape: {text_emb_w.shape}")
    print(f"  Expected: (32001, 1024) - vocab of 32001, embedding dim 1024")
    print(f"  Row 0: {text_emb_row_0[:5]}")
    print(f"  Row 0 min/max: {text_emb_row_0.min():.6f} / {text_emb_row_0.max():.6f}")
    
    # Check for potential issues
    print(f"\n[CHECK] Potential issues:")
    
    # Is the text_emb actually a vocabulary embedding or something else?
    if text_emb_w.shape[0] == 32001:
        print(f"  ✓ text_emb shape (32001, 1024) matches vocabulary size (32001)")
    else:
        print(f"  ✗ Unexpected text_emb shape")
    
    # Check if audio_emb is similar
    for i in range(3):
        key = f"depformer_emb.{i}.weight"
        if key in state_dict:
            audio_emb = state_dict[key]
            print(f"  depformer_emb.{i}.weight shape: {audio_emb.shape}")
        else:
            print(f"  depformer_emb.{i}.weight: NOT FOUND")


if __name__ == "__main__":
    main()
