#!/usr/bin/env python3
"""Check if in_proj_weight slicing is correct.

The in_proj_weight for depth is (49152, 1024) = 16 * 3072.
For step 0, C++ should use slice [0:3072, :].

This script verifies that PyTorch is slicing correctly.
"""

import torch
import numpy as np


def main():
    ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
    state_dict = ckpt.get("state_dict") or ckpt
    
    w_full = state_dict["depformer.layers.0.self_attn.in_proj_weight"].float()
    
    print(f"[CHECK] depformer.layers.0.self_attn.in_proj_weight")
    print(f"  Full shape: {w_full.shape}")
    print(f"  Expected: (49152, 1024) = 16 * (3072, 1024)")
    
    # Check that we can slice correctly
    for step in range(16):
        start = step * 3072
        end = start + 3072
        w_slice = w_full[start:end, :]
        print(f"  Step {step}: slice [{start}:{end}, :] → shape {w_slice.shape}")
        if step == 0:
            print(f"    ^ This is what verify_depth.py uses (in_proj[:3072, :])")
        if step == 1:
            print(f"    ...")
            break
    
    print(f"\n[VERIFY] Slicing logic is: offset = step * 3072")
    print(f"  Step 0: rows [0:3072]")
    print(f"  Step 1: rows [3072:6144]")
    print(f"  ... etc")
    
    # Double check that first few values are different between steps
    slice_0 = w_full[0:3072, 0]
    slice_1 = w_full[3072:6144, 0]
    
    print(f"\n[CHECK] Are step slices different?")
    print(f"  Step 0, column 0, first 3 values: {slice_0[:3]}")
    print(f"  Step 1, column 0, first 3 values: {slice_1[:3]}")
    
    if not torch.allclose(slice_0[:100], slice_1[:100]):
        print(f"  ✓ Yes, they're different (correct)")
    else:
        print(f"  ✗ They're identical?! Unexpected")


if __name__ == "__main__":
    main()
