#!/usr/bin/env python3
"""
Compare first few weight values to see if C++ is using the right slice.
"""

import torch
import numpy as np
from pathlib import Path

ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
state_dict = ckpt.get("state_dict") or ckpt

# Get PyTorch weights for layer 0
w_in_full = state_dict["depformer.layers.0.self_attn.in_proj_weight"].float()
w_out_full = state_dict["depformer.layers.0.self_attn.out_proj.weight"].float()

# Extract step 0 slices (what PyTorch should use)
w_in_step0_pt = w_in_full[:3072, :]  # First 3072 rows
w_out_step0_pt = w_out_full[:1024, :]  # First 1024 rows

print("[PyTorch Step 0 Weights]")
print(f"in_proj_weight slice [0:3072, :]: {w_in_step0_pt.shape}")
print(f"  Row 0, first 5 cols: {w_in_step0_pt[0, :5]}")
print(f"  Row 3071, first 5 cols: {w_in_step0_pt[3071, :5]}")
print()
print(f"out_proj_weight slice [0:1024, :]: {w_out_step0_pt.shape}")
print(f"  Row 0, first 5 cols: {w_out_step0_pt[0, :5]}")
print(f"  Row 1023, first 5 cols: {w_out_step0_pt[1023, :5]}")
print()

# Now check if C++ debug files exist and compare
if Path("cpp_debug_layer_0.bin").exists() and Path("pt_debug_layer_0.bin").exists():
    cpp_out = np.fromfile("cpp_debug_layer_0.bin", dtype=np.float32)
    pt_out = np.fromfile("pt_debug_layer_0.bin", dtype=np.float32)
    
    print("[After Layer 0]")
    print(f"C++: min/max = {cpp_out.min():.6f} / {cpp_out.max():.6f}")
    print(f"PT:  min/max = {pt_out.min():.6f} / {pt_out.max():.6f}")
    
    # Check scaling
    scale = np.abs(cpp_out).max() / (np.abs(pt_out).max() + 1e-9)
    print(f"C++ magnitude is {scale:.2f}x PyTorch")
else:
    print("[No layer 0 debug files - run debug_depth_per_layer.py first]")
