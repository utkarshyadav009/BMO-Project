#!/usr/bin/env python3
"""Check depth transformer weight shapes and concatenation."""

import torch

ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
state_dict = ckpt.get("state_dict") or ckpt

print("Depth transformer weight shapes:")
print()

# Check a single depth layer's attention weights
key = "depformer.layers.0.self_attn.in_proj_weight"
if key in state_dict:
    w = state_dict[key]
    print(f"{key}:")
    print(f"  Shape: {w.shape}")
    print(f"  Should be: (3072 * num_steps, 1024) or similar")
    print()

# Check for per-step weights if they exist
print("Checking for per-step weights:")
for step in range(16):
    key_step = f"depformer.layers.0.self_attn.in_projs.{step}.weight"
    if key_step in state_dict:
        w = state_dict[key_step]
        print(f"  {key_step}: {w.shape}")
        if step == 0:
            print(f"    ^ Step 0 shape")
        if step == 1:
            break

print()
print("Checking depformer_in shapes:")
for step in range(16):
    key = f"depformer_in.{step}.weight"
    if key in state_dict:
        w = state_dict[key]
        print(f"  {key}: {w.shape}")
        if step == 0:
            print(f"    ^ Should be (1024, 4096) for projecting 4096→1024")
        if step == 2:
            break
