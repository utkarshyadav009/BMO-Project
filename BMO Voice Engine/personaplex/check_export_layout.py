#!/usr/bin/env python3
"""
Check how depformer weights are exported to GGUF.

The export might concatenate weights differently than the checkpoint!
"""

import torch

ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
state_dict = ckpt.get("state_dict") or ckpt

print("[CHECKPOINT WEIGHT LAYOUT]")
print()

# Check how in_proj is concatenated in checkpoint
w_in_full = state_dict["depformer.layers.0.self_attn.in_proj_weight"].float()
print(f"Checkpoint depformer.layers.0.self_attn.in_proj_weight:")
print(f"  Shape: {w_in_full.shape}")
print(f"  Rows 0-3070 (step 0): {w_in_full[:3072, 0].min():.6f} to {w_in_full[:3072, 0].max():.6f}")
print(f"  Rows 3072-6143 (step 1): {w_in_full[3072:6144, 0].min():.6f} to {w_in_full[3072:6144, 0].max():.6f}")
print()

# Check out_proj
w_out_full = state_dict["depformer.layers.0.self_attn.out_proj.weight"].float()
print(f"Checkpoint depformer.layers.0.self_attn.out_proj_weight:")
print(f"  Shape: {w_out_full.shape}")
print(f"  Rows 0-1023 (step 0): {w_out_full[:1024, 0].min():.6f} to {w_out_full[:1024, 0].max():.6f}")
print(f"  Rows 1024-2047 (step 1): {w_out_full[1024:2048, 0].min():.6f} to {w_out_full[1024:2048, 0].max():.6f}")
print()

# Now check if export_bmo_gguf.py concatenates them or stores them per-step
print("[POSSIBLE EXPORT LAYOUTS]")
print()
print("Option A: Store all 16 steps concatenated (like checkpoint):")
print("  depformer_layers_0_self_attn_in_proj_weight: (49152, 1024)")
print("  C++ slices: [step * 3072 : (step+1) * 3072]")
print()
print("Option B: Store each step separately:")
print("  depformer_layers_0_self_attn_in_projs_0_weight: (3072, 1024)")
print("  depformer_layers_0_self_attn_in_projs_1_weight: (3072, 1024)")
print("  C++ loads individual step weights")
print()

# Check which keys exist in checkpoint
in_projs_keys = [k for k in state_dict if "in_projs" in k]
print(f"Found {len(in_projs_keys)} per-step in_proj weights:")
if in_projs_keys:
    for key in in_projs_keys[:3]:
        print(f"  {key}: {state_dict[key].shape}")
else:
    print("  None - weights are concatenated by step")
