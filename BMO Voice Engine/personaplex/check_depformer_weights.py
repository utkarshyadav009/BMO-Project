#!/usr/bin/env python3
"""
Check that depformer weight shapes in checkpoint match what C++ expects.

The depth transformer uses 16 different sets of weights (one per codebook step).
Each set has in_proj (3072 cols) and out_proj (1024 cols).

They're concatenated as:
- depformer.layers.i.self_attn.in_proj_weight: (49152, 1024) = [step0:3072, step1:3072, ..., step15:3072]
- depformer.layers.i.self_attn.out_proj.weight: (16384, 1024) = [step0:1024, step1:1024, ..., step15:1024]

For step 0, PyTorch uses [:3072] for in_proj and [:1024] for out_proj.
C++ uses offset = step * 3072 for in_proj and step * 1024 for out_proj.
"""

import torch

ckpt = torch.load("bmo_jetson_ready.pt", map_location="cpu")
state_dict = ckpt.get("state_dict") or ckpt

print("[WEIGHT SHAPES]")
for layer in range(6):
    prefix = f"depformer.layers.{layer}"
    
    w_in = state_dict[f"{prefix}.self_attn.in_proj_weight"]
    w_out = state_dict[f"{prefix}.self_attn.out_proj.weight"]
    
    print(f"\nLayer {layer}:")
    print(f"  in_proj_weight: {w_in.shape}")
    print(f"    Expected: (49152, 1024) = 16 * (3072, 1024)")
    print(f"    Actual:   {w_in.shape[0]} x {w_in.shape[1]}")
    
    if w_in.shape != torch.Size([49152, 1024]):
        print(f"    ✗ MISMATCH!")
    else:
        print(f"    ✓ Correct")
    
    print(f"  out_proj_weight: {w_out.shape}")
    print(f"    Expected: (16384, 1024) = 16 * (1024, 1024)")
    print(f"    Actual:   {w_out.shape[0]} x {w_out.shape[1]}")
    
    if w_out.shape != torch.Size([16384, 1024]):
        print(f"    ✗ MISMATCH!")
    else:
        print(f"    ✓ Correct")
    
    # Verify slicing is different per step
    in_step0 = w_in[:3072]
    in_step1 = w_in[3072:6144]
    
    match = torch.allclose(in_step0, in_step1)
    if match:
        print(f"  ✗ WARNING: Step 0 and Step 1 slices are identical!")
    else:
        print(f"  ✓ Step slices are different")

print("\n[FFN WEIGHTS]")
for layer in range(6):
    prefix = f"depformer.layers.{layer}"
    
    for step in range(16):
        ffn_in_key = f"{prefix}.gating.{step}.linear_in.weight"
        if ffn_in_key not in state_dict:
            if step == 0:
                print(f"Layer {layer}: FFN weights NOT FOUND at {ffn_in_key}")
                print(f"  Available gating keys: {[k for k in state_dict if f'{prefix}.gating' in k][:5]}")
            break
        ffn_in = state_dict[ffn_in_key]
        if step == 0:
            print(f"Layer {layer}: gating.0.linear_in.weight: {ffn_in.shape} ✓")
