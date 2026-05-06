#!/usr/bin/env python3
"""Check how weights are stored in GGUF export."""

import sys
import struct
import numpy as np

# Try to load with gguf-py
try:
    import gguf
    gguf_lib = gguf
except ImportError:
    print("ERROR: gguf module not found. Install with: pip install gguf")
    sys.exit(1)

gguf_path = "bmo_v12_septq.gguf"

print(f"[LOADING GGUF] {gguf_path}\n")

try:
    reader = gguf_lib.GGUFReader(gguf_path)
except Exception as e:
    print(f"ERROR opening GGUF: {e}")
    sys.exit(1)

# Look for depformer in_proj and out_proj weights
print("[GGUF TENSOR KEYS]")
tensor_keys = list(reader.tensors.keys())
depformer_keys = [k for k in tensor_keys if "depformer_layers" in k and "self_attn" in k and "weight" in k]
depformer_keys.sort()

for key in depformer_keys[:20]:  # Show first 20
    tensor = reader.tensors[key]
    shape = tensor.shape
    print(f"  {key}: {shape}")

print(f"\n[TOTAL DEPFORMER ATTENTION WEIGHTS] {len(depformer_keys)}")

# Check layer 0 in_proj specifically
print("\n[LAYER 0 IN_PROJ]")
layer0_in_proj_key = "depformer_layers_0_self_attn_in_proj_weight"
if layer0_in_proj_key in reader.tensors:
    tensor = reader.tensors[layer0_in_proj_key]
    shape = tensor.shape
    print(f"  Key: {layer0_in_proj_key}")
    print(f"  Shape: {shape}")
    
    # Load actual data
    try:
        data = reader.tensors[layer0_in_proj_key].data
        arr = np.frombuffer(data, dtype=np.float16 if "fp16" in str(tensor.data_type) else np.float32)
        arr = arr.reshape(shape)
        
        # Check step 0 and step 1 slices
        print(f"\n  Step 0 slice [0:3072]: min={arr[:3072].min():.6f}, max={arr[:3072].max():.6f}")
        print(f"  Step 1 slice [3072:6144]: min={arr[3072:6144].min():.6f}, max={arr[3072:6144].max():.6f}")
        
        # Also check if layout is (features, hidden) or (hidden, features)
        print(f"\n  Data type: {tensor.data_type}")
        print(f"  Shape interpretation: {shape[0]} features x {shape[1]} hidden (or vice versa)")
    except Exception as e:
        print(f"  Error loading tensor data: {e}")
else:
    print(f"  ERROR: {layer0_in_proj_key} not found in GGUF!")
    print(f"  Available in_proj keys: {[k for k in tensor_keys if 'in_proj' in k]}")

# Check out_proj too
print("\n[LAYER 0 OUT_PROJ]")
layer0_out_proj_key = "depformer_layers_0_self_attn_out_proj_weight"
if layer0_out_proj_key in reader.tensors:
    tensor = reader.tensors[layer0_out_proj_key]
    shape = tensor.shape
    print(f"  Key: {layer0_out_proj_key}")
    print(f"  Shape: {shape}")
else:
    print(f"  ERROR: {layer0_out_proj_key} not found in GGUF!")
