#!/usr/bin/env python3
"""Compare PyTorch per-layer outputs with C++ per-layer outputs."""

import numpy as np
from pathlib import Path


for i in range(6):
    cpp_path = f"cpp_debug_layer_{i}.bin"
    pt_path = f"pt_debug_layer_{i}.bin"
    
    if not Path(cpp_path).exists() or not Path(pt_path).exists():
        print(f"[SKIP] Layer {i}: missing files")
        continue
    
    cpp_data = np.fromfile(cpp_path, dtype=np.float32)
    pt_data = np.fromfile(pt_path, dtype=np.float32)
    
    print(f"[LAYER {i}]")
    print(f"  C++ min/max: {cpp_data.min():.6f} / {cpp_data.max():.6f}")
    print(f"  PT min/max:  {pt_data.min():.6f} / {pt_data.max():.6f}")
    
    if cpp_data.size == pt_data.size and cpp_data.size > 0:
        cosine = float(np.dot(cpp_data, pt_data)) / (np.linalg.norm(cpp_data) * np.linalg.norm(pt_data) + 1e-12)
        mae = float(np.mean(np.abs(cpp_data - pt_data)))
        print(f"  Cosine: {cosine:.8f}")
        print(f"  MAE: {mae:.6f}")
        
        if cosine > 0.99:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ DIVERGE")
            # Show where it first diverges
            diffs = np.abs(cpp_data - pt_data)
            worst_idx = np.argmax(diffs)
            print(f"    Worst diff at index {worst_idx}: C++={cpp_data[worst_idx]:.6f} vs PT={pt_data[worst_idx]:.6f}")
    print()
