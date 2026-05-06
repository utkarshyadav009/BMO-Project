#!/usr/bin/env python3
"""Dump C++ depth computation internals by adding debug output.

Instructions:
1. Modify bmo_compute.cpp depth graph building to dump intermediate tensors
2. This script creates a helper to check what values C++ computed
"""

import numpy as np
from pathlib import Path


def compare_file_pairs(label, cpp_path, pt_path):
    """Compare two binary files (assumed float32)."""
    cpp_file = Path(cpp_path)
    pt_file = Path(pt_path)
    
    if not cpp_file.exists():
        print(f"[MISSING] {cpp_path}")
        return
    if not pt_file.exists():
        print(f"[MISSING] {pt_path}")
        return
    
    cpp_data = np.fromfile(cpp_file, dtype=np.float32)
    pt_data = np.fromfile(pt_file, dtype=np.float32)
    
    print(f"\n{label}:")
    print(f"  C++ size: {cpp_data.size} values")
    print(f"  PT size:  {pt_data.size} values")
    
    if cpp_data.size > 0:
        print(f"  C++ min/max: {cpp_data.min():.6f} / {cpp_data.max():.6f}")
    if pt_data.size > 0:
        print(f"  PT min/max: {pt_data.min():.6f} / {pt_data.max():.6f}")
    
    if cpp_data.size == pt_data.size and cpp_data.size > 0:
        cosine = float(np.dot(cpp_data, pt_data)) / (np.linalg.norm(cpp_data) * np.linalg.norm(pt_data) + 1e-12)
        mae = float(np.mean(np.abs(cpp_data - pt_data)))
        print(f"  Cosine: {cosine:.8f}")
        print(f"  MAE: {mae:.6f}")
        
        if cosine > 0.99:
            print(f"  ✓ MATCH")
        else:
            print(f"  ✗ MISMATCH - values diverge significantly")
            # Find where they first diverge
            diffs = np.abs(cpp_data - pt_data)
            worst_idx = np.argmax(diffs)
            print(f"    Largest diff at index {worst_idx}: C++={cpp_data[worst_idx]:.6f} vs PT={pt_data[worst_idx]:.6f} (diff={diffs[worst_idx]:.6f})")


if __name__ == "__main__":
    print("[DEBUG] Comparing PyTorch debug outputs with C++ outputs")
    
    compare_file_pairs("z_s (depformer_in projection)", 
                      "cpp_debug_z_s.bin", "pt_debug_z_s.bin")
    
    compare_file_pairs("text_emb (embedding row 0)", 
                      "cpp_debug_text_emb.bin", "pt_debug_text_emb.bin")
    
    compare_file_pairs("x_init (after embedding add)", 
                      "cpp_debug_x_init.bin", "pt_debug_x_init.bin")
    
    compare_file_pairs("FINAL depth_out_step_0", 
                      "cpp_depth_out.bin", "pt_depth_out.bin")
