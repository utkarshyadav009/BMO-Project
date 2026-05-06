#!/usr/bin/env python3
"""Inspect depth tensors in GGUF file."""

import sys
sys.path.insert(0, "llama.cpp/ggml-python")

import struct
from pathlib import Path

def inspect_gguf(fname):
    """Read GGUF file and dump tensor info."""
    from ggml import ggml_load_model_from_file
    
    # Try alternative: parse GGUF header manually
    with open(fname, "rb") as f:
        # GGUF magic
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Not a GGUF file: {magic}")
            return
        
        # GGUF version
        version = struct.unpack("<I", f.read(4))[0]
        print(f"GGUF Version: {version}")
        
        # Number of tensors
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        print(f"Number of tensors: {n_tensors}")
        
        # Key-value pairs
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"Number of KV pairs: {n_kv}\n")
        
        # Skip KV pairs (they're complex to parse)
        for i in range(n_kv):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="ignore")
            val_type = struct.unpack("<I", f.read(4))[0]
            # Skip value parsing for now
        
        # Read tensor info
        print("Depth-related tensors:")
        for i in range(n_tensors):
            try:
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode("utf-8", errors="ignore")
                
                if "depth" in name.lower() or "depformer" in name.lower():
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    shape = []
                    for _ in range(n_dims):
                        shape.append(struct.unpack("<Q", f.read(8))[0])
                    ggml_type = struct.unpack("<I", f.read(4))[0]
                    offset = struct.unpack("<Q", f.read(8))[0]
                    print(f"  {name}: shape={shape} type={ggml_type} offset={offset}")
                else:
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    for _ in range(n_dims):
                        struct.unpack("<Q", f.read(8))
                    struct.unpack("<I", f.read(4))
                    struct.unpack("<Q", f.read(8))
            except Exception as e:
                print(f"Error reading tensor {i}: {e}")
                break

if __name__ == "__main__":
    fname = "bmo_weights_v12.gguf" if not sys.argv[1:] else sys.argv[1]
    inspect_gguf(fname)
