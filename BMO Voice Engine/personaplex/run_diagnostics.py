#!/usr/bin/env python3
"""Master diagnostics script for depth cascade debugging.

Run this after running: verify_depth.py, check_depth_shapes.py, debug_depth_computation.py, check_in_proj_slicing.py, verify_export_correctness.py
"""

import subprocess
import sys
from pathlib import Path


def run_script(name, script):
    """Run a Python script and capture output."""
    print(f"\n{'='*70}")
    print(f"Running: {script}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {script} timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run {script}: {e}")
        return False


def check_files():
    """Check for required files."""
    print(f"\n{'='*70}")
    print("Checking for required files...")
    print(f"{'='*70}")
    
    required = [
        "bmo_jetson_ready.pt",
        "bmo_weights_v12.gguf",
        "verify_depth.py",
    ]
    
    for fname in required:
        if Path(fname).exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} - MISSING")
            return False
    
    return True


def main():
    print("[DIAGNOSTICS] BMO Depth Cascade Debug Suite")
    print("[DIAGNOSTICS] This will systematically identify where the divergence occurs\n")
    
    if not check_files():
        print("[ERROR] Missing required files")
        return 1
    
    scripts = [
        ("Export Verification", "verify_export_correctness.py"),
        ("Checkpoint Shapes", "check_depth_shapes.py"),
        ("In-Proj Slicing", "check_in_proj_slicing.py"),
        ("Depth Computation", "debug_depth_computation.py"),
    ]
    
    success = True
    for name, script in scripts:
        if Path(script).exists():
            if not run_script(name, script):
                success = False
        else:
            print(f"[SKIP] {script} not found")
    
    # Now run the actual depth cascade
    print(f"\n{'='*70}")
    print("Running actual depth cascade validation...")
    print(f"{'='*70}\n")
    
    print("[1/3] Running C++ depth cascade...")
    ret1 = subprocess.run(["./build/bmo_main", "bmo_weights_v12.gguf", "--mode", "depth_cascade"], timeout=60)
    
    print("\n[2/3] Generating PyTorch reference...")
    ret2 = subprocess.run([sys.executable, "verify_depth.py", "bmo_jetson_ready.pt"], timeout=60)
    
    print("\n[3/3] Comparing outputs...")
    ret3 = subprocess.run([sys.executable, "compare_tensors.py"], timeout=60)
    
    if ret1.returncode != 0 or ret2.returncode != 0 or ret3.returncode != 0:
        print("\n[ANALYSIS] One or more stages failed")
        return 1
    
    # Now run debug analysis if debug files exist
    if Path("cpp_debug_z_s.bin").exists() and Path("pt_debug_z_s.bin").exists():
        print(f"\n{'='*70}")
        print("Found debug files - analyzing intermediate tensors...")
        print(f"{'='*70}\n")
        subprocess.run([sys.executable, "analyze_depth_debug.py"], timeout=30)
    
    print("\n[DIAGNOSTICS] Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
