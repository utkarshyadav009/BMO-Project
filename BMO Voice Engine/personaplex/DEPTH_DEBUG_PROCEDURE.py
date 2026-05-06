#!/usr/bin/env python3
"""
DEPTH CASCADE DEBUG PROCEDURE
==============================

The depth cascade is producing cosine=0.41888631 instead of ≥0.999.
This diagnostic suite will identify where the C++ and PyTorch diverge.

SYMPTOMS:
- Temporal cascade: ✓ PASSED (cosine 0.99990+)
- Depth cascade: ✗ FAILED (cosine 0.41888)
- PyTorch code looks correct
- Suggests C++ is using wrong weights or embedding

WEIGHT SHAPES (from debug output):
- depformer.layers.0.self_attn.in_proj_weight: (49152, 1024) = 16*3072 
  → Step 0 uses rows [0:3072]
- depformer_text_emb.weight: (32001, 1024)
  → Step 0 uses row 0 (token embedding for token_id=0)
- depformer_in.0.weight: (1024, 4096)
  → Projects temporal output (4096,) → depth input (1024,)

EXECUTION PROCEDURE:
===================

# Step 1: Quick verification scripts (no C++ compilation)
python3 verify_export_correctness.py
python3 check_depth_shapes.py
python3 check_in_proj_slicing.py
python3 debug_depth_computation.py

# Step 2: Add debug dumps to C++ (modify bmo_compute.cpp)
# - Open bmo_compute.cpp
# - See bmo_compute_debug_patch.cpp for exact code locations
# - Copy debug code into bmo_compute.cpp around lines 520, 528, 537
# - Rebuild: cmake --build build -j

# Step 3: Run full validation with debug
rm cpp_debug_*.bin pt_debug_*.bin 2>/dev/null
./build/bmo_main bmo_weights_v12.gguf --mode depth_cascade
python3 verify_depth.py bmo_jetson_ready.pt
python3 analyze_depth_debug.py
python3 compare_tensors.py

EXPECTED OUTCOMES:
==================

IF PyTorch is correct (verify_depth.py):
  - debug_depth_computation.py will show correct values
  - check_in_proj_slicing.py will show step slices are different
  
IF C++ matches PyTorch (after adding debug):
  - cpp_debug_*.bin will match pt_debug_*.bin
  - compare_tensors.py will show cosine ≥ 0.999
  - Depth cascade passes ✓

IF C++ doesn't match (likely scenario):
  - analyze_depth_debug.py will show exactly where divergence occurs:
    - z_s mismatch? → depformer_in export issue
    - text_emb mismatch? → embedding table issue  
    - x_init mismatch? → addition/reshape issue
    - final output mismatch? → transformer computation wrong
    
ROOT CAUSE HYPOTHESES:
======================

1. **C++ using wrong embedding table**
   - Symptom: cpp_debug_text_emb.bin ≠ pt_debug_text_emb.bin
   - Issue: bmo_compute.cpp line ~527 uses wrong tensor
   - Fix: Verify ggml_get_rows(model.text_emb, text_tokens) gets row 0

2. **In_proj_weight slicing wrong**
   - Symptom: First layer attention output diverges
   - Issue: C++ offset calculation incorrect (not step * 3072)
   - Fix: Check bmo_compute.cpp lines ~560-567 slicing logic

3. **Embedding type conversion issue**
   - Symptom: All values are ~0 or NaN
   - Issue: GGUF export lost precision or wrong dtype
   - Fix: Verify export_bmo_gguf.py preserve_half=True on embeddings

4. **Tensor shape/memory layout mismatch**
   - Symptom: Values are permuted/transposed
   - Issue: Row-major vs column-major confusion
   - Fix: Check reshape operations in C++

---

Files Created:
- check_depth_shapes.py - Verify weight dimensions
- debug_depth_computation.py - Show correct PyTorch computation
- check_in_proj_slicing.py - Verify in_proj slicing logic
- verify_export_correctness.py - Check checkpoint values
- bmo_compute_debug_patch.cpp - Debug code to add to C++
- analyze_depth_debug.py - Compare intermediate C++/PT tensors
- run_diagnostics.py - Automated runner for all tests

NEXT STEPS:
===========

1. Run all Python scripts (no C++ changes needed yet):
   python3 run_diagnostics.py    # OR run individually

2. Review output carefully:
   - check_depth_shapes.py should show (49152, 1024) 
   - debug_depth_computation.py should show coherent values
   - check_in_proj_slicing.py should verify different step slices

3. If all Python scripts pass, the bug is in C++:
   - Edit bmo_compute.cpp with debug code from bmo_compute_debug_patch.cpp
   - Rebuild: cmake --build build -j
   - Re-run validation: ./build/bmo_main ... && python3 verify_depth.py ... && python3 analyze_depth_debug.py

4. analyze_depth_debug.py output will show exact point of divergence
"""

print(__doc__)
