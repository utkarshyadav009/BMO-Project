#!/usr/bin/env python3
"""
Kernel sign-off close-out: per-layer residual diff, TRUE single-forward-pass,
no-sampling, OLD (8dfd1ba, mul_mat_vec_bmo_tier_cuda_kernel) vs NEW (803ee57,
mul_mat_vec_bmo_tier_tilemajor_kernel / _rowminor_kernel).

Reads outputs/single_pass_dumps_OLD_8dfd1ba/cpp_out_layer_{0..31}.bin and
outputs/single_pass_dumps_NEW_803ee57/cpp_out_layer_{0..31}.bin (float32, written
by the BMO_DUMP_LAYERS=1 instrumentation added to src/moshi/modules/transformer.h
and src/moshi/models/lm.h in both checkouts, this session).

Deliberately NOT reusing outputs/compare_kernels.py as-is (pointed at the
discredited outputs/old_dumps//new_dumps/ pair — see final report for the
provenance investigation) even though its arithmetic is fine; this script points
at the freshly-produced, provenance-clean directories instead, and is named
distinctly to avoid any ambiguity about which artifact produced which number.

Gate: rel_l2 < 1e-4 per layer (per task; this project's *microbench* gate
elsewhere is 1e-5 -- this per-layer integration gate is 1e-4, per task spec).
"""
import numpy as np
from pathlib import Path

old_dir = Path("/home/jovyan/work/BMO-Project-Repo/BMO-Project/outputs/single_pass_dumps_OLD_8dfd1ba")
new_dir = Path("/home/jovyan/work/BMO-Project-Repo/BMO-Project/outputs/single_pass_dumps_NEW_803ee57")

GATE = 1e-4

print(f"{'layer':>8}  {'n':>6}  {'max_abs_diff':>14}  {'rel_l2':>14}  {'status':>10}")
print("-" * 62)

rows = []
all_pass = True
for i in range(32):
    old_path = old_dir / f"cpp_out_layer_{i}.bin"
    new_path = new_dir / f"cpp_out_layer_{i}.bin"

    if not old_path.exists() or not new_path.exists():
        print(f"layer_{i:>2}  {'MISSING':>6}  {'MISSING':>14}  {'MISSING':>14}  {'FAIL':>10}")
        all_pass = False
        rows.append((i, None, None, "FAIL(MISSING)"))
        continue

    old_data = np.fromfile(old_path, dtype=np.float32)
    new_data = np.fromfile(new_path, dtype=np.float32)

    if old_data.size != new_data.size:
        print(f"layer_{i:>2}  {'SIZE MISMATCH':>14}  {'SIZE MISMATCH':>14}  {'FAIL':>10}")
        all_pass = False
        rows.append((i, None, None, "FAIL(SIZE_MISMATCH)"))
        continue

    abs_diff = np.abs(old_data - new_data)
    max_abs = float(np.max(abs_diff))

    diff_l2 = float(np.linalg.norm(old_data - new_data))
    old_l2 = float(np.linalg.norm(old_data)) + 1e-12
    rel_l2 = diff_l2 / old_l2

    status = "PASS" if rel_l2 < GATE else "FAIL"
    if status == "FAIL":
        all_pass = False
    rows.append((i, max_abs, rel_l2, status))
    print(f"layer_{i:>2}  {old_data.size:>6}  {max_abs:14.8e}  {rel_l2:14.8e}  {status:>10}")

print("-" * 62)
print(f"GATE: rel_l2 < {GATE:.0e} per layer")
print(f"OVERALL: {'PASS_PRODUCTION' if all_pass else 'FAIL'}")
