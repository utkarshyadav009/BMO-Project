#!/usr/bin/env python3
"""compare_tensors.py — 32-layer cascade comparator.

Compares C++ vs PyTorch dumps for every layer output, printing cosine
similarity and MAE per layer.
"""

from pathlib import Path

import numpy as np


def _compare_pair(cpp_path, pt_path):
    cpp = np.fromfile(cpp_path, dtype=np.float32)
    pt = np.fromfile(pt_path, dtype=np.float32)

    if cpp.size == 0 or pt.size == 0:
        return None, None, "EMPTY"

    status = "OK"
    if cpp.size != pt.size:
        n = min(cpp.size, pt.size)
        cpp = cpp[:n]
        pt = pt[:n]
        status = "SIZE_MISMATCH"

    dot = float(np.dot(cpp, pt))
    denom = float(np.linalg.norm(cpp) * np.linalg.norm(pt)) + 1e-12
    cosine = dot / denom
    mae = float(np.mean(np.abs(cpp - pt)))
    return cosine, mae, status


def main():
    rows = []
    for i in range(32):
        name = f"layer_{i}"
        cpp_path = f"cpp_out_layer_{i}.bin"
        pt_path = f"pt_out_layer_{i}.bin"

        if not Path(cpp_path).exists() or not Path(pt_path).exists():
            rows.append((name, None, None, "MISSING"))
            continue

        cosine, mae, status = _compare_pair(cpp_path, pt_path)
        rows.append((name, cosine, mae, status))

    print(f"{'layer':>8}  {'cosine':>12}  {'mae':>12}  {'status':>14}")
    print(f"{'--------':>8}  {'------------':>12}  {'------------':>12}  {'--------------':>14}")
    for name, cosine, mae, status in rows:
        if cosine is None:
            print(f"{name:>8}  {'-':>12}  {'-':>12}  {status:>14}")
        else:
            print(f"{name:>8}  {cosine:12.8f}  {mae:12.8f}  {status:>14}")

    valid = [r for r in rows if r[1] is not None]
    if valid:
        cos_vals = [r[1] for r in valid]
        mae_vals = [r[2] for r in valid]
        print()
        print(f"[summary] cosine min={min(cos_vals):.8f}  mean={np.mean(cos_vals):.8f}")
        print(f"[summary] mae    max={max(mae_vals):.8f}  mean={np.mean(mae_vals):.8f}")
    else:
        print()
        print("[summary] no comparable layer pairs found")


if __name__ == "__main__":
    main()
