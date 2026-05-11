#!/usr/bin/env python3
"""Compare C++ H3 T4 V-slice vs T5 for L=0 (single-token harness).

Loads ``cpp_h3_T4_L0.bin`` (Q||K||V packed as one row, width = 3 * n_embd) and
``cpp_h3_T5_L0.bin`` (post-attention, n_embd). Optionally compares PT tensors from
``.npz`` (keys ``t4``, ``t5`` float32 row vectors).

Usage:
  python scripts/compare_h3_t4_v_slice_vs_t5.py --bin-dir path/to/cpp_h3_bins
  python scripts/compare_h3_t4_v_slice_vs_t5.py --bin-dir ... --pt-npz pt_l0.npz

If ``t4`` width is not 3*n_embd, pass ``--n-embd``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", type=Path, required=True, help="Directory with cpp_h3_T4_L0.bin / cpp_h3_T5_L0.bin")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--n-embd", type=int, default=None, help="If omitted, inferred as len(t5) (expect 4096)")
    ap.add_argument("--pt-npz", type=Path, default=None, help="Optional .npz with float32 keys t4, t5")
    args = ap.parse_args()
    d = args.bin_dir.resolve()
    L = int(args.layer)
    t4p = d / f"cpp_h3_T4_L{L}.bin"
    t5p = d / f"cpp_h3_T5_L{L}.bin"
    if not t4p.is_file() or not t5p.is_file():
        raise SystemExit(f"Missing {t4p.name} or {t5p.name} under {d}")

    t4 = np.fromfile(str(t4p), dtype=np.float32).astype(np.float64)
    t5 = np.fromfile(str(t5p), dtype=np.float32).astype(np.float64)
    n5 = int(t5.size)
    n_embd = int(args.n_embd) if args.n_embd is not None else n5
    if n5 != n_embd:
        raise SystemExit(f"T5 length {n5} != n_embd {n_embd}")
    if t4.size % 3 != 0:
        raise SystemExit(f"T4 length {t4.size} not divisible by 3")
    if t4.size != 3 * n_embd:
        raise SystemExit(f"T4 length {t4.size} != 3*n_embd ({3 * n_embd}); pass --n-embd if layout differs")

    v = t4[2 * n_embd : 3 * n_embd].copy()
    print(f"[C++] L={L}  len(T4)={t4.size}  len(T5)={t5.size}")
    print(f"  L2(V_slice)={np.linalg.norm(v):.6f}  L2(T5)={np.linalg.norm(t5):.6f}")
    print(f"  cos(V_slice, T5)={_cos(v, t5):.8f}")
    print(f"  first8 V: {np.array2string(v[:8], precision=6, floatmode='fixed')}")
    print(f"  first8 T5: {np.array2string(t5[:8], precision=6, floatmode='fixed')}")

    # Simulate C++ eager path: V written to FP16 KV cache then read as F32 for the matvec.
    v_rt = v.astype(np.float32).astype(np.float16).astype(np.float64)
    print(f"  cos(fp16_roundtrip(V), T5)={_cos(v_rt, t5):.8f}  (if ~match, KV FP16 path explains gap vs raw V)")

    if args.pt_npz is not None:
        z = np.load(str(args.pt_npz.resolve()))
        pt4 = np.asarray(z["t4"], dtype=np.float64).ravel()
        pt5 = np.asarray(z["t5"], dtype=np.float64).ravel()
        if pt4.size != 3 * n_embd or pt5.size != n_embd:
            raise SystemExit(f"PT t4/t5 shape mismatch: t4={pt4.size} t5={pt5.size} n_embd={n_embd}")
        pv = pt4[2 * n_embd : 3 * n_embd]
        print(f"[PT_FQ] L={L}")
        print(f"  L2(V_slice)={np.linalg.norm(pv):.6f}  L2(T5)={np.linalg.norm(pt5):.6f}")
        print(f"  cos(V_slice, T5)={_cos(pv, pt5):.8f}")
        print(f"  first8 V: {np.array2string(pv[:8], precision=6, floatmode='fixed')}")
        print(f"  first8 T5: {np.array2string(pt5[:8], precision=6, floatmode='fixed')}")


if __name__ == "__main__":
    main()
