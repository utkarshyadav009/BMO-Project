#!/usr/bin/env python3
# Waveform-parity gate: rel_l2 between the old-kernel and new-kernel decoded
# streams (raw f32, 24 kHz mono, from BENCH_DUMP_PCM). Gate: rel_l2 < 1e-4.
import sys
import numpy as np

old = np.fromfile(sys.argv[1], dtype=np.float32)
new = np.fromfile(sys.argv[2], dtype=np.float32)
print(f"old: {old.size} samples ({old.size / 1920} frames)")
print(f"new: {new.size} samples ({new.size / 1920} frames)")
if old.size != new.size:
    print("WAVEFORM GATE: FAIL (length mismatch)")
    sys.exit(1)

diff = old.astype(np.float64) - new.astype(np.float64)
ref = np.linalg.norm(old.astype(np.float64))
rel_l2 = np.linalg.norm(diff) / ref if ref > 0 else float("inf")
max_abs = np.abs(diff).max()
print(f"rel_l2   = {rel_l2:.6e}")
print(f"max_abs  = {max_abs:.6e}")
print(f"ref_rms  = {ref / np.sqrt(old.size):.6e}")
ok = rel_l2 < 1e-4
print(f"WAVEFORM GATE (rel_l2 < 1e-4): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
