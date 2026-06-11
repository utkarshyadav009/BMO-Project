#!/usr/bin/env python3
"""Smoke test: load turbo2bit LM with Linear2bit and run a tiny CUDA forward."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Turbo2bit inference disables compiled kernels (Triton unavailable on Windows).
os.environ.setdefault("NO_CUDA_GRAPH", "1")
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from native2bit_loader import DEFAULT_WEIGHT, get_moshi_lm_native2bit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; use --device cpu for structure-only checks.")
        sys.exit(1)

    device = args.device
    print(f"Loading native 2-bit LM from {args.weight} on {device}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = get_moshi_lm_native2bit(args.weight, device=device)
    print("Load OK.")

    # Minimal forward through temporal transformer (single frame, batch=1).
    with torch.no_grad():
        x = torch.zeros(1, 1, model.dim, device=device, dtype=torch.bfloat16)
        out = model.transformer(x)
        _ = out.shape

    if torch.cuda.is_available() and device.startswith("cuda"):
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak CUDA memory: {peak_gb:.2f} GB")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
