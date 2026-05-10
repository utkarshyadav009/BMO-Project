#!/usr/bin/env python3
"""Compare text_emb.weight rows in the SOURCE PyTorch checkpoint.

Pairs with `probe_gguf_text_emb_rows.py`. If the source `.pt` shows
non-zero rows where the GGUF shows zeros, the bug is in
`export_bmo_gguf.py`. If the source itself has zero rows, the bug is
upstream (training / source weights).

Usage:

    python probe_pt_text_emb_rows.py \\
      --ckpt /path/to/bmo_jetson_ready.pt \\
      --tok-a 2 --tok-b 100

Optional: --candidates 0 1 2 3 100 1000 31999 32000 — list extra row
ids whose first-4 fp32 values you want printed.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import torch


def _find_text_emb(state_dict) -> tuple[str, torch.Tensor]:
    """Return (key, tensor) for the temporal text embedding."""
    candidates = (
        "text_emb.weight",
        "lm.text_emb.weight",
        "model.text_emb.weight",
    )
    for k in candidates:
        if k in state_dict:
            return k, state_dict[k]
    # Fallback: look for any key ending with text_emb.weight
    for k in state_dict.keys():
        if isinstance(k, str) and k.endswith("text_emb.weight"):
            return k, state_dict[k]
    raise SystemExit(
        f"text_emb.weight not found. Tried {candidates} and any *.text_emb.weight."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Path to bmo_jetson_ready.pt")
    ap.add_argument("--tok-a", type=int, default=2)
    ap.add_argument("--tok-b", type=int, default=100)
    ap.add_argument(
        "--candidates",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 100, 1000, 10000, 21000, 31000, 31999, 32000],
        help="Extra row ids to dump first-4 values for.",
    )
    args = ap.parse_args(argv)

    print(f"[probe_pt] loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        state_dict = ckpt if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        raise SystemExit("could not extract a state_dict from ckpt")

    key, t = _find_text_emb(state_dict)
    print(f"[probe_pt] found tensor key: {key}")
    print(f"[probe_pt] dtype={t.dtype} shape={tuple(t.shape)}")

    # Convert to fp32 for stable stats; do NOT modify on disk.
    arr = t.detach().to(torch.float32).cpu().numpy()
    if arr.ndim != 2:
        raise SystemExit(f"expected 2-d, got {arr.shape}")

    # Heuristic: vocab is the larger axis if one dim is 32000ish.
    if arr.shape[0] in (32000, 32001, 32768) and arr.shape[1] not in (32000, 32001):
        all_rows = arr
        vocab_axis = 0
    elif arr.shape[1] in (32000, 32001, 32768) and arr.shape[0] not in (32000, 32001):
        all_rows = arr.T
        vocab_axis = 1
    elif arr.shape[0] >= arr.shape[1]:
        all_rows = arr
        vocab_axis = 0
    else:
        all_rows = arr.T
        vocab_axis = 1
    n_vocab, dim = all_rows.shape
    print(f"[probe_pt] interpreted vocab_axis={vocab_axis} n_vocab={n_vocab} dim={dim}")

    sqnorms = np.einsum("ij,ij->i", all_rows, all_rows)
    is_zero = sqnorms == 0.0
    is_near = sqnorms < 1e-12
    n_exact = int(is_zero.sum())
    n_near = int(is_near.sum())
    print(f"[probe_pt] rows with sqnorm==0.0:   {n_exact}/{n_vocab}")
    print(f"[probe_pt] rows with sqnorm<1e-12:  {n_near}/{n_vocab}")
    nz = sqnorms[~is_near]
    if nz.size:
        print(
            f"[probe_pt] non-zero-row sqnorm: "
            f"min={float(nz.min()):.6e} median={float(np.median(nz)):.6e} "
            f"max={float(nz.max()):.6e} mean={float(nz.mean()):.6e}"
        )

    print()
    print(f"[probe_pt] sqnorm[tok_a={args.tok_a}] = {sqnorms[args.tok_a]:.10e}")
    print(f"[probe_pt] sqnorm[tok_b={args.tok_b}] = {sqnorms[args.tok_b]:.10e}")
    print(f"[probe_pt] tok_a row[:4] = {all_rows[args.tok_a][:4]}")
    print(f"[probe_pt] tok_b row[:4] = {all_rows[args.tok_b][:4]}")
    print(
        f"[probe_pt] max|row[a]-row[b]| = "
        f"{float(np.abs(all_rows[args.tok_a] - all_rows[args.tok_b]).max()):.6e}"
    )

    print()
    print("[probe_pt] candidate rows first-4 values + sqnorm:")
    for c in args.candidates:
        if c < 0 or c >= n_vocab:
            print(f"  tok={c:6d}  OUT_OF_RANGE")
            continue
        r = all_rows[c]
        print(
            f"  tok={c:6d}  sqnorm={float(np.dot(r, r)):.6e}  "
            f"first4={r[:4].tolist()}"
        )

    # First 64 zero-row indices for pattern-spotting.
    zero_idx = np.where(is_near)[0]
    if zero_idx.size:
        print(
            f"\n[probe_pt] first up-to-64 zero-row indices: "
            f"{zero_idx[:64].tolist()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
