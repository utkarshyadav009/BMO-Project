#!/usr/bin/env python3
"""Compare text_emb.weight rows 2 vs 100 inside a GGUF (no libbmo).

Use when debugging suspected embedding aliasing: if rows match on disk,
the checkpoint/export is wrong; if they differ on disk but C++ yields
identical z, pointer math / accumulation is wrong.

  PYTHONPATH=./llama.cpp/gguf-py python probe_gguf_text_emb_rows.py \\
    --gguf ./bmo_septq_v3.gguf

Depends only on numpy + vendored gguf-py under llama.cpp/gguf-py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _setup_gguf_path() -> None:
    root = Path(__file__).resolve().parent
    cand = root / "llama.cpp" / "gguf-py"
    if cand.is_dir():
        sys.path.insert(0, str(cand))


def _extract_vocab_rows(arr: np.ndarray, tok_a: int, tok_b: int):
    """Return (ra, rb, vocab_axis) for text_emb [vocab, dim] or [dim, vocab]."""
    if arr.ndim != 2:
        raise SystemExit(f"expected 2-d text_emb, got shape={arr.shape}")

    sh = arr.shape
    # Prefer interpreting axis whose length matches typical SPM vocab (+ specials).
    if sh[0] in (32000, 32001, 32768) and sh[1] not in (32000, 32001):
        vocab_ax = 0
        ra = arr[tok_a]
        rb = arr[tok_b]
    elif sh[1] in (32000, 32001, 32768) and sh[0] not in (32000, 32001):
        vocab_ax = 1
        ra = arr[:, tok_a]
        rb = arr[:, tok_b]
    elif sh[0] <= sh[1]:
        # Heuristic: smaller leading dim often embedding width (4096).
        vocab_ax = 1
        ra = arr[:, tok_a]
        rb = arr[:, tok_b]
    else:
        vocab_ax = 0
        ra = arr[tok_a]
        rb = arr[tok_b]
    return ra, rb, vocab_ax


def main(argv=None) -> int:
    _setup_gguf_path()
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise SystemExit(
            "Could not import gguf. Set PYTHONPATH to llama.cpp/gguf-py "
            f"(tried {Path(__file__).resolve().parent / 'llama.cpp' / 'gguf-py'}): {e}"
        ) from e

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="Path to .gguf")
    ap.add_argument("--tok-a", type=int, default=2)
    ap.add_argument("--tok-b", type=int, default=100)
    args = ap.parse_args(argv)

    reader = GGUFReader(args.gguf)
    tensor = None
    for t in reader.tensors:
        if t.name == "text_emb.weight":
            tensor = t
            break
    if tensor is None:
        raise SystemExit("tensor text_emb.weight not found in GGUF")

    arr = np.asarray(tensor.data)
    print(f"[probe_gguf] text_emb.weight shape={arr.shape} dtype={arr.dtype} "
          f"type={tensor.tensor_type.name}")

    ta, tb, vaxis = _extract_vocab_rows(arr, args.tok_a, args.tok_b)
    ta32 = ta.astype(np.float32)
    tb32 = tb.astype(np.float32)
    diff = np.abs(ta32 - tb32)
    print(f"[probe_gguf] interpreted vocab_axis={vaxis} (0=rows are vocab, "
          f"1=cols are vocab)")
    print(f"[probe_gguf] tok_a={args.tok_a} tok_b={args.tok_b}")
    print(f"[probe_gguf] first4_a={ta32[:4]}")
    print(f"[probe_gguf] first4_b={tb32[:4]}")
    print(f"[probe_gguf] max_abs_diff={float(diff.max()):.6g} "
          f"mean_abs_diff={float(diff.mean()):.6g} "
          f"identical_all={bool(np.all(ta32 == tb32))}")
    sq_a = float(np.dot(ta32, ta32))
    sq_b = float(np.dot(tb32, tb32))
    print(f"[probe_gguf] full_vec_sqnorm tok_a={sq_a:.10f} tok_b={sq_b:.10f} "
          f"same_sqnorm={sq_a == sq_b}")

    # Now scan ALL rows and identify which ones are all-zero. This tells us
    # whether the bug is targeted (rows 2/100 only) or systematic (large
    # chunks of vocab missing). Either way, the action is different:
    #
    #   * Few rows zero -> targeted export bug, paste suspect indices for
    #     comparison against PyTorch checkpoint.
    #   * Many rows zero -> chunked write / dtype / quant pipeline issue
    #     in export_bmo_gguf.py, and we look at runs of zero rows.
    print("\n[probe_gguf] scanning ALL vocab rows for zero / near-zero rows ...")
    if vaxis == 0:
        all_rows = arr.astype(np.float32)
    else:
        all_rows = arr.T.astype(np.float32)
    n_vocab, dim = all_rows.shape
    sqnorms = np.einsum("ij,ij->i", all_rows, all_rows)
    is_exact_zero = (sqnorms == 0.0)
    is_near_zero = (sqnorms < 1e-12)
    n_exact = int(is_exact_zero.sum())
    n_near = int(is_near_zero.sum())
    print(f"[probe_gguf] vocab_size={n_vocab} dim={dim}")
    print(f"[probe_gguf] rows with sqnorm==0.0:   {n_exact}/{n_vocab} "
          f"({100.0*n_exact/n_vocab:.2f}%)")
    print(f"[probe_gguf] rows with sqnorm<1e-12:  {n_near}/{n_vocab} "
          f"({100.0*n_near/n_vocab:.2f}%)")

    # Print sqnorm stats over the non-zero rows for sanity.
    nz_mask = ~is_near_zero
    if nz_mask.any():
        nz = sqnorms[nz_mask]
        print(f"[probe_gguf] non-zero-row sqnorm: "
              f"min={float(nz.min()):.6e} median={float(np.median(nz)):.6e} "
              f"max={float(nz.max()):.6e} mean={float(nz.mean()):.6e}")

    # List the first 64 zero-row indices so we can spot patterns
    # (consecutive runs? aligned on power-of-two? scattered?).
    zero_idx = np.where(is_near_zero)[0]
    head = zero_idx[:64].tolist()
    tail = zero_idx[-64:].tolist() if len(zero_idx) > 64 else []
    print(f"[probe_gguf] first up-to-64 zero-row indices: {head}")
    if tail:
        print(f"[probe_gguf] last  up-to-64 zero-row indices: {tail}")

    # Find longest contiguous run of zero rows -- chunked-write bug
    # signature.
    if len(zero_idx) > 0:
        gaps = np.diff(zero_idx)
        run_starts = np.concatenate(([0], np.where(gaps != 1)[0] + 1))
        run_ends = np.concatenate((np.where(gaps != 1)[0], [len(zero_idx) - 1]))
        runs = [(int(zero_idx[s]), int(zero_idx[e]), int(zero_idx[e] - zero_idx[s] + 1))
                for s, e in zip(run_starts, run_ends)]
        runs.sort(key=lambda r: -r[2])
        print("[probe_gguf] longest contiguous zero-row runs (start, end, length):")
        for r in runs[:10]:
            print(f"            [{r[0]}..{r[1]}]  len={r[2]}")

    # Also scan the audio embedding tables to see if the same pattern is there.
    print("\n[probe_gguf] scanning audio codebook embedding tables (emb.{k}.weight) ...")
    for t in reader.tensors:
        if not (t.name.startswith("emb.") and t.name.endswith(".weight")):
            continue
        a = np.asarray(t.data).astype(np.float32)
        if a.ndim != 2:
            continue
        # Heuristic: the larger axis is vocab.
        if a.shape[0] > a.shape[1]:
            rows = a
        else:
            rows = a.T
        sq = np.einsum("ij,ij->i", rows, rows)
        n_zero = int((sq < 1e-12).sum())
        nz_min = float(sq[sq >= 1e-12].min()) if (sq >= 1e-12).any() else float("nan")
        nz_max = float(sq[sq >= 1e-12].max()) if (sq >= 1e-12).any() else float("nan")
        shape_str = str(tuple(a.shape))
        print(f"[probe_gguf]   {t.name:30s} shape={shape_str:16s} "
              f"zero_rows={n_zero}/{rows.shape[0]:5d}  "
              f"nz_sqnorm_range=[{nz_min:.4e}, {nz_max:.4e}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
