#!/usr/bin/env python3
"""probe_text_aliasing.py -- targeted probe for the text=2 / text=100 alias.

probe_locked_residual.py / test 0e showed that two different text token
ids (2 and 100) produce BIT-IDENTICAL final-z and bit-identical layer-0
output. That means the embedding-stage row lookup is collapsing
distinct tokens into the same row -- the upstream cause of the
"DC-attractor" symptom.

This script does the absolute minimum to confirm or refute that
hypothesis. It runs three forwards back-to-back on the same engine:

    1) text=PAD (3),  moshi=SIL, user=SIL    -- baseline
    2) text=  2,      moshi=SIL, user=SIL    -- "EOS"
    3) text=100,      moshi=SIL, user=SIL    -- regular wordpiece

For each forward we set BMO_LOG_EMBED=1 in the parent shell so the
C++ side prints the [bmo_embed] sum stats AND the raw text_emb_row.
If the row pointers really are different but the sums are equal, the
bug is in the per-row reduction. If the row pointers are equal, the
bug is in the index computation (tok * t->nb[1]). If the sums are
different but the L=0 / final-z is identical, the bug is downstream
in the temporal kernel chain (very unlikely given the existing data).

Usage (mirror the locked-residual probe):

    BMO_LOG_EMBED=1 \\
    PYTHONPATH=./moshi BMO_SO_PATH="$PWD/build_jetson/libbmo.so" \\
      python probe_text_aliasing.py \\
        --gguf "$PWD/bmo_septq_v3.gguf" \\
        --mimi "$PWD/tokenizer-e351c8d8-checkpoint125.safetensors" \\
        --n-ctx 128

The script forces n_ctx low so init is cheap; we don't need a long
history. Output is ~30 lines; paste it back wholesale.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from bmo_inference import (
    SILENCE_TOKENS,
    TEXT_PAD_ID,
    DEFAULT_N_MOSHI_CODEBOOKS,
    _load_engine_class,
    _load_mimi,
)


def _fwd(engine, label, text_token, n_moshi, n_user):
    K = engine.n_codebooks
    toks = np.zeros(K, dtype=np.int32)
    toks[0] = int(text_token)
    if n_moshi > 0:
        toks[1:1 + n_moshi] = SILENCE_TOKENS[:n_moshi].astype(np.int32, copy=False)
    if n_user > 0:
        toks[1 + n_moshi:1 + n_moshi + n_user] = SILENCE_TOKENS[:n_user].astype(
            np.int32, copy=False)

    sys.stdout.write(f"\n=== {label}  text_token={int(text_token)}  toks={toks.tolist()} ===\n")
    sys.stdout.flush()
    # Force a clean engine state per call so KV history can't affect the
    # comparison. We are at pos_base 0 every time.
    engine.reset()
    z, lt = engine.forward_temporal(toks)
    z = np.asarray(z, dtype=np.float32)
    lt = np.asarray(lt, dtype=np.float32)
    print(
        f"    z: mean={float(z.mean()):+.6f} std={float(z.std()):.6f} "
        f"|z|={float(np.linalg.norm(z)):.4f} z[0..3]={z[0]:+.4f},{z[1]:+.4f},{z[2]:+.4f},{z[3]:+.4f}"
    )
    print(
        f"   lt: argmax={int(np.argmax(lt))} max={float(np.max(lt)):+.4f} "
        f"lt[0]={float(lt[0]):+.4f} lt[1]={float(lt[1]):+.4f} "
        f"lt[2]={float(lt[2]):+.4f} lt[3]={float(lt[3]):+.4f}"
    )
    return z, lt


def main(argv=None):
    p = argparse.ArgumentParser("probe_text_aliasing")
    p.add_argument("--gguf", required=True)
    p.add_argument("--mimi", required=True)
    p.add_argument("--n-ctx", type=int, default=128)
    args = p.parse_args(argv)

    if not os.environ.get("BMO_LOG_EMBED"):
        sys.stderr.write(
            "[probe_text_aliasing] WARNING: BMO_LOG_EMBED not set in the "
            "environment; you'll only see Python-side stats. Re-run with "
            "BMO_LOG_EMBED=1 to also see the C++ [bmo_embed] line.\n")

    print(f"[probe] gguf={args.gguf} n_ctx={args.n_ctx}")
    Engine = _load_engine_class()
    # mimi is required by Engine some setups but we don't actually need it
    # to run forward_temporal. Load it anyway so the engine init path is
    # the same as in the bigger probes.
    _ = _load_mimi(args, device="cpu")
    engine = Engine(args.gguf, n_ctx=args.n_ctx)

    K = engine.n_codebooks
    n_audio = max(0, K - 1)
    n_moshi = min(DEFAULT_N_MOSHI_CODEBOOKS, n_audio)
    n_user = n_audio - n_moshi
    print(f"[probe] K={K} n_moshi={n_moshi} n_user={n_user}")

    cases = [
        ("baseline (text=PAD=3)", TEXT_PAD_ID),
        ("text=2 (EOS)",          2),
        ("text=100 (wordpiece)",  100),
        # extra cases for cross-checks (these are *known* to differ from
        # the above in test 0e, so if they also collapse we have a much
        # bigger problem)
        ("text=1 (BOS)",          1),
        ("text=1000",             1000),
    ]

    zs = []
    for (label, tok) in cases:
        z, _ = _fwd(engine, label, tok, n_moshi, n_user)
        zs.append((label, tok, z.copy()))

    # Cross-pair: paint a small table of cos(z[i],z[j]) and max|diff|.
    print("\n=== pairwise z comparison ===")
    print(f"  {'pair':40s} | cos(zi,zj)   max|zi-zj|   mean|zi-zj|")
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            li, ti, zi = zs[i]
            lj, tj, zj = zs[j]
            zi64 = zi.astype(np.float64)
            zj64 = zj.astype(np.float64)
            ni = float(np.linalg.norm(zi64))
            nj = float(np.linalg.norm(zj64))
            cos = float(np.dot(zi64, zj64) / (ni * nj)) if (ni > 0 and nj > 0) else float("nan")
            diff = np.abs(zi64 - zj64)
            tag = ""
            if cos > 0.999999:
                tag = "  <-- IDENTICAL"
            elif cos > 0.99:
                tag = "  <-- near-identical"
            print(f"  {li[:18]:18s} vs {lj[:18]:18s} | {cos:+.8f}  {diff.max():.6f}    {diff.mean():.6f}{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
