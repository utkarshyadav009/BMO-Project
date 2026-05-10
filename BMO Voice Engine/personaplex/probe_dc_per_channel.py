#!/usr/bin/env python3
"""Per-channel DC-injection probe.

Goal: at pos=0 with empty KV, feed PAD on the text channel and the
canonical SILENCE / SINE codes on every audio channel.  Then flip ONE
channel at a time to a "non-silence" value and watch which channel
produces a >10x jump in z_mean.

  * If only the text channel injects DC -> the temporal stack reacts
    badly to any wordpiece embedding and the bug is in the temporal
    weights (or how we apply them).
  * If audio cb1..cb7 inject DC -> our SILENCE/SINE codewords or
    delay wiring don't match the training distribution.
  * If the user channels (offsets 9..16) inject DC -> same as above
    but on the user side.

Only the C++ engine is touched; no PyTorch reference is required.
Per-call we engine.reset() so KV history can never leak across cases.

Run:
  PYTHONPATH=./moshi BMO_SO_PATH="$PWD/build_jetson/libbmo.so" \\
    python probe_dc_per_channel.py \\
      --gguf $PWD/bmo_septq_v3.gguf \\
      --mimi $PWD/tokenizer-e351c8d8-checkpoint125.safetensors \\
      --n-ctx 128 \\
      2>&1 | tee probe_dc_per_channel.log
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np


def _load_engine():
    from bmo_inference import _load_engine_class
    return _load_engine_class()


def _silence_baseline(engine) -> np.ndarray:
    """Return the canonical silent input vector (text=PAD, all audio=silence)."""
    from bmo_inference import (SILENCE_TOKENS, TEXT_PAD_ID,
                               DEFAULT_N_MOSHI_CODEBOOKS)
    K = engine.n_codebooks
    n_audio = max(0, K - 1)
    n_moshi = min(DEFAULT_N_MOSHI_CODEBOOKS, n_audio)
    n_user = n_audio - n_moshi

    toks = np.zeros(K, dtype=np.int32)
    toks[0] = TEXT_PAD_ID
    if n_moshi > 0:
        toks[1:1 + n_moshi] = SILENCE_TOKENS[:n_moshi].astype(np.int32, copy=False)
    if n_user > 0:
        toks[1 + n_moshi:1 + n_moshi + n_user] = SILENCE_TOKENS[:n_user].astype(np.int32, copy=False)
    return toks


def _stat(z: np.ndarray) -> Tuple[float, float, float, float]:
    z64 = z.astype(np.float64)
    return (float(z64.mean()), float(z64.std()),
            float(np.linalg.norm(z64)), float(np.abs(z64 - z64.mean()).max()))


def _fwd(engine, label: str, toks: np.ndarray):
    engine.reset()
    z, _ = engine.forward_temporal(toks)
    z = np.asarray(z, dtype=np.float32)
    m, s, n, _ = _stat(z)
    print(f"  {label:48s}  mean={m:+.5f}  std={s:.5f}  |z|={n:8.3f}  toks[0..4]={toks[:5].tolist()}")
    return z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mimi", required=True, help="(unused but kept for parity with other probes)")
    ap.add_argument("--n-ctx", type=int, default=128)
    args = ap.parse_args(argv)

    Engine = _load_engine()
    engine = Engine(args.gguf, n_ctx=args.n_ctx)

    base = _silence_baseline(engine)
    K = engine.n_codebooks
    print(f"[probe] K={K}")
    print(f"[probe] baseline toks={base.tolist()}\n")

    print("=== A: baseline (everything silent) ===")
    z_base = _fwd(engine, "baseline", base)

    print("\n=== B: flip ONLY the text channel ===")
    for tid, label in [(1, "text=BOS=1"),
                       (1000, "text=1000"),
                       (9267, "text=9267 (model's repeat-pred)"),
                       (4831, "text='system'=4831"),
                       (493, "text=' You'=493")]:
        toks = base.copy()
        toks[0] = tid
        _fwd(engine, label, toks)

    print("\n=== C: flip ONLY one moshi audio channel (cb0..cb7) ===")
    for cb in range(8):
        toks = base.copy()
        ch = 1 + cb  # text occupies index 0
        if ch < K:
            toks[ch] = (toks[ch] + 1) % 2048  # neighbouring codeword
            _fwd(engine, f"moshi cb{cb} -> +1", toks)

    print("\n=== D: flip ONLY one user audio channel (cb0..cb7) ===")
    for cb in range(8):
        toks = base.copy()
        ch = 1 + 8 + cb
        if ch < K:
            toks[ch] = (toks[ch] + 1) % 2048
            _fwd(engine, f"user  cb{cb} -> +1", toks)

    print("\n=== E: flip text + moshi cb0 together (lightest non-silent both sides) ===")
    for tid in (1, 1000):
        toks = base.copy()
        toks[0] = tid
        if K > 1:
            toks[1] = (toks[1] + 1) % 2048
        _fwd(engine, f"text={tid} + moshi cb0+=1", toks)

    print("\n=== F: zero out ALL audio channels (just text=PAD, everything else=0) ===")
    toks = np.zeros(K, dtype=np.int32)
    toks[0] = 3
    _fwd(engine, "text=PAD, audio=ZEROS", toks)

    print("\n=== Summary ===")
    print("Look for cases where mean shifts >0.3 vs baseline. Those channels")
    print("are the ones injecting DC.  If only text-channel cases shift the")
    print("mean, the DC source is in the temporal weights' projection of the")
    print("text embedding.  If audio-channel cases shift mean, the canonical")
    print("SILENCE/SINE codewords are wrong or delays are mis-wired.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
