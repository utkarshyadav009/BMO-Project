"""Bisect probe #2: does the temporal text head EVER predict PAD/EPAD?

We saw at frame 0 the model emits real wordpieces with text_max ~+8.7. That
might be normal for a fresh sequence; the canonical PyTorch path runs a 4-step
warmup (with all-zeros input) before any real generation, and PersonaPlex was
trained with substantial KV history.

This probe runs N successive forward_temporal calls feeding the canonical
"prompt phase" inputs (text=PAD=3, moshi=SILENCE_TOKENS, user=SINE_TOKENS) and
prints text_argmax for each. Outcomes:

  * text_argmax converges to 0 (EPAD) or 3 (PAD) within ~10-20 frames:
      -> head is fine, sequence just needs warmup. Add explicit warmup to
         mode_stream.

  * text_argmax stays clustered in 21300+ tokens forever:
      -> head is genuinely producing OOD distribution. Suggests one of:
         (a) text_linear weight loaded/transposed wrong.
         (b) out_norm gamma not applied correctly.
         (c) GGUF tensor dtype mismatch for text_linear.

  * text_argmax oscillates wildly between random tokens:
      -> KV cache is being corrupted, or the residual stream is degenerate.
         Likely a layer norm / residual add bug.

Run from personaplex root:

    PYTHONPATH=./moshi BMO_SO_PATH="$PWD/build_jetson/libbmo.so" \
        python probe_text_warmup.py --gguf "$PWD/bmo_septq_v3.gguf" --n-frames 40
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

_HERE = Path(__file__).resolve().parent
_MOSHI = _HERE / "moshi"
if _MOSHI.exists() and str(_MOSHI) not in sys.path:
    sys.path.insert(0, str(_MOSHI))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n-ctx", type=int, default=128)
    ap.add_argument("--n-frames", type=int, default=40,
                    help="how many successive forward_temporal calls to make")
    ap.add_argument("--tokenizer", default=None,
                    help="optional sentencepiece .model so we can decode argmax labels")
    args = ap.parse_args()

    from moshi.bmo_engine import BMOEngine
    e = BMOEngine(args.gguf, n_ctx=args.n_ctx)

    sp = None
    if args.tokenizer:
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(args.tokenizer)
        except ImportError:
            sp = None

    SPECIAL = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}

    def label(idx: int) -> str:
        if idx in SPECIAL:
            return SPECIAL[idx]
        if sp is not None:
            try:
                return sp.id_to_piece(idx).replace("\u2581", " ")
            except Exception:
                return f"<id:{idx}>"
        return f"<id:{idx}>"

    K = e.n_codebooks
    n_audio = K - 1
    n_moshi = min(8, n_audio)
    n_user = n_audio - n_moshi
    SILENCE = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int32)
    SINE    = np.array([430, 1268, 381, 1611, 1095, 1495,   56,  472], dtype=np.int32)

    def make_canonical_token() -> np.ndarray:
        tok = np.zeros(K, dtype=np.int32)
        tok[0] = 3                                  # PAD = zero_text_code
        tok[1:1 + n_moshi] = SILENCE[:n_moshi]
        if n_user > 0:
            tok[1 + n_moshi:1 + n_moshi + n_user] = SINE[:n_user]
        return tok

    def make_zeros_token() -> np.ndarray:
        return np.zeros(K, dtype=np.int32)

    print(f"[warmup] K={K} n_moshi={n_moshi} n_user={n_user} text_vocab={e.text_vocab}")
    print(f"[warmup] sweeping {args.n_frames} successive frames per scenario\n")

    # ------ Scenario A: constant canonical input every frame ------
    # If KV/RoPE work, |z| should still drift slightly as kv_len grows (history
    # is averaged in). If output is truly bit-identical across frames, RoPE
    # position is not advancing.
    print("--- scenario: canonical_prompt (CONSTANT input) ---")
    e.reset()
    seen_a: List[int] = []
    z_first = None
    for t in range(args.n_frames):
        tok = make_canonical_token()
        z, lt = e.forward_temporal(tok)
        argmax = int(np.argmax(lt))
        top5_idx = np.argsort(lt)[::-1][:5]
        top5_str = " ".join(f"{int(i)}({float(lt[i]):+.2f})" for i in top5_idx)
        seen_a.append(argmax)
        if z_first is None:
            z_first = z.copy()
        z_diff_from_t0 = float(np.linalg.norm(z - z_first))
        if t < 5 or t == args.n_frames - 1:
            print(f"[warmup] canonical t={t:3d} "
                  f"argmax={argmax:5d}({label(argmax)!s:>8s}) "
                  f"max={float(np.max(lt)):+.4f} "
                  f"|z|={float(np.linalg.norm(z)):.4f} "
                  f"|z - z@t=0|={z_diff_from_t0:.4f}")
    print(f"[warmup] canonical summary: unique argmax values={len(set(seen_a))}")

    # ------ Scenario B: VARYING input each frame ------
    # Different tokens at each step. If KV+RoPE work, output at step N depends
    # on the history of inputs at steps 0..N-1, not just the current input.
    # We therefore expect frame N's z to DIFFER from a "cold" forward of the
    # same input alone (after a fresh reset).
    print("\n--- scenario: varying input ---")
    e.reset()
    var_zs = []
    var_inputs = []
    for t in range(min(args.n_frames, 8)):
        tok = make_canonical_token()
        # vary text token across frames so each frame has different input
        tok[0] = (3 if t % 2 == 0 else 5 + t)        # PAD vs random word ids
        var_inputs.append(tok.copy())
        z, lt = e.forward_temporal(tok)
        var_zs.append(z.copy())
        argmax = int(np.argmax(lt))
        print(f"[warmup] varying t={t:3d} text_in={int(tok[0]):5d} "
              f"argmax={argmax:5d}({label(argmax)!s:>8s}) "
              f"|z|={float(np.linalg.norm(z)):.4f}")

    # Now run COLD forwards of the same inputs and compare
    print("\n--- scenario: COLD versions of same varying inputs ---")
    cold_zs = []
    for t, tok in enumerate(var_inputs):
        e.reset()
        z, lt = e.forward_temporal(tok)
        cold_zs.append(z.copy())
        argmax = int(np.argmax(lt))
        print(f"[warmup] cold    t={t:3d} text_in={int(tok[0]):5d} "
              f"argmax={argmax:5d}({label(argmax)!s:>8s}) "
              f"|z|={float(np.linalg.norm(z)):.4f}")

    # Compare hot vs cold. If KV+RoPE work, frame N hot z should differ from
    # cold z because hot has history attended.
    print("\n--- HOT vs COLD diffs (any value > 0 means KV history is felt) ---")
    for t in range(len(var_zs)):
        d = float(np.linalg.norm(var_zs[t] - cold_zs[t]))
        rel = d / max(float(np.linalg.norm(cold_zs[t])), 1e-9)
        verdict = "DIFFERENT" if rel > 1e-3 else "IDENTICAL  <-- KV/RoPE not active"
        print(f"[warmup] t={t}: |z_hot - z_cold|={d:.4f} rel={rel:.3e}  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
