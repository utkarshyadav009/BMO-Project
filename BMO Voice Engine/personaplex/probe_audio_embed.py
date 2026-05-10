"""Bisect probe: does the C++ engine actually USE the audio-channel inputs?

If the temporal output `z` is the same when we feed all-zeros vs SILENCE_TOKENS
vs SINE_TOKENS in the audio channels (with everything else identical), then
the audio embedding lookup in bmo_compute.cpp::bmo_embed_input_tokens is
silently dropping non-zero audio tokens. That would explain why our
voice-prompt fix changed nothing -- the prefill content never reaches the
KV cache at all.

Run it from the personaplex root after scp'ing this file:

    PYTHONPATH=./moshi BMO_SO_PATH="$PWD/build_jetson/libbmo.so" \
        python probe_audio_embed.py --gguf "$PWD/bmo_septq_v3.gguf"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_MOSHI = _HERE / "moshi"
if _MOSHI.exists() and str(_MOSHI) not in sys.path:
    sys.path.insert(0, str(_MOSHI))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--n-ctx", type=int, default=128)
    args = ap.parse_args()

    so_path = os.environ.get("BMO_SO_PATH", "./build_jetson/libbmo.so")
    if not Path(so_path).exists():
        print(f"[probe] BMO_SO_PATH={so_path!r} missing; pass --so-path or "
              f"export BMO_SO_PATH.", file=sys.stderr)
        return 2

    from moshi.bmo_engine import BMOEngine
    e = BMOEngine(args.gguf, n_ctx=args.n_ctx)
    K = e.n_codebooks
    print(f"[probe] K={K} dep_q={e.dep_q} d_embd={e.n_embd} text_vocab={e.text_vocab}")

    SILENCE = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int32)
    SINE    = np.array([430, 1268, 381, 1611, 1095, 1495,   56,  472], dtype=np.int32)

    def make_tokens(text_id: int, moshi_arr=None, user_arr=None) -> np.ndarray:
        tok = np.zeros(K, dtype=np.int32)
        tok[0] = int(text_id)
        n_audio = K - 1
        n_moshi = min(8, n_audio)
        n_user = n_audio - n_moshi
        if moshi_arr is not None and n_moshi > 0:
            n = min(moshi_arr.size, n_moshi)
            tok[1:1 + n] = moshi_arr[:n]
        if user_arr is not None and n_user > 0:
            n = min(user_arr.size, n_user)
            tok[1 + n_moshi:1 + n_moshi + n] = user_arr[:n]
        return tok

    # Three contrasting single-step inputs. ALL use text=3 so any z difference
    # must come from the audio channels.
    cases = [
        ("zeros",   make_tokens(3, np.zeros(8, np.int32), np.zeros(8, np.int32))),
        ("silence", make_tokens(3, SILENCE,               np.zeros(8, np.int32))),
        ("sine",    make_tokens(3, np.zeros(8, np.int32), SINE)),
        ("both",    make_tokens(3, SILENCE,               SINE)),
        ("scramble", make_tokens(3,
                                np.array([1, 100, 500, 1000, 1500, 2000, 100, 1], np.int32),
                                np.array([42, 7, 999, 13, 17, 19, 23, 29],          np.int32))),
    ]

    results = []
    for name, tok in cases:
        e.reset()
        z, lt = e.forward_temporal(tok)
        results.append((name, tok.copy(), z.copy(), lt.copy()))
        print(f"[probe] case={name:8s} "
              f"z[:4]={z[:4]} "
              f"|z|={np.linalg.norm(z):.3f} "
              f"text_argmax={int(np.argmax(lt)):5d} "
              f"text_max={float(np.max(lt)):+.3f} "
              f"text_finite={bool(np.all(np.isfinite(lt)))}")

    print()
    print("[probe] pairwise z-diff norms (any pair >> 0 means audio embeddings ARE wired):")
    base = results[0][2]
    base_name = results[0][0]
    for name, _, z, _ in results[1:]:
        diff = float(np.linalg.norm(z - base))
        rel = diff / max(float(np.linalg.norm(base)), 1e-9)
        verdict = "DIFFERENT" if rel > 1e-3 else "IDENTICAL  <-- BROKEN"
        print(f"[probe]   {base_name} vs {name:8s}: |dz|={diff:.3f} rel={rel:.3e}  {verdict}")

    # If ALL cases are identical, audio embeddings are dead.
    z_all_same = all(
        float(np.linalg.norm(results[i][2] - base)) / max(float(np.linalg.norm(base)), 1e-9) <= 1e-3
        for i in range(1, len(results))
    )
    if z_all_same:
        print()
        print("[probe] CONCLUSION: audio-channel inputs do NOT affect z.")
        print("[probe]   bmo_embed_input_tokens / depformer_in path is silently zeroing")
        print("[probe]   the audio embeddings. This is why the voice-prompt fix changed")
        print("[probe]   nothing -- the prefill is effectively all-text, and identical to")
        print("[probe]   feeding zeros.")
        return 1

    print()
    print("[probe] CONCLUSION: audio embeddings ARE wired. The gibberish must come")
    print("[probe]   from elsewhere -- delays, depth, sampling, or n_past. Next step:")
    print("[probe]   bisect the prompt phases by running stream WITHOUT --voice-prompt")
    print("[probe]   and WITHOUT --text-prompt to see if minimal prefill still gibbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
