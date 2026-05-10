#!/usr/bin/env python3
"""Check whether the prefilled text channel ever feeds dead-row token ids.

Background:
  `probe_gguf_text_emb_rows.py` showed that 110 / 32001 rows of
  `text_emb.weight` are zero in the GGUF (and ~zero in the .pt source).
  Their indices are dominated by a contiguous block [36..130] -- exactly
  the SentencePiece byte-fallback range for printable ASCII bytes
  '$' (0x24) through '~' (0x82-ish).

  If `wrap_with_system_tags()` + sp.EncodeAsIds() falls into byte-
  fallback for any character in `<system>` or the prompt itself, every
  one of those tokens produces a *zero* text embedding at runtime --
  i.e. the model literally cannot see the prompt.

This script:
  * Loads the GGUF and computes the dead-row set offline.
  * Loads the SentencePiece tokenizer.
  * Tokenizes the wrapped system prompt (and any extra strings you
    pass via --text).
  * Reports per-token whether each id is dead or live, and a percent
    summary.

Usage:
  PYTHONPATH=./llama.cpp/gguf-py python probe_text_prompt_in_dead_rows.py \\
    --gguf $PWD/bmo_septq_v3.gguf \\
    --tokenizer $PWD/tokenizer-e351c8d8-checkpoint125.safetensors  \\
    --text "You are a helpful assistant."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import numpy as np


def _setup_gguf_path() -> None:
    root = Path(__file__).resolve().parent
    cand = root / "llama.cpp" / "gguf-py"
    if cand.is_dir():
        sys.path.insert(0, str(cand))


def _load_dead_rows(gguf_path: str) -> Set[int]:
    _setup_gguf_path()
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == "text_emb.weight":
            arr = np.asarray(t.data).astype(np.float32)
            if arr.shape[0] not in (32000, 32001, 32768):
                arr = arr.T
            sq = np.einsum("ij,ij->i", arr, arr)
            return set(np.where(sq < 1e-12)[0].tolist())
    raise SystemExit("text_emb.weight not found in GGUF")


def _wrap_with_system_tags(text: str) -> str:
    """Match bmo_inference.wrap_with_system_tags()."""
    cleaned = (text or "").strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="Path to sentencepiece .model used by bmo_inference")
    ap.add_argument(
        "--text",
        nargs="*",
        default=[
            "You are a helpful assistant.",
            "Hello world.",
            "Hi.",
            "system prompt",
        ],
        help="Strings to wrap and tokenize (defaults if not passed)",
    )
    args = ap.parse_args(argv)

    print(f"[probe] gguf={args.gguf}")
    dead = _load_dead_rows(args.gguf)
    print(f"[probe] dead-row count: {len(dead)} / 32001")

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise SystemExit(f"sentencepiece not installed: {e}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    print(f"[probe] tokenizer vocab size: {sp.GetPieceSize()}")

    # Special token sanity: check whether the BMO special ids 0/1/2/3 are dead.
    print("\n[probe] special-id dead-status:")
    for tid in (0, 1, 2, 3):
        print(f"   id={tid:5d}  dead={'YES' if tid in dead else 'no':3s}  "
              f"piece={sp.IdToPiece(tid)!r}")

    # Walk each --text string through wrap_with_system_tags + EncodeAsIds.
    for text in args.text:
        wrapped = _wrap_with_system_tags(text)
        ids: List[int] = sp.EncodeAsIds(wrapped)
        n_dead = sum(1 for tid in ids if tid in dead)
        n_total = len(ids)
        pct = 100.0 * n_dead / n_total if n_total else 0.0

        print(f"\n=== text={text!r} ===")
        print(f"   wrapped: {wrapped!r}")
        print(f"   n_tokens={n_total}  dead={n_dead}  ({pct:.1f}% dead)")
        print("   per-token:")
        for tid in ids:
            piece = sp.IdToPiece(tid)
            tag = "[DEAD]" if tid in dead else "      "
            print(f"      id={tid:6d} {tag} piece={piece!r}")

    # Also check what happens if we tokenize WITHOUT the <system> wrap.
    if args.text:
        bare = args.text[0]
        ids = sp.EncodeAsIds(bare)
        n_dead = sum(1 for tid in ids if tid in dead)
        n_total = len(ids)
        pct = 100.0 * n_dead / n_total if n_total else 0.0
        print(f"\n[probe] sanity: bare {bare!r} -> n_tokens={n_total} "
              f"dead={n_dead} ({pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
