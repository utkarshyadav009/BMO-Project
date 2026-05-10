#!/usr/bin/env python3
"""compare_divergence.py — Compare flat float32 dumps from PyTorch vs C++ numerical harness.

Expects pairs `pt_<name>.bin` and `cpp_<name>.bin` in the current working directory
for each stage name in `STAGES` (embed_sum, layer-0 attention internals, per-layer
`layer{L}_{x_in,post_attn,...}` for L in DEEP_LAYERS, post_out_norm, final_logits).
Missing files are reported without crashing.

Exit code: 0 if every present stage passes thresholds; 1 if any stage diverges or is missing on one side.

Residual / hidden stages: cosine >= 0.999 required (unless NaN).
final_logits: same cosine rule, plus divergence if argmax(pt) != argmax(cpp).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEEP_LAYERS = [0, 1, 8, 15, 23, 31]
SUB_STAGES = ["x_in", "post_attn", "post_attn_residual", "post_ffn", "residual_out"]

EXISTING_LAYER0_STAGES = [
    "layer0_post_norm1",
    "layer0_q_pre_rope",
    "layer0_q_post_rope",
    "layer0_k_pre_rope",
    "layer0_k_post_rope",
    "layer0_v",
    "layer0_post_attn",
    "layer0_post_norm2",
    "layer0_post_ffn",
    "layer0_residual_out",
]


def _ordered_unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


STAGES = _ordered_unique(
    ["embed_sum"]
    + EXISTING_LAYER0_STAGES
    + [f"layer{L}_{s}" for L in DEEP_LAYERS for s in SUB_STAGES]
    + ["post_out_norm", "final_logits"]
)

COS_THRESHOLD = 0.999
TEXT_PAD_ID = 3
FINAL_LOGITS_TOPK = 5


def stats(pt: np.ndarray, cpp: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """cosine, max_abs, rel_l2, mean_abs_pt, mean_abs_cpp, count"""
    pt = pt.astype(np.float64).ravel()
    cpp = cpp.astype(np.float64).ravel()
    n = pt.size
    n_pt = float(np.linalg.norm(pt))
    n_cpp = float(np.linalg.norm(cpp))
    cos = float(np.dot(pt, cpp) / (n_pt * n_cpp)) if n_pt > 0 and n_cpp > 0 else float("nan")
    diff = pt - cpp
    max_abs = float(np.max(np.abs(diff))) if n else 0.0
    denom = max(n_pt, 1e-30)
    rel_l2 = float(np.linalg.norm(diff) / denom)
    mean_abs_pt = float(np.mean(np.abs(pt))) if n else 0.0
    mean_abs_cpp = float(np.mean(np.abs(cpp))) if n else 0.0
    return cos, max_abs, rel_l2, mean_abs_pt, mean_abs_cpp, float(n)


def logit_argmax(x: np.ndarray) -> int:
    return int(np.argmax(x.ravel()))


def logit_rank_pad(x: np.ndarray, pad_id: int = TEXT_PAD_ID) -> int:
    scores = x.ravel().astype(np.float64)
    order = np.argsort(-scores)
    idx = np.where(order == pad_id)[0]
    return int(idx[0])


def logit_topk_indices(x: np.ndarray, k: int) -> list[int]:
    scores = x.ravel().astype(np.float64)
    k = min(k, scores.size)
    order = np.argsort(-scores)
    return [int(i) for i in order[:k]]


def main() -> int:
    diverged = False
    print(
        f"{'stage':<26} {'n':>10} {'cosine':>10} {'max|diff|':>12} {'rel_L2':>10} {'mean|pt|':>10} {'mean|cpp|':>10}  notes"
    )
    print("-" * 120)

    for name in STAGES:
        p_pt = Path(f"pt_{name}.bin")
        p_cpp = Path(f"cpp_{name}.bin")
        miss = []
        if not p_pt.is_file():
            miss.append("pt")
        if not p_cpp.is_file():
            miss.append("cpp")
        if miss:
            note = "MISSING (" + "|".join(miss) + ")"
            print(f"{name:<26} {'-':>10} {'-':>10} {'-':>12} {'-':>10} {'-':>10} {'-':>10}  {note}")
            diverged = True
            continue

        pt = np.fromfile(p_pt, dtype=np.float32)
        cpp = np.fromfile(p_cpp, dtype=np.float32)
        if pt.size != cpp.size:
            note = f"<-- DIVERGENCE (count pt={pt.size} cpp={cpp.size})"
            print(f"{name:<26} {pt.size:>10} {'-':>10} {'-':>12} {'-':>10} {'-':>10} {'-':>10}  {note}")
            diverged = True
            continue

        cos, max_abs, rel_l2, m_pt, m_cpp, n = stats(pt, cpp)
        flag = ""
        if name == "final_logits":
            arg_pt = logit_argmax(pt)
            arg_cpp = logit_argmax(cpp)
            r_pt = logit_rank_pad(pt)
            r_cpp = logit_rank_pad(cpp)
            t5_pt = logit_topk_indices(pt, FINAL_LOGITS_TOPK)
            t5_cpp = logit_topk_indices(cpp, FINAL_LOGITS_TOPK)
            note_extra = (
                f"PAD_rank pt={r_pt} cpp={r_cpp}  argmax pt={arg_pt} cpp={arg_cpp}  "
                f"top{FINAL_LOGITS_TOPK}_pt={t5_pt}  top{FINAL_LOGITS_TOPK}_cpp={t5_cpp}"
            )
            if cos < COS_THRESHOLD or np.isnan(cos) or arg_pt != arg_cpp:
                flag = "<-- DIVERGENCE"
                diverged = True
            print(
                f"{name:<26} {int(n):>10} {cos:>10.6f} {max_abs:>12.6g} {rel_l2:>10.6g} "
                f"{m_pt:>10.6g} {m_cpp:>10.6g}  {flag} {note_extra}"
            )
            continue

        if cos < COS_THRESHOLD or np.isnan(cos):
            flag = "<-- DIVERGENCE"
            diverged = True
        print(
            f"{name:<26} {int(n):>10} {cos:>10.6f} {max_abs:>12.6g} {rel_l2:>10.6g} "
            f"{m_pt:>10.6g} {m_cpp:>10.6g}  {flag}"
        )

    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
