#!/usr/bin/env python3
"""Deliverable 3: C++ v5 GGUF runtime vs FP16 safetensors reference (single harness step).

Compares one C++ temporal forward (pos=0) against PyTorch ``forward_codes``:
``bmo_forward_temporal2`` when capture is on, ``bmo_forward_temporal`` when ``--no-capture``.
with the same token vector. Writes ``path_b_day3_e2e_report.txt``.

Requires: built ``libbmo.so`` with ``bmo_forward_temporal2``, ``PYTHONPATH=./moshi``,
CUDA, checkpoint + GGUF paths on the server.

Use ``--milestone-residual-cosines`` with ``--septq-meta`` for full-tensor cos(C++,PT_FQ) vs PT baselines
(Jetson ``libbmo.so`` writes ``cpp_residual_L*.bin`` via ``BMO_H3_DUMP_RESIDUAL_BINS``; no ``bmo_api`` capture).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
_MOSHI = _REPO / "moshi"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if _MOSHI.is_dir() and str(_MOSHI) not in sys.path:
    sys.path.insert(0, str(_MOSHI))

from apply_septq import get_temporal_layers  # noqa: E402
from moshi.models.loaders import get_moshi_lm  # noqa: E402
from pt_fakequant_vs_fp16 import (  # noqa: E402
    H3_T2_LAYERS,
    append_h3_cpp_vs_pt_fq_per_op_table,
    format_h3_pt_report,
    format_pt_fq_h3_tap_t2_line,
    gather_qat_entries,
    load_harness_ids,
    qat_selection_from_ckpt,
    register_fake_quant_for_entries,
    resolve_quant_sources,
    run_forward_h3_pt_captures,
    run_forward_with_captures_all_residuals,
    write_milestone_residual_bin,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _capture_slot_stats(
    cap_buf: ctypes.Array, *, n_layers: int, n_embd: int, layer_index: int
) -> tuple[np.ndarray, float]:
    """One row of the C++ capture buffer: slot ``layer_index`` = ``out_layer_{L}`` when layers are 0..N-1 in order."""
    total = int(n_layers) * int(n_embd)
    flat = np.frombuffer(memoryview(cap_buf), dtype=np.float32, count=total)
    off = int(layer_index) * int(n_embd)
    row = flat[off : off + int(n_embd)].astype(np.float64, copy=False)
    l2 = float(np.linalg.norm(row))
    return row, l2


def _print_capture_user_buf_taps(
    cap_buf: ctypes.Array, *, n_layers: int, n_embd: int, max_layer: int = 1
) -> None:
    """Step-1 diagnostic: same bytes the e2e table uses; compare stderr to ``[h3_tap_T2] L=...``."""
    for L in range(min(max_layer + 1, int(n_layers))):
        row, l2 = _capture_slot_stats(cap_buf, n_layers=n_layers, n_embd=n_embd, layer_index=L)
        first8 = row[:8]
        parts = " ".join(f"{float(x):+.6e}" for x in first8)
        print(
            f"[capture_user_buf] L={L} first8={parts} l2={l2:.6f}",
            file=sys.stderr,
            flush=True,
        )


def _load_residual_bin_f32(path: Path, n: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    raw = np.fromfile(str(path), dtype=np.float32)
    if int(raw.size) != int(n):
        return None
    return raw.astype(np.float64, copy=False)


def _append_milestone_residual_cosine_table(
    lines: list[str],
    *,
    bin_root: Path,
    milestones: tuple[int, ...],
    n_embd: int,
    cap_fp16: dict[str, Any],
    cap_fq: dict[str, Any],
) -> None:
    """Full-vector cosines from ``cpp_residual_L{L}.bin`` (C++ T2 tap dump) vs PT captures."""
    lines.append("")
    lines.append("=== Milestone T2 FULL-TENSOR cosines (C++ device tap .bin vs PT) ===")
    lines.append(f"bin_dir: {bin_root}")
    lines.append(
        "gap = cos(PT_FP16, PT_FQ) - cos(C++, PT_FQ)  (C++ error beyond PT fakequant vs FP16 drift)"
    )
    lines.append("| L | cos(C++,PT_FQ) | cos(C++,PT_FP16) | cos(PT_FP16,PT_FQ) | gap |")
    lines.append("|---|----------------|------------------|--------------------|-----|")
    for L in milestones:
        key = f"layer{L}_residual_out"
        if key not in cap_fp16 or key not in cap_fq:
            lines.append(f"| {L} | MISSING_PT_CAP | MISSING_PT_CAP | MISSING_PT_CAP | nan |")
            continue
        pt_fp = cap_fp16[key].detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        pt_fq = cap_fq[key].detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
        c_fp_fq = _cosine(pt_fp, pt_fq)
        cpp = _load_residual_bin_f32(bin_root / f"cpp_residual_L{L}.bin", n_embd)
        if cpp is None:
            lines.append(
                f"| {L} | MISSING_CPP_BIN | nan | {c_fp_fq:.6f} | nan |  "
                f"(need Jetson libbmo + BMO_H3_DUMP_RESIDUAL_BINS; see script env)"
            )
            continue
        c_cq_fq = _cosine(cpp, pt_fq)
        c_cq_fp = _cosine(cpp, pt_fp)
        gap = c_fp_fq - c_cq_fq
        lines.append(f"| {L} | {c_cq_fq:.6f} | {c_cq_fp:.6f} | {c_fp_fq:.6f} | {gap:.6f} |")


def _kl_logits(p: np.ndarray, q: np.ndarray) -> float:
    """KL( softmax(p) || softmax(q) ), natural nats."""
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.size != q.size:
        return float("nan")
    lp = p - np.max(p)
    lq = q - np.max(q)
    sp = np.exp(lp)
    sq = np.exp(lq)
    sp /= np.sum(sp) + 1e-30
    sq /= np.sum(sq) + 1e-30
    sp = np.clip(sp, 1e-12, 1.0)
    sq = np.clip(sq, 1e-12, 1.0)
    return float(np.sum(sp * (np.log(sp) - np.log(sq))))


def _load_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.bmo_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.bmo_init.restype = ctypes.c_void_p
    lib.bmo_free.argtypes = [ctypes.c_void_p]
    lib.bmo_free.restype = None
    lib.bmo_reset.argtypes = [ctypes.c_void_p]
    lib.bmo_reset.restype = None
    for name in ("bmo_get_n_layers", "bmo_get_n_embd", "bmo_get_n_codebooks", "bmo_get_text_vocab"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
    lib.bmo_last_error.argtypes = [ctypes.c_void_p]
    lib.bmo_last_error.restype = ctypes.c_char_p

    lib.bmo_forward_temporal2 = getattr(lib, "bmo_forward_temporal2", None)
    if lib.bmo_forward_temporal2 is None:
        raise RuntimeError("libbmo.so missing bmo_forward_temporal2 (rebuild bmo_shared)")
    lib.bmo_forward_temporal2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bmo_forward_temporal2.restype = ctypes.c_int

    lib.bmo_forward_temporal = getattr(lib, "bmo_forward_temporal", None)
    if lib.bmo_forward_temporal is None:
        raise RuntimeError("libbmo.so missing bmo_forward_temporal (rebuild bmo_shared)")
    lib.bmo_forward_temporal.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.bmo_forward_temporal.restype = ctypes.c_int
    return lib


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True, help="bmo_septq_v5.gguf")
    ap.add_argument("--fp16", type=Path, required=True, help="v5_step1500_split.safetensors")
    ap.add_argument("--so-path", type=Path, default=None, help="libbmo.so (default: build/libbmo.so)")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument(
        "--harness-input",
        type=Path,
        default=None,
        help="JSON input_ids (default: harness_input.json or pt_dump_final/harness_input.json)",
    )
    ap.add_argument("--report", type=Path, default=here / "path_b_day3_e2e_report.txt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--no-capture",
        action="store_true",
        help="Call bmo_forward_temporal (no out_layer_{L} capture). "
        "Used to test whether the temporal2 capture memcpy corrupts state.",
    )
    ap.add_argument(
        "--septq-meta",
        type=Path,
        default=None,
        help="Optional SEPTQ multitier .pt for PT fakequant mirror (per-layer PT_FP16 vs PT_FQ vs C++ table).",
    )
    ap.add_argument(
        "--h3-dump-pt",
        action="store_true",
        help="Run hooked H3 forwards before heavy captures: PT FP16 + PT fakequant lines "
        "([PT_h3_tap_*] / [PT_FQ_h3_tap_*]), set cpp bin dump env, append per-op cos table vs PT_FQ. "
        "Requires --septq-meta.",
    )
    ap.add_argument(
        "--milestone-residual-cosines",
        action="store_true",
        help="With --septq-meta: write float32 pt_fp16/pt_fq/cpp_residual_L*.bin per milestone layer, "
        "print [PT_FQ_h3_tap_T2] lines, set BMO_H3_RESIDUAL_BIN_DIR and BMO_H3_DUMP_RESIDUAL_BINS for C++ "
        "(Jetson bmo_h3_tap T2 dump), append full-tensor cosine gap table to the report.",
    )
    args = ap.parse_args()
    if args.h3_dump_pt and args.septq_meta is None:
        raise SystemExit("--h3-dump-pt requires --septq-meta (PT FQ H3 + per-op table)")
    if args.milestone_residual_cosines and args.septq_meta is None:
        raise SystemExit("--milestone-residual-cosines requires --septq-meta")

    so = (args.so_path or (here / "build" / "libbmo.so")).resolve()
    if not so.is_file():
        raise SystemExit(f"missing libbmo.so: {so}")

    harness = args.harness_input
    if harness is None:
        for cand in (here / "harness_input.json", here / "pt_dump_final" / "harness_input.json"):
            if cand.is_file():
                harness = cand
                break
    if harness is None or not harness.is_file():
        raise SystemExit("Provide --harness-input (input_ids JSON)")

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    fp16_path = args.fp16.resolve()
    septq_path = args.septq_meta.resolve() if args.septq_meta is not None else None

    print(f"Loading FP16 reference: {fp16_path}")
    model_pt = get_moshi_lm(str(fp16_path), device=dev, dtype=dtype, copy_missing_weights=True)
    model_pt.eval()
    num_cb = int(model_pt.num_codebooks)
    ids = load_harness_ids(harness.resolve(), num_cb)
    seq = torch.tensor(ids, dtype=torch.long, device=dev).view(1, num_cb, 1)

    def _qat_payload_for_fp16_path(p: Path) -> dict[str, Any]:
        if str(p).lower().endswith(".safetensors"):
            return {}
        payload = torch.load(str(p), map_location="cpu")
        if not isinstance(payload, dict):
            raise SystemExit(f"Expected QAT checkpoint dict from {p}, got {type(payload)}")
        return payload

    cap_fq: dict[str, torch.Tensor] | None = None
    model_fq: torch.nn.Module | None = None
    h3_pt_fp16_lines: list[str] = []
    h3_pt_fq_lines: list[str] = []
    h3_cap_fq_for_cpp_table: dict[str, Any] | None = None
    milestone_bin_root: Path | None = None
    milestone_fq_tap_lines: list[str] = []
    milestone_n_embd: int | None = None
    with torch.no_grad():
        # H3 hooked forwards first (clean KV) so tap tensors match the harness step.
        if args.h3_dump_pt:
            h3_cap_fp16 = run_forward_h3_pt_captures(model_pt, seq)
            h3_pt_fp16_lines = format_h3_pt_report(h3_cap_fp16, family="PT_h3")
            for ln in h3_pt_fp16_lines:
                print(ln, flush=True)

        cap_fp16 = run_forward_with_captures_all_residuals(model_pt, seq)
        logits_pt = cap_fp16["final_logits"].detach().float().cpu().numpy().reshape(-1)

        if septq_path is not None:
            if not septq_path.is_file():
                raise SystemExit(f"--septq-meta not found: {septq_path}")
            qat_payload = _qat_payload_for_fp16_path(fp16_path)
            quant_ckpt, tier_masks_uint2, tier_masks_meta = resolve_quant_sources(
                qat_payload, fp16_path, septq_path
            )
            model_fq = get_moshi_lm(str(fp16_path), device=dev, dtype=dtype, copy_missing_weights=True)
            model_fq.eval()
            tl = get_temporal_layers(model_fq)
            if not tl:
                raise SystemExit("get_temporal_layers returned empty for fakequant model")
            n_temp = len(tl)
            selected_layers, skip_filters = qat_selection_from_ckpt(qat_payload, n_temp)
            entries, _excluded = gather_qat_entries(
                student=model_fq,
                selected_layers=selected_layers,
                skip_module_filters=skip_filters,
            )
            if not entries:
                raise SystemExit("gather_qat_entries returned no modules for fakequant model")
            register_fake_quant_for_entries(
                entries=entries,
                quant_checkpoint=quant_ckpt,
                tier_masks_uint2=tier_masks_uint2,
                tier_masks_meta=tier_masks_meta,
            )
            if args.h3_dump_pt:
                h3_cap_fq_for_cpp_table = run_forward_h3_pt_captures(model_fq, seq)
                h3_pt_fq_lines = format_h3_pt_report(h3_cap_fq_for_cpp_table, family="PT_FQ_h3")
                for ln in h3_pt_fq_lines:
                    print(ln, flush=True)

            cap_fq = run_forward_with_captures_all_residuals(model_fq, seq)
            if args.milestone_residual_cosines:
                report_path = args.report.resolve()
                milestone_bin_root = report_path.parent / "milestone_residual_bins"
                milestone_bin_root.mkdir(parents=True, exist_ok=True)
                milestone_n_embd = int(cap_fp16["layer0_residual_out"].numel())
                for L in H3_T2_LAYERS:
                    k = f"layer{L}_residual_out"
                    write_milestone_residual_bin(
                        milestone_bin_root / f"pt_fp16_residual_L{L}.bin",
                        cap_fp16[k],
                        expect_n=milestone_n_embd,
                    )
                    write_milestone_residual_bin(
                        milestone_bin_root / f"pt_fq_residual_L{L}.bin",
                        cap_fq[k],
                        expect_n=milestone_n_embd,
                    )
                    ln = format_pt_fq_h3_tap_t2_line(L, cap_fq[k])
                    print(ln, flush=True)
                    milestone_fq_tap_lines.append(ln)

    cpp_h3_dump_dir: Path | None = milestone_bin_root
    if args.h3_dump_pt:
        rp = args.report.resolve()
        alt = rp.parent / "cpp_h3_bins"
        alt.mkdir(parents=True, exist_ok=True)
        if cpp_h3_dump_dir is None:
            cpp_h3_dump_dir = alt
        os.environ["BMO_H3_RESIDUAL_BIN_DIR"] = str(cpp_h3_dump_dir.resolve())
        os.environ["BMO_H3_DUMP_RESIDUAL_BINS"] = "1"
        os.environ["BMO_H3_TAP_BIN_DIR"] = str(cpp_h3_dump_dir.resolve())
        os.environ["BMO_H3_DUMP_TAP_BINS"] = "1"
    elif milestone_bin_root is not None:
        os.environ["BMO_H3_RESIDUAL_BIN_DIR"] = str(milestone_bin_root.resolve())
        os.environ["BMO_H3_DUMP_RESIDUAL_BINS"] = "1"

    lib = _load_lib(so)
    h = lib.bmo_init(str(args.gguf.resolve()).encode("utf-8"), int(args.n_ctx))
    if not h:
        raise SystemExit("bmo_init returned NULL (see stderr)")

    n_embd = lib.bmo_get_n_embd(h)
    if milestone_n_embd is not None and int(n_embd) != int(milestone_n_embd):
        lib.bmo_free(h)
        raise SystemExit(
            f"n_embd mismatch: C++ bmo_get_n_embd={n_embd} vs PT milestone capture width={milestone_n_embd}"
        )
    n_layers_cpp = lib.bmo_get_n_layers(h)
    n_pt_layers = len(get_temporal_layers(model_pt))
    if n_pt_layers != int(n_layers_cpp):
        lib.bmo_free(h)
        raise SystemExit(f"layer count mismatch: PT temporal layers={n_pt_layers} vs bmo_get_n_layers={n_layers_cpp}")
    text_vocab = lib.bmo_get_text_vocab(h)
    tok = (ctypes.c_int32 * num_cb)(*map(int, ids))
    z_cpp = (ctypes.c_float * int(n_embd))()
    lg_cpp = (ctypes.c_float * int(text_vocab))()
    layer_list = list(range(int(n_layers_cpp)))
    layers = (ctypes.c_int32 * int(n_layers_cpp))(*layer_list)
    cap_buf = (ctypes.c_float * (int(n_layers_cpp) * int(n_embd)))()

    lib.bmo_reset(h)
    if args.no_capture:
        rc = lib.bmo_forward_temporal(
            h,
            tok,
            num_cb,
            0,
            z_cpp,
            lg_cpp,
        )
        if rc != 0:
            err = lib.bmo_last_error(h)
            msg = err.decode("utf-8", errors="replace") if err else f"rc={rc}"
            lib.bmo_free(h)
            raise SystemExit(f"bmo_forward_temporal failed: {msg}")
    else:
        rc = lib.bmo_forward_temporal2(
            h,
            tok,
            num_cb,
            0,
            z_cpp,
            lg_cpp,
            layers,
            int(n_layers_cpp),
            cap_buf,
        )
        if rc != 0:
            err = lib.bmo_last_error(h)
            msg = err.decode("utf-8", errors="replace") if err else f"rc={rc}"
            lib.bmo_free(h)
            raise SystemExit(f"bmo_forward_temporal2 failed: {msg}")
        _print_capture_user_buf_taps(cap_buf, n_layers=n_layers_cpp, n_embd=n_embd, max_layer=1)

    logits_cpp = np.frombuffer(
        (ctypes.c_float * int(text_vocab)).from_address(ctypes.addressof(lg_cpp)),
        dtype=np.float32,
        count=int(text_vocab),
    ).copy()

    cos_final = _cosine(logits_pt, logits_cpp)
    kl = _kl_logits(logits_pt, logits_cpp)

    logits_fq_np: np.ndarray | None = None
    cos_final_fq_cpp = float("nan")
    cos_final_fp16_fq = float("nan")
    if cap_fq is not None and "final_logits" in cap_fq:
        logits_fq_np = cap_fq["final_logits"].detach().float().cpu().numpy().reshape(-1)
        cos_final_fq_cpp = _cosine(logits_fq_np, logits_cpp)
        cos_final_fp16_fq = _cosine(logits_pt, logits_fq_np)

    lines: list[str] = []
    lines.append("path_b_day3_e2e (C++ v5 GGUF vs FP16 safetensors, single harness step pos=0)")
    lines.append(f"gguf: {args.gguf}")
    lines.append(f"fp16: {args.fp16}")
    if septq_path is not None:
        lines.append(f"septq_meta (PT fakequant): {septq_path}")
    lines.append(f"harness: {harness}")
    if args.no_capture:
        lines.append("n_capture_layers: 0 (capture disabled)")
    else:
        lines.append(f"n_capture_layers: {n_layers_cpp} (full stack)")
    lines.append(f"final_logits_cosine_PTfp16_vs_C++: {cos_final:.8f}")
    if cap_fq is not None:
        lines.append(f"final_logits_cosine_PTfq_vs_C++: {cos_final_fq_cpp:.8f}")
        lines.append(f"final_logits_cosine_PTfp16_vs_PTfq: {cos_final_fp16_fq:.8f}")
    lines.append(f"final_logits_kl_softmax_p_q: {kl:.8f}")
    if not args.no_capture:
        lines.append("")
        lines.append(
            "per_layer_residual (PT layer{L}_residual_out vs C++ out_layer_L); "
            "cos_fp16_cpp / cos_fq_cpp / cos_fp16_fq (last two only with --septq-meta):"
        )
        first_diverge: int | None = None
        for i, L in enumerate(layer_list):
            key = f"layer{L}_residual_out"
            off = i * int(n_embd)
            cap_np = np.frombuffer(
                memoryview(cap_buf), dtype=np.float32, count=int(n_layers_cpp) * int(n_embd)
            )
            cpp_v = cap_np[off : off + int(n_embd)].astype(np.float64, copy=False)
            if key not in cap_fp16:
                lines.append(f"  L{L:2d}: MISSING layer{L}_residual_out in PT capture")
                continue
            pt_fp = cap_fp16[key].detach().float().cpu().numpy().reshape(-1)
            c_fp_cpp = _cosine(pt_fp, cpp_v)
            if cap_fq is None or key not in cap_fq:
                lines.append(f"  L{L:2d}: cos_fp16_cpp={c_fp_cpp:.8f}")
                continue
            pt_fq = cap_fq[key].detach().float().cpu().numpy().reshape(-1)
            c_fq_cpp = _cosine(pt_fq, cpp_v)
            c_fp_fq = _cosine(pt_fp, pt_fq)
            lines.append(
                f"  L{L:2d}: cos_fp16_cpp={c_fp_cpp:.8f}  cos_fq_cpp={c_fq_cpp:.8f}  cos_fp16_fq={c_fp_fq:.8f}"
            )
            if first_diverge is None and (c_fp_fq - c_fq_cpp) > 0.05:
                first_diverge = L
        lines.append("")
        if cap_fq is not None:
            if first_diverge is not None:
                lines.append(
                    f"triage_B_first_layer_C++_vs_PTfq_gap_gt_0p05_vs_PTfp16_vs_PTfq: L={first_diverge} "
                    f"(cos_fp16_fq - cos_fq_cpp > 0.05)"
                )
            else:
                lines.append(
                    "triage_B_first_layer_C++_vs_PTfq_gap_gt_0p05_vs_PTfp16_vs_PTfq: none (no layer exceeded 0.05 gap)"
                )
    else:
        lines.append("")
        lines.append("(no-capture enabled: skipped per-layer residual table)")

    if milestone_bin_root is not None and milestone_n_embd is not None and cap_fq is not None:
        lines.append("")
        lines.append("=== [PT_FQ_h3_tap_T2] PT fakequant milestone residuals (same harness as C++ T2) ===")
        lines.extend(milestone_fq_tap_lines)
        _append_milestone_residual_cosine_table(
            lines,
            bin_root=milestone_bin_root,
            milestones=H3_T2_LAYERS,
            n_embd=milestone_n_embd,
            cap_fp16=cap_fp16,
            cap_fq=cap_fq,
        )
        lines.append("")
        lines.append(
            "Milestone verdict hints: max gap <= 0.05 all L → C++ tracks PT_FQ (runtime OK vs fakequant); "
            "gap grows from L=0 → extra C++ error from the start; gap widens mid-stack → localized runtime drift."
        )

    lines.append("")
    if cos_final >= 0.95:
        gate = "TARGET_MET (>=0.95)"
    elif cos_final >= 0.90:
        gate = "ACCEPTABLE (>=0.90 go/no-go listen)"
    elif cos_final >= 0.85:
        gate = "MARGINAL (<0.95 target)"
    else:
        gate = "STOP (<0.85) — root-cause before listen test"
    lines.append(f"gate: {gate}")
    if h3_pt_fp16_lines or h3_pt_fq_lines:
        lines.append("")
        lines.extend(h3_pt_fp16_lines)
        lines.extend(h3_pt_fq_lines)
        lines.append("")
        lines.append(
            "C++ h3 taps: BMO_H3_TAPS=1 (stderr). With --h3-dump-pt, C++ also writes cpp_h3_*.bin / "
            "cpp_residual_*.bin under the dump dir for the per-op table."
        )
    if (
        args.h3_dump_pt
        and h3_cap_fq_for_cpp_table is not None
        and cpp_h3_dump_dir is not None
    ):
        append_h3_cpp_vs_pt_fq_per_op_table(
            lines,
            bin_dir=cpp_h3_dump_dir,
            cap_fq=h3_cap_fq_for_cpp_table,
        )

    rep = args.report.resolve()
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {rep}")

    lib.bmo_free(h)

    if cos_final < 0.85:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
