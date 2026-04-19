import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from moshi.models import loaders
from verify_int4_rollout_drift import build_forced_tokens, parse_bool, parse_dtype, register_post_bridge_hook


def run_zs_drift(
    teacher_ckpt: str,
    student_ckpt: str,
    *,
    steps: int,
    seed: int,
    device: str,
    input_wav: str,
    voice_prompt_wav: str,
    text_prompt: str,
    mimi_weight: str,
    tokenizer_path: str,
    voice_ratio: float,
    teacher_dtype: str,
    student_dtype: str,
) -> List[Dict[str, float | int]]:
    teacher_dt = parse_dtype(teacher_dtype)
    student_dt = None if str(student_dtype).strip().lower() == "auto" else parse_dtype(student_dtype)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    print(f"[RUN] Loading teacher model: {teacher_ckpt}")
    teacher = loaders.get_moshi_lm(
        teacher_ckpt,
        device=device,
        dtype=teacher_dt,
        cpu_offload=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[RUN] Loading student model: {student_ckpt}")
    student = loaders.get_moshi_lm(
        student_ckpt,
        device=device,
        dtype=student_dt,
        cpu_offload=False,
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    print(
        "[INFO] Eval dtypes: "
        f"teacher={str(teacher_dt).replace('torch.', '')} "
        f"student={str(student_dt).replace('torch.', '') if student_dt is not None else 'auto'}"
    )

    if int(teacher.num_codebooks) != int(student.num_codebooks):
        raise RuntimeError(
            f"num_codebooks mismatch: teacher={teacher.num_codebooks} student={student.num_codebooks}"
        )

    forced_tokens = build_forced_tokens(
        teacher,
        int(steps),
        device,
        input_wav=input_wav,
        voice_prompt_wav=voice_prompt_wav,
        text_prompt=text_prompt,
        mimi_weight=mimi_weight,
        tokenizer_path=tokenizer_path,
        voice_ratio=float(voice_ratio),
    )

    k = int(teacher.num_codebooks)
    t_bridge_cache, t_bridge_handle = register_post_bridge_hook(teacher)
    s_bridge_cache, s_bridge_handle = register_post_bridge_hook(student)

    per_step: List[Dict[str, float | int]] = []

    try:
        with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
            for t in range(int(steps)):
                token_codes = forced_tokens[t]
                teacher_seq = token_codes.view(1, k, 1).to(device)
                student_seq = token_codes.view(1, k, 1).to(device)

                teacher.forward_codes(teacher_seq)
                student.forward_codes(student_seq)

                if not t_bridge_cache or not s_bridge_cache:
                    raise RuntimeError("Post-bridge caches are empty; out_norm hooks did not capture activations")

                teacher_post = t_bridge_cache.pop().cpu().contiguous()
                student_post = s_bridge_cache.pop().cpu().contiguous()

                rows = min(int(teacher_post.shape[0]), int(student_post.shape[0]))
                dims = min(int(teacher_post.shape[-1]), int(student_post.shape[-1]))
                if rows <= 0 or dims <= 0:
                    raise RuntimeError("Invalid z_s activation shape captured by hooks")

                teacher_vec = teacher_post[:rows, :dims].reshape(-1)
                student_vec = student_post[:rows, :dims].reshape(-1)
                diff = teacher_vec - student_vec

                cos = float(F.cosine_similarity(teacher_vec, student_vec, dim=0).item())
                mse = float(F.mse_loss(teacher_vec, student_vec).item())
                max_abs = float(torch.max(torch.abs(diff)).item())

                per_step.append(
                    {
                        "step": int(t),
                        "cos": cos,
                        "mse": mse,
                        "max_abs": max_abs,
                    }
                )
    finally:
        if t_bridge_handle is not None:
            t_bridge_handle.remove()
        if s_bridge_handle is not None:
            s_bridge_handle.remove()

        del teacher
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return per_step


def summarize_steps(rows: List[Dict[str, float | int]]) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("No z_s drift rows captured")

    cos = torch.tensor([float(r["cos"]) for r in rows], dtype=torch.float32)
    mse = torch.tensor([float(r["mse"]) for r in rows], dtype=torch.float32)
    max_abs = torch.tensor([float(r["max_abs"]) for r in rows], dtype=torch.float32)

    q = torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32)
    cos_q = torch.quantile(cos, q)
    mse_q = torch.quantile(mse, q)
    max_abs_q = torch.quantile(max_abs, q)

    worst_cos_idx = int(torch.argmin(cos).item())
    worst_mse_idx = int(torch.argmax(mse).item())

    return {
        "steps": int(len(rows)),
        "cos_min": float(cos.min().item()),
        "cos_p10": float(cos_q[0].item()),
        "cos_median": float(cos_q[1].item()),
        "cos_p90": float(cos_q[2].item()),
        "cos_mean": float(cos.mean().item()),
        "mse_min": float(mse.min().item()),
        "mse_p10": float(mse_q[0].item()),
        "mse_median": float(mse_q[1].item()),
        "mse_p90": float(mse_q[2].item()),
        "mse_mean": float(mse.mean().item()),
        "max_abs_min": float(max_abs.min().item()),
        "max_abs_p10": float(max_abs_q[0].item()),
        "max_abs_median": float(max_abs_q[1].item()),
        "max_abs_p90": float(max_abs_q[2].item()),
        "max_abs_mean": float(max_abs.mean().item()),
        "worst_cos_step": int(rows[worst_cos_idx]["step"]),
        "worst_cos_value": float(rows[worst_cos_idx]["cos"]),
        "worst_mse_step": int(rows[worst_mse_idx]["step"]),
        "worst_mse_value": float(rows[worst_mse_idx]["mse"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare z_s drift between teacher and SEPTQ checkpoint under teacher-forced tokens."
    )
    parser.add_argument("--teacher", default="v5_step1500_split.safetensors")
    parser.add_argument("--student", default="bmo_temporal_septq_3bit_diag.pt")
    parser.add_argument("--steps", type=int, default=125)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--voice-prompt-wav", default="bmo_621.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--voice-ratio", type=float, default=0.25)
    parser.add_argument(
        "--teacher-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Loader dtype for teacher model.",
    )
    parser.add_argument(
        "--student-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Loader dtype for student model.",
    )
    parser.add_argument(
        "--runtime-patch",
        type=parse_bool,
        default=False,
        help=(
            "If true, import test_rtx_edge to apply runtime monkey patches. "
            "Defaults to false for native-path drift verification."
        ),
    )
    parser.add_argument(
        "--min-median-cos",
        type=float,
        default=0.997,
        help="Pass threshold for median z_s cosine.",
    )
    parser.add_argument(
        "--min-step-cos",
        type=float,
        default=0.99,
        help="Pass threshold for minimum z_s cosine.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to write detailed per-step and summary JSON.",
    )
    parser.add_argument(
        "--report-worst-k",
        type=int,
        default=10,
        help="How many lowest-cosine steps to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.steps) <= 0:
        raise ValueError(f"steps must be > 0, got {args.steps}")
    if float(args.voice_ratio) < 0.0 or float(args.voice_ratio) >= 1.0:
        raise ValueError(f"voice-ratio must be in [0, 1), got {args.voice_ratio}")

    if bool(args.runtime_patch):
        # Importing this module applies runtime monkey patches used in some test pipelines.
        import test_rtx_edge  # noqa: F401
    else:
        print("[INFO] Runtime patch disabled: using native loader/attention path.")

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required when --device is cuda")

    rows = run_zs_drift(
        teacher_ckpt=str(args.teacher),
        student_ckpt=str(args.student),
        steps=int(args.steps),
        seed=int(args.seed),
        device=str(args.device),
        input_wav=str(args.input_wav),
        voice_prompt_wav=str(args.voice_prompt_wav),
        text_prompt=str(args.text_prompt),
        mimi_weight=str(args.mimi_weight),
        tokenizer_path=str(args.tokenizer),
        voice_ratio=float(args.voice_ratio),
        teacher_dtype=str(args.teacher_dtype),
        student_dtype=str(args.student_dtype),
    )

    summary = summarize_steps(rows)

    print("\n=== Z_S DRIFT SUMMARY ===")
    print(f"[RESULT] steps = {summary['steps']}")
    print(
        f"[RESULT] cos: min={summary['cos_min']:.6f} p10={summary['cos_p10']:.6f} "
        f"median={summary['cos_median']:.6f} p90={summary['cos_p90']:.6f} "
        f"mean={summary['cos_mean']:.6f}"
    )
    print(
        f"[RESULT] mse: min={summary['mse_min']:.6e} p10={summary['mse_p10']:.6e} "
        f"median={summary['mse_median']:.6e} p90={summary['mse_p90']:.6e} "
        f"mean={summary['mse_mean']:.6e}"
    )
    print(
        f"[RESULT] max_abs: min={summary['max_abs_min']:.6e} p10={summary['max_abs_p10']:.6e} "
        f"median={summary['max_abs_median']:.6e} p90={summary['max_abs_p90']:.6e} "
        f"mean={summary['max_abs_mean']:.6e}"
    )
    print(
        f"[RESULT] worst_cos_step = {summary['worst_cos_step']} "
        f"worst_cos = {summary['worst_cos_value']:.6f}"
    )
    print(
        f"[RESULT] worst_mse_step = {summary['worst_mse_step']} "
        f"worst_mse = {summary['worst_mse_value']:.6e}"
    )

    threshold_ok = (
        float(summary["cos_median"]) >= float(args.min_median_cos)
        and float(summary["cos_min"]) >= float(args.min_step_cos)
    )
    print(
        f"[RESULT] pass_threshold = {threshold_ok} "
        f"(median>={float(args.min_median_cos):.6f} and min>={float(args.min_step_cos):.6f})"
    )

    worst_k = max(1, int(args.report_worst_k))
    print(f"\n=== WORST {worst_k} STEPS BY COSINE ===")
    for row in sorted(rows, key=lambda r: float(r["cos"]))[:worst_k]:
        print(
            f"step={int(row['step']):03d} cos={float(row['cos']):.6f} "
            f"mse={float(row['mse']):.6e} max_abs={float(row['max_abs']):.6e}"
        )

    if str(args.save_json).strip():
        out_path = Path(str(args.save_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "thresholds": {
                "min_median_cos": float(args.min_median_cos),
                "min_step_cos": float(args.min_step_cos),
            },
            "pass_threshold": bool(threshold_ok),
            "per_step": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[RESULT] wrote_json = {out_path}")


if __name__ == "__main__":
    main()
