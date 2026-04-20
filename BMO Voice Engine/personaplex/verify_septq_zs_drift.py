import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from moshi.models import loaders
from verify_int4_rollout_drift import (
    build_forced_tokens,
    get_temporal_layers,
    parse_bool,
    parse_dtype,
    register_post_bridge_hook,
)


def _unwrap_tensor_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


def register_temporal_layer_hooks(model):
    layers = get_temporal_layers(model)
    caches: List[List[torch.Tensor]] = [[] for _ in layers]
    handles = []

    for layer_idx, layer in enumerate(layers):
        def _hook(_module, _inputs, output, idx=layer_idx):
            y = _unwrap_tensor_output(output)
            if not torch.is_tensor(y):
                return
            if y.ndim == 1:
                y = y.view(1, -1)
            caches[idx].append(y.detach().float().reshape(-1, y.shape[-1]).cpu().contiguous())

        handles.append(layer.register_forward_hook(_hook))

    return caches, handles, int(len(layers))


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
) -> Dict[str, Any]:
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
    t_layer_cache, t_layer_handles, t_layer_count = register_temporal_layer_hooks(teacher)
    s_layer_cache, s_layer_handles, s_layer_count = register_temporal_layer_hooks(student)

    layer_count = min(int(t_layer_count), int(s_layer_count))
    if layer_count <= 0:
        raise RuntimeError("Temporal layer hooks are unavailable; could not capture per-layer z_s drift")
    if int(t_layer_count) != int(s_layer_count):
        print(
            "[WARN] Temporal layer count mismatch: "
            f"teacher={t_layer_count} student={s_layer_count}; using first {layer_count}"
        )
    print(
        "[INFO] Temporal layer hooks enabled: "
        f"teacher={t_layer_count} student={s_layer_count} using={layer_count}"
    )

    per_step: List[Dict[str, float | int]] = []
    per_step_layer: List[Dict[str, float | int]] = []

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

                for layer_idx in range(int(layer_count)):
                    if not t_layer_cache[layer_idx] or not s_layer_cache[layer_idx]:
                        raise RuntimeError(
                            "Layer hook caches are empty at step "
                            f"{t}, layer {layer_idx}; temporal forward outputs were not captured"
                        )

                    teacher_layer = t_layer_cache[layer_idx].pop().contiguous()
                    student_layer = s_layer_cache[layer_idx].pop().contiguous()

                    rows = min(int(teacher_layer.shape[0]), int(student_layer.shape[0]))
                    dims = min(int(teacher_layer.shape[-1]), int(student_layer.shape[-1]))
                    if rows <= 0 or dims <= 0:
                        raise RuntimeError(
                            f"Invalid temporal layer activation shape at step {t}, layer {layer_idx}"
                        )

                    teacher_vec = teacher_layer[:rows, :dims].reshape(-1)
                    student_vec = student_layer[:rows, :dims].reshape(-1)
                    layer_diff = teacher_vec - student_vec

                    per_step_layer.append(
                        {
                            "step": int(t),
                            "layer": int(layer_idx),
                            "cos": float(F.cosine_similarity(teacher_vec, student_vec, dim=0).item()),
                            "mse": float(F.mse_loss(teacher_vec, student_vec).item()),
                            "max_abs": float(torch.max(torch.abs(layer_diff)).item()),
                        }
                    )
    finally:
        if t_bridge_handle is not None:
            t_bridge_handle.remove()
        if s_bridge_handle is not None:
            s_bridge_handle.remove()
        for handle in t_layer_handles:
            handle.remove()
        for handle in s_layer_handles:
            handle.remove()

        del teacher
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "per_step": per_step,
        "per_step_layer": per_step_layer,
        "layer_count": int(layer_count),
    }


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


def summarize_layer_steps(rows: List[Dict[str, float | int]], cliff_threshold: float) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("No temporal per-layer z_s drift rows captured")

    layers = sorted({int(r["layer"]) for r in rows})
    per_layer: List[Dict[str, float | int]] = []

    for layer_idx in layers:
        layer_rows = [r for r in rows if int(r["layer"]) == int(layer_idx)]
        cos = torch.tensor([float(r["cos"]) for r in layer_rows], dtype=torch.float32)
        mse = torch.tensor([float(r["mse"]) for r in layer_rows], dtype=torch.float32)
        max_abs = torch.tensor([float(r["max_abs"]) for r in layer_rows], dtype=torch.float32)

        q = torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32)
        cos_q = torch.quantile(cos, q)
        mse_q = torch.quantile(mse, q)
        max_abs_q = torch.quantile(max_abs, q)

        per_layer.append(
            {
                "layer": int(layer_idx),
                "steps": int(len(layer_rows)),
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
            }
        )

    if not per_layer:
        raise RuntimeError("No temporal per-layer summaries were produced")

    layer0_med = float(per_layer[0]["cos_median"])
    cumulative_positive_drop = 0.0
    for idx, row in enumerate(per_layer):
        if idx == 0:
            row["drop_from_prev_median"] = 0.0
            row["cumulative_positive_drop_median"] = 0.0
            row["cumulative_drop_from_layer0_median"] = 0.0
            continue
        prev_med = float(per_layer[idx - 1]["cos_median"])
        cur_med = float(row["cos_median"])
        drop_prev = prev_med - cur_med
        cumulative_positive_drop += max(0.0, drop_prev)
        row["drop_from_prev_median"] = float(drop_prev)
        row["cumulative_positive_drop_median"] = float(cumulative_positive_drop)
        row["cumulative_drop_from_layer0_median"] = float(layer0_med - cur_med)

    first_below = -1
    for row in per_layer:
        if float(row["cos_median"]) < float(cliff_threshold):
            first_below = int(row["layer"])
            break

    drift_mode_hint = "none"
    if first_below >= 0:
        first_idx = int(first_below)
        if first_idx <= 1:
            drift_mode_hint = "early_cliff"
        else:
            first_drop = float(per_layer[first_idx]["drop_from_prev_median"])
            tail_pos_drops = [
                max(0.0, float(per_layer[i]["drop_from_prev_median"]))
                for i in range(first_idx + 1, len(per_layer))
            ]
            tail_mean = float(sum(tail_pos_drops) / len(tail_pos_drops)) if tail_pos_drops else 0.0
            if first_drop >= max(0.0025, 3.0 * tail_mean):
                drift_mode_hint = "sharp_cliff"
            else:
                drift_mode_hint = "smooth_decay"

    return {
        "layers": int(len(per_layer)),
        "cliff_threshold": float(cliff_threshold),
        "first_layer_below_threshold": int(first_below),
        "drift_mode_hint": str(drift_mode_hint),
        "per_layer": per_layer,
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
    parser.add_argument(
        "--layer-cliff-threshold",
        type=float,
        default=0.995,
        help="Layer median cosine threshold used to detect first cliff layer.",
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

    run_payload = run_zs_drift(
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
    rows = list(run_payload["per_step"])
    layer_rows = list(run_payload["per_step_layer"])

    summary = summarize_steps(rows)
    layer_summary = summarize_layer_steps(layer_rows, cliff_threshold=float(args.layer_cliff_threshold))

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

    print("\n=== PER-LAYER Z_S DRIFT (TEMPORAL) ===")
    print(f"[RESULT] temporal_layers = {int(layer_summary['layers'])}")
    print(f"[RESULT] layer_cliff_threshold = {float(layer_summary['cliff_threshold']):.6f}")
    print(f"[RESULT] first_layer_below_threshold = {int(layer_summary['first_layer_below_threshold'])}")
    print(f"[RESULT] drift_mode_hint = {str(layer_summary['drift_mode_hint'])}")

    first_layer_idx = int(layer_summary["first_layer_below_threshold"])
    if first_layer_idx >= 0:
        first_layer_row = next(
            (r for r in layer_summary["per_layer"] if int(r["layer"]) == first_layer_idx),
            None,
        )
        if first_layer_row is not None:
            print(
                f"[RESULT] first_layer_cos_median = {float(first_layer_row['cos_median']):.6f} "
                f"first_layer_cos_min = {float(first_layer_row['cos_min']):.6f}"
            )

    for row in layer_summary["per_layer"]:
        print(
            f"layer={int(row['layer']):02d} "
            f"cos_median={float(row['cos_median']):.6f} "
            f"cos_min={float(row['cos_min']):.6f} "
            f"drop_prev={float(row['drop_from_prev_median']):+.6f} "
            f"cum_drop={float(row['cumulative_positive_drop_median']):.6f}"
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
            "layer_summary": layer_summary,
            "thresholds": {
                "min_median_cos": float(args.min_median_cos),
                "min_step_cos": float(args.min_step_cos),
                "layer_cliff_threshold": float(args.layer_cliff_threshold),
            },
            "pass_threshold": bool(threshold_ok),
            "per_step": rows,
            "per_step_layer": layer_rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[RESULT] wrote_json = {out_path}")


if __name__ == "__main__":
    main()
