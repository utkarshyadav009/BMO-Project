import argparse
import math
from pathlib import Path

import torch

from apply_septq import gather_audio_files, resolve_local_path
from moshi.models import loaders
from moshi.models.lm import load_audio


def summarize(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }

    t = torch.tensor(values, dtype=torch.float32)
    q = torch.quantile(t, torch.tensor([0.50, 0.90, 0.99], dtype=torch.float32))
    return {
        "count": int(t.numel()),
        "min": float(t.min().item()),
        "p50": float(q[0].item()),
        "p90": float(q[1].item()),
        "p99": float(q[2].item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect calibration clip coverage and estimated Mimi frame steps."
    )
    parser.add_argument("--calibration-clips", required=True)
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-steps-per-clip", type=int, default=750)
    parser.add_argument("--target-samples", type=int, default=16384)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    calibration_path = resolve_local_path(root, args.calibration_clips)
    mimi_weight = resolve_local_path(root, args.mimi_weight)

    max_clips = int(args.max_clips)
    if max_clips < 0:
        raise SystemExit("[ERROR] --max-clips must be >= 0")
    clip_cap = max_clips if max_clips > 0 else 1_000_000_000

    files = gather_audio_files(calibration_path, max_clips=clip_cap)
    if not files:
        raise SystemExit(f"[ERROR] No audio clips found under: {calibration_path}")

    if not mimi_weight.exists():
        raise SystemExit(f"[ERROR] Mimi checkpoint not found: {mimi_weight}")

    mimi = loaders.get_mimi(str(mimi_weight), args.device)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    raw_steps: list[int] = []
    capped_steps: list[int] = []

    step_cap = int(args.max_steps_per_clip)
    if step_cap <= 0:
        raise SystemExit("[ERROR] --max-steps-per-clip must be > 0")

    for p in files:
        sample_pcm = load_audio(str(p), mimi.sample_rate)
        num_samples = int(sample_pcm.shape[-1])
        if num_samples <= 0:
            continue

        steps = int(math.ceil(num_samples / float(frame_size)))
        raw_steps.append(steps)
        capped_steps.append(min(steps, step_cap))

    if not capped_steps:
        raise SystemExit("[ERROR] No usable clips after loading audio.")

    raw_total = int(sum(raw_steps))
    capped_total = int(sum(capped_steps))

    raw_summary = summarize(raw_steps)
    capped_summary = summarize(capped_steps)

    print(f"[INFO] calibration_path = {calibration_path}")
    print(f"[INFO] clip_count = {len(files)}")
    print(f"[INFO] mimi_sample_rate = {mimi.sample_rate}")
    print(f"[INFO] mimi_frame_rate = {mimi.frame_rate}")
    print(f"[INFO] frame_size_samples = {frame_size}")
    print(f"[INFO] max_steps_per_clip = {step_cap}")
    print(f"[RESULT] total_steps_raw = {raw_total}")
    print(f"[RESULT] total_steps_capped = {capped_total}")

    print(
        "[RESULT] raw_steps_stats = "
        f"min={raw_summary['min']:.1f} p50={raw_summary['p50']:.1f} "
        f"p90={raw_summary['p90']:.1f} p99={raw_summary['p99']:.1f} "
        f"max={raw_summary['max']:.1f} mean={raw_summary['mean']:.2f}"
    )
    print(
        "[RESULT] capped_steps_stats = "
        f"min={capped_summary['min']:.1f} p50={capped_summary['p50']:.1f} "
        f"p90={capped_summary['p90']:.1f} p99={capped_summary['p99']:.1f} "
        f"max={capped_summary['max']:.1f} mean={capped_summary['mean']:.2f}"
    )

    target_samples = int(args.target_samples)
    if target_samples > 0:
        clips_needed = int(math.ceil(target_samples / max(capped_summary["mean"], 1e-6)))
        print(f"[RESULT] estimated_clips_needed_for_{target_samples} = {clips_needed}")


if __name__ == "__main__":
    main()
