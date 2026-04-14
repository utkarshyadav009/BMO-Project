import argparse
import json
from pathlib import Path

import torch


QUANT_SUFFIX = ".quant_state.bitsandbytes__nf4"


def classify_group(base_key: str) -> str:
    if base_key.endswith(".self_attn.in_proj_weight"):
        return "self_attn.in_proj_weight"
    if base_key.endswith(".self_attn.out_proj.weight"):
        return "self_attn.out_proj.weight"
    if base_key.endswith(".gating.linear_in.weight"):
        return "gating.linear_in.weight"
    if base_key.endswith(".gating.linear_out.weight"):
        return "gating.linear_out.weight"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Generate LQEC manifest from INT4 base checkpoint")
    parser.add_argument("--teacher", default="v5_step1500.safetensors")
    parser.add_argument("--student", default="bmo_temporal_int4_base.pt")
    parser.add_argument("--out", default="lqec_manifest.json")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    teacher_path = (root / args.teacher).resolve()
    student_path = (root / args.student).resolve()
    out_path = (root / args.out).resolve()

    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_path}")

    ckpt = torch.load(student_path, map_location="cpu", mmap=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        awq_meta = ckpt.get("awq_meta", {})
    else:
        state_dict = ckpt
        awq_meta = {}

    quant_bases = sorted(
        k[: -len(QUANT_SUFFIX)]
        for k in state_dict.keys()
        if k.endswith(QUANT_SUFFIX)
    )

    grouped = {}
    for base in quant_bases:
        g = classify_group(base)
        grouped[g] = grouped.get(g, 0) + 1

    manifest = {
        "teacher_checkpoint": str(teacher_path),
        "student_checkpoint": str(student_path),
        "teacher_size_gb": teacher_path.stat().st_size / 1e9,
        "student_size_gb": student_path.stat().st_size / 1e9,
        "awq_meta": awq_meta,
        "quantized_group_count": len(quant_bases),
        "quantized_group_breakdown": grouped,
        "quantized_groups": quant_bases,
        "lqec_plan": {
            "target": "all quantized temporal linear groups",
            "rank": args.rank,
            "alpha": args.alpha,
            "loss": "mse(student_text_logits, teacher_text_logits)",
            "steps": args.steps,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Wrote manifest: {out_path}")
    print(f"[INFO] Quantized groups: {len(quant_bases)}")
    print(f"[INFO] Breakdown: {grouped}")


if __name__ == "__main__":
    main()
