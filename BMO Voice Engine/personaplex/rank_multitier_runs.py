import argparse
import json
from pathlib import Path

import torch


def parse_run_spec(raw: str) -> tuple[str, Path, Path]:
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "Each --run must be NAME:CHECKPOINT_PATH:DRIFT_JSON_PATH"
        )
    name = parts[0].strip()
    ckpt = Path(parts[1].strip()).resolve()
    drift = Path(parts[2].strip()).resolve()
    if not name:
        raise ValueError("Run name must be non-empty")
    return name, ckpt, drift


def load_checkpoint_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
    meta = payload.get("septq_meta")
    if not isinstance(meta, dict):
        meta = {}

    return {
        "estimated_weight_gib": float(meta.get("estimated_weight_gib", float("nan"))),
        "effective_bpw": float(meta.get("effective_bpw", float("nan"))),
        "ratio_fp16": float(meta.get("ratio_fp16", float("nan"))),
        "ratio_int8": float(meta.get("ratio_int8", float("nan"))),
        "ratio_lowbit": float(meta.get("ratio_lowbit", float("nan"))),
        "low_bits": int(meta.get("low_bits", 0)) if "low_bits" in meta else 0,
    }


def load_drift_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Drift JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported drift payload: {type(payload)}")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    return {
        "cos_median": float(summary.get("cos_median", float("nan"))),
        "cos_min": float(summary.get("cos_min", float("nan"))),
        "pass_threshold": bool(payload.get("pass_threshold", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank multi-tier SEPTQ runs by z_s drift quality and weight budget. "
            "Run format: NAME:CHECKPOINT:DRIFT_JSON"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec NAME:CHECKPOINT_PATH:DRIFT_JSON_PATH (repeat for each run).",
    )
    parser.add_argument("--min-median-cos", type=float, default=0.997)
    parser.add_argument("--max-weight-gib", type=float, default=3.6)
    args = parser.parse_args()

    rows = []
    for raw in args.run:
        name, ckpt_path, drift_path = parse_run_spec(raw)
        ckpt_metrics = load_checkpoint_metrics(ckpt_path)
        drift_metrics = load_drift_metrics(drift_path)

        row = {
            "name": name,
            "checkpoint": str(ckpt_path),
            "drift_json": str(drift_path),
            **ckpt_metrics,
            **drift_metrics,
        }
        row["meets_success"] = (
            row["cos_median"] >= float(args.min_median_cos)
            and row["estimated_weight_gib"] <= float(args.max_weight_gib)
        )
        rows.append(row)

    print("[INFO] Candidate runs:")
    for r in rows:
        print(
            f"[RESULT] {r['name']}: cos_median={r['cos_median']:.6f} "
            f"cos_min={r['cos_min']:.6f} weight_gib={r['estimated_weight_gib']:.6f} "
            f"effective_bpw={r['effective_bpw']:.6f} meets_success={r['meets_success']}"
        )

    feasible = [r for r in rows if r["meets_success"]]
    if not feasible:
        print("[RESULT] best = NONE (no run satisfied both thresholds)")
        raise SystemExit(1)

    feasible.sort(key=lambda r: (-r["cos_median"], r["estimated_weight_gib"]))
    best = feasible[0]

    print("[RESULT] best = " + best["name"])
    print(
        f"[RESULT] best_metrics: cos_median={best['cos_median']:.6f} "
        f"cos_min={best['cos_min']:.6f} weight_gib={best['estimated_weight_gib']:.6f} "
        f"effective_bpw={best['effective_bpw']:.6f}"
    )


if __name__ == "__main__":
    main()
