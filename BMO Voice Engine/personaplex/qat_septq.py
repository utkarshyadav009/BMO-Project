import argparse
import gc
import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import bitsandbytes as bnb

from moshi.models import loaders
from moshi.models.lm import LMModel

from apply_septq import (
    build_calibration_sequences,
    build_lm_kwargs,
    build_quantization_entries,
    gather_audio_files,
    get_module_name_map,
    get_temporal_layers,
    parse_quantize_layers,
    parse_skip_module_filters,
    read_config_override_from_payload,
    resolve_local_path,
)


def parse_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError("dtype must be one of: bfloat16, float16, float32")


def resolve_runtime_device(requested: str) -> str:
    dev = str(requested).strip().lower()
    if not dev:
        return "cpu"

    if not dev.startswith("cuda"):
        return dev

    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA device requested but torch.cuda.is_available() is False")

    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise SystemExit("[ERROR] CUDA device requested but no CUDA devices are visible")

    if dev == "cuda":
        return "cuda:0"

    if ":" not in dev:
        return "cuda:0"

    _, ordinal_text = dev.split(":", 1)
    try:
        ordinal = int(ordinal_text)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Invalid --device value: {requested}") from exc

    if ordinal < 0 or ordinal >= visible:
        raise SystemExit(
            "[ERROR] Invalid CUDA device ordinal for current visibility: "
            f"requested={requested} visible_count={visible}. "
            "If CUDA_VISIBLE_DEVICES is set to a single GPU, use --device cuda or --device cuda:0."
        )

    return f"cuda:{ordinal}"


def unwrap_zs(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 1:
        return z.view(1, -1)
    return z.reshape(-1, z.shape[-1])


def align_zs(student_z: torch.Tensor, teacher_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = unwrap_zs(student_z)
    t = unwrap_zs(teacher_z)
    rows = min(int(s.shape[0]), int(t.shape[0]))
    dims = min(int(s.shape[1]), int(t.shape[1]))
    if rows <= 0 or dims <= 0:
        raise RuntimeError("Invalid z_s tensor shape while aligning teacher/student outputs")
    return s[:rows, :dims], t[:rows, :dims]


def zs_kl_div(student_z: torch.Tensor, teacher_z: torch.Tensor, temperature: float) -> torch.Tensor:
    s, t = align_zs(student_z, teacher_z)
    temp = max(1e-6, float(temperature))
    s_log_prob = F.log_softmax(s.float() / temp, dim=-1)
    t_prob = F.softmax(t.float() / temp, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp * temp)


def zs_cosine(student_z: torch.Tensor, teacher_z: torch.Tensor) -> float:
    s, t = align_zs(student_z, teacher_z)
    s_flat = s.reshape(-1).float()
    t_flat = t.reshape(-1).float()
    denom = float((torch.norm(s_flat) * torch.norm(t_flat)).item())
    if denom <= 0.0:
        return 1.0
    return float(torch.sum(s_flat * t_flat).item() / denom)


def register_zs_hook(model: nn.Module, keep_grad: bool) -> tuple[Dict[str, torch.Tensor | None], Any]:
    cache: Dict[str, torch.Tensor | None] = {"value": None}
    out_norm = getattr(model, "out_norm", None)
    if out_norm is None:
        raise RuntimeError("Model has no out_norm module; cannot capture z_s")

    def pre_hook(_module, inputs):
        if not inputs:
            cache["value"] = None
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            cache["value"] = None
            return
        cache["value"] = x if keep_grad else x.detach()

    handle = out_norm.register_forward_pre_hook(pre_hook)
    return cache, handle


def _detach_streaming_obj_inplace(obj: Any, visited: set[int]) -> Any:
    if obj is None:
        return None

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    if torch.is_tensor(obj):
        return obj.detach()

    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _detach_streaming_obj_inplace(obj[k], visited)
        return obj

    if isinstance(obj, list):
        for idx in range(len(obj)):
            obj[idx] = _detach_streaming_obj_inplace(obj[idx], visited)
        return obj

    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            detached = _detach_streaming_obj_inplace(value, visited)
            if detached is not value:
                setattr(obj, name, detached)
        return obj

    return obj


def detach_model_streaming_state_(model: nn.Module) -> None:
    getter = getattr(model, "get_streaming_state", None)
    if getter is None:
        return
    state_by_name = getter()
    visited: set[int] = set()
    for state in state_by_name.values():
        if state is None:
            continue
        _detach_streaming_obj_inplace(state, visited)


def fake_quantize_affine_ste(w: torch.Tensor, bits: int, min_range: float) -> torch.Tensor:
    qmin = 0.0
    qmax = float((1 << int(bits)) - 1)

    w_min = torch.min(w)
    w_max = torch.max(w)
    span = (w_max - w_min).clamp_min(float(min_range))

    scale = span / max(1.0, qmax - qmin)
    zero_point = torch.round(-w_min / scale).clamp(qmin, qmax)

    q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax)
    dequant = scale * (q - zero_point)

    # Straight-through estimator: forward uses dequant, backward sees identity.
    return w + (dequant - w).detach()


class FakeQuantSTEParam(nn.Module):
    def __init__(self, bits: int, min_range: float):
        super().__init__()
        self.bits = int(bits)
        self.min_range = float(min_range)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return fake_quantize_affine_ste(w, bits=self.bits, min_range=self.min_range)


def gather_qat_entries(
    student: nn.Module,
    selected_layers: List[int],
    skip_module_filters: List[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    temporal_layers = get_temporal_layers(student)
    if not temporal_layers:
        raise RuntimeError("Could not resolve temporal transformer layers for student model")

    name_map = get_module_name_map(student)
    layer_plan = build_quantization_entries(
        temporal_layers=temporal_layers,
        selected_indices=selected_layers,
        name_map=name_map,
        skip_module_filters=skip_module_filters,
    )

    entries: List[Dict[str, Any]] = []
    excluded: List[str] = []
    for idx in selected_layers:
        pack = layer_plan[idx]
        entries.extend(pack["entries"])
        excluded.extend(list(pack.get("excluded_entries", [])))

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for e in entries:
        name = str(e["name"])
        if name in seen:
            continue
        seen.add(name)
        deduped.append(e)

    return deduped, sorted(set(excluded))


def register_fake_quant_for_entries(
    entries: List[Dict[str, Any]],
    bits: int,
    min_range: float,
) -> Dict[str, Tuple[nn.Module, str]]:
    state_key_to_param: Dict[str, Tuple[nn.Module, str]] = {}
    seen = set()

    for e in entries:
        module = e["module"]
        param_name = "weight" if e["kind"] == "linear" else str(e["param_name"])
        state_key = str(e["name"])
        dedup_key = (id(module), param_name)
        if dedup_key in seen:
            state_key_to_param[state_key] = (module, param_name)
            continue
        seen.add(dedup_key)

        target = getattr(module, param_name, None)
        if not torch.is_tensor(target):
            raise RuntimeError(f"Cannot register fake quant: missing tensor param {state_key}")

        has_parametrization = (
            hasattr(module, "parametrizations") and param_name in getattr(module, "parametrizations", {})
        )
        if not has_parametrization:
            parametrize.register_parametrization(
                module,
                param_name,
                FakeQuantSTEParam(bits=int(bits), min_range=float(min_range)),
            )

        original = module.parametrizations[param_name].original
        original.requires_grad = True
        state_key_to_param[state_key] = (module, param_name)

    return state_key_to_param


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def export_dense_state_dict(
    model: nn.Module,
    parametrized_state_keys: Dict[str, Tuple[nn.Module, str]],
) -> Dict[str, torch.Tensor]:
    raw_sd = model.state_dict()
    export_sd: Dict[str, torch.Tensor] = {}

    for key, tensor in raw_sd.items():
        if ".parametrizations." in key:
            continue
        export_sd[key] = tensor.detach().to(device="cpu").contiguous()

    for state_key, (module, param_name) in parametrized_state_keys.items():
        export_sd[state_key] = getattr(module, param_name).detach().to(device="cpu").contiguous()

    return export_sd


def evaluate_zs_drift(
    teacher: nn.Module,
    student: nn.Module,
    sequences: List[torch.Tensor],
    *,
    teacher_cache: Dict[str, torch.Tensor | None],
    student_cache: Dict[str, torch.Tensor | None],
    device: str,
    max_eval_clips: int,
    max_eval_steps_per_clip: int,
    temperature: float,
) -> Dict[str, float | int]:
    was_student_training = bool(student.training)
    teacher.eval()
    student.eval()

    clip_count = len(sequences) if int(max_eval_clips) <= 0 else min(int(max_eval_clips), len(sequences))
    if clip_count <= 0:
        raise RuntimeError("No sequences available for z_s evaluation")

    cos_values: List[float] = []
    kl_values: List[float] = []

    with torch.no_grad():
        for clip_idx in range(clip_count):
            seq = sequences[clip_idx]
            steps = int(seq.shape[0])
            if int(max_eval_steps_per_clip) > 0:
                steps = min(steps, int(max_eval_steps_per_clip))
            if steps <= 0:
                continue

            with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
                for t in range(steps):
                    token = seq[t].view(1, seq.shape[1], 1).to(device=device)
                    teacher_cache["value"] = None
                    student_cache["value"] = None

                    teacher.forward_codes(token)
                    student.forward_codes(token)

                    t_z = teacher_cache.get("value")
                    s_z = student_cache.get("value")
                    if t_z is None or s_z is None:
                        continue

                    cos_values.append(zs_cosine(s_z, t_z))
                    kl_values.append(float(zs_kl_div(s_z, t_z, temperature=temperature).item()))

    if was_student_training:
        student.train()

    if not cos_values:
        raise RuntimeError("Evaluation captured no valid z_s pairs")

    cos_t = torch.tensor(cos_values, dtype=torch.float32)
    kl_t = torch.tensor(kl_values, dtype=torch.float32)
    q = torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32)
    cos_q = torch.quantile(cos_t, q)
    kl_q = torch.quantile(kl_t, q)

    return {
        "clip_count": int(clip_count),
        "steps": int(cos_t.numel()),
        "cos_min": float(cos_t.min().item()),
        "cos_p10": float(cos_q[0].item()),
        "cos_median": float(cos_q[1].item()),
        "cos_p90": float(cos_q[2].item()),
        "cos_mean": float(cos_t.mean().item()),
        "kl_min": float(kl_t.min().item()),
        "kl_p10": float(kl_q[0].item()),
        "kl_median": float(kl_q[1].item()),
        "kl_p90": float(kl_q[2].item()),
        "kl_mean": float(kl_t.mean().item()),
    }


def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total = max(1, int(total_steps))
    warmup = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        s = int(step)
        if warmup > 0 and s < warmup:
            return float(s + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        progress = float(s - warmup) / float(max(1, total - warmup))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_qat_checkpoint(
    out_path: Path,
    student: nn.Module,
    parametrized_state_keys: Dict[str, Tuple[nn.Module, str]],
    source_cfg: Dict[str, Any] | None,
    qat_meta: Dict[str, Any],
) -> None:
    export_sd = export_dense_state_dict(student, parametrized_state_keys)
    payload = {
        "state_dict": export_sd,
        "config_override": source_cfg,
        "model_mode": "qat_septq_dense",
        "force_dense": True,
        "qat_meta": qat_meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))


def strict_verify_load(export_state_dict: Dict[str, torch.Tensor], source_checkpoint: Path, dtype: torch.dtype) -> None:
    lm_kwargs = build_lm_kwargs(source_checkpoint)
    verify_model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
    verify_model.load_state_dict(export_state_dict, strict=True, assign=True)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QAT for SEPTQ 4-bit initialization with STE fake-quantized weights and "
            "z_s KL distillation against a teacher model."
        )
    )
    parser.add_argument("--teacher", default="v5_step1500_split.safetensors")
    parser.add_argument("--student-init", default="bmo_temporal_septq_4bit.pt")
    parser.add_argument("--out-dir", default="qat_septq_runs")

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--teacher-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--student-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--calibration-clips", required=True)
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--max-clips", type=int, default=857)
    parser.add_argument("--max-steps-per-clip", type=int, default=750)

    parser.add_argument("--train-layers", default="0-29")
    parser.add_argument("--skip-modules", default="self_attn.out_proj")

    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--quant-min-range", type=float, default=1e-6)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--min-train-steps", type=int, default=500)
    parser.add_argument(
        "--train-max-steps-per-clip",
        type=int,
        default=0,
        help="If >0, truncate each training clip to this many token steps to reduce memory.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-clips", type=int, default=64)
    parser.add_argument("--eval-max-steps-per-clip", type=int, default=0)
    parser.add_argument("--target-median-cos", type=float, default=0.997)
    parser.add_argument("--flatline-median-cos", type=float, default=0.99)
    parser.add_argument("--flatline-window", type=int, default=3)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verify-load-on-checkpoint", action="store_true")
    parser.add_argument("--no-copy-missing-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.max_train_steps) <= 0:
        raise SystemExit("[ERROR] --max-train-steps must be > 0")
    if int(args.min_train_steps) < 0:
        raise SystemExit("[ERROR] --min-train-steps must be >= 0")
    if int(args.min_train_steps) > int(args.max_train_steps):
        raise SystemExit("[ERROR] --min-train-steps must be <= --max-train-steps")
    if int(args.max_steps_per_clip) <= 0:
        raise SystemExit("[ERROR] --max-steps-per-clip must be > 0")
    if int(args.max_clips) <= 0:
        raise SystemExit("[ERROR] --max-clips must be > 0")
    if int(args.checkpoint_every) <= 0:
        raise SystemExit("[ERROR] --checkpoint-every must be > 0")
    if float(args.quant_min_range) <= 0.0:
        raise SystemExit("[ERROR] --quant-min-range must be > 0")
    if float(args.lr) <= 0.0:
        raise SystemExit("[ERROR] --lr must be > 0")
    if int(args.flatline_window) <= 0:
        raise SystemExit("[ERROR] --flatline-window must be > 0")

    runtime_device = resolve_runtime_device(str(args.device))

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    start_time = time.perf_counter()
    root = Path(__file__).resolve().parent

    teacher_path = resolve_local_path(root, str(args.teacher))
    student_init_path = resolve_local_path(root, str(args.student_init))
    calibration_path = resolve_local_path(root, str(args.calibration_clips))
    mimi_weight = resolve_local_path(root, str(args.mimi_weight))
    out_dir = resolve_local_path(root, str(args.out_dir))

    if not teacher_path.exists():
        raise SystemExit(f"[ERROR] teacher checkpoint not found: {teacher_path}")
    if not student_init_path.exists():
        raise SystemExit(f"[ERROR] student initialization checkpoint not found: {student_init_path}")
    if not mimi_weight.exists():
        raise SystemExit(f"[ERROR] mimi weight not found: {mimi_weight}")

    teacher_dtype = parse_dtype(args.teacher_dtype)
    student_dtype = None if str(args.student_dtype).strip().lower() == "auto" else parse_dtype(args.student_dtype)
    verify_dtype = parse_dtype(args.dtype)

    print(f"[INFO] teacher = {teacher_path}")
    print(f"[INFO] student_init = {student_init_path}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] device = {runtime_device}")
    print(
        f"[INFO] bits={args.bits} train_layers={args.train_layers} "
        f"max_train_steps={args.max_train_steps} warmup_steps={args.warmup_steps}"
    )
    if int(args.train_max_steps_per_clip) > 0:
        print(f"[INFO] train_max_steps_per_clip = {int(args.train_max_steps_per_clip)}")

    print("[INFO] Loading teacher model...")
    teacher = loaders.get_moshi_lm(
        str(teacher_path),
        copy_missing_weights=not bool(args.no_copy_missing_weights),
        device=runtime_device,
        dtype=teacher_dtype,
        cpu_offload=False,
    )
    teacher.eval()
    freeze_all_params(teacher)

    print("[INFO] Loading student model...")
    student = loaders.get_moshi_lm(
        str(student_init_path),
        copy_missing_weights=not bool(args.no_copy_missing_weights),
        device=runtime_device,
        dtype=student_dtype,
        cpu_offload=False,
    )
    student.train()

    temporal_layers = get_temporal_layers(student)
    if not temporal_layers:
        raise SystemExit("[ERROR] Could not resolve temporal layers in student model")

    try:
        selected_layers = parse_quantize_layers(str(args.train_layers), len(temporal_layers))
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    skip_filters = parse_skip_module_filters(args.skip_modules)
    entries, excluded_entries = gather_qat_entries(
        student=student,
        selected_layers=selected_layers,
        skip_module_filters=skip_filters,
    )
    if not entries:
        raise SystemExit("[ERROR] No quantizable entries selected for QAT")

    print(
        f"[INFO] QAT modules selected: {len(entries)} "
        f"excluded_by_filter={len(excluded_entries)}"
    )

    freeze_all_params(student)
    parametrized_state_keys = register_fake_quant_for_entries(
        entries=entries,
        bits=int(args.bits),
        min_range=float(args.quant_min_range),
    )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not trainable_params:
        raise SystemExit("[ERROR] No trainable parameters after fake-quant registration")

    total_trainable = count_trainable_params(student)
    print(f"[INFO] trainable_params = {total_trainable}")

    calibration_files = gather_audio_files(calibration_path, max_clips=int(args.max_clips))
    if not calibration_files:
        raise SystemExit(f"[ERROR] No calibration clips found under {calibration_path}")

    print(f"[INFO] Building training sequences from {len(calibration_files)} clips...")
    sequences, total_steps_from_clips = build_calibration_sequences(
        model=teacher,
        calibration_files=calibration_files,
        mimi_weight=mimi_weight,
        device=runtime_device,
        max_steps_per_clip=int(args.max_steps_per_clip),
    )
    if not sequences:
        raise SystemExit("[ERROR] Failed to build any token sequences for QAT")

    print(
        f"[INFO] sequence_count={len(sequences)} total_steps_from_clips={total_steps_from_clips}"
    )

    optimizer = bnb.optim.AdamW8bit(
        trainable_params,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = make_cosine_warmup_scheduler(
        optimizer,
        total_steps=int(args.max_train_steps),
        warmup_steps=int(args.warmup_steps),
    )

    teacher_cache, teacher_hook = register_zs_hook(teacher, keep_grad=False)
    student_cache, student_hook = register_zs_hook(student, keep_grad=True)

    log_path = out_dir / "qat_train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_cfg = read_config_override_from_payload(student_init_path)

    train_order = list(range(len(sequences)))
    rng = random.Random(int(args.seed))
    rng.shuffle(train_order)
    train_ptr = 0

    def next_clip_index() -> int:
        nonlocal train_ptr
        if train_ptr >= len(train_order):
            rng.shuffle(train_order)
            train_ptr = 0
        idx = train_order[train_ptr]
        train_ptr += 1
        return int(idx)

    best_median = -1.0
    best_step = 0
    med_tail = deque(maxlen=int(args.flatline_window))
    stop_reason = "max_steps"

    try:
        print("[INFO] Running baseline z_s evaluation before QAT...")
        baseline_eval = evaluate_zs_drift(
            teacher=teacher,
            student=student,
            sequences=sequences,
            teacher_cache=teacher_cache,
            student_cache=student_cache,
            device=runtime_device,
            max_eval_clips=int(args.eval_clips),
            max_eval_steps_per_clip=int(args.eval_max_steps_per_clip),
            temperature=float(args.temperature),
        )
        print(
            f"[RESULT] baseline_eval: cos_median={baseline_eval['cos_median']:.6f} "
            f"cos_min={baseline_eval['cos_min']:.6f} kl_median={baseline_eval['kl_median']:.6e}"
        )
        if baseline_eval['cos_median'] < 0.5:
            raise SystemExit(f"[ERROR] Baseline median cosine is critically low ({baseline_eval['cos_median']:.6f}). The student initialization is broken or over-quantized. Aborting.")

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                json.dumps(
                    {
                        "type": "baseline_eval",
                        "time_sec": float(time.perf_counter() - start_time),
                        "metrics": baseline_eval,
                    }
                )
                + "\n"
            )

            for step in range(1, int(args.max_train_steps) + 1):
                student.train()
                clip_idx = next_clip_index()
                seq = sequences[clip_idx]
                seq_steps = int(seq.shape[0])
                if int(args.train_max_steps_per_clip) > 0:
                    seq_steps = min(seq_steps, int(args.train_max_steps_per_clip))
                if seq_steps <= 0:
                    print(f"[WARN] Step {step}: clip {clip_idx} has 0 usable steps; skipping")
                    continue

                optimizer.zero_grad(set_to_none=True)
                valid_steps = 0
                kl_sum = 0.0

                with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
                    for t in range(seq_steps):
                        token = seq[t].view(1, seq.shape[1], 1).to(device=runtime_device)

                        teacher_cache["value"] = None
                        student_cache["value"] = None

                        with torch.no_grad():
                            teacher.forward_codes(token)
                        student.forward_codes(token)

                        t_z = teacher_cache.get("value")
                        s_z = student_cache.get("value")
                        if t_z is None or s_z is None:
                            continue

                        kl_t = zs_kl_div(s_z, t_z, temperature=float(args.temperature))
                        kl_sum += float(kl_t.item())
                        # Backprop per token to avoid retaining an entire-clip graph in memory.
                        (kl_t / float(seq_steps)).backward()
                        # Truncated-BPTT style detach for streaming caches to avoid
                        # reusing freed graph references on the next token step.
                        detach_model_streaming_state_(student)
                        valid_steps += 1

                if valid_steps <= 0:
                    print(f"[WARN] Step {step}: no valid z_s captures; skipping optimizer step")
                    continue

                loss_value = float(kl_sum / max(1, valid_steps))

                if float(args.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))

                optimizer.step()
                scheduler.step()

                if step % max(1, int(args.log_every)) == 0 or step == 1:
                    current_lr = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"[TRAIN] step={step} clip_idx={clip_idx} seq_steps={seq_steps} "
                        f"kl={loss_value:.6e} lr={current_lr:.3e}"
                    )

                run_eval = (step % int(args.checkpoint_every) == 0) or (step == int(args.max_train_steps))
                if not run_eval:
                    continue

                eval_t0 = time.perf_counter()
                eval_metrics = evaluate_zs_drift(
                    teacher=teacher,
                    student=student,
                    sequences=sequences,
                    teacher_cache=teacher_cache,
                    student_cache=student_cache,
                    device=runtime_device,
                    max_eval_clips=int(args.eval_clips),
                    max_eval_steps_per_clip=int(args.eval_max_steps_per_clip),
                    temperature=float(args.temperature),
                )
                eval_elapsed = time.perf_counter() - eval_t0
                med = float(eval_metrics["cos_median"])
                med_tail.append(med)

                print(
                    f"[EVAL] step={step} cos_median={med:.6f} cos_min={float(eval_metrics['cos_min']):.6f} "
                    f"kl_median={float(eval_metrics['kl_median']):.6e} elapsed={eval_elapsed:.1f}s"
                )

                qat_meta = {
                    "source_teacher": str(teacher_path),
                    "source_student_init": str(student_init_path),
                    "step": int(step),
                    "seed": int(args.seed),
                    "bits": int(args.bits),
                    "quant_min_range": float(args.quant_min_range),
                    "train_layers": [int(i) for i in selected_layers],
                    "skip_modules_filters": skip_filters,
                    "excluded_modules": excluded_entries,
                    "optimizer": "AdamW",
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "scheduler": "cosine_with_warmup",
                    "warmup_steps": int(args.warmup_steps),
                    "max_train_steps": int(args.max_train_steps),
                    "dataset_clip_count": int(len(sequences)),
                    "dataset_total_steps": int(total_steps_from_clips),
                    "temperature": float(args.temperature),
                    "target_median_cos": float(args.target_median_cos),
                    "flatline_median_cos": float(args.flatline_median_cos),
                    "flatline_window": int(args.flatline_window),
                    "eval_metrics": eval_metrics,
                    "loss_kl": float(loss_value),
                    "elapsed_sec": float(time.perf_counter() - start_time),
                }

                step_ckpt = out_dir / f"qat_step_{step:04d}.pt"
                save_qat_checkpoint(
                    out_path=step_ckpt,
                    student=student,
                    parametrized_state_keys=parametrized_state_keys,
                    source_cfg=source_cfg,
                    qat_meta=qat_meta,
                )

                if bool(args.verify_load_on_checkpoint):
                    export_sd = export_dense_state_dict(student, parametrized_state_keys)
                    strict_verify_load(export_sd, teacher_path, verify_dtype)

                if med > best_median:
                    best_median = med
                    best_step = int(step)
                    best_path = out_dir / "qat_best.pt"
                    save_qat_checkpoint(
                        out_path=best_path,
                        student=student,
                        parametrized_state_keys=parametrized_state_keys,
                        source_cfg=source_cfg,
                        qat_meta=qat_meta,
                    )

                log_file.write(
                    json.dumps(
                        {
                            "type": "eval",
                            "step": int(step),
                            "metrics": eval_metrics,
                            "loss_kl": float(loss_value),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "time_sec": float(time.perf_counter() - start_time),
                            "checkpoint": str(step_ckpt),
                        }
                    )
                    + "\n"
                )
                log_file.flush()

                if int(step) >= int(args.min_train_steps) and med >= float(args.target_median_cos):
                    stop_reason = "target_reached"
                    print(
                        f"[STOP] Target reached at step {step}: median={med:.6f} >= {float(args.target_median_cos):.6f}"
                    )
                    break

                if (
                    int(step) >= int(args.min_train_steps)
                    and len(med_tail) >= int(args.flatline_window)
                    and max(med_tail) <= float(args.flatline_median_cos)
                ):
                    stop_reason = "flatline"
                    print(
                        f"[STOP] Flatline detected at step {step}: "
                        f"window_max_median={max(med_tail):.6f} <= {float(args.flatline_median_cos):.6f}"
                    )
                    break

    finally:
        teacher_hook.remove()
        student_hook.remove()

        del teacher
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time
    print(f"[RESULT] stop_reason = {stop_reason}")
    print(f"[RESULT] best_step = {best_step}")
    print(f"[RESULT] best_median_cos = {best_median:.6f}")
    print(f"[RESULT] elapsed_sec = {elapsed:.3f}")
    print(f"[RESULT] out_dir = {out_dir}")
    print(f"[RESULT] train_log = {log_path}")


if __name__ == "__main__":
    main()
