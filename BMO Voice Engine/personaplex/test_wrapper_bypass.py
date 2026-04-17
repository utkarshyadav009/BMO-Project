import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from moshi.models import loaders
from verify_int4_rollout_drift import (
    build_forced_tokens,
    get_temporal_layers,
    parse_bool,
    parse_dtype,
)


def _unwrap_tensor_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    return None


@torch.no_grad()
def capture_teacher_step(
    teacher,
    forced_tokens: torch.Tensor,
    step_idx: int,
    device: str,
    use_streaming: bool,
):
    layers = get_temporal_layers(teacher)
    if not layers:
        raise RuntimeError("Teacher temporal layers are unavailable")

    teacher_input = None
    layer_inputs = [None for _ in range(len(layers))]
    layer_outputs = [None for _ in range(len(layers))]

    def pre_hook(_module, inputs):
        nonlocal teacher_input
        if inputs and torch.is_tensor(inputs[0]):
            teacher_input = inputs[0].detach().to(dtype=torch.float32)

    def make_pre_hook(idx):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                layer_inputs[idx] = inputs[0].detach().to(dtype=torch.float32)

        return hook

    def make_hook(idx):
        def hook(_module, _inputs, output):
            y = _unwrap_tensor_output(output)
            if torch.is_tensor(y):
                layer_outputs[idx] = y.detach().to(dtype=torch.float32)

        return hook

    handles = [layers[0].register_forward_pre_hook(pre_hook)]
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
        handles.append(layer.register_forward_hook(make_hook(idx)))

    k = int(teacher.num_codebooks)
    try:
        if bool(use_streaming):
            with teacher.streaming(batch_size=1):
                for t in range(int(forced_tokens.shape[0])):
                    seq = forced_tokens[t].view(1, k, 1).to(device)
                    teacher.forward_codes(seq)
                    if int(t) == int(step_idx):
                        break
        else:
            for t in range(int(forced_tokens.shape[0])):
                seq = forced_tokens[t].view(1, k, 1).to(device)
                teacher.forward_codes(seq)
                if int(t) == int(step_idx):
                    break
    finally:
        for handle in handles:
            handle.remove()

    if teacher_input is None:
        raise RuntimeError("Failed to capture teacher transformer input at the requested step")
    if any(row is None for row in layer_inputs):
        raise RuntimeError("Failed to capture one or more teacher layer inputs")
    if any(row is None for row in layer_outputs):
        raise RuntimeError("Failed to capture one or more teacher layer outputs")

    return teacher_input, layer_inputs, layer_outputs


@torch.no_grad()
def capture_student_wrapper_step(
    student,
    forced_tokens: torch.Tensor,
    step_idx: int,
    device: str,
    use_streaming: bool,
):
    layers = get_temporal_layers(student)
    if not layers:
        raise RuntimeError("Student temporal layers are unavailable")

    inner_input = None
    layer_outputs = [None for _ in range(len(layers))]

    def pre_hook(_module, inputs):
        nonlocal inner_input
        if inputs and torch.is_tensor(inputs[0]):
            inner_input = inputs[0].detach().to(dtype=torch.float32)

    def make_hook(idx):
        def hook(_module, _inputs, output):
            y = _unwrap_tensor_output(output)
            if torch.is_tensor(y):
                layer_outputs[idx] = y.detach().to(dtype=torch.float32)

        return hook

    handles = [layers[0].register_forward_pre_hook(pre_hook)]
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    k = int(student.num_codebooks)
    try:
        if bool(use_streaming):
            with student.streaming(batch_size=1):
                for t in range(int(forced_tokens.shape[0])):
                    seq = forced_tokens[t].view(1, k, 1).to(device)
                    student.forward_codes(seq)
                    if int(t) == int(step_idx):
                        break
        else:
            for t in range(int(forced_tokens.shape[0])):
                seq = forced_tokens[t].view(1, k, 1).to(device)
                student.forward_codes(seq)
                if int(t) == int(step_idx):
                    break
    finally:
        for handle in handles:
            handle.remove()

    if inner_input is None:
        raise RuntimeError("Failed to capture student inner-input at the requested step")
    if any(row is None for row in layer_outputs):
        raise RuntimeError("Failed to capture one or more student wrapper layer outputs")

    return inner_input, layer_outputs


@torch.no_grad()
def run_manual_bypass(student, teacher_input: torch.Tensor, use_streaming: bool):
    transformer = getattr(student, "transformer", None)
    if transformer is None:
        raise RuntimeError("Student model has no transformer")

    inner = getattr(transformer, "inner", None)
    input_proj = getattr(transformer, "input_proj", None)
    if inner is None or input_proj is None:
        raise RuntimeError("Student model is missing TemporalProjectedTransformer.inner/input_proj")

    layers = list(inner.layers)
    if not layers:
        raise RuntimeError("Student inner layer list is empty")

    # Row-vector convention: input_proj weight stores Q^T and forward computes x @ Q.
    w_in = input_proj.weight.detach().to(device=teacher_input.device, dtype=torch.float32)
    q_matrix = w_in.T.contiguous()

    x_rot_module = input_proj(teacher_input.to(dtype=input_proj.weight.dtype)).to(dtype=torch.float32)
    x_rot_manual = teacher_input @ q_matrix
    input_proj_max_abs = float((x_rot_module - x_rot_manual).abs().max().item())
    input_proj_mse = float(F.mse_loss(x_rot_module, x_rot_manual).item())

    bypass_layer_outputs = []
    if bool(use_streaming):
        with inner.streaming(batch_size=int(teacher_input.shape[0])):
            x_rot = x_rot_manual
            for layer in layers:
                out = layer(x_rot)
                y = _unwrap_tensor_output(out)
                if not torch.is_tensor(y):
                    raise RuntimeError("Bypass layer returned non-tensor output")
                x_rot = y.to(dtype=torch.float32)
                bypass_layer_outputs.append(x_rot.detach())
    else:
        x_rot = x_rot_manual
        for layer in layers:
            out = layer(x_rot)
            y = _unwrap_tensor_output(out)
            if not torch.is_tensor(y):
                raise RuntimeError("Bypass layer returned non-tensor output")
            x_rot = y.to(dtype=torch.float32)
            bypass_layer_outputs.append(x_rot.detach())

    return bypass_layer_outputs, w_in, input_proj_max_abs, input_proj_mse


@torch.no_grad()
def _capture_layer_with_stages(layer, x: torch.Tensor, use_streaming: bool):
    cache = {}
    handles = []

    def make_hook(key: str):
        def hook(_module, _inputs, output):
            y = _unwrap_tensor_output(output)
            if torch.is_tensor(y):
                cache[key] = y.detach().to(dtype=torch.float32)

        return hook

    if hasattr(layer, "norm1"):
        handles.append(layer.norm1.register_forward_hook(make_hook("norm1")))
    if hasattr(layer, "self_attn"):
        handles.append(layer.self_attn.register_forward_hook(make_hook("self_attn")))
    if hasattr(layer, "norm2"):
        handles.append(layer.norm2.register_forward_hook(make_hook("norm2")))

    gating = getattr(layer, "gating", None)
    if isinstance(gating, nn.Module) and not isinstance(gating, nn.ModuleList):
        handles.append(gating.register_forward_hook(make_hook("gating")))

    try:
        if bool(use_streaming):
            with layer.streaming(batch_size=int(x.shape[0])):
                out = layer(x)
        else:
            out = layer(x)
    finally:
        for handle in handles:
            handle.remove()

    y = _unwrap_tensor_output(out)
    if not torch.is_tensor(y):
        raise RuntimeError("Layer returned non-tensor output during isolated stage capture")
    return y.detach().to(dtype=torch.float32), cache


@torch.no_grad()
def run_isolated_layer_map(
    teacher_layers,
    student_layers,
    teacher_layer_inputs,
    unrotation_weight: torch.Tensor,
    use_streaming: bool,
):
    isolated_outputs = []
    isolated_stage_rows = []
    q_matrix = unrotation_weight.T.contiguous()

    for idx, (teacher_layer, student_layer, teacher_x) in enumerate(
        zip(teacher_layers, student_layers, teacher_layer_inputs)
    ):
        x_rot = teacher_x @ q_matrix
        student_y, student_cache = _capture_layer_with_stages(student_layer, x_rot, use_streaming)
        _, teacher_cache = _capture_layer_with_stages(teacher_layer, teacher_x, use_streaming)
        isolated_outputs.append(student_y)

        for stage_name in ("self_attn", "gating"):
            if stage_name not in teacher_cache or stage_name not in student_cache:
                continue

            t_stage = teacher_cache[stage_name]
            s_stage = student_cache[stage_name] @ unrotation_weight

            rows = min(int(t_stage.shape[0]), int(s_stage.shape[0]))
            steps = min(int(t_stage.shape[1]), int(s_stage.shape[1]))
            dims = min(int(t_stage.shape[2]), int(s_stage.shape[2]))
            if rows <= 0 or steps <= 0 or dims <= 0:
                continue

            t_vec = t_stage[:rows, :steps, :dims].reshape(-1)
            s_vec = s_stage[:rows, :steps, :dims].reshape(-1)
            diff = t_vec - s_vec

            isolated_stage_rows.append(
                {
                    "layer": int(idx),
                    "stage": stage_name,
                    "cos": float(F.cosine_similarity(t_vec, s_vec, dim=0).item()),
                    "max_abs": float(diff.abs().max().item()),
                    "mse": float(F.mse_loss(t_vec, s_vec).item()),
                }
            )

    return isolated_outputs, isolated_stage_rows


def compare_ladder(
    teacher_outputs,
    student_rot_outputs,
    student_unrotation: torch.Tensor,
    label: str,
    selected_layers: set[int] | None,
):
    print(f"\n=== {label} ===")
    for idx, (teacher_y, student_rot_y) in enumerate(zip(teacher_outputs, student_rot_outputs)):
        if selected_layers is not None and idx not in selected_layers:
            continue
        student_nat = student_rot_y @ student_unrotation
        t_vec = teacher_y.reshape(-1)
        s_vec = student_nat.reshape(-1)
        diff = t_vec - s_vec
        cos = float(F.cosine_similarity(t_vec, s_vec, dim=0).item())
        max_abs = float(diff.abs().max().item())
        mse = float(F.mse_loss(t_vec, s_vec).item())
        print(f"layer={idx:02d} cos={cos:.6f} max_abs={max_abs:.6f} mse={mse:.6f}")


def print_stage_rows(stage_rows, selected_layers: set[int] | None):
    print("\n=== ISOLATED STAGE LADDER (teacher_x -> student layer on teacher_x@Q) ===")
    if not stage_rows:
        print("No isolated stage rows captured.")
        return

    for row in stage_rows:
        if selected_layers is not None and int(row["layer"]) not in selected_layers:
            continue
        print(
            f"layer={int(row['layer']):02d} stage={row['stage']} "
            f"cos={row['cos']:.6f} max_abs={row['max_abs']:.6f} mse={row['mse']:.6f}"
        )


def parse_layers(raw: str, n_layers: int) -> set[int] | None:
    text = str(raw).strip().lower()
    if text in {"", "all", "*"}:
        return None

    out = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= int(n_layers):
            raise ValueError(f"Invalid layer index {idx}; valid range is [0, {n_layers - 1}]")
        out.add(idx)
    return out


def main():
    parser = argparse.ArgumentParser(description="Wrapper bypass probe for SliceGPT identity checkpoints")
    parser.add_argument("--bf16", default="v5_step1500.safetensors")
    parser.add_argument("--student", default="bmo_slicegpt_4096_identity_fp32.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runtime-patch",
        type=parse_bool,
        default=False,
        help="If true, import test_rtx_edge runtime monkey patches before loading models.",
    )
    parser.add_argument(
        "--teacher-dtype",
        default="float32",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--student-dtype",
        default="float32",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--step-idx", type=int, default=0)
    parser.add_argument("--layers", default="0,4,5,15,31", help="Comma-separated layer list or 'all'")
    parser.add_argument(
        "--streaming-mode",
        choices=["streaming", "non-streaming"],
        default="streaming",
        help="Execution mode for probes. streaming uses KV cache path; non-streaming bypasses cache state.",
    )
    parser.add_argument(
        "--compact-kv",
        type=parse_bool,
        default=None,
        help=(
            "Optional override for MOSHI_STREAMING_COMPACT_KV (true/false). "
            "If omitted, keeps current environment setting."
        ),
    )

    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--voice-prompt-wav", default="bmo_621.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--voice-ratio", type=float, default=0.25)
    args = parser.parse_args()

    if int(args.steps) <= 0:
        raise ValueError(f"steps must be > 0, got {args.steps}")
    if int(args.step_idx) < 0 or int(args.step_idx) >= int(args.steps):
        raise ValueError(f"step-idx must be in [0, {args.steps - 1}], got {args.step_idx}")
    if float(args.voice_ratio) < 0.0 or float(args.voice_ratio) >= 1.0:
        raise ValueError(f"voice-ratio must be in [0, 1), got {args.voice_ratio}")

    use_streaming = str(args.streaming_mode).strip().lower() == "streaming"

    if args.compact_kv is not None:
        os.environ["MOSHI_STREAMING_COMPACT_KV"] = "1" if bool(args.compact_kv) else "0"

    if bool(args.runtime_patch):
        import test_rtx_edge  # noqa: F401

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")

    teacher_dtype = parse_dtype(args.teacher_dtype)
    student_dtype = None if str(args.student_dtype).strip().lower() == "auto" else parse_dtype(args.student_dtype)

    print(f"[INFO] Loading teacher: {args.bf16}")
    teacher = loaders.get_moshi_lm(
        args.bf16,
        device=args.device,
        dtype=teacher_dtype,
        cpu_offload=False,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[INFO] Loading student: {args.student}")
    student = loaders.get_moshi_lm(
        args.student,
        device=args.device,
        dtype=student_dtype,
        cpu_offload=False,
    )
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    student_transformer = getattr(student, "transformer", None)
    student_inner = getattr(student_transformer, "inner", None)
    if student_inner is None:
        raise RuntimeError("Student transformer is not projected; expected transformer.inner to exist")

    teacher_layers = get_temporal_layers(teacher)
    student_layers = get_temporal_layers(student)
    if len(teacher_layers) != len(student_layers):
        raise RuntimeError(
            f"Layer count mismatch: teacher={len(teacher_layers)} student={len(student_layers)}"
        )
    selected_layers = parse_layers(args.layers, len(teacher_layers))

    print(
        "[INFO] Positional embedding modes: "
        f"teacher={getattr(teacher.transformer, 'positional_embedding', 'unknown')} "
        f"student_inner={getattr(student_inner, 'positional_embedding', 'unknown')}"
    )
    print(f"[INFO] probe_streaming_mode={str(args.streaming_mode).strip().lower()}")
    print(
        "[INFO] compact_kv_env="
        f"{os.environ.get('MOSHI_STREAMING_COMPACT_KV', '<unset>')}"
    )
    if len(teacher_layers) > 0 and len(student_layers) > 0:
        teacher_compact = getattr(teacher_layers[0].self_attn, "compact_kv_cache", None)
        student_compact = getattr(student_layers[0].self_attn, "compact_kv_cache", None)
        print(
            "[INFO] compact_kv_layer0: "
            f"teacher={teacher_compact} student={student_compact}"
        )

    forced_tokens = build_forced_tokens(
        teacher,
        int(args.steps),
        args.device,
        input_wav=args.input_wav,
        voice_prompt_wav=args.voice_prompt_wav,
        text_prompt=args.text_prompt,
        mimi_weight=args.mimi_weight,
        tokenizer_path=args.tokenizer,
        voice_ratio=float(args.voice_ratio),
    )

    teacher_input, teacher_layer_inputs, teacher_layer_outputs = capture_teacher_step(
        teacher,
        forced_tokens,
        int(args.step_idx),
        args.device,
        use_streaming,
    )

    wrapper_inner_input, wrapper_layer_outputs = capture_student_wrapper_step(
        student,
        forced_tokens,
        int(args.step_idx),
        args.device,
        use_streaming,
    )

    bypass_layer_outputs, unrotation, proj_max_abs, proj_mse = run_manual_bypass(
        student,
        teacher_input,
        use_streaming,
    )
    isolated_layer_outputs, isolated_stage_rows = run_isolated_layer_map(
        teacher_layers,
        student_layers,
        teacher_layer_inputs,
        unrotation,
        use_streaming,
    )

    wrapper_input_max_abs = float((wrapper_inner_input - (teacher_input @ unrotation.T)).abs().max().item())
    wrapper_input_mse = float(F.mse_loss(wrapper_inner_input, teacher_input @ unrotation.T).item())

    print("\n=== SANITY ===")
    print(f"input_proj_manual_vs_module_max_abs={proj_max_abs:.6e} mse={proj_mse:.6e}")
    print(
        "wrapper_inner_input_vs_teacher_rotated_max_abs="
        f"{wrapper_input_max_abs:.6e} mse={wrapper_input_mse:.6e}"
    )

    compare_ladder(
        teacher_layer_outputs,
        wrapper_layer_outputs,
        unrotation,
        label="WRAPPER PATH LADDER (student.forward_codes)",
        selected_layers=selected_layers,
    )

    compare_ladder(
        teacher_layer_outputs,
        bypass_layer_outputs,
        unrotation,
        label="MANUAL BYPASS LADDER (no TemporalProjectedTransformer.forward)",
        selected_layers=selected_layers,
    )

    compare_ladder(
        teacher_layer_outputs,
        isolated_layer_outputs,
        unrotation,
        label="ISOLATED LAYER MAP LADDER (each layer fed teacher input@Q)",
        selected_layers=selected_layers,
    )

    print_stage_rows(isolated_stage_rows, selected_layers)


if __name__ == "__main__":
    main()
