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


def _offset_from_state(state) -> int | None:
    if state is None:
        return None

    offset = getattr(state, "offset", None)
    if torch.is_tensor(offset):
        return int(offset.item())
    if offset is None:
        return None

    try:
        return int(offset)
    except Exception:
        return None


def _offset_from_module(module: nn.Module) -> int | None:
    return _offset_from_state(getattr(module, "_streaming_state", None))


def _apply_headwise_transform(
    values: torch.Tensor,
    q_matrix: torch.Tensor,
    *,
    use_transpose: bool,
) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"Expected values with shape [B, H, T, D], got {tuple(values.shape)}")

    bsz, n_heads, steps, head_dim = values.shape
    total_dim = int(n_heads * head_dim)
    if tuple(q_matrix.shape) != (total_dim, total_dim):
        raise ValueError(
            "Q shape mismatch for headwise transform: "
            f"q={tuple(q_matrix.shape)} expected={(total_dim, total_dim)}"
        )

    out = torch.empty_like(values)
    for h in range(int(n_heads)):
        start = int(h * head_dim)
        end = int(start + head_dim)
        q_block = q_matrix[start:end, start:end]
        if bool(use_transpose):
            q_block = q_block.T

        flat = values[:, h, :, :].reshape(-1, int(head_dim))
        out[:, h, :, :] = (flat @ q_block).reshape(int(bsz), int(steps), int(head_dim))

    return out


def _extract_latest_kv_from_attn(attn: nn.Module):
    state = getattr(attn, "_streaming_state", None)
    if state is None:
        raise RuntimeError("Attention streaming state is unavailable")

    kv_cache = getattr(state, "kv_cache", None)
    if kv_cache is None:
        raise RuntimeError("Attention streaming state has no kv_cache")

    cache = getattr(kv_cache, "cache", None)
    if not torch.is_tensor(cache):
        raise RuntimeError("kv_cache.cache is unavailable or not a tensor")
    if cache.ndim != 5 or int(cache.shape[0]) < 2:
        raise RuntimeError(f"Unexpected kv_cache.cache shape: {tuple(cache.shape)}")

    end_offset = getattr(kv_cache, "end_offset", None)
    if torch.is_tensor(end_offset):
        end_offset_int = int(end_offset.item())
    elif end_offset is not None:
        end_offset_int = int(end_offset)
    else:
        raise RuntimeError("kv_cache.end_offset is unavailable")

    capacity = int(cache.shape[3])
    if end_offset_int <= 0:
        raise RuntimeError("KV cache appears empty (end_offset <= 0)")

    latest_slot = int((end_offset_int - 1) % capacity)
    keys = cache[0, :, :, latest_slot : latest_slot + 1, :].detach().to(dtype=torch.float32).cpu().contiguous()
    values = cache[1, :, :, latest_slot : latest_slot + 1, :].detach().to(dtype=torch.float32).cpu().contiguous()
    return keys, values, end_offset_int, latest_slot, capacity


def _multi_linear_projection(weight: torch.Tensor, x: torch.Tensor, num_linear: int, offset_cpu: int) -> torch.Tensor:
    if int(num_linear) <= 0:
        raise ValueError(f"num_linear must be > 0, got {num_linear}")

    bsz, t_len, _ = x.shape
    out_ch, in_ch = weight.shape
    if int(out_ch) % int(num_linear) != 0:
        raise ValueError(
            "Invalid multi-linear weight shape: "
            f"weight={tuple(weight.shape)} num_linear={num_linear}"
        )

    weight_view = weight.view(int(num_linear), int(out_ch // num_linear), int(in_ch))
    ys = []
    for t in range(int(t_len)):
        idx = int(offset_cpu) + int(t)
        if idx < 0 or idx >= int(num_linear):
            raise RuntimeError(
                "multi-linear projection index out of bounds: "
                f"idx={idx} num_linear={num_linear} offset_cpu={offset_cpu} t={t}"
            )
        ys.append(F.linear(x[:, t], weight_view[idx]))
    return torch.stack(ys, dim=1)


def _compute_layer_qkv_snapshot(layer: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm1 = getattr(layer, "norm1", None)
    if not isinstance(norm1, nn.Module):
        raise RuntimeError("Layer has no norm1 for qkv snapshot")

    attn = getattr(layer, "self_attn", None)
    if not isinstance(attn, nn.Module):
        raise RuntimeError("Layer has no self_attn for qkv snapshot")

    x_norm = norm1(x)

    state = getattr(attn, "_streaming_state", None)
    offset_cpu = int(getattr(state, "offset_cpu", 0)) if state is not None else 0
    weights_per_step = int(getattr(attn, "weights_per_step", 0))

    if int(weights_per_step) > 0:
        projected = _multi_linear_projection(attn.in_proj_weight, x_norm, int(weights_per_step), int(offset_cpu))
    else:
        projected = F.linear(x_norm, attn.in_proj_weight)

    bsz, t_len, _ = projected.shape
    n_heads = int(attn.num_heads)
    head_dim = int(attn.embed_dim // attn.num_heads)

    qkv = projected.view(int(bsz), int(t_len), 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    offset = getattr(state, "offset", None)
    if not torch.is_tensor(offset):
        offset = torch.zeros((1,), device=x.device, dtype=torch.long)

    rope = getattr(attn, "rope", None)
    if rope is not None:
        q, k = rope(q, k, offset, time_before_heads=False)

    return (
        q.detach().to(dtype=torch.float32),
        k.detach().to(dtype=torch.float32),
        v.detach().to(dtype=torch.float32),
    )


def _compact_attn_used_kv(
    attn: nn.Module,
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    *,
    prior_cache_len: int,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = getattr(attn, "_streaming_state", None)
    if state is None:
        pos = torch.arange(k_current.shape[2], device=k_current.device, dtype=torch.long)
        return k_current, v_current, pos

    compact = bool(getattr(attn, "compact_kv_cache", False))
    if compact and int(prior_cache_len) == 0:
        pos = torch.arange(k_current.shape[2], device=k_current.device, dtype=torch.long)
        return k_current, v_current, pos

    k_used = keys
    v_used = values
    pos_k = positions

    if compact:
        valid = pos_k >= 0
        if bool(valid.any()):
            if not bool(valid.all()):
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                k_used = k_used.index_select(2, valid_idx)
                v_used = v_used.index_select(2, valid_idx)
                pos_k = pos_k.index_select(0, valid_idx)
        else:
            k_used = k_used[:, :, :1, :]
            v_used = v_used[:, :, :1, :]
            pos_k = torch.zeros((1,), device=k_used.device, dtype=torch.long)

    return k_used, v_used, pos_k


def _materialize_attn_used_kv(
    attn: nn.Module,
    k_current: torch.Tensor,
    v_current: torch.Tensor,
    *,
    prior_cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = getattr(attn, "_streaming_state", None)
    if state is None:
        pos = torch.arange(k_current.shape[2], device=k_current.device, dtype=torch.long)
        return k_current, v_current, pos

    kv_cache = getattr(state, "kv_cache", None)
    if kv_cache is None:
        raise RuntimeError("Attention state has no kv_cache during attn debug probe")

    cache = getattr(kv_cache, "cache", None)
    if not torch.is_tensor(cache):
        raise RuntimeError("kv_cache.cache is unavailable during attn debug probe")

    end_offset = getattr(kv_cache, "end_offset", None)
    if torch.is_tensor(end_offset):
        end_offset_int = int(end_offset.item())
    elif end_offset is not None:
        end_offset_int = int(end_offset)
    else:
        raise RuntimeError("kv_cache.end_offset is unavailable during attn debug probe")

    capacity = int(cache.shape[3])
    indexes = torch.arange(capacity, device=cache.device, dtype=torch.long)
    invalid = indexes >= end_offset_int

    end_index = int(end_offset_int % capacity)
    delta = indexes - end_index
    end_offset_tensor = torch.tensor(end_offset_int, device=cache.device, dtype=torch.long)
    positions = torch.where(
        delta <= 0,
        end_offset_tensor + delta,
        end_offset_tensor + delta - capacity,
    )
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    keys = cache[0]
    values = cache[1]

    return _compact_attn_used_kv(
        attn,
        keys,
        values,
        positions,
        prior_cache_len=int(prior_cache_len),
        k_current=k_current,
        v_current=v_current,
    )


def _build_attn_target_step_debug(
    attn: nn.Module,
    query_norm: torch.Tensor,
    offset: torch.Tensor,
    *,
    prior_cache_len: int,
    offset_cpu: int,
    runtime_capture: dict | None = None,
) -> dict:
    in_proj_weight = getattr(attn, "in_proj_weight", None)
    if not torch.is_tensor(in_proj_weight):
        raise RuntimeError("Attention has no in_proj_weight for attn debug probe")

    q_in = query_norm.to(device=in_proj_weight.device, dtype=in_proj_weight.dtype)
    offset_local = offset.to(device=q_in.device, dtype=torch.long)
    weights_per_step = int(getattr(attn, "weights_per_step", 0))
    offset_cpu_local = int(offset_cpu)

    if int(weights_per_step) > 0:
        projected = _multi_linear_projection(
            in_proj_weight,
            q_in,
            int(weights_per_step),
            int(offset_cpu_local),
        )
    else:
        projected = F.linear(q_in, in_proj_weight)
    bsz, t_len, _ = projected.shape
    n_heads = int(attn.num_heads)
    head_dim = int(attn.embed_dim // attn.num_heads)

    qkv = projected.view(int(bsz), int(t_len), 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
    q_current = qkv[0]
    k_current = qkv[1]
    v_current = qkv[2]

    rope = getattr(attn, "rope", None)
    if rope is not None:
        q_current, k_current = rope(q_current, k_current, offset_local, time_before_heads=False)

    if runtime_capture is not None and {
        "k_current_runtime",
        "v_current_runtime",
        "kv_keys_runtime",
        "kv_values_runtime",
        "kv_positions_runtime",
    }.issubset(set(runtime_capture.keys())):
        k_current_runtime = runtime_capture["k_current_runtime"].to(device=q_current.device, dtype=torch.float32)
        v_current_runtime = runtime_capture["v_current_runtime"].to(device=q_current.device, dtype=torch.float32)
        kv_keys_runtime = runtime_capture["kv_keys_runtime"].to(device=q_current.device, dtype=torch.float32)
        kv_values_runtime = runtime_capture["kv_values_runtime"].to(device=q_current.device, dtype=torch.float32)
        kv_positions_runtime = runtime_capture["kv_positions_runtime"].to(device=q_current.device, dtype=torch.long)
        prior_cache_runtime = int(runtime_capture.get("prior_cache_len_runtime", prior_cache_len))

        k_used, v_used, pos_k = _compact_attn_used_kv(
            attn,
            kv_keys_runtime,
            kv_values_runtime,
            kv_positions_runtime,
            prior_cache_len=int(prior_cache_runtime),
            k_current=k_current_runtime,
            v_current=v_current_runtime,
        )
        k_current = k_current_runtime
        v_current = v_current_runtime
        prior_cache_len = int(prior_cache_runtime)
    else:
        k_used, v_used, pos_k = _materialize_attn_used_kv(
            attn,
            k_current,
            v_current,
            prior_cache_len=int(prior_cache_len),
        )

    t_steps = int(q_current.shape[2])
    if bool(getattr(attn, "causal", False)):
        pos_k_row = pos_k.view(1, -1)
        pos_q = offset_local + torch.arange(t_steps, device=q_current.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k_row
        attn_bias = (pos_k_row >= 0) & (delta >= 0)
        context = getattr(attn, "context", None)
        if context is not None:
            attn_bias = attn_bias & (delta < int(context))
    else:
        attn_bias = torch.ones((t_steps, int(k_used.shape[2])), device=q_current.device, dtype=torch.bool)

    q32 = q_current.to(dtype=torch.float32)
    k32 = k_used.to(dtype=torch.float32)
    v32 = v_used.to(dtype=torch.float32)

    scale = float(int(q32.shape[-1]) ** -0.5)
    attn_scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale

    bias4 = attn_bias.view(1, 1, int(attn_bias.shape[0]), int(attn_bias.shape[1]))
    masked_scores = attn_scores.masked_fill(~bias4, -1.0e9)
    attn_weights = torch.softmax(masked_scores, dim=-1)
    valid_rows = bias4.any(dim=-1, keepdim=True)
    attn_weights = torch.where(valid_rows, attn_weights, torch.zeros_like(attn_weights))

    attn_heads = torch.matmul(attn_weights, v32)

    state = getattr(attn, "_streaming_state", None)
    kv_cache = getattr(state, "kv_cache", None) if state is not None else None
    post_cache_end = getattr(kv_cache, "end_offset", None) if kv_cache is not None else None
    if torch.is_tensor(post_cache_end):
        post_cache_len = int(post_cache_end.item())
    elif post_cache_end is not None:
        post_cache_len = int(post_cache_end)
    else:
        post_cache_len = None

    return {
        "q_current": q_current.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "k_current": k_current.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "v_current": v_current.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "k_used": k_used.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "v_used": v_used.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "pos_k": pos_k.detach().to(dtype=torch.long, device="cpu").contiguous(),
        "attn_bias": attn_bias.detach().to(dtype=torch.bool, device="cpu").contiguous(),
        "attn_scores": attn_scores.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "attn_weights": attn_weights.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "attn_heads": attn_heads.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "offset_start": int(offset_local.item()),
        "offset_cpu_start": int(offset_cpu_local),
        "weights_per_step": int(weights_per_step),
        "prior_cache_len": int(prior_cache_len),
        "post_cache_len": post_cache_len,
    }


@torch.no_grad()
def collect_streaming_offset_trace(
    teacher,
    student,
    forced_tokens: torch.Tensor,
    step_idx: int,
    device: str,
):
    teacher_layers = get_temporal_layers(teacher)
    student_layers = get_temporal_layers(student)
    n_layers = min(len(teacher_layers), len(student_layers))
    if n_layers <= 0:
        raise RuntimeError("Temporal layers are unavailable for offset trace")

    teacher_transformer = getattr(teacher, "transformer", None)
    student_transformer = getattr(student, "transformer", None)
    student_inner = getattr(student_transformer, "inner", None)

    trace = {
        "teacher_transformer": [],
        "student_outer": [],
        "student_inner": [],
        "teacher_mha": [],
        "student_mha": [],
    }

    teacher_mha_calls = [0 for _ in range(n_layers)]
    student_mha_calls = [0 for _ in range(n_layers)]
    handles = []

    def make_transformer_hook(trace_key: str):
        def hook(module, _inputs):
            trace[trace_key].append(
                {
                    "call": int(len(trace[trace_key])),
                    "offset_in": _offset_from_module(module),
                }
            )

        return hook

    def make_mha_hook(trace_key: str, call_counts: list[int], layer_idx: int):
        def hook(module, _inputs):
            state = getattr(module, "_streaming_state", None)
            kv_cache = getattr(state, "kv_cache", None) if state is not None else None
            end_offset = getattr(kv_cache, "end_offset", None) if kv_cache is not None else None
            if torch.is_tensor(end_offset):
                cache_end = int(end_offset.item())
            elif end_offset is not None:
                cache_end = int(end_offset)
            else:
                cache_end = None

            trace[trace_key].append(
                {
                    "layer": int(layer_idx),
                    "call": int(call_counts[layer_idx]),
                    "offset_in": _offset_from_module(module),
                    "cache_end": cache_end,
                }
            )
            call_counts[layer_idx] += 1

        return hook

    if isinstance(teacher_transformer, nn.Module):
        handles.append(teacher_transformer.register_forward_pre_hook(make_transformer_hook("teacher_transformer")))
    if isinstance(student_transformer, nn.Module):
        handles.append(student_transformer.register_forward_pre_hook(make_transformer_hook("student_outer")))
    if isinstance(student_inner, nn.Module):
        handles.append(student_inner.register_forward_pre_hook(make_transformer_hook("student_inner")))

    for idx in range(n_layers):
        teacher_attn = getattr(teacher_layers[idx], "self_attn", None)
        student_attn = getattr(student_layers[idx], "self_attn", None)
        if isinstance(teacher_attn, nn.Module):
            handles.append(
                teacher_attn.register_forward_pre_hook(
                    make_mha_hook("teacher_mha", teacher_mha_calls, idx)
                )
            )
        if isinstance(student_attn, nn.Module):
            handles.append(
                student_attn.register_forward_pre_hook(
                    make_mha_hook("student_mha", student_mha_calls, idx)
                )
            )

    k = int(teacher.num_codebooks)
    try:
        with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
            for t in range(int(step_idx) + 1):
                seq = forced_tokens[t].view(1, k, 1).to(device)
                teacher.forward_codes(seq)
                student.forward_codes(seq)
    finally:
        for handle in handles:
            handle.remove()

    return trace


def _fmt_opt_int(value: int | None) -> str:
    return "None" if value is None else str(int(value))


def _callable_origin(fn) -> str:
    if fn is None:
        return "<none>"

    target = fn.__func__ if hasattr(fn, "__func__") else fn
    module = getattr(target, "__module__", "<unknown>")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", "<callable>"))
    code = getattr(target, "__code__", None)
    if code is None:
        return f"{module}:{qualname}"
    return f"{module}:{qualname}@{code.co_filename}:{int(code.co_firstlineno)}"


def print_offset_trace(trace: dict, selected_layers: set[int] | None):
    print("\n=== STREAMING OFFSET TRACE ===")

    for row in trace.get("teacher_transformer", []):
        print(
            "[teacher-transformer] "
            f"call={int(row['call']):02d} offset_in={_fmt_opt_int(row['offset_in'])}"
        )
    for row in trace.get("student_outer", []):
        print(
            "[student-outer-transformer] "
            f"call={int(row['call']):02d} offset_in={_fmt_opt_int(row['offset_in'])}"
        )
    for row in trace.get("student_inner", []):
        print(
            "[student-inner-transformer] "
            f"call={int(row['call']):02d} offset_in={_fmt_opt_int(row['offset_in'])}"
        )

    print("\n--- teacher self_attn offsets ---")
    for row in trace.get("teacher_mha", []):
        if selected_layers is not None and int(row["layer"]) not in selected_layers:
            continue
        print(
            "[teacher-mha] "
            f"layer={int(row['layer']):02d} call={int(row['call']):02d} "
            f"offset_in={_fmt_opt_int(row['offset_in'])} cache_end={_fmt_opt_int(row['cache_end'])}"
        )

    print("\n--- student self_attn offsets ---")
    for row in trace.get("student_mha", []):
        if selected_layers is not None and int(row["layer"]) not in selected_layers:
            continue
        print(
            "[student-mha] "
            f"layer={int(row['layer']):02d} call={int(row['call']):02d} "
            f"offset_in={_fmt_opt_int(row['offset_in'])} cache_end={_fmt_opt_int(row['cache_end'])}"
        )


@torch.no_grad()
def run_kv_cache_basis_probe(
    teacher,
    student,
    forced_tokens: torch.Tensor,
    step_idx: int,
    layer_idx: int,
    device: str,
):
    teacher_layers = get_temporal_layers(teacher)
    student_layers = get_temporal_layers(student)
    n_layers = min(len(teacher_layers), len(student_layers))
    if n_layers <= 0:
        raise RuntimeError("Temporal layers are unavailable for KV cache probe")
    if int(layer_idx) < 0 or int(layer_idx) >= n_layers:
        raise ValueError(f"probe-kv-layer must be in [0, {n_layers - 1}], got {layer_idx}")
    if int(step_idx) < 0 or int(step_idx) >= int(forced_tokens.shape[0]):
        raise ValueError(
            f"probe-kv-step must be in [0, {int(forced_tokens.shape[0]) - 1}], got {step_idx}"
        )

    k = int(teacher.num_codebooks)
    with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
        for t in range(int(step_idx) + 1):
            seq = forced_tokens[t].view(1, k, 1).to(device)
            teacher.forward_codes(seq)
            student.forward_codes(seq)

        teacher_attn = getattr(teacher_layers[int(layer_idx)], "self_attn", None)
        student_attn = getattr(student_layers[int(layer_idx)], "self_attn", None)
        if not isinstance(teacher_attn, nn.Module) or not isinstance(student_attn, nn.Module):
            raise RuntimeError("Target layer has no self_attn module for KV cache probe")

        t_k, t_v, t_end, t_slot, t_cap = _extract_latest_kv_from_attn(teacher_attn)
        s_k, s_v, s_end, s_slot, s_cap = _extract_latest_kv_from_attn(student_attn)

    print("\n=== KV CACHE BASIS PROBE ===")
    print(f"layer={int(layer_idx):02d} probe_step={int(step_idx)}")
    print(
        "teacher_cache: "
        f"end_offset={int(t_end)} latest_slot={int(t_slot)} capacity={int(t_cap)}"
    )
    print(
        "student_cache: "
        f"end_offset={int(s_end)} latest_slot={int(s_slot)} capacity={int(s_cap)}"
    )

    def pair_metrics(a: torch.Tensor, b: torch.Tensor):
        return float((a - b).abs().max().item()), float(F.mse_loss(a, b).item())

    k_max_abs, k_mse = pair_metrics(t_k, s_k)
    v_max_abs, v_mse = pair_metrics(t_v, s_v)
    print(f"K direct: max_abs={k_max_abs:.6e} mse={k_mse:.6e}")
    print(f"V direct: max_abs={v_max_abs:.6e} mse={v_mse:.6e}")

    student_transformer = getattr(student, "transformer", None)
    input_proj = getattr(student_transformer, "input_proj", None)
    if input_proj is None or not hasattr(input_proj, "weight"):
        print("[WARN] KV basis check skipped: student.input_proj is unavailable")
        return

    q_matrix = input_proj.weight.detach().to(dtype=torch.float32).cpu().T.contiguous()
    total_dim = int(t_k.shape[1] * t_k.shape[3])
    if tuple(q_matrix.shape) != (total_dim, total_dim):
        print(
            "[WARN] KV basis check skipped: unexpected Q shape for headwise transform "
            f"q={tuple(q_matrix.shape)} expected={(total_dim, total_dim)}"
        )
        return

    s_k_qt = _apply_headwise_transform(s_k, q_matrix, use_transpose=True)
    s_k_q = _apply_headwise_transform(s_k, q_matrix, use_transpose=False)
    s_v_qt = _apply_headwise_transform(s_v, q_matrix, use_transpose=True)
    s_v_q = _apply_headwise_transform(s_v, q_matrix, use_transpose=False)

    k_qt_max_abs, k_qt_mse = pair_metrics(t_k, s_k_qt)
    k_q_max_abs, k_q_mse = pair_metrics(t_k, s_k_q)
    v_qt_max_abs, v_qt_mse = pair_metrics(t_v, s_v_qt)
    v_q_max_abs, v_q_mse = pair_metrics(t_v, s_v_q)

    print(f"K student@Q^T vs teacher: max_abs={k_qt_max_abs:.6e} mse={k_qt_mse:.6e}")
    print(f"K student@Q   vs teacher: max_abs={k_q_max_abs:.6e} mse={k_q_mse:.6e}")
    print(f"V student@Q^T vs teacher: max_abs={v_qt_max_abs:.6e} mse={v_qt_mse:.6e}")
    print(f"V student@Q   vs teacher: max_abs={v_q_max_abs:.6e} mse={v_q_mse:.6e}")


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
    teacher_step_inputs = []
    layer_inputs = [None for _ in range(len(layers))]
    layer_outputs = [None for _ in range(len(layers))]

    def pre_hook(_module, inputs):
        nonlocal teacher_input
        if inputs and torch.is_tensor(inputs[0]):
            captured = inputs[0].detach().to(dtype=torch.float32)
            teacher_input = captured
            teacher_step_inputs.append(captured)

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
    if len(teacher_step_inputs) <= int(step_idx):
        raise RuntimeError(
            "Failed to capture teacher transformer input history up to the requested step"
        )
    if any(row is None for row in layer_inputs):
        raise RuntimeError("Failed to capture one or more teacher layer inputs")
    if any(row is None for row in layer_outputs):
        raise RuntimeError("Failed to capture one or more teacher layer outputs")

    return teacher_input, layer_inputs, layer_outputs, teacher_step_inputs


@torch.no_grad()
def capture_teacher_layer_input_history(
    teacher,
    forced_tokens: torch.Tensor,
    step_idx: int,
    device: str,
    use_streaming: bool,
):
    layers = get_temporal_layers(teacher)
    if not layers:
        raise RuntimeError("Teacher temporal layers are unavailable")

    max_step = int(step_idx)
    if max_step < 0:
        raise ValueError(f"Invalid step_idx={step_idx}; expected >=0")

    history = [[None for _ in range(len(layers))] for _ in range(max_step + 1)]
    active_step = -1

    def make_pre_hook(idx):
        def hook(_module, inputs):
            nonlocal active_step
            if (
                active_step >= 0
                and active_step < len(history)
                and inputs
                and torch.is_tensor(inputs[0])
            ):
                history[active_step][idx] = inputs[0].detach().to(dtype=torch.float32)

        return hook

    handles = []
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(idx)))

    k = int(teacher.num_codebooks)
    try:
        if bool(use_streaming):
            with teacher.streaming(batch_size=1):
                for t in range(max_step + 1):
                    active_step = int(t)
                    seq = forced_tokens[t].view(1, k, 1).to(device)
                    teacher.forward_codes(seq)
        else:
            for t in range(max_step + 1):
                active_step = int(t)
                seq = forced_tokens[t].view(1, k, 1).to(device)
                teacher.forward_codes(seq)
    finally:
        for handle in handles:
            handle.remove()

    for t in range(max_step + 1):
        if any(row is None for row in history[t]):
            raise RuntimeError(f"Failed to capture one or more teacher layer inputs at step={t}")

    return history


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
def run_manual_bypass(
    student,
    teacher_step_inputs: list[torch.Tensor],
    step_idx: int,
    use_streaming: bool,
    replay_history: bool,
):
    if not teacher_step_inputs:
        raise RuntimeError("teacher_step_inputs is empty")
    if int(step_idx) < 0 or int(step_idx) >= len(teacher_step_inputs):
        raise ValueError(
            f"Invalid step_idx={step_idx}; expected [0, {len(teacher_step_inputs) - 1}]"
        )

    teacher_input = teacher_step_inputs[int(step_idx)]

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
            if bool(replay_history):
                step_indices = range(int(step_idx) + 1)
            else:
                step_indices = [int(step_idx)]

            for t in step_indices:
                x_rot = teacher_step_inputs[int(t)] @ q_matrix
                cur_outputs = []
                for layer in layers:
                    out = layer(x_rot)
                    y = _unwrap_tensor_output(out)
                    if not torch.is_tensor(y):
                        raise RuntimeError("Bypass layer returned non-tensor output")
                    x_rot = y.to(dtype=torch.float32)
                    cur_outputs.append(x_rot.detach())

                if int(t) == int(step_idx):
                    bypass_layer_outputs = cur_outputs

            if not bypass_layer_outputs:
                raise RuntimeError("Failed to capture manual bypass outputs at requested step")
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


@torch.no_grad()
def run_single_layer_history_probe(
    teacher_layers,
    student_layers,
    teacher_layer_input_history,
    unrotation_weight: torch.Tensor,
    *,
    layer_idx: int,
    step_idx: int,
    use_streaming: bool,
    probe_qkv: bool,
    probe_attn_debug: bool,
):
    if not bool(use_streaming):
        raise RuntimeError("single-layer-history probe requires --streaming-mode=streaming")

    n_layers = min(len(teacher_layers), len(student_layers))
    if int(layer_idx) < 0 or int(layer_idx) >= n_layers:
        raise ValueError(f"probe-layer-history-layer must be in [0, {n_layers - 1}], got {layer_idx}")
    if int(step_idx) < 0 or int(step_idx) >= len(teacher_layer_input_history):
        raise ValueError(
            f"probe-layer-history-step must be in [0, {len(teacher_layer_input_history) - 1}], got {step_idx}"
        )

    teacher_inputs = [teacher_layer_input_history[t][int(layer_idx)] for t in range(int(step_idx) + 1)]
    if any(x is None for x in teacher_inputs):
        raise RuntimeError("Teacher layer-input history is incomplete for single-layer probe")

    teacher_layer = teacher_layers[int(layer_idx)]
    student_layer = student_layers[int(layer_idx)]

    probe_device = teacher_inputs[0].device
    unrotation_local = unrotation_weight.to(device=probe_device, dtype=torch.float32).contiguous()
    q_matrix = unrotation_local.T.contiguous()
    q_matrix_cpu = q_matrix.detach().to(dtype=torch.float32, device="cpu").contiguous()
    target_step = int(step_idx)
    active_step = -1

    teacher_stage = {}
    student_stage = {}
    handles = []

    def make_stage_hook(cache_ref, stage_key):
        def hook(_module, _inputs, output):
            if int(active_step) != target_step:
                return
            y = _unwrap_tensor_output(output)
            if torch.is_tensor(y):
                cache_ref[stage_key] = y.detach().to(dtype=torch.float32)

        return hook

    teacher_attn = getattr(teacher_layer, "self_attn", None)
    student_attn = getattr(student_layer, "self_attn", None)
    teacher_gating = getattr(teacher_layer, "gating", None)
    student_gating = getattr(student_layer, "gating", None)

    teacher_forward_origin = _callable_origin(getattr(teacher_attn, "forward", None))
    student_forward_origin = _callable_origin(getattr(student_attn, "forward", None))
    teacher_init_state_origin = _callable_origin(getattr(teacher_attn, "_init_streaming_state", None))
    student_init_state_origin = _callable_origin(getattr(student_attn, "_init_streaming_state", None))

    teacher_attn_pre = {}
    student_attn_pre = {}
    teacher_attn_runtime = {}
    student_attn_runtime = {}
    teacher_attn_complete_kv_orig = None
    student_attn_complete_kv_orig = None

    if isinstance(teacher_attn, nn.Module):
        handles.append(teacher_attn.register_forward_hook(make_stage_hook(teacher_stage, "self_attn")))
    if isinstance(student_attn, nn.Module):
        handles.append(student_attn.register_forward_hook(make_stage_hook(student_stage, "self_attn")))
    if isinstance(teacher_gating, nn.Module) and not isinstance(teacher_gating, nn.ModuleList):
        handles.append(teacher_gating.register_forward_hook(make_stage_hook(teacher_stage, "gating")))
    if isinstance(student_gating, nn.Module) and not isinstance(student_gating, nn.ModuleList):
        handles.append(student_gating.register_forward_hook(make_stage_hook(student_stage, "gating")))

    def make_attn_pre_hook(cache_ref: dict):
        def hook(module, inputs):
            if int(active_step) != target_step:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                return

            query_norm = inputs[0]
            state = getattr(module, "_streaming_state", None)
            kv_cache = getattr(state, "kv_cache", None) if state is not None else None
            cache_end = getattr(kv_cache, "end_offset", None) if kv_cache is not None else None
            if torch.is_tensor(cache_end):
                prior_cache_len = int(cache_end.item())
            elif cache_end is not None:
                prior_cache_len = int(cache_end)
            else:
                prior_cache_len = 0

            offset = getattr(state, "offset", None)
            if not torch.is_tensor(offset):
                offset = torch.zeros((1,), device=query_norm.device, dtype=torch.long)

            offset_cpu = 0
            if state is not None:
                offset_cpu = int(getattr(state, "offset_cpu", 0))

            cache_ref["query_norm"] = query_norm.detach()
            cache_ref["offset"] = offset.detach().clone()
            cache_ref["prior_cache_len"] = int(prior_cache_len)
            cache_ref["offset_cpu"] = int(offset_cpu)

        return hook

    def install_complete_kv_capture(attn_module: nn.Module, cache_ref: dict):
        original_complete_kv = getattr(attn_module, "_complete_kv", None)
        if original_complete_kv is None:
            raise RuntimeError("Target attention module has no _complete_kv for runtime capture")

        def wrapped_complete_kv(k, v):
            state = getattr(attn_module, "_streaming_state", None)
            kv_cache = getattr(state, "kv_cache", None) if state is not None else None
            cache_end = getattr(kv_cache, "end_offset", None) if kv_cache is not None else None
            if torch.is_tensor(cache_end):
                prior_cache_len_runtime = int(cache_end.item())
            elif cache_end is not None:
                prior_cache_len_runtime = int(cache_end)
            else:
                prior_cache_len_runtime = 0

            offset_cpu_runtime = int(getattr(state, "offset_cpu", 0)) if state is not None else 0
            kv_cache_class = (
                f"{type(kv_cache).__module__}.{type(kv_cache).__name__}"
                if kv_cache is not None
                else "<none>"
            )
            cache_buffer = getattr(kv_cache, "cache", None) if kv_cache is not None else None
            cache_buffer_dtype = str(cache_buffer.dtype) if torch.is_tensor(cache_buffer) else "<none>"

            step_key = int(active_step)
            should_capture_step = int(0) <= step_key <= int(target_step)
            step_entry = None
            if should_capture_step:
                by_step = cache_ref.setdefault("by_step", {})
                step_entry = by_step.setdefault(step_key, {})
                step_entry["prior_cache_len_runtime"] = int(prior_cache_len_runtime)
                step_entry["offset_cpu_runtime"] = int(offset_cpu_runtime)
                step_entry["weights_per_step_runtime"] = int(getattr(attn_module, "weights_per_step", 0))
                step_entry["kv_cache_class_runtime"] = kv_cache_class
                step_entry["kv_cache_buffer_dtype_runtime"] = cache_buffer_dtype
                step_entry["k_current_runtime"] = k.detach().to(dtype=torch.float32, device="cpu").contiguous()
                step_entry["v_current_runtime"] = v.detach().to(dtype=torch.float32, device="cpu").contiguous()

            if int(active_step) == target_step:
                cache_ref["prior_cache_len_runtime"] = int(prior_cache_len_runtime)
                cache_ref["offset_cpu_runtime"] = int(offset_cpu_runtime)
                cache_ref["weights_per_step_runtime"] = int(getattr(attn_module, "weights_per_step", 0))
                cache_ref["k_current_runtime"] = k.detach().to(dtype=torch.float32, device="cpu").contiguous()
                cache_ref["v_current_runtime"] = v.detach().to(dtype=torch.float32, device="cpu").contiguous()

            result = original_complete_kv(k, v)

            if step_entry is not None:
                step_entry["kv_keys_runtime"] = result.keys.detach().to(dtype=torch.float32, device="cpu").contiguous()
                step_entry["kv_values_runtime"] = result.values.detach().to(dtype=torch.float32, device="cpu").contiguous()
                step_entry["kv_positions_runtime"] = result.positions.detach().to(dtype=torch.long, device="cpu").contiguous()

                seq_len = int(k.shape[2])
                capacity = int(result.keys.shape[2])
                write_indexes = (
                    torch.arange(seq_len, device=result.keys.device, dtype=torch.long)
                    + int(prior_cache_len_runtime)
                ) % int(capacity)
                write_positions = result.positions.index_select(0, write_indexes)
                k_written = result.keys.index_select(2, write_indexes)
                v_written = result.values.index_select(2, write_indexes)

                k_vec = k.reshape(-1).to(dtype=torch.float32)
                k_written_vec = k_written.reshape(-1).to(dtype=torch.float32)
                v_vec = v.reshape(-1).to(dtype=torch.float32)
                v_written_vec = v_written.reshape(-1).to(dtype=torch.float32)
                k_diff = k_vec - k_written_vec
                v_diff = v_vec - v_written_vec

                step_entry["cache_dtype_runtime"] = str(result.keys.dtype)
                step_entry["k_input_dtype_runtime"] = str(k.dtype)
                step_entry["v_input_dtype_runtime"] = str(v.dtype)
                step_entry["k_write_self_cos_runtime"] = float(F.cosine_similarity(k_vec, k_written_vec, dim=0).item())
                step_entry["k_write_self_max_abs_runtime"] = float(k_diff.abs().max().item())
                step_entry["k_write_self_mse_runtime"] = float(F.mse_loss(k_vec, k_written_vec).item())
                step_entry["v_write_self_cos_runtime"] = float(F.cosine_similarity(v_vec, v_written_vec, dim=0).item())
                step_entry["v_write_self_max_abs_runtime"] = float(v_diff.abs().max().item())
                step_entry["v_write_self_mse_runtime"] = float(F.mse_loss(v_vec, v_written_vec).item())

                step_entry["write_indexes_runtime"] = write_indexes.detach().to(dtype=torch.long, device="cpu").contiguous()
                step_entry["write_positions_runtime"] = write_positions.detach().to(dtype=torch.long, device="cpu").contiguous()
                step_entry["k_written_runtime"] = k_written.detach().to(dtype=torch.float32, device="cpu").contiguous()
                step_entry["v_written_runtime"] = v_written.detach().to(dtype=torch.float32, device="cpu").contiguous()

            if int(active_step) == target_step:
                cache_ref["kv_keys_runtime"] = result.keys.detach().to(dtype=torch.float32, device="cpu").contiguous()
                cache_ref["kv_values_runtime"] = result.values.detach().to(dtype=torch.float32, device="cpu").contiguous()
                cache_ref["kv_positions_runtime"] = result.positions.detach().to(dtype=torch.long, device="cpu").contiguous()

            return result

        setattr(attn_module, "_complete_kv", wrapped_complete_kv)
        return original_complete_kv

    if bool(probe_attn_debug):
        if not isinstance(teacher_attn, nn.Module) or not isinstance(student_attn, nn.Module):
            raise RuntimeError("probe-layer-history-attn-debug requires self_attn modules")
        handles.append(teacher_attn.register_forward_pre_hook(make_attn_pre_hook(teacher_attn_pre)))
        handles.append(student_attn.register_forward_pre_hook(make_attn_pre_hook(student_attn_pre)))
        teacher_attn_complete_kv_orig = install_complete_kv_capture(teacher_attn, teacher_attn_runtime)
        student_attn_complete_kv_orig = install_complete_kv_capture(student_attn, student_attn_runtime)

    teacher_out = None
    student_out = None
    teacher_qkv = None
    student_qkv = None
    teacher_attn_debug = None
    student_attn_debug = None

    try:
        with teacher_layer.streaming(batch_size=1), student_layer.streaming(batch_size=1):
            for t in range(target_step + 1):
                active_step = int(t)
                teacher_x = teacher_inputs[t]
                student_x = teacher_x @ q_matrix

                if bool(probe_qkv) and int(t) == target_step:
                    teacher_qkv = _compute_layer_qkv_snapshot(teacher_layer, teacher_x)
                    student_qkv = _compute_layer_qkv_snapshot(student_layer, student_x)

                t_out = teacher_layer(teacher_x)
                s_out = student_layer(student_x)

                t_y = _unwrap_tensor_output(t_out)
                s_y = _unwrap_tensor_output(s_out)
                if not torch.is_tensor(t_y) or not torch.is_tensor(s_y):
                    raise RuntimeError("Single-layer history probe encountered non-tensor layer output")

                if int(t) == target_step:
                    teacher_out = t_y.detach().to(dtype=torch.float32)
                    student_out = s_y.detach().to(dtype=torch.float32)

                    if bool(probe_attn_debug):
                        if (
                            "query_norm" not in teacher_attn_pre
                            or "offset" not in teacher_attn_pre
                            or "prior_cache_len" not in teacher_attn_pre
                        ):
                            raise RuntimeError("Teacher attn pre-hook did not capture target-step inputs")
                        if (
                            "query_norm" not in student_attn_pre
                            or "offset" not in student_attn_pre
                            or "prior_cache_len" not in student_attn_pre
                        ):
                            raise RuntimeError("Student attn pre-hook did not capture target-step inputs")

                        teacher_attn_debug = _build_attn_target_step_debug(
                            teacher_attn,
                            teacher_attn_pre["query_norm"],
                            teacher_attn_pre["offset"],
                            prior_cache_len=int(teacher_attn_pre["prior_cache_len"]),
                            offset_cpu=int(teacher_attn_pre["offset_cpu"]),
                            runtime_capture=teacher_attn_runtime,
                        )
                        student_attn_debug = _build_attn_target_step_debug(
                            student_attn,
                            student_attn_pre["query_norm"],
                            student_attn_pre["offset"],
                            prior_cache_len=int(student_attn_pre["prior_cache_len"]),
                            offset_cpu=int(student_attn_pre["offset_cpu"]),
                            runtime_capture=student_attn_runtime,
                        )
    finally:
        if bool(probe_attn_debug):
            if isinstance(teacher_attn, nn.Module) and teacher_attn_complete_kv_orig is not None:
                setattr(teacher_attn, "_complete_kv", teacher_attn_complete_kv_orig)
            if isinstance(student_attn, nn.Module) and student_attn_complete_kv_orig is not None:
                setattr(student_attn, "_complete_kv", student_attn_complete_kv_orig)
        for handle in handles:
            handle.remove()

    if teacher_out is None or student_out is None:
        raise RuntimeError("Failed to capture single-layer history probe outputs at requested step")

    student_out_nat = student_out @ unrotation_local
    t_vec = teacher_out.reshape(-1)
    s_vec = student_out_nat.reshape(-1)
    diff = t_vec - s_vec

    print("\n=== SINGLE LAYER HISTORY PROBE ===")
    print(f"layer={int(layer_idx):02d} target_step={target_step}")
    print(
        "layer_output "
        f"cos={float(F.cosine_similarity(t_vec, s_vec, dim=0).item()):.6f} "
        f"max_abs={float(diff.abs().max().item()):.6f} "
        f"mse={float(F.mse_loss(t_vec, s_vec).item()):.6f}"
    )

    for stage_name in ("self_attn", "gating"):
        if stage_name not in teacher_stage or stage_name not in student_stage:
            continue

        t_stage = teacher_stage[stage_name]
        s_stage_nat = student_stage[stage_name] @ unrotation_local
        t_stage_vec = t_stage.reshape(-1)
        s_stage_vec = s_stage_nat.reshape(-1)
        d_stage = t_stage_vec - s_stage_vec

        print(
            f"{stage_name} "
            f"cos={float(F.cosine_similarity(t_stage_vec, s_stage_vec, dim=0).item()):.6f} "
            f"max_abs={float(d_stage.abs().max().item()):.6f} "
            f"mse={float(F.mse_loss(t_stage_vec, s_stage_vec).item()):.6f}"
        )

    if bool(probe_qkv):
        if teacher_qkv is None or student_qkv is None:
            raise RuntimeError("qkv diagnostics requested but qkv snapshots were not captured")

        print("\n--- qkv basis diagnostics (target step) ---")

        def pair_metrics(a: torch.Tensor, b: torch.Tensor):
            return float((a - b).abs().max().item()), float(F.mse_loss(a, b).item())

        for name, t_tensor, s_tensor in (
            ("q", teacher_qkv[0], student_qkv[0]),
            ("k", teacher_qkv[1], student_qkv[1]),
            ("v", teacher_qkv[2], student_qkv[2]),
        ):
            t_cpu = t_tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
            s_cpu = s_tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()

            direct_max_abs, direct_mse = pair_metrics(t_cpu, s_cpu)
            s_qt = _apply_headwise_transform(s_cpu, q_matrix_cpu, use_transpose=True)
            s_q = _apply_headwise_transform(s_cpu, q_matrix_cpu, use_transpose=False)
            qt_max_abs, qt_mse = pair_metrics(t_cpu, s_qt)
            q_max_abs, q_mse = pair_metrics(t_cpu, s_q)

            print(
                f"{name} direct: max_abs={direct_max_abs:.6e} mse={direct_mse:.6e}"
            )
            print(
                f"{name} student@Q^T vs teacher: max_abs={qt_max_abs:.6e} mse={qt_mse:.6e}"
            )
            print(
                f"{name} student@Q   vs teacher: max_abs={q_max_abs:.6e} mse={q_mse:.6e}"
            )

    if bool(probe_attn_debug):
        if teacher_attn_debug is None or student_attn_debug is None:
            raise RuntimeError("attn diagnostics requested but target-step attn snapshots were not captured")

        def tensor_metrics(a: torch.Tensor, b: torch.Tensor):
            a_vec = a.reshape(-1)
            b_vec = b.reshape(-1)
            d_vec = a_vec - b_vec
            return (
                float(F.cosine_similarity(a_vec, b_vec, dim=0).item()),
                float(d_vec.abs().max().item()),
                float(F.mse_loss(a_vec, b_vec).item()),
            )

        def print_metric_line(prefix: str, a: torch.Tensor, b: torch.Tensor):
            cos, max_abs, mse = tensor_metrics(a, b)
            print(f"{prefix} cos={cos:.6f} max_abs={max_abs:.6e} mse={mse:.6e}")

        print("\n--- attention recurrence diagnostics (target step) ---")
        print(
            "runtime_origins "
            f"teacher_forward={teacher_forward_origin} "
            f"teacher_init_state={teacher_init_state_origin}"
        )
        print(
            "runtime_origins "
            f"student_forward={student_forward_origin} "
            f"student_init_state={student_init_state_origin}"
        )
        print(
            "cache_lens "
            f"offset={int(teacher_attn_debug['offset_start'])} "
            f"offset_cpu={int(teacher_attn_debug['offset_cpu_start'])} "
            f"weights_per_step={int(teacher_attn_debug['weights_per_step'])} "
            f"teacher_prior={int(teacher_attn_debug['prior_cache_len'])} "
            f"teacher_post={_fmt_opt_int(teacher_attn_debug['post_cache_len'])} "
            f"student_prior={int(student_attn_debug['prior_cache_len'])} "
            f"student_post={_fmt_opt_int(student_attn_debug['post_cache_len'])}"
        )

        t_pos = teacher_attn_debug["pos_k"]
        s_pos = student_attn_debug["pos_k"]
        pos_shape_match = tuple(t_pos.shape) == tuple(s_pos.shape)
        if pos_shape_match:
            pos_diff = t_pos - s_pos
            pos_max_abs = int(pos_diff.abs().max().item()) if t_pos.numel() > 0 else 0
            pos_mismatch = int((pos_diff != 0).sum().item())
            pos_total = int(t_pos.numel())
            print(
                "pos_k "
                f"shape={tuple(t_pos.shape)} max_abs={pos_max_abs} mismatches={pos_mismatch}/{pos_total}"
            )
        else:
            print(
                "pos_k "
                f"shape_mismatch teacher={tuple(t_pos.shape)} student={tuple(s_pos.shape)}"
            )

        t_bias = teacher_attn_debug["attn_bias"]
        s_bias = student_attn_debug["attn_bias"]
        bias_shape_match = tuple(t_bias.shape) == tuple(s_bias.shape)
        if bias_shape_match:
            bias_mismatch = int((t_bias != s_bias).sum().item())
            bias_total = int(t_bias.numel())
            print(f"attn_bias mismatches={bias_mismatch}/{bias_total}")
        else:
            print(
                "attn_bias "
                f"shape_mismatch teacher={tuple(t_bias.shape)} student={tuple(s_bias.shape)}"
            )

        for name in (
            "q_current",
            "k_current",
            "v_current",
            "k_used",
            "v_used",
            "attn_scores",
            "attn_weights",
            "attn_heads",
        ):
            t_tensor = teacher_attn_debug[name]
            s_tensor = student_attn_debug[name]

            if tuple(t_tensor.shape) != tuple(s_tensor.shape):
                print(
                    f"{name} shape_mismatch teacher={tuple(t_tensor.shape)} student={tuple(s_tensor.shape)}"
                )
                continue

            print_metric_line(name, t_tensor, s_tensor)

            if name in {"k_used", "v_used"}:
                s_qt = _apply_headwise_transform(s_tensor, q_matrix_cpu, use_transpose=True)
                s_q = _apply_headwise_transform(s_tensor, q_matrix_cpu, use_transpose=False)
                print_metric_line(f"{name} student@Q^T", t_tensor, s_qt)
                print_metric_line(f"{name} student@Q", t_tensor, s_q)

        if pos_shape_match:
            t_used_k = teacher_attn_debug["k_used"]
            s_used_k = student_attn_debug["k_used"]
            t_used_v = teacher_attn_debug["v_used"]
            s_used_v = student_attn_debug["v_used"]

            t_pos_list = [int(x) for x in t_pos.tolist()]
            s_pos_list = [int(x) for x in s_pos.tolist()]
            common_positions = sorted(set(t_pos_list).intersection(s_pos_list))

            print("\ncache position diagnostics:")
            max_positions = 8
            for idx_pos, pos in enumerate(common_positions):
                if idx_pos >= max_positions:
                    print(f"... truncated to first {max_positions} positions")
                    break

                t_idx = t_pos_list.index(int(pos))
                s_idx = s_pos_list.index(int(pos))

                t_k_slice = t_used_k[:, :, t_idx : t_idx + 1, :]
                s_k_slice = s_used_k[:, :, s_idx : s_idx + 1, :]
                t_v_slice = t_used_v[:, :, t_idx : t_idx + 1, :]
                s_v_slice = s_used_v[:, :, s_idx : s_idx + 1, :]

                print_metric_line(f"pos={pos} k_used", t_k_slice, s_k_slice)
                print_metric_line(
                    f"pos={pos} k_used student@Q^T",
                    t_k_slice,
                    _apply_headwise_transform(s_k_slice, q_matrix_cpu, use_transpose=True),
                )
                print_metric_line(
                    f"pos={pos} k_used student@Q",
                    t_k_slice,
                    _apply_headwise_transform(s_k_slice, q_matrix_cpu, use_transpose=False),
                )

                print_metric_line(f"pos={pos} v_used", t_v_slice, s_v_slice)
                print_metric_line(
                    f"pos={pos} v_used student@Q^T",
                    t_v_slice,
                    _apply_headwise_transform(s_v_slice, q_matrix_cpu, use_transpose=True),
                )
                print_metric_line(
                    f"pos={pos} v_used student@Q",
                    t_v_slice,
                    _apply_headwise_transform(s_v_slice, q_matrix_cpu, use_transpose=False),
                )

            target_pos = int(teacher_attn_debug["offset_start"])
            if target_pos in t_pos_list and target_pos in s_pos_list:
                t_idx = t_pos_list.index(target_pos)
                s_idx = s_pos_list.index(target_pos)
                print("\ncurrent-position cache vs current-token diagnostics:")
                print_metric_line(
                    "teacher k_used[pos=offset] vs k_current",
                    t_used_k[:, :, t_idx : t_idx + 1, :],
                    teacher_attn_debug["k_current"],
                )
                print_metric_line(
                    "student k_used[pos=offset] vs k_current",
                    s_used_k[:, :, s_idx : s_idx + 1, :],
                    student_attn_debug["k_current"],
                )
                print_metric_line(
                    "teacher v_used[pos=offset] vs v_current",
                    t_used_v[:, :, t_idx : t_idx + 1, :],
                    teacher_attn_debug["v_current"],
                )
                print_metric_line(
                    "student v_used[pos=offset] vs v_current",
                    s_used_v[:, :, s_idx : s_idx + 1, :],
                    student_attn_debug["v_current"],
                )

            teacher_by_step = teacher_attn_runtime.get("by_step", {})
            student_by_step = student_attn_runtime.get("by_step", {})
            if teacher_by_step and student_by_step:
                print("\nstep-to-cache persistence diagnostics:")
                step_keys = sorted(
                    set(int(k) for k in teacher_by_step.keys()).intersection(
                        int(k) for k in student_by_step.keys()
                    )
                )
                max_steps = 8
                emitted = 0
                for step in step_keys:
                    if emitted >= max_steps:
                        print(f"... truncated to first {max_steps} steps")
                        break
                    if step not in t_pos_list or step not in s_pos_list:
                        continue

                    t_step = teacher_by_step.get(step, {})
                    s_step = student_by_step.get(step, {})
                    if (
                        "k_current_runtime" not in t_step
                        or "k_current_runtime" not in s_step
                        or "v_current_runtime" not in t_step
                        or "v_current_runtime" not in s_step
                    ):
                        continue

                    t_pos_idx = t_pos_list.index(step)
                    s_pos_idx = s_pos_list.index(step)

                    t_k_cache = t_used_k[:, :, t_pos_idx : t_pos_idx + 1, :]
                    s_k_cache = s_used_k[:, :, s_pos_idx : s_pos_idx + 1, :]
                    t_v_cache = t_used_v[:, :, t_pos_idx : t_pos_idx + 1, :]
                    s_v_cache = s_used_v[:, :, s_pos_idx : s_pos_idx + 1, :]

                    t_k_step = t_step["k_current_runtime"]
                    s_k_step = s_step["k_current_runtime"]
                    t_v_step = t_step["v_current_runtime"]
                    s_v_step = s_step["v_current_runtime"]

                    t_write_idx = t_step.get("write_indexes_runtime", None)
                    s_write_idx = s_step.get("write_indexes_runtime", None)
                    t_write_pos = t_step.get("write_positions_runtime", None)
                    s_write_pos = s_step.get("write_positions_runtime", None)
                    if torch.is_tensor(t_write_idx) and torch.is_tensor(s_write_idx):
                        t_idx_list = [int(v) for v in t_write_idx.tolist()]
                        s_idx_list = [int(v) for v in s_write_idx.tolist()]
                        t_pos_runtime = [int(v) for v in t_write_pos.tolist()] if torch.is_tensor(t_write_pos) else []
                        s_pos_runtime = [int(v) for v in s_write_pos.tolist()] if torch.is_tensor(s_write_pos) else []
                        t_cache_dtype = t_step.get("cache_dtype_runtime", "?")
                        s_cache_dtype = s_step.get("cache_dtype_runtime", "?")
                        t_k_dtype = t_step.get("k_input_dtype_runtime", "?")
                        s_k_dtype = s_step.get("k_input_dtype_runtime", "?")
                        t_cache_class = t_step.get("kv_cache_class_runtime", "?")
                        s_cache_class = s_step.get("kv_cache_class_runtime", "?")
                        t_cache_buf_dtype = t_step.get("kv_cache_buffer_dtype_runtime", "?")
                        s_cache_buf_dtype = s_step.get("kv_cache_buffer_dtype_runtime", "?")
                        print(
                            f"step={step} write_slots teacher_idx={t_idx_list} student_idx={s_idx_list} "
                            f"teacher_pos={t_pos_runtime} student_pos={s_pos_runtime} "
                            f"teacher_cache_dtype={t_cache_dtype} student_cache_dtype={s_cache_dtype} "
                            f"teacher_k_dtype={t_k_dtype} student_k_dtype={s_k_dtype} "
                            f"teacher_cache_class={t_cache_class} student_cache_class={s_cache_class} "
                            f"teacher_cache_buf_dtype={t_cache_buf_dtype} student_cache_buf_dtype={s_cache_buf_dtype}"
                        )

                    if (
                        "k_write_self_cos_runtime" in t_step
                        and "v_write_self_cos_runtime" in t_step
                        and "k_write_self_cos_runtime" in s_step
                        and "v_write_self_cos_runtime" in s_step
                    ):
                        print(
                            f"step={step} teacher cache_write_self_k "
                            f"cos={float(t_step['k_write_self_cos_runtime']):.6f} "
                            f"max_abs={float(t_step['k_write_self_max_abs_runtime']):.6e} "
                            f"mse={float(t_step['k_write_self_mse_runtime']):.6e}"
                        )
                        print(
                            f"step={step} student cache_write_self_k "
                            f"cos={float(s_step['k_write_self_cos_runtime']):.6f} "
                            f"max_abs={float(s_step['k_write_self_max_abs_runtime']):.6e} "
                            f"mse={float(s_step['k_write_self_mse_runtime']):.6e}"
                        )
                        print(
                            f"step={step} teacher cache_write_self_v "
                            f"cos={float(t_step['v_write_self_cos_runtime']):.6f} "
                            f"max_abs={float(t_step['v_write_self_max_abs_runtime']):.6e} "
                            f"mse={float(t_step['v_write_self_mse_runtime']):.6e}"
                        )
                        print(
                            f"step={step} student cache_write_self_v "
                            f"cos={float(s_step['v_write_self_cos_runtime']):.6f} "
                            f"max_abs={float(s_step['v_write_self_max_abs_runtime']):.6e} "
                            f"mse={float(s_step['v_write_self_mse_runtime']):.6e}"
                        )

                    if "k_written_runtime" in t_step and "k_written_runtime" in s_step:
                        t_k_written = t_step["k_written_runtime"]
                        s_k_written = s_step["k_written_runtime"]
                        print_metric_line(
                            f"step={step} teacher cache_write_k[idx] vs step_k_current",
                            t_k_written,
                            t_k_step,
                        )
                        print_metric_line(
                            f"step={step} student cache_write_k[idx] vs step_k_current",
                            s_k_written,
                            s_k_step,
                        )
                        print_metric_line(
                            f"step={step} teacher_vs_student cache_write_k[idx]",
                            t_k_written,
                            s_k_written,
                        )

                    if "v_written_runtime" in t_step and "v_written_runtime" in s_step:
                        t_v_written = t_step["v_written_runtime"]
                        s_v_written = s_step["v_written_runtime"]
                        print_metric_line(
                            f"step={step} teacher cache_write_v[idx] vs step_v_current",
                            t_v_written,
                            t_v_step,
                        )
                        print_metric_line(
                            f"step={step} student cache_write_v[idx] vs step_v_current",
                            s_v_written,
                            s_v_step,
                        )
                        print_metric_line(
                            f"step={step} teacher_vs_student cache_write_v[idx]",
                            t_v_written,
                            s_v_written,
                        )

                    print_metric_line(
                        f"step={step} teacher cache_k[pos] vs step_k_current",
                        t_k_cache,
                        t_k_step,
                    )
                    print_metric_line(
                        f"step={step} student cache_k[pos] vs step_k_current",
                        s_k_cache,
                        s_k_step,
                    )
                    print_metric_line(
                        f"step={step} teacher_vs_student step_k_current",
                        t_k_step,
                        s_k_step,
                    )

                    print_metric_line(
                        f"step={step} teacher cache_v[pos] vs step_v_current",
                        t_v_cache,
                        t_v_step,
                    )
                    print_metric_line(
                        f"step={step} student cache_v[pos] vs step_v_current",
                        s_v_cache,
                        s_v_step,
                    )
                    print_metric_line(
                        f"step={step} teacher_vs_student step_v_current",
                        t_v_step,
                        s_v_step,
                    )
                    emitted += 1


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
    parser.add_argument(
        "--manual-bypass-replay-history",
        type=parse_bool,
        default=True,
        help=(
            "If true and --streaming-mode=streaming, manual bypass replays prior step inputs "
            "to warm KV cache before evaluating --step-idx."
        ),
    )
    parser.add_argument(
        "--probe-offset-trace",
        type=parse_bool,
        default=False,
        help=(
            "If true, log streaming offsets for teacher/student transformer calls and per-layer self_attn calls "
            "up to --step-idx."
        ),
    )
    parser.add_argument(
        "--probe-offset-trace-layers",
        default="0,31",
        help="Comma-separated self_attn layer list for offset trace, or 'all'.",
    )
    parser.add_argument(
        "--probe-kv-cache",
        type=parse_bool,
        default=False,
        help="If true, dump and compare teacher/student KV cache tensors at --probe-kv-step.",
    )
    parser.add_argument(
        "--probe-kv-step",
        type=int,
        default=0,
        help="Step index used for KV cache probe replay.",
    )
    parser.add_argument(
        "--probe-kv-layer",
        type=int,
        default=0,
        help="Layer index used for KV cache probe.",
    )
    parser.add_argument(
        "--probe-layer-history",
        type=parse_bool,
        default=False,
        help=(
            "If true, run a warm-history single-layer replay using teacher-captured per-layer inputs "
            "to isolate recurrence mismatch at one layer."
        ),
    )
    parser.add_argument(
        "--probe-layer-history-step",
        type=int,
        default=1,
        help="Target step index for single-layer history probe.",
    )
    parser.add_argument(
        "--probe-layer-history-layer",
        type=int,
        default=0,
        help="Layer index for single-layer history probe.",
    )
    parser.add_argument(
        "--probe-layer-history-qkv",
        type=parse_bool,
        default=False,
        help="If true, print q/k/v basis diagnostics at the single-layer target step.",
    )
    parser.add_argument(
        "--probe-layer-history-attn-debug",
        type=parse_bool,
        default=False,
        help=(
            "If true, print target-step attention internals (cache-derived K/V, positions, bias, "
            "scores, weights, and head outputs) in single-layer history probe."
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
    if int(args.probe_kv_step) < 0 or int(args.probe_kv_step) >= int(args.steps):
        raise ValueError(f"probe-kv-step must be in [0, {args.steps - 1}], got {args.probe_kv_step}")
    if int(args.probe_layer_history_step) < 0 or int(args.probe_layer_history_step) >= int(args.steps):
        raise ValueError(
            f"probe-layer-history-step must be in [0, {args.steps - 1}], got {args.probe_layer_history_step}"
        )

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

    if bool(args.probe_offset_trace):
        trace_layers = parse_layers(args.probe_offset_trace_layers, len(teacher_layers))
        offset_trace = collect_streaming_offset_trace(
            teacher,
            student,
            forced_tokens,
            int(args.step_idx),
            args.device,
        )
        print_offset_trace(offset_trace, trace_layers)

    if bool(args.probe_kv_cache):
        run_kv_cache_basis_probe(
            teacher,
            student,
            forced_tokens,
            step_idx=int(args.probe_kv_step),
            layer_idx=int(args.probe_kv_layer),
            device=args.device,
        )

    if bool(args.probe_layer_history):
        probe_history = capture_teacher_layer_input_history(
            teacher,
            forced_tokens,
            int(args.probe_layer_history_step),
            args.device,
            use_streaming,
        )
        probe_unrotation = student.transformer.input_proj.weight.detach().to(dtype=torch.float32).cpu().contiguous()
        run_single_layer_history_probe(
            teacher_layers,
            student_layers,
            probe_history,
            probe_unrotation,
            layer_idx=int(args.probe_layer_history_layer),
            step_idx=int(args.probe_layer_history_step),
            use_streaming=use_streaming,
            probe_qkv=bool(args.probe_layer_history_qkv),
            probe_attn_debug=bool(args.probe_layer_history_attn_debug),
        )

    teacher_input, teacher_layer_inputs, teacher_layer_outputs, teacher_step_inputs = capture_teacher_step(
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
        teacher_step_inputs,
        int(args.step_idx),
        use_streaming,
        replay_history=bool(args.manual_bypass_replay_history),
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
