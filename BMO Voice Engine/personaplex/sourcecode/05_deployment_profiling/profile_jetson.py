import gc
import importlib
import os
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any, Dict

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F


# Must be set before importing moshi modules so torch compile/cudagraph wrappers
# stay disabled for edge-friendly eager execution.
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["NO_CUDA_GRAPH"] = "1"

from moshi.models import loaders
from moshi.models.lm import LMModel
import moshi.offline
import moshi.modules.transformer as moshi_transformer
from moshi.modules.gating import ActivationGating
from moshi.modules.transformer import StreamingMultiheadAttention


moshi_compile_utils = importlib.import_module("moshi.utils.compile")
moshi_gating = importlib.import_module("moshi.modules.gating")


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def _resolve_cuda_device(device: str | torch.device = "cuda:0") -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    if isinstance(device, torch.device):
        if device.type == "cuda":
            return device
        return torch.device("cuda:0")
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.device(device)
    return torch.device("cuda:0")


def _print_memory_snapshot(label: str, device: str | torch.device = "cuda:0") -> None:
    message = f"[JETSON][MEM] {label}: rss={_rss_gb():.2f} GB"
    cuda_device = _resolve_cuda_device(device)
    if cuda_device is not None:
        allocated = torch.cuda.memory_allocated(cuda_device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(cuda_device) / (1024 ** 3)
        message += (
            f" | cuda_allocated={allocated:.2f} GB"
            f" cuda_reserved={reserved:.2f} GB"
        )
    print(message)


def unpack_tier_mask_uint2(packed: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    total = int(target_shape[0]) * int(target_shape[1])
    expanded = torch.zeros(total, dtype=torch.uint8, device=packed.device)
    for i in range(4):
        expanded[i::4] = (packed >> (i * 2)) & 0b11
    return expanded[:total].reshape(target_shape)


def _build_module_meta_lookup(per_layer_stats: list[dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for layer in per_layer_stats:
        if not isinstance(layer, dict):
            continue
        modules = layer.get("modules")
        if not isinstance(modules, list):
            continue
        for mod in modules:
            if not isinstance(mod, dict):
                continue
            name = mod.get("name")
            if isinstance(name, str) and name:
                lookup[name] = mod
    return lookup


def _module_meta_float(module_meta: Dict[str, Any], key: str, module_name: str) -> float:
    fallback_key = ""
    if key == "quant_scale_low":
        fallback_key = "quant_scale"
    elif key == "quant_zero_point_low":
        fallback_key = "quant_zero_point"

    value = module_meta.get(key)
    if value is None and fallback_key:
        value = module_meta.get(fallback_key)
    if value is None:
        raise RuntimeError(
            f"Missing quantization metadata '{key}' for module {module_name}"
        )

    if torch.is_tensor(value):
        if int(value.numel()) != 1:
            raise RuntimeError(
                f"Quantization metadata '{key}' for module {module_name} must be scalar"
            )
        return float(value.detach().item())
    return float(value)


class MultiTierLinearInference(nn.Module):
    def __init__(
        self,
        dense_weight: torch.Tensor,
        packed_tier_mask: torch.Tensor,
        module_meta: Dict[str, Any],
        bias: torch.Tensor | None = None,
        device: str | torch.device = "cuda:0",
        module_name: str = "",
    ):
        super().__init__()
        target_device = _resolve_cuda_device(device) or torch.device("cpu")
        rows, cols = int(dense_weight.shape[0]), int(dense_weight.shape[1])

        # Move raw weights directly to the target device in bfloat16.
        w_gpu = dense_weight.to(device=target_device, dtype=torch.bfloat16)
        packed_gpu = packed_tier_mask.to(device=target_device, dtype=torch.uint8)

        scale_low = _module_meta_float(module_meta, "quant_scale_low", module_name)
        zp_low = _module_meta_float(module_meta, "quant_zero_point_low", module_name)
        scale_int4 = _module_meta_float(module_meta, "quant_scale_int4", module_name)
        zp_int4 = _module_meta_float(module_meta, "quant_zero_point_int4", module_name)
        scale_int8 = _module_meta_float(module_meta, "quant_scale_int8", module_name)
        zp_int8 = _module_meta_float(module_meta, "quant_zero_point_int8", module_name)

        self.register_buffer(
            "scale_low",
            torch.tensor(scale_low, dtype=torch.bfloat16, device=target_device),
        )
        self.register_buffer(
            "zp_low",
            torch.tensor(zp_low, dtype=torch.bfloat16, device=target_device),
        )
        self.register_buffer(
            "scale_int4",
            torch.tensor(scale_int4, dtype=torch.bfloat16, device=target_device),
        )
        self.register_buffer(
            "zp_int4",
            torch.tensor(zp_int4, dtype=torch.bfloat16, device=target_device),
        )
        self.register_buffer(
            "scale_int8",
            torch.tensor(scale_int8, dtype=torch.bfloat16, device=target_device),
        )
        self.register_buffer(
            "zp_int8",
            torch.tensor(zp_int8, dtype=torch.bfloat16, device=target_device),
        )

        # Vectorized quantization path on target device.
        flat_w = w_gpu.view(-1)
        shift = torch.tensor([0, 2, 4, 6], device=target_device, dtype=torch.uint8)
        unpacked_mask = (packed_gpu.unsqueeze(1) >> shift) & 0b11
        unpacked_mask = unpacked_mask.view(-1)[: flat_w.numel()]

        q_out = torch.zeros_like(flat_w, dtype=torch.uint8)

        t3 = unpacked_mask >= 3
        if t3.any():
            q_out[t3] = torch.clamp(
                torch.round(flat_w[t3] / scale_low + zp_low), 0, 3
            ).to(torch.uint8)

        t2 = unpacked_mask == 2
        if t2.any():
            q_out[t2] = torch.clamp(
                torch.round(flat_w[t2] / scale_int4 + zp_int4), 0, 15
            ).to(torch.uint8)

        t1 = unpacked_mask == 1
        if t1.any():
            q_out[t1] = torch.clamp(
                torch.round(flat_w[t1] / scale_int8 + zp_int8), 0, 255
            ).to(torch.uint8)

        t0 = unpacked_mask == 0
        indices = t0.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        values = flat_w[t0]

        self.out_features = rows
        self.in_features = cols

        self.register_buffer(
            "tier_mask_packed",
            packed_gpu.contiguous(),
        )
        self.register_buffer(
            "q_weight",
            q_out.contiguous(),
        )
        self.register_buffer(
            "fp16_indices",
            indices.contiguous(),
        )
        self.register_buffer(
            "fp16_values",
            values.contiguous(),
        )

        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=target_device, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        # Explicitly release large intermediates.
        del w_gpu, flat_w, unpacked_mask, t3, t2, t1, t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del dense_weight

    def _dequantize_weight_bf16(self) -> torch.Tensor:
        flat_q = self.q_weight.view(-1).to(torch.bfloat16)

        shift = torch.tensor([0, 2, 4, 6], device=flat_q.device, dtype=torch.uint8)
        unpacked_mask = (self.tier_mask_packed.unsqueeze(1) >> shift) & 0b11
        unpacked_mask = unpacked_mask.view(-1)[: flat_q.numel()]

        out = (flat_q - self.zp_low) * self.scale_low
        out = torch.where(unpacked_mask == 2, (flat_q - self.zp_int4) * self.scale_int4, out)
        out = torch.where(unpacked_mask == 1, (flat_q - self.zp_int8) * self.scale_int8, out)

        if self.fp16_indices.numel() > 0:
            out[self.fp16_indices.long()] = self.fp16_values

        del flat_q, unpacked_mask

        return out.view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight_bf16()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        out = F.linear(x, weight, self.bias)
        del weight
        return out


def _print_linear_and_parameter_audit(model: nn.Module, *, threshold_mb: float = 1.0, top_k: int = 30) -> None:
    rows = []
    dtype_totals_mb: Dict[str, float] = {}
    type_totals_mb: Dict[str, float] = {}

    for name, param in model.named_parameters():
        if param is None:
            continue
        size_mb = (param.numel() * param.element_size()) / 1e6
        dtype_name = str(param.dtype).replace("torch.", "")
        type_name = type(param).__name__
        device_name = str(param.device)

        dtype_totals_mb[dtype_name] = dtype_totals_mb.get(dtype_name, 0.0) + size_mb
        type_totals_mb[type_name] = type_totals_mb.get(type_name, 0.0) + size_mb

        if size_mb >= float(threshold_mb):
            rows.append(
                {
                    "name": name,
                    "ptype": type_name,
                    "dtype": dtype_name,
                    "size_mb": size_mb,
                    "device": device_name,
                }
            )

    rows.sort(key=lambda item: item["size_mb"], reverse=True)

    multitier_count = sum(1 for m in model.modules() if isinstance(m, MultiTierLinearInference))
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    attn_proj_count = sum(
        1
        for m in model.modules()
        if isinstance(m, StreamingMultiheadAttention) and hasattr(m, "in_proj_quantized")
    )

    print(
        "[JETSON][AUDIT] module counts: "
        f"MultiTierLinearInference={multitier_count} "
        f"torch.nn.Linear={linear_count} "
        f"StreamingMHA.in_proj_quantized={attn_proj_count}"
    )

    print(
        "[JETSON][AUDIT] parameters >= "
        f"{float(threshold_mb):.2f} MB (top {top_k} by size):"
    )
    if not rows:
        print("[JETSON][AUDIT]   (none)")
    else:
        for row in rows[: int(top_k)]:
            print(
                "[JETSON][AUDIT]   "
                f"size_mb={row['size_mb']:.2f} "
                f"dtype={row['dtype']} "
                f"ptype={row['ptype']} "
                f"device={row['device']} "
                f"name={row['name']}"
            )

    print("[JETSON][AUDIT] per-dtype totals (MB):")
    for dtype_name, total_mb in sorted(
        dtype_totals_mb.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"[JETSON][AUDIT]   dtype={dtype_name} total_mb={total_mb:.2f}")

    print("[JETSON][AUDIT] per-parameter-type totals (MB):")
    for type_name, total_mb in sorted(
        type_totals_mb.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"[JETSON][AUDIT]   ptype={type_name} total_mb={total_mb:.2f}")


def _print_state_dict_remaining(state_dict: Dict[str, Any], *, top_k: int = 20) -> None:
    remaining_rows = []
    total_mb = 0.0
    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        size_mb = (tensor.numel() * tensor.element_size()) / 1e6
        total_mb += size_mb
        remaining_rows.append((size_mb, str(tensor.dtype).replace("torch.", ""), key))

    remaining_rows.sort(key=lambda item: item[0], reverse=True)
    print(
        "[JETSON][RAM] remaining state_dict tensors after load: "
        f"count={len(remaining_rows)} total_mb={total_mb:.2f}"
    )
    for size_mb, dtype_name, key in remaining_rows[: int(top_k)]:
        print(
            "[JETSON][RAM]   "
            f"size_mb={size_mb:.2f} dtype={dtype_name} key={key}"
        )


def _instrumented_warmup(mimi, other_mimi, lm_gen, device, frame_size):
    cuda_device = _resolve_cuda_device(device)
    printed_first_step = False

    if cuda_device is not None:
        _print_memory_snapshot("warmup entry", cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            if not printed_first_step:
                _print_memory_snapshot(
                    "right before first lm_gen.step in warmup", cuda_device or device
                )
                printed_first_step = True
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        _print_memory_snapshot("warmup end current", cuda_device)
        max_allocated = torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 3)
        max_reserved = torch.cuda.max_memory_reserved(cuda_device) / (1024 ** 3)
        print(
            "[JETSON][MEM] warmup peak: "
            f"cuda_max_allocated={max_allocated:.2f} GB "
            f"cuda_max_reserved={max_reserved:.2f} GB"
        )


moshi.offline.warmup = _instrumented_warmup


def _patched_activation_gating_forward(self, x: torch.Tensor) -> torch.Tensor:
    lin_in_weight = getattr(self.linear_in, "weight", None)
    if torch.is_tensor(lin_in_weight) and x.dtype != lin_in_weight.dtype:
        x = x.to(lin_in_weight.dtype)

    x = self.linear_in(x)
    bsz, tlen, _ = x.shape
    x = x.view(bsz, tlen, 2, -1)
    x = self.activation(x[..., 0, :]) * x[..., 1, :]

    lin_out_weight = getattr(self.linear_out, "weight", None)
    if torch.is_tensor(lin_out_weight) and x.dtype != lin_out_weight.dtype:
        x = x.to(lin_out_weight.dtype)

    return self.linear_out(x)


ActivationGating.forward = _patched_activation_gating_forward


def _patched_streaming_mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    state = self._streaming_state
    t_steps = query.shape[1]

    if state is None:
        offset = torch.zeros(1, device=query.device, dtype=torch.long)
        offset_cpu = 0
    else:
        assert self.causal, "Streaming only available for causal"
        offset = state.offset
        offset_cpu = state.offset_cpu

    if hasattr(self, "in_proj_quantized") and self.in_proj_quantized is not None:
        projected = self.in_proj_quantized(query)
    elif self.weights_per_step:
        proj_input = query
        if proj_input.dtype != self.in_proj_weight.dtype:
            proj_input = proj_input.to(self.in_proj_weight.dtype)
        projected = moshi_transformer.multi_linear(
            self.weights_per_step,
            self.in_proj_weight,
            proj_input,
            offset_cpu,
        )
    else:
        proj_input = query
        if proj_input.dtype != self.in_proj_weight.dtype:
            proj_input = proj_input.to(self.in_proj_weight.dtype)
        projected = F.linear(proj_input, self.in_proj_weight)

    q, k, v = moshi_transformer.rearrange(
        projected,
        "b t (p h d) -> p b h t d",
        p=3,
        h=self.num_heads,
    )

    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)

    prior_cache_len = 0
    if state is not None and self.compact_kv_cache:
        prior_cache_len = int(state.kv_cache.end_offset.item())

    kv_res = self._complete_kv(k, v)

    if state is not None and self.compact_kv_cache and prior_cache_len == 0:
        k, v, pos_k = moshi_transformer.KVCacheResult.from_kv(k, v)
    else:
        k, v, pos_k = kv_res
        if state is not None and self.compact_kv_cache:
            valid = pos_k >= 0
            if bool(valid.any()):
                if not bool(valid.all()):
                    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    k = k.index_select(2, valid_idx)
                    v = v.index_select(2, valid_idx)
                    pos_k = pos_k.index_select(0, valid_idx)
            else:
                k = k[:, :, :1, :]
                v = v[:, :, :1, :]
                pos_k = torch.zeros((1,), device=q.device, dtype=torch.long)

    if self.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(t_steps, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
    else:
        attn_bias = None

    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    x = moshi_transformer.rearrange(x, "b h t d -> b t (h d)")

    if self.weights_per_step:
        x = moshi_transformer.multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)
    else:
        x = self.out_proj(x)

    if state is not None:
        state.offset.add_(t_steps)
        state.offset_cpu += t_steps
    return x


def _patched_streaming_mha_init_streaming_state(self, batch_size: int):
    if self.context is None:
        if self.weights_per_step:
            capacity = self.weights_per_step
        else:
            raise RuntimeError(
                "Cannot create a streaming KVCache without a context to estimate capacity."
            )
    else:
        capacity = self.context

    in_proj_weight = getattr(self, "in_proj_weight", None)
    if torch.is_tensor(in_proj_weight):
        device = in_proj_weight.device
        dtype = in_proj_weight.dtype
    elif hasattr(self, "in_proj_quantized") and self.in_proj_quantized is not None:
        device = self.in_proj_quantized.q_weight.device
        dtype = torch.bfloat16
    else:
        first_param = next(self.parameters(), None)
        if first_param is None:
            raise RuntimeError("StreamingMultiheadAttention has no parameters to infer device")
        device = first_param.device
        dtype = first_param.dtype if first_param.is_floating_point() else torch.bfloat16

    dim_per_head = self.embed_dim // self.num_heads
    kv_cache = moshi_transformer.RingKVCache(
        batch_size,
        self.num_heads,
        dim_per_head,
        capacity,
        device,
        dtype,
    )
    return moshi_transformer._MHAState(
        kv_cache,
        offset=torch.zeros(1, device=device, dtype=torch.long),
        offset_cpu=0,
    )


StreamingMultiheadAttention.forward = _patched_streaming_mha_forward
StreamingMultiheadAttention._init_streaming_state = _patched_streaming_mha_init_streaming_state


def _load_direct_parameters_and_buffers(
    module: nn.Module,
    state_dict: Dict[str, Any],
    device: str | torch.device,
    prefix: str,
) -> None:
    for pname, param in list(module.named_parameters(recurse=False)):
        full_key = f"{prefix}{pname}"
        source_tensor = state_dict.pop(full_key, None)
        if source_tensor is not None:
            if not torch.is_tensor(source_tensor):
                raise RuntimeError(f"State entry '{full_key}' is not a tensor")
            param.data = (
                source_tensor.detach()
                .clone()
                .contiguous()
                .to(device=device, dtype=param.dtype)
            )
        elif param.device.type == "cpu" and str(device) != "cpu":
            param.data = param.data.to(device=device)

    for bname, buf in list(module.named_buffers(recurse=False)):
        if buf is None:
            continue
        full_key = f"{prefix}{bname}"
        source_buffer = state_dict.pop(full_key, None)
        if source_buffer is not None:
            if not torch.is_tensor(source_buffer):
                raise RuntimeError(f"Buffer state entry '{full_key}' is not a tensor")
            module._buffers[bname] = (
                source_buffer.detach()
                .clone()
                .contiguous()
                .to(device=device, dtype=buf.dtype)
            )
        elif buf.device.type == "cpu" and str(device) != "cpu":
            module._buffers[bname] = buf.to(device=device)


def _create_multitier_module_for_weight(
    weight_key: str,
    dense_weight: torch.Tensor,
    packed_tier_mask: torch.Tensor,
    module_meta_lookup: Dict[str, Dict[str, Any]],
    bias: torch.Tensor | None,
    device: str | torch.device,
) -> MultiTierLinearInference:
    module_meta = module_meta_lookup.get(weight_key)
    if not isinstance(module_meta, dict):
        raise RuntimeError(
            "Missing metadata in septq_meta.per_layer_stats for module "
            f"{weight_key}"
        )

    return MultiTierLinearInference(
        dense_weight=dense_weight,
        packed_tier_mask=packed_tier_mask,
        module_meta=module_meta,
        bias=bias,
        device=device,
        module_name=weight_key,
    )


def _replace_linear_child_with_multitier(
    parent_module: nn.Module,
    child_name: str,
    child_module: nn.Linear,
    full_name: str,
    state_dict: Dict[str, Any],
    tier_masks_uint2: Dict[str, torch.Tensor],
    module_meta_lookup: Dict[str, Dict[str, Any]],
    device: str | torch.device,
) -> bool:
    weight_key = f"{full_name}.weight"
    packed_tier_mask = tier_masks_uint2.get(weight_key)
    if packed_tier_mask is None:
        return False

    source_weight = state_dict.pop(weight_key, None)
    if source_weight is None:
        source_weight = child_module.weight.detach().to(device="cpu")
    elif not torch.is_tensor(source_weight):
        raise RuntimeError(f"State entry '{weight_key}' is not a tensor")

    source_bias = None
    if child_module.bias is not None:
        bias_key = f"{full_name}.bias"
        source_bias = state_dict.pop(bias_key, None)
        if source_bias is None:
            source_bias = child_module.bias.detach().to(device="cpu")
        elif not torch.is_tensor(source_bias):
            raise RuntimeError(f"State entry '{bias_key}' is not a tensor")

    quantized_layer = _create_multitier_module_for_weight(
        weight_key=weight_key,
        dense_weight=source_weight,
        packed_tier_mask=packed_tier_mask,
        module_meta_lookup=module_meta_lookup,
        bias=source_bias,
        device=device,
    )
    setattr(parent_module, child_name, quantized_layer)
    return True


def _load_dense_linear_child(
    child_module: nn.Linear,
    full_name: str,
    state_dict: Dict[str, Any],
    device: str | torch.device,
) -> None:
    weight_key = f"{full_name}.weight"
    source_weight = state_dict.pop(weight_key, None)
    if source_weight is not None:
        if not torch.is_tensor(source_weight):
            raise RuntimeError(f"State entry '{weight_key}' is not a tensor")
        child_module.weight.data = (
            source_weight.detach()
            .clone()
            .contiguous()
            .to(device=device, dtype=child_module.weight.dtype)
        )
    elif child_module.weight.device.type == "cpu" and str(device) != "cpu":
        child_module.weight.data = child_module.weight.data.to(device=device)

    if child_module.bias is not None:
        bias_key = f"{full_name}.bias"
        source_bias = state_dict.pop(bias_key, None)
        if source_bias is not None:
            if not torch.is_tensor(source_bias):
                raise RuntimeError(f"State entry '{bias_key}' is not a tensor")
            child_module.bias.data = (
                source_bias.detach()
                .clone()
                .contiguous()
                .to(device=device, dtype=child_module.bias.dtype)
            )
        elif child_module.bias.device.type == "cpu" and str(device) != "cpu":
            child_module.bias.data = child_module.bias.data.to(device=device)


def _attach_quantized_attention_in_proj(
    attn_module: StreamingMultiheadAttention,
    full_name: str,
    state_dict: Dict[str, Any],
    tier_masks_uint2: Dict[str, torch.Tensor],
    module_meta_lookup: Dict[str, Dict[str, Any]],
    device: str | torch.device,
) -> bool:
    weight_key = f"{full_name}.in_proj_weight"
    packed_tier_mask = tier_masks_uint2.get(weight_key)
    if packed_tier_mask is None:
        return False

    source_weight = state_dict.pop(weight_key, None)
    if source_weight is None:
        in_proj_weight = getattr(attn_module, "in_proj_weight", None)
        if torch.is_tensor(in_proj_weight):
            source_weight = in_proj_weight.detach().to(device="cpu")
    elif not torch.is_tensor(source_weight):
        raise RuntimeError(f"State entry '{weight_key}' is not a tensor")

    if source_weight is None:
        raise RuntimeError(
            f"Could not resolve source tensor for attention projection {weight_key}"
        )

    quantized_proj = _create_multitier_module_for_weight(
        weight_key=weight_key,
        dense_weight=source_weight,
        packed_tier_mask=packed_tier_mask,
        module_meta_lookup=module_meta_lookup,
        bias=None,
        device=device,
    )
    attn_module.in_proj_quantized = quantized_proj

    state_dict.pop(f"{full_name}.in_proj_bias", None)

    if hasattr(attn_module, "_parameters"):
        attn_module._parameters.pop("in_proj_weight", None)
    if hasattr(attn_module, "in_proj_weight"):
        delattr(attn_module, "in_proj_weight")

    return True


def load_and_compress_layer_by_layer(
    module: nn.Module,
    state_dict: Dict[str, Any],
    tier_masks_uint2: Dict[str, torch.Tensor],
    module_meta_lookup: Dict[str, Dict[str, Any]],
    device: str | torch.device = "cuda:0",
    prefix: str = "",
    stats: Dict[str, int] | None = None,
) -> Dict[str, int]:
    if stats is None:
        stats = {
            "linear_quantized": 0,
            "linear_dense_fallback": 0,
            "attention_in_proj_quantized": 0,
        }

    _load_direct_parameters_and_buffers(module, state_dict, device, prefix)

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}{child_name}"

        if isinstance(child, StreamingMultiheadAttention):
            replaced_attn = _attach_quantized_attention_in_proj(
                attn_module=child,
                full_name=full_name,
                state_dict=state_dict,
                tier_masks_uint2=tier_masks_uint2,
                module_meta_lookup=module_meta_lookup,
                device=device,
            )
            if replaced_attn:
                stats["attention_in_proj_quantized"] += 1

        if isinstance(child, nn.Linear) and "out_proj" not in full_name:
            replaced_linear = _replace_linear_child_with_multitier(
                parent_module=module,
                child_name=child_name,
                child_module=child,
                full_name=full_name,
                state_dict=state_dict,
                tier_masks_uint2=tier_masks_uint2,
                module_meta_lookup=module_meta_lookup,
                device=device,
            )
            if replaced_linear:
                stats["linear_quantized"] += 1
            else:
                _load_dense_linear_child(
                    child_module=child,
                    full_name=full_name,
                    state_dict=state_dict,
                    device=device,
                )
                stats["linear_dense_fallback"] += 1
            continue

        load_and_compress_layer_by_layer(
            child,
            state_dict,
            tier_masks_uint2,
            module_meta_lookup,
            device=device,
            prefix=full_name + ".",
            stats=stats,
        )

    return stats


def jetson_get_moshi_lm(
    weight_path,
    copy_missing_weights=False,
    device="cpu",
    dtype=torch.bfloat16,
    cpu_offload=False,
):
    del copy_missing_weights
    del cpu_offload

    resolved_weight_path = os.path.abspath(weight_path)
    checkpoint_size_gb = os.path.getsize(resolved_weight_path) / (1024 ** 3)
    checkpoint_is_zip = zipfile.is_zipfile(resolved_weight_path)

    print("\n[JETSON] 1. Reading weights via SSD memory-map (System RAM stays at ~0 GB)...")
    print(
        "[JETSON][RAM] checkpoint: "
        f"path={resolved_weight_path} "
        f"size={checkpoint_size_gb:.2f} GB "
        f"zipfile={checkpoint_is_zip}"
    )
    _print_memory_snapshot("before torch.load", "cuda:0")

    ckpt = torch.load(resolved_weight_path, map_location="cpu", mmap=True)
    _print_memory_snapshot("after torch.load(mmap=True)", "cuda:0")
    if not checkpoint_is_zip:
        print(
            "[JETSON][RAM][WARN] checkpoint is not zipfile-serialized; "
            "torch mmap may be ineffective for this file format."
        )

    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint must contain a root 'state_dict' dict")

    tier_masks_raw = ckpt.get("tier_masks_uint2")
    if not isinstance(tier_masks_raw, dict) or not tier_masks_raw:
        raise RuntimeError("Checkpoint missing root 'tier_masks_uint2' dictionary")

    septq_meta = ckpt.get("septq_meta")
    if not isinstance(septq_meta, dict):
        raise RuntimeError("Checkpoint missing 'septq_meta' dictionary")

    per_layer_stats = septq_meta.get("per_layer_stats")
    if not isinstance(per_layer_stats, list) or not per_layer_stats:
        raise RuntimeError("Checkpoint missing 'septq_meta[\"per_layer_stats\"]'")

    module_meta_lookup = _build_module_meta_lookup(per_layer_stats)
    if not module_meta_lookup:
        raise RuntimeError("No module entries found in septq_meta.per_layer_stats")

    tier_masks_uint2: Dict[str, torch.Tensor] = {}
    for name, packed in tier_masks_raw.items():
        if not isinstance(name, str):
            continue
        if not torch.is_tensor(packed):
            continue
        tier_masks_uint2[name] = (
            packed.detach().to(device="cpu", dtype=torch.uint8).contiguous().reshape(-1)
        )

    if not tier_masks_uint2:
        raise RuntimeError("No valid tensor entries found in checkpoint tier_masks_uint2")

    lm_kwargs = loaders._lm_kwargs.copy()
    if "config_override" in ckpt and ckpt["config_override"]:
        lm_kwargs.update(ckpt["config_override"])

    # Must be forced before skeleton creation to avoid depformer shape mismatch.
    lm_kwargs["dep_q"] = 16

    print("[JETSON] 2. Building empty Moshi skeleton...")
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    target_device = device
    if target_device == "cpu" and torch.cuda.is_available():
        target_device = "cuda:0"

    print("[JETSON] 3. Replacing modules with native multi-tier inference kernels...")
    load_stats = load_and_compress_layer_by_layer(
        model,
        state_dict,
        tier_masks_uint2,
        module_meta_lookup,
        device=target_device,
    )
    print(
        "[JETSON] replacement stats: "
        f"linear_quantized={load_stats['linear_quantized']} "
        f"attention_in_proj_quantized={load_stats['attention_in_proj_quantized']} "
        f"linear_dense_fallback={load_stats['linear_dense_fallback']}"
    )

    _print_state_dict_remaining(state_dict, top_k=20)
    _print_linear_and_parameter_audit(model, threshold_mb=1.0, top_k=30)
    _print_memory_snapshot("after layer-by-layer load (pre-cleanup)", target_device)

    del state_dict
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _print_memory_snapshot("after ckpt/state_dict cleanup", target_device)
    print("[JETSON] Model successfully loaded with native multi-tier inference!\n")

    model.eval()
    return model


loaders.get_moshi_lm = jetson_get_moshi_lm


BASE = "."
MODEL = "bmo_jetson_ready.pt"
MIMI = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOK = "tokenizer_spm_32k_3.model"
VOICE = "bmo_621.wav"
VOICE_DIR = BASE
INPUT_WAV = "silence.wav"
OUT_DIR = "outputs/profiler_run"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "Explain airplane flight in detail. Please provide a very long and detailed explanation."
GPU_ID = os.environ.get("GPU_ID", "0")


def print_gpu_process_snapshot(gpu_id: str) -> None:
    smi_cmd = (
        f"nvidia-smi --id={gpu_id} "
        "--query-compute-apps=pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    try:
        out = subprocess.check_output(
            smi_cmd,
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            print(f"[JETSON][GPU] active compute processes on GPU {gpu_id}:")
            for line in out.splitlines():
                print(f"[JETSON][GPU]   {line.strip()}")
        else:
            print(f"[JETSON][GPU] no active compute processes on GPU {gpu_id} before run")
    except Exception as exc:
        print(f"[JETSON][GPU][WARN] could not query nvidia-smi process list: {exc}")


keep_monitoring = True
peak_vram_mb = 0
peak_ram_mb = 0


def monitor_memory(pid: int) -> None:
    global keep_monitoring, peak_vram_mb, peak_ram_mb
    try:
        process = psutil.Process(pid)
    except Exception:
        return

    while keep_monitoring:
        try:
            ram_mb = process.memory_info().rss / (1024 * 1024)
            if ram_mb > peak_ram_mb:
                peak_ram_mb = ram_mb

            smi_cmd = (
                f"nvidia-smi --id={GPU_ID} "
                "--query-gpu=memory.used --format=csv,nounits,noheader"
            )
            res = subprocess.check_output(
                smi_cmd,
                shell=True,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            vram_mb = int(res.strip())
            if vram_mb > peak_vram_mb:
                peak_vram_mb = vram_mb
        except Exception:
            pass
        time.sleep(0.1)


print("\n=== BMO JETSON DEPLOYMENT PROFILER ===")
print(f"Target Model : {MODEL}")
print(f"Target GPU   : {GPU_ID}")
print("======================================\n")

sys.argv = [
    "moshi.offline",
    "--moshi-weight",
    MODEL,
    "--mimi-weight",
    MIMI,
    "--tokenizer",
    TOK,
    "--voice-prompt",
    VOICE,
    "--voice-prompt-dir",
    VOICE_DIR,
    "--input-wav",
    INPUT_WAV,
    "--text-prompt",
    PROMPT,
    "--output-wav",
    f"{OUT_DIR}/jetson_profiler.wav",
    "--output-text",
    f"{OUT_DIR}/jetson_profiler.json",
    "--device",
    "cuda:0",
]

start_time = time.time()
monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),))
monitor_thread.start()

print(
    "[JETSON] compile controls: "
    f"NO_TORCH_COMPILE={os.environ.get('NO_TORCH_COMPILE')} "
    f"NO_CUDA_GRAPH={os.environ.get('NO_CUDA_GRAPH')} "
    f"torch_compile_lazy={moshi_compile_utils.torch_compile_lazy} "
    f"gating_wrapped={hasattr(moshi_gating.gating_forward_kernel, '__wrapped__')} "
    f"gating_forward={ActivationGating.forward.__name__}"
)
print_gpu_process_snapshot(GPU_ID)

run_error: Exception | None = None
try:
    moshi.offline.main()
    _print_memory_snapshot("after offline.main", "cuda:0")
except Exception as exc:
    run_error = exc
finally:
    keep_monitoring = False
    monitor_thread.join()
    elapsed_time = time.time() - start_time

    print("\n=== JETSON PROFILING RESULTS ===")
    print(f"Generation Time : {elapsed_time:.2f} seconds")
    print(f"Peak System RAM : {peak_ram_mb / 1024:.2f} GB")
    print(f"Peak GPU VRAM   : {peak_vram_mb / 1024:.2f} GB")
    print("================================\n")

if run_error is None:
    if (peak_vram_mb / 1024) < 7.5 and (peak_ram_mb / 1024) < 7.5:
        print("[SUCCESS] The deployment pipeline is safe for the Jetson Orin Nano!")
    else:
        print("[WARNING] The footprint exceeds 8GB. Review KV cache limits.")
else:
    print(f"[ERROR] offline run failed: {run_error}")
    raise run_error