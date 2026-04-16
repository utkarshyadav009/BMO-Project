import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Also set env var so compile.py's torch.compile() calls degrade gracefully
import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
import atexit

import json
import torch
import torch.nn as nn
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
from moshi import offline
from moshi.models import loaders
from moshi.models.lm import LMModel
from moshi.modules.transformer import StreamingMultiheadAttention
import moshi.modules.transformer as moshi_transformer
from moshi.modules.transformer import KVCacheResult


import moshi.utils.compile as moshi_compile

ORIGINAL_GET_MOSHI_LM = loaders.get_moshi_lm

activation_scales = {}
_activation_probe_handles = []


def outlier_probe_hook(name):
    def hook(module, input, output):
        if not input:
            return
        x = input[0]
        if not isinstance(x, torch.Tensor):
            return
        x = x.detach().float().abs()
        if x.dim() != 3:
            return
        # input[0] is expected [batch, seq_len, embed_dim]
        max_vals = x.max(dim=0)[0].max(dim=0)[0]
        prev = activation_scales.get(name)
        activation_scales[name] = max_vals if prev is None else torch.maximum(prev, max_vals)

    return hook


def print_activation_outlier_analysis():
    if not activation_scales:
        return
    print("\n--- Activation Outlier Analysis ---")
    for name, scales in activation_scales.items():
        mean_val = scales.mean().item()
        max_val = scales.max().item()
        ratio = max_val / (mean_val + 1e-9)
        print(
            f"[{name}] Mean Magnitude: {mean_val:.4f} | "
            f"Max Outlier: {max_val:.4f} | "
            f"Outlier-to-Mean Ratio: {ratio:.2f}x"
        )


def _attach_activation_probes(model):
    global _activation_probe_handles
    if _activation_probe_handles:
        return
    print("[INFO] Attaching Activation Outlier Probes...")

    try:
        attn0 = model.transformer.layers[0].self_attn
    except Exception as e:
        print(f"[WARN] Unable to find first temporal self-attention layer for probing: {e}")
        return

    # Preferred target: explicit q_proj if architecture exposes it.
    if hasattr(attn0, "q_proj") and isinstance(attn0.q_proj, nn.Module):
        _activation_probe_handles.append(
            attn0.q_proj.register_forward_hook(outlier_probe_hook("layer_0_q_proj"))
        )
    else:
        # Fallback for fused-projection attention blocks.
        _activation_probe_handles.append(
            attn0.register_forward_hook(outlier_probe_hook("layer_0_attn_input"))
        )

    if hasattr(attn0, "out_proj") and isinstance(attn0.out_proj, nn.Module):
        _activation_probe_handles.append(
            attn0.out_proj.register_forward_hook(outlier_probe_hook("layer_0_out_proj"))
        )
    else:
        print("[WARN] self_attn.out_proj not found; out-proj probe skipped")


atexit.register(print_activation_outlier_analysis)

# Replace the compiled-function wrapper with a plain passthrough
class _EagerGraphed:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args):
        result = self.func(*args)
        return result if isinstance(result, tuple) else (result,)

moshi_compile.GraphedModule = _EagerGraphed

moshi_compile.GraphedModule = _EagerGraphed  # patch before models load

# --- 1. TURBOQUANT KV CACHE ---
class TurboRingKVCache(nn.Module):
    def __init__(self, batch_size, num_heads, dim_per_head, capacity,
                 device=torch.device("cuda"), dtype=torch.bfloat16):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("cache", torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=torch.int8))
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(dim_per_head, dim_per_head, device=device))
            self.register_buffer("rotation_q", q.to(dtype))
        self.register_buffer("scales", torch.ones((2, batch_size, num_heads, capacity, 1), device=device, dtype=dtype))
        self.register_buffer("offset", torch.zeros(1, device=device, dtype=torch.long))  # was end_offset
        self.offset_cpu = 0  # plain int, mirrors offset on CPU

    def _rebuild_for_runtime_shape(self, batch_size, num_heads, dim_per_head, device, dtype):
        # Some layers may surface a different per-head dim than what we inferred at init.
        # Rebuild buffers to match runtime K/V shape.
        self.cache = torch.zeros(
            (2, batch_size, num_heads, self.capacity, dim_per_head),
            device=device,
            dtype=torch.int8,
        )
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(dim_per_head, dim_per_head, device=device))
            self.rotation_q = q.to(dtype)
        self.scales = torch.ones(
            (2, batch_size, num_heads, self.capacity, 1),
            device=device,
            dtype=dtype,
        )
        self.offset.zero_()
        self.offset_cpu = 0

    def reset(self):
        self.offset.zero_()
        self.offset_cpu = 0

    def complete(self, k, v):
        B, H, T, D = k.shape
        if self.rotation_q.shape[-1] != D:
            self._rebuild_for_runtime_shape(
                batch_size=B,
                num_heads=H,
                dim_per_head=D,
                device=k.device,
                dtype=k.dtype,
            )
        k = k.to(self.rotation_q.dtype)
        v = v.to(self.rotation_q.dtype)
        dtype = k.dtype
        k_rot, v_rot = k @ self.rotation_q, v @ self.rotation_q
        num_levels = 2 ** 2.5
        k_s = k_rot.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / num_levels
        v_s = v_rot.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / num_levels
        k_int = (k_rot / k_s).round().clamp(-3, 3).to(torch.int8)
        v_int = (v_rot / v_s).round().clamp(-3, 3).to(torch.int8)

        indexes = (torch.arange(T, device=self.offset.device, dtype=self.offset.dtype) + self.offset) % self.capacity
        self.cache[0].index_copy_(2, indexes, k_int)
        self.cache[1].index_copy_(2, indexes, v_int)
        self.scales[0].index_copy_(2, indexes, k_s.to(dtype))
        self.scales[1].index_copy_(2, indexes, v_s.to(dtype))
        self.offset.add_(T)          # keep tensor in sync
        self.offset_cpu += T         # keep int in sync

        keys   = (self.cache[0].to(dtype) * self.scales[0]) @ self.rotation_q.T
        values = (self.cache[1].to(dtype) * self.scales[1]) @ self.rotation_q.T
        indexes_all = torch.arange(self.capacity, device=self.offset.device, dtype=torch.long)
        invalid   = indexes_all >= self.offset
        delta     = indexes_all - (self.offset % self.capacity)
        positions = torch.where(delta <= 0, self.offset + delta, self.offset + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)
        return KVCacheResult(keys, values, positions)


class _MHAStreamingState:
    """Minimal streaming state expected by StreamingMultiheadAttention._complete_kv"""
    def __init__(self, kv_cache: TurboRingKVCache, device):
        self.kv_cache = kv_cache
        self.offset = torch.zeros(1, device=device, dtype=torch.long)
        self.offset_cpu = 0

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0



# --- 2. LORA + INT4 INJECTION ARCHITECTURE ---
class LoRA4bitLinear(nn.Module):
    def __init__(self, base_layer, r=64, alpha=16):
        super().__init__()
        self.base = base_layer
        if hasattr(base_layer, 'in_features'):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            out_features, in_features = base_layer.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.bfloat16))
        self.scaling = alpha / r

    def forward(self, x):
        x_lora = x if x.dtype == self.lora_A.dtype else x.to(self.lora_A.dtype)
        base_out = self.base(x_lora)
        return base_out + (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling

def patched_mha_forward(self, query, key, value):
    state = self._streaming_state
    T = query.shape[1]
    offset = state.offset if state else torch.zeros(1, device=query.device, dtype=torch.long)
    offset_cpu = state.offset_cpu if state else 0

    if hasattr(self, 'int4_in_proj'):
        projected = self.int4_in_proj(query)
    elif self.weights_per_step:
        from moshi.modules.transformer import multi_linear
        projected = multi_linear(self.weights_per_step, self.in_proj_weight, query, offset_cpu)
    else:
        projected = torch.nn.functional.linear(query, self.in_proj_weight)

    from einops import rearrange
    import torch.nn.functional as F
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)

    # --- FIX: keep K/V floating before cache completion/rotation ---
    if not k.is_floating_point():
        k = k.to(query.dtype)
    if not v.is_floating_point():
        v = v.to(query.dtype)

    # Align to cache dtype only when cache dtype is floating.
    if state is not None and getattr(state, "kv_cache", None) is not None:
        cache_dtype = None
        if hasattr(state.kv_cache, "dtype"):
            cache_dtype = state.kv_cache.dtype
        elif hasattr(state.kv_cache, "cache"):
            cache_dtype = state.kv_cache.cache.dtype

        if cache_dtype is not None:
            try:
                if torch.empty((), dtype=cache_dtype).is_floating_point():
                    if k.dtype != cache_dtype:
                        k = k.to(cache_dtype)
                    if v.dtype != cache_dtype:
                        v = v.to(cache_dtype)
            except Exception:
                pass

    k, v, pos_k = self._complete_kv(k, v)

    # k and v may now be bfloat16 (from cache) — cast q to match
    q = q.to(k.dtype)

    attn_bias = None
    if self.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)

    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    x = rearrange(x, "b h t d -> b t (h d)")

    x_in_dtype = x.dtype
    out_proj_weight_dtype = None
    if hasattr(self.out_proj, "lora_A") and self.out_proj.lora_A is not None:
        out_proj_weight_dtype = self.out_proj.lora_A.dtype
    elif hasattr(self.out_proj, "weight") and self.out_proj.weight is not None:
        out_proj_weight_dtype = self.out_proj.weight.dtype
    elif hasattr(self.out_proj, "base") and hasattr(self.out_proj.base, "weight") and self.out_proj.base.weight is not None:
        out_proj_weight_dtype = self.out_proj.base.weight.dtype
    else:
        first_param = next(self.out_proj.parameters(), None)
        if first_param is not None:
            out_proj_weight_dtype = first_param.dtype

    if out_proj_weight_dtype is not None and x.dtype != out_proj_weight_dtype:
        x = x.to(out_proj_weight_dtype)

    if self.weights_per_step:
        from moshi.modules.transformer import multi_linear
        x = multi_linear(self.weights_per_step, self.out_proj.weight, x, offset_cpu)
    else:
        x = self.out_proj(x)

    if x.dtype != x_in_dtype:
        x = x.to(x_in_dtype)

    if state is not None:
        state.offset.add_(T)
        state.offset_cpu += T
    return x

StreamingMultiheadAttention.forward = patched_mha_forward

def patched_init_streaming_state(self, batch_size: int):
    if self.context is None:
        if self.weights_per_step:
            capacity = self.weights_per_step
        else:
            raise RuntimeError(
                "Cannot create a streaming KVCache without a context to estimate capacity."
            )
    else:
        capacity = self.context

    if hasattr(self, "in_proj_weight") and self.in_proj_weight is not None:
        device = self.in_proj_weight.device
        dtype = self.in_proj_weight.dtype
    elif hasattr(self, "int4_in_proj"):
        first_param = next(self.int4_in_proj.parameters())
        device = first_param.device
        dtype = first_param.dtype
    else:
        first_param = next(self.out_proj.parameters())
        device = first_param.device
        dtype = first_param.dtype

    dim_per_head = self.embed_dim // self.num_heads
    kv_cache = moshi_transformer.RingKVCache(
        batch_size,
        self.num_heads,
        dim_per_head,
        capacity,
        device,
        dtype,
    )
    return _MHAStreamingState(kv_cache, device)  # <-- wrap it, don't return raw cache

StreamingMultiheadAttention._init_streaming_state = patched_init_streaming_state

from moshi.modules.gating import ActivationGating
def patched_gating_forward(self, x: torch.Tensor):
    x = self.linear_in(x)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = self.activation(x[..., 0, :]) * x[..., 1, :]
    return self.linear_out(x)
ActivationGating.forward = patched_gating_forward

# --- 3. THE ZERO-RAM EDGE LOADER ---
def patched_get_moshi(*args, **kwargs):
    checkpoint_path = None
    if len(args) > 0 and isinstance(args[0], str):
        checkpoint_path = args[0]
    elif isinstance(kwargs.get("repo_or_path"), str):
        checkpoint_path = kwargs.get("repo_or_path")
    else:
        checkpoint_path = "bmo_mixed_precision.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Moshi checkpoint not found: {checkpoint_path}")

    device = kwargs.get("device", "cuda")
    cpu_offload = kwargs.get("cpu_offload", False)

    load_dtype = kwargs.get("dtype", torch.bfloat16)
    if isinstance(load_dtype, str):
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        load_dtype = _dtype_map.get(load_dtype.lower(), torch.bfloat16)

    def _cast_to_load_dtype(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != load_dtype:
            return x.to(dtype=load_dtype)
        return x

    if checkpoint_path.lower().endswith((".safetensors", ".sft", ".sfts")):
        print(f"[INFO] BF16 control path: loading native safetensors checkpoint: {checkpoint_path}")
        model = ORIGINAL_GET_MOSHI_LM(
            checkpoint_path,
            copy_missing_weights=True,
            device=device,
            dtype=load_dtype,
            cpu_offload=cpu_offload,
        )
        _attach_activation_probes(model)
        return model

    ckpt_size_gb = os.path.getsize(checkpoint_path) / 1e9
    print(f"[INFO] Memory-Mapping {ckpt_size_gb:.2f} GB Payload from Disk: {checkpoint_path}")
    loaded_obj = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    state_dict = loaded_obj["state_dict"] if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj else loaded_obj

    model_mode = "int4_prequant"
    config_override = None
    force_dense = False
    if isinstance(loaded_obj, dict):
        model_mode = str(loaded_obj.get("model_mode", model_mode))
        config_override = loaded_obj.get("config_override")
        force_dense = bool(loaded_obj.get("force_dense", False)) or model_mode in {
            "dense_bf16",
            "slicegpt_dense",
        }

    if force_dense:
        print(f"[INFO] Dense checkpoint mode detected: model_mode={model_mode} (delegating to native loader)")
        del loaded_obj
        model = ORIGINAL_GET_MOSHI_LM(
            checkpoint_path,
            copy_missing_weights=True,
            device=device,
            dtype=load_dtype,
            cpu_offload=cpu_offload,
        )
        _attach_activation_probes(model)
        return model

    print("[INFO] Loading Config Skeleton...")
    from moshi.models.lm import LMModel
    with open("bmo_config.json", "r") as f:
        config_dict = json.load(f)
    if isinstance(config_override, dict):
        print("[INFO] Applying checkpoint config override")
        config_dict.update(config_override)
    config_dict.pop("model_type", None)
    
    print("[INFO] Initializing architecture on META device...")
    with torch.device("meta"):
        model = LMModel(dtype=load_dtype, **config_dict)

    if not force_dense:
        print("[INFO] Applying Mixed-Precision structure...")
        SKIP_MODULES = {"text_emb", "audio_emb", "out_norm", "audio_heads", "text_heads", "extra_heads", "mimi"}

        def _swap(module, prefix=""):
            for child_name, child in list(module.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name

                if "depformer" in full_name or any(skip in full_name for skip in SKIP_MODULES):
                    continue

                if isinstance(child, nn.Linear):
                    int4_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                    )
                    setattr(module, child_name, int4_layer)
                else:
                    _swap(child, full_name)

        _swap(model)
    else:
        print(f"[INFO] Dense checkpoint mode detected: model_mode={model_mode}")
    
    print("[INFO] Materializing to VRAM (Expect System RAM Spillover)...")
    quant_suffix = ".quant_state.bitsandbytes__nf4"
    quant_bases = [k[:-len(quant_suffix)] for k in state_dict.keys() if k.endswith(quant_suffix)]

    if quant_bases and not force_dense:
        print(f"[INFO] Detected {len(quant_bases)} pre-quantized NF4 weight groups in checkpoint")
        quant_base_set = set(quant_bases)

        # Partial-ablation checkpoints may only quantize a subset of Linear layers.
        # Restore non-quantized Linear4bit modules back to dense nn.Linear so they
        # can receive normal BF16 weights from state_dict.
        reverted_dense = 0
        modules_snapshot = dict(model.named_modules())
        for module_name, module in list(modules_snapshot.items()):
            if not isinstance(module, bnb.nn.Linear4bit):
                continue

            weight_key = f"{module_name}.weight"
            if weight_key in quant_base_set:
                continue

            if "." in module_name:
                parent_name, child_name = module_name.rsplit(".", 1)
                parent_module = modules_snapshot.get(parent_name)
            else:
                parent_module = model
                child_name = module_name

            if parent_module is None:
                continue

            dense_layer = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device="meta",
                dtype=torch.bfloat16,
            )
            setattr(parent_module, child_name, dense_layer)
            reverted_dense += 1

        if reverted_dense:
            print(f"[INFO] Restored non-quantized layers to dense BF16: {reverted_dense}")

        dense_sd = {}
        for k, v in state_dict.items():
            if k.endswith(".absmax") or k.endswith(".quant_map") or k.endswith(".nested_absmax") or k.endswith(".nested_quant_map") or k.endswith(".quant_state.bitsandbytes__nf4"):
                continue
            if k in quant_base_set:
                continue
            if k.endswith(".weight") and k in quant_base_set:
                continue
            dense_sd[k] = _cast_to_load_dtype(v)

        incompat = model.load_state_dict(dense_sd, strict=False, assign=True)
        if incompat.unexpected_keys:
            print(f"[WARN] Unexpected dense keys while loading: {len(incompat.unexpected_keys)}")

        modules = dict(model.named_modules())
        loaded_prequant = 0
        skipped_prequant = 0
        for base in quant_bases:
            target_module = None
            if base.endswith(".weight"):
                module_name = base[:-len(".weight")]
                module = modules.get(module_name)
                if module is None or not isinstance(module, bnb.nn.Linear4bit):
                    skipped_prequant += 1
                    continue
                target_module = module
            elif base.endswith(".in_proj_weight"):
                attn_name = base[:-len(".in_proj_weight")]
                attn_module = modules.get(attn_name)
                if attn_module is None or not hasattr(attn_module, "in_proj_weight"):
                    skipped_prequant += 1
                    continue

                out_features, in_features = attn_module.in_proj_weight.shape
                int4_in_proj = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=False,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                )
                setattr(attn_module, "int4_in_proj", int4_in_proj)
                if getattr(attn_module.in_proj_weight, "is_meta", False):
                    attn_module.in_proj_weight = nn.Parameter(
                        torch.zeros(
                            (out_features, in_features),
                            dtype=torch.bfloat16,
                            device="cpu",
                        ),
                        requires_grad=False,
                    )
                target_module = int4_in_proj
            else:
                skipped_prequant += 1
                continue

            packed_weight = state_dict[base]
            stats = {}
            stats_prefix = base + "."
            for sk, sv in state_dict.items():
                if sk.startswith(stats_prefix):
                    stats[sk[len(stats_prefix):]] = sv

            target_module.weight = Params4bit.from_prequantized(
                packed_weight,
                stats,
                requires_grad=False,
                device="cpu",
                module=target_module,
            )
            loaded_prequant += 1

        print(f"[INFO] Loaded pre-quantized weights: {loaded_prequant}")
        if skipped_prequant:
            print(f"[WARN] Skipped pre-quantized groups (non-Linear4bit targets): {skipped_prequant}")
    else:
        dense_sd = {}
        for k, v in state_dict.items():
            if k.endswith(".absmax") or k.endswith(".quant_map") or k.endswith(".nested_absmax") or k.endswith(".nested_quant_map") or k.endswith(".quant_state.bitsandbytes__nf4"):
                continue
            dense_sd[k] = _cast_to_load_dtype(v)

        incompat = model.load_state_dict(dense_sd, strict=False, assign=True)
        if incompat.unexpected_keys:
            print(f"[WARN] Unexpected dense keys while loading: {len(incompat.unexpected_keys)}")
        if incompat.missing_keys:
            print(f"[WARN] Missing dense keys while loading: {len(incompat.missing_keys)}")

    model.cuda()
    _attach_activation_probes(model)
    
    return model

loaders.get_moshi_lm = patched_get_moshi
print("[INFO] Bypassing TurboQuant: Using Native FP16 RingKVCache for isolation test.")

if __name__ == "__main__":
    print("[INFO] Booting Ultimate Edge Simulator on RTX 3070...")
    try:
        offline.main()
    finally:
        if torch.cuda.is_available():
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
