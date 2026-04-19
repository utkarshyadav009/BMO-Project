import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from moshi.models import loaders
from moshi.models.lm import LMModel, _iterate_audio, encode_from_sphn, load_audio


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def parse_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError("--dtype must be one of: bfloat16, float32")


def resolve_local_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def is_safetensors(path: Path) -> bool:
    return path.suffix.lower() in {".safetensors", ".sft", ".sfts"}


def read_config_override_from_payload(path: Path) -> Dict[str, Any] | None:
    if is_safetensors(path):
        return None
    with open(path, "rb") as handle:
        loaded_obj = torch.load(handle, map_location="cpu")
    if isinstance(loaded_obj, dict):
        cfg = loaded_obj.get("config_override")
        if isinstance(cfg, dict):
            return cfg
    return None


def get_input_state_keys(path: Path) -> List[str]:
    if is_safetensors(path):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return list(handle.keys())

    with open(path, "rb") as handle:
        loaded_obj = torch.load(handle, map_location="cpu")

    if isinstance(loaded_obj, dict) and isinstance(loaded_obj.get("state_dict"), dict):
        return list(loaded_obj["state_dict"].keys())

    if isinstance(loaded_obj, dict):
        return [k for k, v in loaded_obj.items() if torch.is_tensor(v)]

    raise TypeError(f"Unsupported checkpoint payload type: {type(loaded_obj)}")


def build_lm_kwargs(checkpoint_path: Path) -> Dict[str, Any]:
    lm_kwargs = dict(loaders._lm_kwargs)
    lm_kwargs["dep_q"] = 16
    cfg = read_config_override_from_payload(checkpoint_path)
    if isinstance(cfg, dict):
        lm_kwargs.update(cfg)
    return lm_kwargs


def get_temporal_layers(model: nn.Module) -> List[nn.Module]:
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        return []

    inner = getattr(transformer, "inner", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)

    layers = getattr(transformer, "layers", None)
    if layers is not None:
        return list(layers)

    return []


def gather_audio_files(calibration_path: Path, max_clips: int) -> List[Path]:
    files: List[Path] = []

    if calibration_path.is_dir():
        for p in sorted(calibration_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
                files.append(p)
    elif calibration_path.is_file():
        if calibration_path.suffix.lower() in AUDIO_SUFFIXES:
            files = [calibration_path]
        else:
            for raw in calibration_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = calibration_path.parent / p
                p = p.resolve()
                if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
                    files.append(p)
    else:
        raise FileNotFoundError(f"Calibration source not found: {calibration_path}")

    if max_clips > 0:
        files = files[:max_clips]

    return files


def encode_clip_to_audio_frames(
    mimi,
    clip_path: Path,
    max_steps_per_clip: int,
) -> List[torch.Tensor]:
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_pcm = load_audio(str(clip_path), mimi.sample_rate)
    chunks = _iterate_audio(sample_pcm, frame_size, max_len=max_steps_per_clip, pad=True)

    frames: List[torch.Tensor] = []
    with torch.no_grad():
        for encoded in encode_from_sphn(mimi, chunks):
            if not torch.is_tensor(encoded) or encoded.ndim != 3:
                continue
            codes = encoded[0].detach().cpu()  # [K, F]
            if codes.ndim != 2:
                continue
            for frame_idx in range(codes.shape[1]):
                frames.append(codes[:, frame_idx].to(torch.long).contiguous())
                if len(frames) >= max_steps_per_clip:
                    return frames

    return frames


def build_calibration_sequences(
    model: nn.Module,
    calibration_files: List[Path],
    mimi_weight: Path,
    device: str,
    max_steps_per_clip: int,
) -> tuple[List[torch.Tensor], int]:
    if not calibration_files:
        raise RuntimeError("No calibration audio files available.")

    if not mimi_weight.exists():
        raise FileNotFoundError(f"Mimi weight not found: {mimi_weight}")

    # Use the same stable Mimi loader path as offline/server for version compatibility.
    mimi = loaders.get_mimi(str(mimi_weight), device)

    num_codebooks = int(model.num_codebooks)
    num_audio_codebooks = int(model.num_audio_codebooks)
    audio_pad = int(model.card)
    text_pad = 0

    sequences: List[torch.Tensor] = []
    total_steps = 0

    for clip in calibration_files:
        try:
            frames = encode_clip_to_audio_frames(
                mimi=mimi,
                clip_path=clip,
                max_steps_per_clip=max_steps_per_clip,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to encode calibration clip {clip}: {exc}")
            continue

        if not frames:
            print(f"[WARN] No frames extracted from calibration clip: {clip}")
            continue

        seq = torch.full(
            (len(frames), num_codebooks),
            audio_pad,
            dtype=torch.long,
        )
        seq[:, 0] = text_pad

        for t, frame in enumerate(frames):
            usable = min(num_audio_codebooks, int(frame.numel()))
            if usable > 0:
                seq[t, 1 : 1 + usable] = frame[:usable]

        sequences.append(seq)
        total_steps += int(seq.shape[0])

    del mimi
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    if not sequences:
        raise RuntimeError("Failed to construct any calibration token sequences.")

    return sequences, total_steps


def get_module_name_map(model: nn.Module) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for name, module in model.named_modules():
        out[id(module)] = name
    return out


def _is_activation_gating_module(module: nn.Module) -> bool:
    linear_in = getattr(module, "linear_in", None)
    linear_out = getattr(module, "linear_out", None)
    return isinstance(linear_in, nn.Linear) and isinstance(linear_out, nn.Linear)


def build_quantization_entries(
    temporal_layers: List[nn.Module],
    selected_indices: List[int],
    name_map: Dict[int, str],
) -> Dict[int, Dict[str, Any]]:
    by_layer: Dict[int, Dict[str, Any]] = {}

    for idx in selected_indices:
        layer = temporal_layers[idx]
        layer_name = name_map.get(id(layer), f"transformer.layers.{idx}")
        entries: List[Dict[str, Any]] = []

        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, nn.Module):
            in_proj = getattr(attn, "in_proj_weight", None)
            if torch.is_tensor(in_proj) and in_proj.ndim == 2:
                entries.append(
                    {
                        "name": f"{layer_name}.self_attn.in_proj_weight",
                        "kind": "param",
                        "hook_module": attn,
                        "module": attn,
                        "param_name": "in_proj_weight",
                        "in_features": int(in_proj.shape[1]),
                    }
                )

        for sub_name, sub_module in layer.named_modules():
            full_name = layer_name if not sub_name else f"{layer_name}.{sub_name}"

            if _is_activation_gating_module(sub_module):
                gating_module = sub_module
                lin_in = gating_module.linear_in
                lin_out = gating_module.linear_out
                lin_in_w = getattr(lin_in, "weight", None)
                lin_out_w = getattr(lin_out, "weight", None)
                if not torch.is_tensor(lin_in_w) or lin_in_w.ndim != 2:
                    continue
                if not torch.is_tensor(lin_out_w) or lin_out_w.ndim != 2:
                    continue

                linear_in_name = f"{full_name}.linear_in.weight"
                linear_out_name = f"{full_name}.linear_out.weight"

                entries.append(
                    {
                        "name": linear_in_name,
                        "kind": "linear",
                        "hook_module": gating_module,
                        "hook_kind": "gating_parent",
                        "gating_role": "linear_in",
                        "gating_pair_name": linear_out_name,
                        "module": lin_in,
                        "in_features": int(lin_in_w.shape[1]),
                    }
                )
                entries.append(
                    {
                        "name": linear_out_name,
                        "kind": "linear",
                        "hook_module": gating_module,
                        "hook_kind": "gating_parent",
                        "gating_role": "linear_out",
                        "gating_pair_name": linear_in_name,
                        "module": lin_out,
                        "in_features": int(lin_out_w.shape[1]),
                    }
                )
                continue

            if not isinstance(sub_module, nn.Linear):
                continue

            if ".gating." in full_name and (
                full_name.endswith("linear_in") or full_name.endswith("linear_out")
            ):
                # Captured via parent ActivationGating forward hook.
                continue

            allowed = (
                "self_attn.out_proj" in full_name
                or full_name.endswith("linear1")
                or full_name.endswith("linear2")
            )
            if not allowed:
                continue

            w = getattr(sub_module, "weight", None)
            if not torch.is_tensor(w) or w.ndim != 2:
                continue

            entries.append(
                {
                    "name": f"{full_name}.weight",
                    "kind": "linear",
                    "hook_module": sub_module,
                    "hook_kind": "pre_input",
                    "module": sub_module,
                    "in_features": int(w.shape[1]),
                }
            )

        seen = set()
        deduped = []
        for e in entries:
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            deduped.append(e)

        by_layer[idx] = {
            "layer_name": layer_name,
            "entries": deduped,
        }

    return by_layer


def _all_collectors_full(counts: Dict[str, int], max_samples: int) -> bool:
    for _, count in counts.items():
        if count < max_samples:
            return False
    return True


def collect_inputs_for_entries(
    model: nn.Module,
    sequences: List[torch.Tensor],
    entries: List[Dict[str, Any]],
    device: str,
    max_samples: int,
) -> Dict[str, torch.Tensor]:
    buffers: Dict[str, List[torch.Tensor]] = {e["name"]: [] for e in entries}
    counts: Dict[str, int] = {e["name"]: 0 for e in entries}
    entry_by_name: Dict[str, Dict[str, Any]] = {e["name"]: e for e in entries}
    handles = []

    def _append_flat(name: str, flat: torch.Tensor) -> None:
        if name not in counts:
            return
        if counts[name] >= max_samples:
            return
        if flat.ndim == 1:
            flat = flat.view(1, -1)
        else:
            flat = flat.reshape(-1, flat.shape[-1])

        expected = int(entry_by_name[name]["in_features"])
        if flat.shape[-1] != expected:
            return

        remain = max_samples - counts[name]
        if remain <= 0:
            return
        if flat.shape[0] > remain:
            flat = flat[:remain]
        if flat.shape[0] <= 0:
            return

        cpu_flat = flat.to(device="cpu", dtype=torch.float32).contiguous()
        buffers[name].append(cpu_flat)
        counts[name] += int(cpu_flat.shape[0])

    def make_pre_hook(name: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def hook_fn(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if counts[name] >= max_samples:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            _append_flat(name, x.detach())

        return hook_fn

    def make_gating_hook(
        linear_in_name: str,
        linear_out_name: str,
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook_fn(module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return

            x_detached = x.detach()
            if counts[linear_in_name] < max_samples:
                _append_flat(linear_in_name, x_detached)

            if counts[linear_out_name] >= max_samples:
                return

            linear_in = getattr(module, "linear_in", None)
            activation = getattr(module, "activation", None)
            if not isinstance(linear_in, nn.Linear) or activation is None:
                return

            x_linear = x_detached
            if x_linear.dtype != linear_in.weight.dtype:
                x_linear = x_linear.to(linear_in.weight.dtype)

            lin_in_out = F.linear(x_linear, linear_in.weight)
            if lin_in_out.ndim == 2:
                lin_in_out = lin_in_out.unsqueeze(1)
            if lin_in_out.ndim != 3:
                return

            bsz, steps, _ = lin_in_out.shape
            split = lin_in_out.view(bsz, steps, 2, -1)
            lin_out_in = activation(split[..., 0, :]) * split[..., 1, :]
            _append_flat(linear_out_name, lin_out_in)

        return hook_fn

    gating_groups: Dict[int, Dict[str, Any]] = {}
    for e in entries:
        hook_kind = str(e.get("hook_kind", "pre_input"))
        if hook_kind == "gating_parent":
            module = e["hook_module"]
            key = id(module)
            group = gating_groups.setdefault(
                key,
                {
                    "module": module,
                    "linear_in_name": None,
                    "linear_out_name": None,
                },
            )
            role = str(e.get("gating_role", ""))
            if role == "linear_in":
                group["linear_in_name"] = e["name"]
            elif role == "linear_out":
                group["linear_out_name"] = e["name"]
            continue

        hook = e["hook_module"].register_forward_pre_hook(make_pre_hook(name=e["name"]))
        handles.append(hook)

    for group in gating_groups.values():
        linear_in_name = group["linear_in_name"]
        linear_out_name = group["linear_out_name"]
        if not linear_in_name or not linear_out_name:
            continue
        hook = group["module"].register_forward_hook(
            make_gating_hook(
                linear_in_name=linear_in_name,
                linear_out_name=linear_out_name,
            )
        )
        handles.append(hook)

    with torch.no_grad():
        for seq in sequences:
            with model.streaming(batch_size=1):
                for t in range(seq.shape[0]):
                    token = seq[t].view(1, seq.shape[1], 1).to(device=device)
                    model.forward_codes(token)
                    if _all_collectors_full(counts, max_samples):
                        break
            if _all_collectors_full(counts, max_samples):
                break

    for handle in handles:
        handle.remove()

    out: Dict[str, torch.Tensor] = {}
    for name, chunks in buffers.items():
        if not chunks:
            continue
        out[name] = torch.cat(chunks, dim=0)

    return out


def quantize_vector_rtn_affine(
    w: torch.Tensor,
    bits: int,
    scale: float | None = None,
    zero_point: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    # Eq. (2): affine RTN quantization with scale/zero-point and clipping.
    qmin = 0
    qmax = (1 << bits) - 1

    if scale is None or zero_point is None:
        w_min = float(w.min().item())
        w_max = float(w.max().item())
        if w_max - w_min <= 1e-12:
            return w.clone(), 1.0, 0.0

        inferred_scale = (w_max - w_min) / float(qmax - qmin)
        if inferred_scale <= 0.0:
            return w.clone(), 1.0, 0.0

        scale = float(inferred_scale)
        zero_point = float(round(-w_min / scale))

    scale = float(scale)
    zero_point = float(zero_point)
    if scale <= 0.0:
        return w.clone(), 1.0, 0.0

    zero_point = float(max(qmin, min(qmax, zero_point)))

    q = torch.round(w / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    dequant = scale * (q - zero_point)
    return dequant, float(scale), float(zero_point)


def find_affine_params_mse(
    w: torch.Tensor,
    bits: int,
    grid_steps: int = 64,
    min_shrink: float = 0.20,
    sample_size: int = 262144,
) -> tuple[float, float]:
    qmin = 0.0
    qmax = float((1 << bits) - 1)
    half_levels = (qmax - qmin) / 2.0
    zero_point = (qmax + qmin) / 2.0

    flat = w.detach().reshape(-1).float().cpu()
    if flat.numel() == 0:
        return 1.0, zero_point

    if flat.numel() > sample_size:
        step = max(1, int(flat.numel() // sample_size))
        sample = flat[::step][:sample_size]
    else:
        sample = flat

    max_abs = float(sample.abs().max().item())
    if max_abs <= 0.0:
        return 1.0, zero_point

    levels = max(half_levels, 1e-6)
    best_scale = max_abs / levels
    best_mse = float("inf")

    shrink_values = torch.linspace(1.0, min_shrink, steps=max(2, int(grid_steps)))
    for shrink in shrink_values.tolist():
        clipped_abs = max_abs * float(shrink)
        scale = clipped_abs / levels
        if scale <= 0.0:
            continue
        q = torch.clamp(torch.round(sample / scale + zero_point), qmin, qmax)
        deq = scale * (q - zero_point)
        mse = float(torch.mean((sample - deq) ** 2).item())
        if mse < best_mse:
            best_mse = mse
            best_scale = float(scale)

    return float(best_scale), float(zero_point)


def compute_hessian_inverse(
    activations: torch.Tensor,
    hessian_damp: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x = activations.detach().float().cpu().contiguous()
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])

    h = 2.0 * (x.t() @ x)
    mean_diag = float(torch.mean(torch.diag(h)).item())
    damp_ratio = float(hessian_damp) if hessian_damp > 0.0 else 0.01
    damp = damp_ratio * max(mean_diag, 1e-8)
    h = h + damp * torch.eye(h.shape[0], dtype=h.dtype)

    if os.environ.get("SEPTQ_LOG_HESSIAN_COND", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            cond = torch.linalg.cond(h)
            log10_cond = float(torch.log10(cond.clamp_min(1e-12)).item())
            print(f"[DEBUG] log10_cond(H)={log10_cond:.4f}")
        except RuntimeError as exc:
            print(f"[WARN] Failed to compute Hessian condition number: {exc}")

    try:
        h_inv = torch.linalg.inv(h)
    except RuntimeError:
        h_inv = torch.linalg.pinv(h)

    h_inv = 0.5 * (h_inv + h_inv.t())

    try:
        # SEPTQ/GPTQ-style update path uses cholesky(inv(H), upper=True).
        h_inv_chol = torch.linalg.cholesky(h_inv, upper=True)
    except RuntimeError:
        jitter = 1e-8
        h_inv_chol = torch.linalg.cholesky(
            h_inv + jitter * torch.eye(h_inv.shape[0], dtype=h_inv.dtype),
            upper=True,
        )

    print(torch.allclose(h_inv_chol, torch.triu(h_inv_chol), atol=1e-6))

    return h_inv.contiguous(), h_inv_chol.contiguous(), float(damp)


def build_static_global_mask(
    weight: torch.Tensor,
    weight_rtn: torch.Tensor,
    h_inv: torch.Tensor,
    ratio_p: float,
) -> tuple[torch.Tensor, int]:
    w = weight.detach().float().cpu().contiguous()
    wq = weight_rtn.detach().float().cpu().contiguous()
    in_dim = int(w.shape[1])

    diag = torch.diag(h_inv).abs().clamp_min(1e-12)
    if int(diag.numel()) != in_dim:
        raise ValueError(
            f"H^-1 diag size mismatch: got {int(diag.numel())}, expected {in_dim}"
        )

    # Eq. (4): s_{i,j} = (w_ij - quant(w_ij))^2 / (2 * [XX^T]^{-1}_{j,j}).
    scores = (w - wq).pow(2) / (2.0 * diag.unsqueeze(0))

    ratio = float(min(1.0, max(0.0, ratio_p)))
    total = int(scores.numel())
    reserved = int(round(ratio * total))
    reserved = max(0, min(total, reserved))

    mask = torch.zeros_like(scores, dtype=torch.bool)
    if reserved > 0:
        top = torch.topk(scores.reshape(-1), k=reserved, largest=True).indices
        mask.reshape(-1)[top] = True

    return mask.contiguous(), int(reserved)


def septq_quantize_weight(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int,
    ratio_p: float,
    block_size: int,
    hessian_damp: float,
) -> tuple[torch.Tensor, Dict[str, float | int]]:
    w = weight.detach().float().cpu().contiguous()
    x = activations.detach().float().cpu().contiguous()

    out_dim, in_dim = w.shape

    h_inv, h_inv_chol, damp_used = compute_hessian_inverse(x, hessian_damp=hessian_damp)

    quant_scale, quant_zero_point = find_affine_params_mse(w, bits=bits)

    w_rtn = torch.empty_like(w)
    for j in range(in_dim):
        w_rtn[:, j], _, _ = quantize_vector_rtn_affine(
            w[:, j],
            bits,
            scale=quant_scale,
            zero_point=quant_zero_point,
        )

    mask, reserved_elements = build_static_global_mask(
        weight=w,
        weight_rtn=w_rtn,
        h_inv=h_inv,
        ratio_p=ratio_p,
    )

    q = torch.zeros_like(w)
    w_work = w.clone()

    for start in range(0, in_dim, block_size):
        end = min(in_dim, start + block_size)
        if end <= start:
            continue

        e = torch.zeros((out_dim, end - start), dtype=w.dtype)

        for j in range(start, end):
            q_col, _, _ = quantize_vector_rtn_affine(
                w_work[:, j],
                bits,
                scale=quant_scale,
                zero_point=quant_zero_point,
            )

            m_col = mask[:, j]
            if torch.any(m_col):
                q_col = q_col + m_col.to(q_col.dtype) * (w_work[:, j] - q_col)

            q[:, j] = q_col

            denom = float(h_inv_chol[j, j].item())
            if abs(denom) < 1e-10:
                denom = 1e-10 if denom >= 0 else -1e-10

            err = (w_work[:, j] - q_col) / denom
            e[:, j - start] = err

            h_row = h_inv_chol[j, j:end].to(w_work.dtype)
            w_work[:, j:end] -= err.unsqueeze(1) * h_row.unsqueeze(0)

        if end < in_dim:
            w_work[:, end:] -= e @ h_inv_chol[start:end, end:].to(w_work.dtype)

    mse = float(torch.mean((w - q) ** 2).item())
    denom = float((torch.norm(w) * torch.norm(q)).item())
    cos = 1.0 if denom <= 0.0 else float(torch.sum(w * q).item() / denom)
    cos = max(-1.0, min(1.0, cos))

    return q, {
        "mse": mse,
        "cos": cos,
        # Kept for compatibility with downstream logging keys.
        "salient_cols": int(reserved_elements),
        "total_cols": int(w.numel()),
        "reserved_elements": int(reserved_elements),
        "total_elements": int(w.numel()),
        "hessian_damp": float(damp_used),
        "quant_scale": float(quant_scale),
        "quant_zero_point": float(quant_zero_point),
    }


def get_entry_weight_tensor(entry: Dict[str, Any]) -> torch.Tensor:
    if entry["kind"] == "linear":
        return entry["module"].weight
    if entry["kind"] == "param":
        return getattr(entry["module"], entry["param_name"])
    raise ValueError(f"Unsupported entry kind: {entry['kind']}")


def assign_entry_weight(entry: Dict[str, Any], new_weight_cpu: torch.Tensor) -> None:
    target = get_entry_weight_tensor(entry)
    with torch.no_grad():
        target.copy_(new_weight_cpu.to(device=target.device, dtype=target.dtype))


def sanitize_layer_stats(layer_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for item in layer_stats:
        clean = {
            "layer_idx": int(item["layer_idx"]),
            "layer_name": str(item["layer_name"]),
            "module_count": int(item.get("module_count", 0)),
            "cos_mean": float(item.get("cos_mean", 1.0)),
            "mse_mean": float(item.get("mse_mean", 0.0)),
            "modules": [],
        }
        modules = item.get("modules", [])
        for m in modules:
            clean["modules"].append(
                {
                    "name": str(m.get("name", "")),
                    "cos": float(m.get("cos", 1.0)),
                    "mse": float(m.get("mse", 0.0)),
                    "salient_cols": int(m.get("salient_cols", 0)),
                    "total_cols": int(m.get("total_cols", 0)),
                    "reserved_elements": int(
                        m.get("reserved_elements", m.get("salient_cols", 0))
                    ),
                    "total_elements": int(
                        m.get("total_elements", m.get("total_cols", 0))
                    ),
                    "hessian_damp": float(m.get("hessian_damp", 0.0)),
                    "quant_scale": float(m.get("quant_scale", 0.0)),
                    "quant_zero_point": float(m.get("quant_zero_point", 0.0)),
                    "calibration_samples": int(m.get("calibration_samples", 0)),
                }
            )
        sanitized.append(clean)
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply temporal-only SEPTQ-style post-training quantization to Moshi checkpoints."
    )
    parser.add_argument("--bf16", "--input", dest="input", default="v5_step1500_split.safetensors")
    parser.add_argument("--out", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--ratio-p", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--hessian-damp",
        type=float,
        default=0.0,
        help="Damping added to Hessian diagonal before inversion; <=0 uses automatic damping.",
    )
    parser.add_argument("--skip-first-n-temporal", type=int, default=1)
    parser.add_argument("--skip-last-n-temporal", type=int, default=2)
    parser.add_argument("--calibration-clips", required=True)
    parser.add_argument("--max-clips", type=int, default=64)
    parser.add_argument("--max-steps-per-clip", type=int, default=750)
    parser.add_argument("--max-calibration-samples", type=int, default=32768)
    parser.add_argument(
        "--no-copy-missing-weights",
        action="store_true",
        help="Disable loader fallback that copies missing depformer groups 0..7 -> 8..15.",
    )
    parser.add_argument(
        "--skip-verify-load",
        action="store_true",
        help="Skip strict state_dict verification against LMModel skeleton.",
    )
    args = parser.parse_args()

    if args.block_size <= 0:
        print("[ERROR] --block-size must be > 0")
        sys.exit(1)
    if args.max_steps_per_clip <= 0:
        print("[ERROR] --max-steps-per-clip must be > 0")
        sys.exit(1)
    if args.max_calibration_samples <= 0:
        print("[ERROR] --max-calibration-samples must be > 0")
        sys.exit(1)
    if args.ratio_p < 0.0 or args.ratio_p > 1.0:
        print("[ERROR] --ratio-p must be in [0, 1]")
        sys.exit(1)

    start = time.perf_counter()
    root = Path(__file__).resolve().parent

    input_path = resolve_local_path(root, args.input)
    if not input_path.exists():
        print(f"[ERROR] input checkpoint not found: {input_path}")
        sys.exit(1)

    if args.out.strip():
        output_path = resolve_local_path(root, args.out)
    else:
        output_path = resolve_local_path(root, f"bmo_temporal_septq_{args.bits}bit.pt")

    calibration_path = resolve_local_path(root, args.calibration_clips)
    mimi_weight = resolve_local_path(root, args.mimi_weight)

    dtype = parse_dtype(args.dtype)
    calibration_files = gather_audio_files(calibration_path, max_clips=int(args.max_clips))
    if not calibration_files:
        print(f"[ERROR] No calibration audio files found in: {calibration_path}")
        sys.exit(1)
    if len(calibration_files) < 50:
        print(
            f"[WARN] calibration files selected is below 50 ({len(calibration_files)}). "
            "For 2-bit runs, use >=50 clips when available."
        )

    print(f"[INFO] Input checkpoint: {input_path}")
    print(f"[INFO] Output checkpoint: {output_path}")
    print(f"[INFO] bits={args.bits} ratio_p={args.ratio_p} block_size={args.block_size}")
    print(
        f"[INFO] skip_first_n_temporal={args.skip_first_n_temporal} "
        f"skip_last_n_temporal={args.skip_last_n_temporal}"
    )
    print(f"[INFO] calibration files selected: {len(calibration_files)}")

    model = loaders.get_moshi_lm(
        str(input_path),
        copy_missing_weights=not bool(args.no_copy_missing_weights),
        device=str(args.device),
        dtype=dtype,
        cpu_offload=False,
    )
    model.eval()

    temporal_layers = get_temporal_layers(model)
    if not temporal_layers:
        print("[ERROR] Could not resolve temporal transformer layers from model.transformer")
        sys.exit(1)

    layer_name_map = get_module_name_map(model)

    skip_first_n = max(0, int(args.skip_first_n_temporal))
    skip_last_n = max(0, int(args.skip_last_n_temporal))

    start_idx = min(skip_first_n, len(temporal_layers))
    end_idx = max(start_idx, len(temporal_layers) - skip_last_n)

    selected_indices = list(range(start_idx, end_idx))
    quant_layer_count = len(selected_indices)
    if quant_layer_count <= 0:
        print(
            "[ERROR] No temporal layers left after skip rule. "
            "Reduce --skip-first-n-temporal and/or --skip-last-n-temporal."
        )
        sys.exit(1)

    skipped_indices = list(range(0, start_idx)) + list(range(end_idx, len(temporal_layers)))
    skipped_layers = [
        layer_name_map.get(id(temporal_layers[idx]), f"transformer.layers.{idx}")
        for idx in skipped_indices
    ]

    sequences, calibration_steps = build_calibration_sequences(
        model=model,
        calibration_files=calibration_files,
        mimi_weight=mimi_weight,
        device=str(args.device),
        max_steps_per_clip=int(args.max_steps_per_clip),
    )
    print(
        f"[INFO] Calibration token sequences: clips={len(sequences)} total_steps={calibration_steps}"
    )

    layer_plan = build_quantization_entries(
        temporal_layers=temporal_layers,
        selected_indices=selected_indices,
        name_map=layer_name_map,
    )

    layer_stats: List[Dict[str, Any]] = []
    quantized_modules = 0
    skipped_modules: List[str] = []

    for layer_idx in selected_indices:
        layer_name = layer_plan[layer_idx]["layer_name"]
        entries = layer_plan[layer_idx]["entries"]

        if not entries:
            print(f"[WARN] No quantizable modules found in {layer_name}; skipping layer")
            layer_stats.append(
                {
                    "layer_idx": layer_idx,
                    "layer_name": layer_name,
                    "module_count": 0,
                    "cos_mean": 1.0,
                    "mse_mean": 0.0,
                    "modules": [],
                }
            )
            continue

        print(f"[INFO] Quantizing layer {layer_idx}: {layer_name} ({len(entries)} module(s))")
        inputs = collect_inputs_for_entries(
            model=model,
            sequences=sequences,
            entries=entries,
            device=str(args.device),
            max_samples=int(args.max_calibration_samples),
        )

        module_stats: List[Dict[str, Any]] = []
        for entry in entries:
            name = entry["name"]
            weight = get_entry_weight_tensor(entry)
            x = inputs.get(name)

            if x is None or x.numel() == 0:
                print(f"[WARN] Missing calibration inputs for {name}; skipping module")
                skipped_modules.append(name)
                continue

            if x.shape[1] != weight.shape[1]:
                print(
                    f"[WARN] Activation width mismatch for {name}: "
                    f"X={tuple(x.shape)} W={tuple(weight.shape)}; skipping module"
                )
                skipped_modules.append(name)
                continue

            if int(x.shape[0]) < 8192:
                print(
                    f"[WARN] Low calibration samples for {name}: {int(x.shape[0])}. "
                    "Target >= 8192 (prefer 16384+) for stable Hessian statistics."
                )

            q_weight, stats = septq_quantize_weight(
                weight=weight,
                activations=x,
                bits=int(args.bits),
                ratio_p=float(args.ratio_p),
                block_size=int(args.block_size),
                hessian_damp=float(args.hessian_damp),
            )
            assign_entry_weight(entry, q_weight)

            mod = {
                "name": name,
                "cos": float(stats["cos"]),
                "mse": float(stats["mse"]),
                "salient_cols": int(stats["salient_cols"]),
                "total_cols": int(stats["total_cols"]),
                "reserved_elements": int(stats["reserved_elements"]),
                "total_elements": int(stats["total_elements"]),
                "hessian_damp": float(stats["hessian_damp"]),
                "quant_scale": float(stats["quant_scale"]),
                "quant_zero_point": float(stats["quant_zero_point"]),
                "calibration_samples": int(x.shape[0]),
            }
            module_stats.append(mod)
            quantized_modules += 1
            print(
                f"[INFO]   {name}: cos={mod['cos']:.6f} mse={mod['mse']:.6e} "
                f"reserved={mod['reserved_elements']}/{mod['total_elements']} "
                f"samples={mod['calibration_samples']}"
            )

        if module_stats:
            cos_mean = float(sum(m["cos"] for m in module_stats) / len(module_stats))
            mse_mean = float(sum(m["mse"] for m in module_stats) / len(module_stats))
        else:
            cos_mean = 1.0
            mse_mean = 0.0

        layer_stats.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "module_count": len(module_stats),
                "cos_mean": cos_mean,
                "mse_mean": mse_mean,
                "modules": module_stats,
            }
        )

    export_sd = {
        key: tensor.detach().to(device="cpu").contiguous()
        for key, tensor in model.state_dict().items()
    }

    input_keys = get_input_state_keys(input_path)
    output_keys = list(export_sd.keys())
    input_key_set = set(input_keys)
    output_key_set = set(output_keys)
    added_keys = sorted(output_key_set - input_key_set)
    removed_keys = sorted(input_key_set - output_key_set)

    source_cfg = read_config_override_from_payload(input_path)
    payload = {
        "state_dict": export_sd,
        "config_override": source_cfg,
        "model_mode": "septq_dense",
        "force_dense": True,
        "septq_meta": {
            "source_checkpoint": str(input_path),
            "bits": int(args.bits),
            "ratio_p": float(args.ratio_p),
            "block_size": int(args.block_size),
            "skip_first_n_temporal": int(skip_first_n),
            "skip_last_n_temporal": int(skip_last_n),
            "skip_depformer": True,
            "skip_embeddings": True,
            "calibration_source": str(calibration_path),
            "calibration_clip_count": int(len(sequences)),
            "calibration_total_steps": int(calibration_steps),
            "calibration_files": [str(p) for p in calibration_files],
            "skipped_temporal_layers": skipped_layers,
            "skipped_modules": skipped_modules,
            "per_layer_stats": sanitize_layer_stats(layer_stats),
            "added_state_keys": added_keys,
            "removed_state_keys": removed_keys,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))

    if not args.skip_verify_load:
        print("[INFO] Running strict load verification against LMModel skeleton...")
        lm_kwargs = build_lm_kwargs(input_path)
        verify_model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)
        try:
            verify_model.load_state_dict(export_sd, strict=True, assign=True)
        except RuntimeError as exc:
            print("[RESULT] strict_load = FAIL")
            print(f"[ERROR] {exc}")
            sys.exit(1)
        print("[RESULT] strict_load = PASS")

    elapsed = time.perf_counter() - start

    print(f"[INFO] Input key count: {len(input_keys)}")
    print(f"[INFO] Output key count: {len(output_keys)}")
    print(f"[INFO] Added keys: {len(added_keys)}")
    print(f"[INFO] Removed keys: {len(removed_keys)}")
    if added_keys:
        print(f"[INFO] First added key: {added_keys[0]}")
    if removed_keys:
        print(f"[INFO] First removed key: {removed_keys[0]}")

    print(f"[RESULT] output = {output_path}")
    print(f"[RESULT] bits = {args.bits}")
    print(f"[RESULT] ratio_p = {args.ratio_p}")
    print(f"[RESULT] block_size = {args.block_size}")
    print(f"[RESULT] calibration_clip_count = {len(sequences)}")
    print(f"[RESULT] calibration_total_steps = {calibration_steps}")
    print(f"[RESULT] quantized_temporal_layers = {quant_layer_count}")
    print(f"[RESULT] quantized_modules = {quantized_modules}")
    print(f"[RESULT] elapsed_sec = {elapsed:.3f}")


if __name__ == "__main__":
    main()
