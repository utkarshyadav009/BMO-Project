import argparse
from pathlib import Path

import torch
import bitsandbytes.functional as BF
from moshi.models import loaders


def main():
    parser = argparse.ArgumentParser(description="Apply AWQ absorption to fused attention in_proj and export prequantized checkpoint")
    parser.add_argument("--bf16", default="v5_step1500.safetensors")
    parser.add_argument("--scales", default="bmo_awq_scales.pt")
    parser.add_argument("--out", default="bmo_awq_attention.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--scale-stat", choices=["max", "p995", "p999"], default="max")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = Path(__file__).resolve().parent
    bf16_path = (root / args.bf16).resolve()
    scales_path = (root / args.scales).resolve()
    out_path = (root / args.out).resolve()

    if not bf16_path.exists():
        raise FileNotFoundError(f"BF16 checkpoint not found: {bf16_path}")
    if not scales_path.exists():
        raise FileNotFoundError(f"AWQ scales file not found: {scales_path}")

    print(f"[INFO] Loading BF16 model: {bf16_path}")
    model = loaders.get_moshi_lm(str(bf16_path), device=args.device, dtype=torch.bfloat16, cpu_offload=False)
    model.eval()

    print(f"[INFO] Loading AWQ scales: {scales_path}")
    awq_scales = torch.load(scales_path, map_location="cpu")

    sd = model.state_dict()
    num_layers = len(model.transformer.layers)

    quantized_layers = 0
    skipped_layers = 0
    quantized_by_group = {
        "self_attn.in_proj_weight_awq": 0,
        "gating.linear_in_awq": 0,
        "self_attn.out_proj_naive": 0,
        "gating.linear_out_naive": 0,
    }
    skipped_by_group = {
        "self_attn.in_proj_weight_awq": 0,
        "gating.linear_in_awq": 0,
        "self_attn.out_proj_naive": 0,
        "gating.linear_out_naive": 0,
    }

    def resolve_scale_tensor(scale_entry, scale_key: str):
        if isinstance(scale_entry, torch.Tensor):
            # Legacy format: value is already per-channel max tensor.
            if scale_key != "max":
                raise KeyError("requested percentile stat is unavailable in legacy scales file")
            return scale_entry
        if isinstance(scale_entry, dict) and scale_key in scale_entry:
            return scale_entry[scale_key]
        raise KeyError(f"missing scale stat '{scale_key}'")

    def absorb_upstream_norm(scale: torch.Tensor, norm_w_key: str, norm_b_key: str, norm_alpha_key: str) -> bool:
        if norm_w_key in sd:
            sd[norm_w_key] = (sd[norm_w_key].float() / scale).to(sd[norm_w_key].dtype)
            if norm_b_key in sd and sd[norm_b_key] is not None:
                sd[norm_b_key] = (sd[norm_b_key].float() / scale).to(sd[norm_b_key].dtype)
            return True

        if norm_alpha_key in sd:
            s_alpha = scale.view(1, 1, -1).to(sd[norm_alpha_key].dtype)
            sd[norm_alpha_key] = (sd[norm_alpha_key] / s_alpha).to(sd[norm_alpha_key].dtype)
            return True

        return False

    def quantize_and_store(weight_key: str, group_name: str, scale: torch.Tensor | None = None):
        nonlocal quantized_layers, skipped_layers

        if weight_key not in sd:
            skipped_layers += 1
            skipped_by_group[group_name] += 1
            return

        w = sd[weight_key].to(device=args.device, dtype=torch.bfloat16)
        if scale is not None:
            w = w * scale.view(1, -1).to(w.dtype)

        packed, qstate = BF.quantize_4bit(w, quant_type="nf4")
        stats = qstate.as_dict(packed=True)

        sd[weight_key] = packed.detach().cpu()
        for sk, sv in stats.items():
            sd[f"{weight_key}.{sk}"] = sv.detach().cpu() if isinstance(sv, torch.Tensor) else sv

        quantized_layers += 1
        quantized_by_group[group_name] += 1

    for i in range(num_layers):
        layer_prefix = f"transformer.layers.{i}"

        # 1) AWQ: self-attn fused in_proj_weight (absorb via norm1).
        inproj_scale_key = f"layers.{i}.self_attn.in_proj_weight"
        inproj_weight_key = f"{layer_prefix}.self_attn.in_proj_weight"
        norm1_w_key = f"{layer_prefix}.norm1.weight"
        norm1_b_key = f"{layer_prefix}.norm1.bias"
        norm1_alpha_key = f"{layer_prefix}.norm1.alpha"

        if inproj_scale_key in awq_scales and inproj_weight_key in sd:
            try:
                inproj_scale = resolve_scale_tensor(awq_scales[inproj_scale_key], args.scale_stat)
                inproj_scale = torch.clamp(inproj_scale.to(device=sd[inproj_weight_key].device, dtype=torch.float32), min=1e-4).pow(args.alpha)
                if absorb_upstream_norm(inproj_scale, norm1_w_key, norm1_b_key, norm1_alpha_key):
                    quantize_and_store(inproj_weight_key, "self_attn.in_proj_weight_awq", scale=inproj_scale)
                else:
                    skipped_layers += 1
                    skipped_by_group["self_attn.in_proj_weight_awq"] += 1
            except KeyError:
                skipped_layers += 1
                skipped_by_group["self_attn.in_proj_weight_awq"] += 1
        else:
            skipped_layers += 1
            skipped_by_group["self_attn.in_proj_weight_awq"] += 1

        # 2) AWQ: gating.linear_in.weight (absorb via norm2).
        linin_scale_key = f"layers.{i}.gating.linear_in"
        linin_weight_key = f"{layer_prefix}.gating.linear_in.weight"
        norm2_w_key = f"{layer_prefix}.norm2.weight"
        norm2_b_key = f"{layer_prefix}.norm2.bias"
        norm2_alpha_key = f"{layer_prefix}.norm2.alpha"

        if linin_scale_key in awq_scales and linin_weight_key in sd:
            try:
                linin_scale = resolve_scale_tensor(awq_scales[linin_scale_key], args.scale_stat)
                linin_scale = torch.clamp(linin_scale.to(device=sd[linin_weight_key].device, dtype=torch.float32), min=1e-4).pow(args.alpha)
                if absorb_upstream_norm(linin_scale, norm2_w_key, norm2_b_key, norm2_alpha_key):
                    quantize_and_store(linin_weight_key, "gating.linear_in_awq", scale=linin_scale)
                else:
                    skipped_layers += 1
                    skipped_by_group["gating.linear_in_awq"] += 1
            except KeyError:
                skipped_layers += 1
                skipped_by_group["gating.linear_in_awq"] += 1
        else:
            skipped_layers += 1
            skipped_by_group["gating.linear_in_awq"] += 1

        # 3) Naive NF4: self-attn out_proj.weight.
        outproj_weight_key = f"{layer_prefix}.self_attn.out_proj.weight"
        quantize_and_store(outproj_weight_key, "self_attn.out_proj_naive", scale=None)

        # 4) Naive NF4: gating.linear_out.weight.
        linout_weight_key = f"{layer_prefix}.gating.linear_out.weight"
        quantize_and_store(linout_weight_key, "gating.linear_out_naive", scale=None)

    payload = {
        "state_dict": sd,
        "awq_meta": {
            "alpha": args.alpha,
            "scale_stat": args.scale_stat,
            "quantized_target": "temporal_transformer_full_hybrid",
            "quantized_layers": quantized_layers,
            "skipped_layers": skipped_layers,
            "quantized_by_group": quantized_by_group,
            "skipped_by_group": skipped_by_group,
            "scale_source": str(scales_path.name),
        },
    }

    torch.save(payload, out_path)
    print(f"[INFO] Saved checkpoint: {out_path}")
    print(f"[INFO] Quantized layers: {quantized_layers} | Skipped layers: {skipped_layers}")
    print(f"[INFO] Quantized by group: {quantized_by_group}")
    print(f"[INFO] Skipped by group: {skipped_by_group}")


if __name__ == "__main__":
    main()
