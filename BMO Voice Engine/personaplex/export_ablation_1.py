import argparse
import torch
import torch.nn as nn
import bitsandbytes as bnb

from moshi.models import loaders


def quantize_temporal(module: nn.Module, prefix: str = "") -> tuple[int, int]:
    quantized = 0
    visited = 0
    SKIP_MODULES = {"text_emb", "audio_emb", "out_norm", "audio_heads", "text_heads", "extra_heads", "mimi"}

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        visited += 1

        if "depformer" in full_name or any(skip in full_name for skip in SKIP_MODULES):
            continue

        # Ablation split 1: quantize only attention linear projections.
        if isinstance(child, nn.Linear) and "self_attn" in full_name:
            int4_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            ).to(device=child.weight.device)

            q_weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
                quant_type="nf4",
            ).to(device=child.weight.device)
            int4_layer.weight = q_weight

            if child.bias is not None:
                int4_layer.bias = nn.Parameter(child.bias.data.clone())

            setattr(module, child_name, int4_layer)
            quantized += 1
        else:
            q_sub, v_sub = quantize_temporal(child, full_name)
            quantized += q_sub
            visited += v_sub

    return quantized, visited


def main():
    parser = argparse.ArgumentParser(description="Export ablation-1 checkpoint: quantize attention only")
    parser.add_argument("--input", default="v5_step1500.safetensors")
    parser.add_argument("--output", default="bmo_ablation_1.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[INFO] Loading BF16 base checkpoint: {args.input}")
    model = loaders.get_moshi_lm(
        args.input,
        copy_missing_weights=True,
        device=args.device,
        dtype=torch.bfloat16,
        cpu_offload=False,
    )
    model.eval()

    print("[INFO] Applying Ablation Split 1 (self_attn Linear only -> NF4)")
    quantized, visited = quantize_temporal(model)
    print(f"[INFO] Modules visited: {visited}")
    print(f"[INFO] Quantized self_attn nn.Linear modules: {quantized}")

    sd = model.state_dict()
    quant_state_keys = [k for k in sd.keys() if k.endswith(".quant_state.bitsandbytes__nf4")]
    print(f"[INFO] Quantized state entries found: {len(quant_state_keys)}")
    if quant_state_keys:
        print("[INFO] Sample quantized keys:")
        for k in quant_state_keys[:8]:
            print("  -", k)

    torch.save(sd, args.output)
    print(f"[INFO] Saved ablation checkpoint: {args.output}")


if __name__ == "__main__":
    main()
