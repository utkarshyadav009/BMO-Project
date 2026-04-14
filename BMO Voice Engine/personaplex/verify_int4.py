import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
from safetensors.torch import load_file

BF16_CKPT = "v5_step1500.safetensors"
INT4_CKPT = "bmo_mixed_precision.pt"
LAYER_KEY = "transformer.layers.0.self_attn.out_proj.weight"


def build_int4_layer_from_prequant(state_dict: dict, layer_key: str, device: torch.device):
    packed = state_dict[layer_key]
    stats_prefix = layer_key + "."
    stats = {
        k[len(stats_prefix):]: v.to(device)
        for k, v in state_dict.items()
        if k.startswith(stats_prefix)
    }

    out_features = 4096
    in_features = 4096
    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).to(device)

    packed = packed.to(device)
    layer.weight = Params4bit.from_prequantized(
        packed,
        stats,
        requires_grad=False,
        device=device.type,
        module=layer,
    )
    return layer


def build_bf16_layer(bf16_sd: dict, layer_key: str, device: torch.device):
    weight = bf16_sd[layer_key].to(device=device, dtype=torch.bfloat16)
    out_features, in_features = weight.shape
    layer = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")

    device = torch.device("cuda")

    bf16_sd = load_file(BF16_CKPT)
    int4_sd = torch.load(INT4_CKPT, map_location="cpu", mmap=True)

    if LAYER_KEY not in bf16_sd:
        raise KeyError(f"Missing BF16 key: {LAYER_KEY}")
    if LAYER_KEY not in int4_sd:
        raise KeyError(f"Missing INT4 packed key: {LAYER_KEY}")
    if f"{LAYER_KEY}.quant_state.bitsandbytes__nf4" not in int4_sd:
        raise KeyError(f"Missing INT4 quant_state key for: {LAYER_KEY}")

    torch.manual_seed(0)
    x = torch.randn(1, 16, 4096, dtype=torch.bfloat16, device=device)

    bf16_layer = build_bf16_layer(bf16_sd, LAYER_KEY, device).eval()
    int4_layer = build_int4_layer_from_prequant(int4_sd, LAYER_KEY, device).eval()

    with torch.no_grad():
        out_bf16 = bf16_layer(x)
        out_int4 = int4_layer(x)

    out_bf16_f = out_bf16.float()
    out_int4_f = out_int4.float()

    cos = F.cosine_similarity(out_bf16_f.flatten(), out_int4_f.flatten(), dim=0)
    mse = F.mse_loss(out_bf16_f, out_int4_f)
    max_abs = torch.max(torch.abs(out_bf16_f - out_int4_f))

    print(f"Layer: {LAYER_KEY}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Cosine Similarity: {cos.item():.8f}")
    print(f"MSE: {mse.item():.8f}")
    print(f"Max Absolute Difference: {max_abs.item():.8f}")


if __name__ == "__main__":
    main()
