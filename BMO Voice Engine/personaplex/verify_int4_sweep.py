import argparse
import gc
import math
from statistics import mean

import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def build_int4_layer_from_prequant(state_dict: dict, layer_key: str, out_features: int, in_features: int, device: torch.device):
    packed = state_dict[layer_key].to(device)
    stats_prefix = layer_key + "."
    stats = {
        k[len(stats_prefix):]: v.to(device)
        for k, v in state_dict.items()
        if k.startswith(stats_prefix)
    }

    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).to(device)

    layer.weight = Params4bit.from_prequantized(
        packed,
        stats,
        requires_grad=False,
        device=device.type,
        module=layer,
    )
    return layer.eval()


def build_bf16_layer(weight: torch.Tensor, device: torch.device):
    weight = weight.to(device=device, dtype=torch.bfloat16)
    out_features, in_features = weight.shape
    layer = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer.eval()


def layer_metrics(bf16_layer, int4_layer, in_features: int, seq_len: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    x = torch.randn(1, seq_len, in_features, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out_bf16 = bf16_layer(x)
        out_int4 = int4_layer(x)

    a = out_bf16.float()
    b = out_int4.float()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    mse = F.mse_loss(a, b).item()
    max_abs = torch.max(torch.abs(a - b)).item()
    return cos, mse, max_abs


def main():
    parser = argparse.ArgumentParser(description="Sweep BF16 vs INT4 layer fidelity")
    parser.add_argument("--bf16", default="v5_step1500.safetensors")
    parser.add_argument("--int4", default="bmo_mixed_precision.pt")
    parser.add_argument("--max-layers", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-cos", type=float, default=0.95)
    parser.add_argument("--threshold-mse", type=float, default=0.1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    print(f"Loading BF16 checkpoint: {args.bf16}")
    bf16_sd = load_file(args.bf16)
    print(f"Loading INT4 checkpoint: {args.int4}")
    int4_sd = torch.load(args.int4, map_location="cpu", mmap=True)

    quant_suffix = ".weight.quant_state.bitsandbytes__nf4"
    quant_keys = [k[:-len(".quant_state.bitsandbytes__nf4")] for k in int4_sd.keys() if k.endswith(quant_suffix)]

    candidates = []
    for k in sorted(quant_keys):
        if k not in bf16_sd:
            continue
        w = bf16_sd[k]
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        candidates.append(k)

    if not candidates:
        raise RuntimeError("No comparable quantized 2D weight layers found")

    selected = candidates[: args.max_layers]
    print(f"Comparable quantized layers found: {len(candidates)}; selected: {len(selected)}")

    results = []
    for idx, key in enumerate(selected):
        w = bf16_sd[key]
        out_features, in_features = w.shape
        bf16_layer = build_bf16_layer(w, device)
        int4_layer = build_int4_layer_from_prequant(int4_sd, key, out_features, in_features, device)

        cos, mse, max_abs = layer_metrics(
            bf16_layer,
            int4_layer,
            in_features=in_features,
            seq_len=args.seq_len,
            seed=args.seed + idx,
            device=device,
        )
        results.append((key, cos, mse, max_abs, in_features, out_features))

        status = "OK"
        if cos < args.threshold_cos or mse > args.threshold_mse:
            status = "FAIL"
        print(f"[{idx+1:02d}/{len(selected)}] {status} cos={cos:.6f} mse={mse:.6f} max_abs={max_abs:.6f} :: {key}")

        del bf16_layer, int4_layer
        gc.collect()
        torch.cuda.empty_cache()

    worst_cos = sorted(results, key=lambda x: x[1])[:10]
    worst_mse = sorted(results, key=lambda x: x[2], reverse=True)[:10]
    failed = [r for r in results if (r[1] < args.threshold_cos or r[2] > args.threshold_mse)]

    print("\n=== SUMMARY ===")
    print(f"Layers tested: {len(results)}")
    print(f"Mean cosine: {mean([r[1] for r in results]):.6f}")
    print(f"Mean MSE: {mean([r[2] for r in results]):.6f}")
    print(f"Worst cosine: {min([r[1] for r in results]):.6f}")
    print(f"Worst MSE: {max([r[2] for r in results]):.6f}")
    print(f"Failed thresholds (cos<{args.threshold_cos} or mse>{args.threshold_mse}): {len(failed)}")

    print("\n=== WORST 10 BY COSINE ===")
    for key, cos, mse, max_abs, in_f, out_f in worst_cos:
        print(f"cos={cos:.6f} mse={mse:.6f} max_abs={max_abs:.6f} shape=({out_f},{in_f}) :: {key}")

    print("\n=== WORST 10 BY MSE ===")
    for key, cos, mse, max_abs, in_f, out_f in worst_mse:
        print(f"mse={mse:.6f} cos={cos:.6f} max_abs={max_abs:.6f} shape=({out_f},{in_f}) :: {key}")


if __name__ == "__main__":
    main()
