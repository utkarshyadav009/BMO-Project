import argparse
import time

import torch

from apply_septq import septq_quantize_weight


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float((torch.norm(a) * torch.norm(b)).item())
    if denom <= 0.0:
        return 1.0
    return float(torch.sum(a * b).item() / denom)


def parse_bits(raw: str) -> list[int]:
    out = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        bit = int(t)
        if bit not in {2, 3, 4}:
            raise ValueError(f"Unsupported bit-width in --bits: {bit}")
        out.append(bit)
    if not out:
        raise ValueError("--bits must contain at least one value")
    return out


def run_case(
    dim: int,
    samples: int,
    bits: int,
    ratio_p: float,
    block_size: int,
    hessian_damp: float,
    seed: int,
) -> dict:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + bits)

    weight = torch.randn((dim, dim), generator=g, dtype=torch.float32)
    activations = torch.randn((samples, dim), generator=g, dtype=torch.float32)

    t0 = time.perf_counter()
    quantized, stats = septq_quantize_weight(
        weight=weight,
        activations=activations,
        bits=bits,
        ratio_p=ratio_p,
        block_size=block_size,
        hessian_damp=hessian_damp,
    )
    elapsed = time.perf_counter() - t0

    cos = cosine_similarity(weight, quantized)
    mse = float(torch.mean((weight - quantized) ** 2).item())

    return {
        "bits": int(bits),
        "cos": float(cos),
        "mse": float(mse),
        "elapsed_sec": float(elapsed),
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEPTQ sanity check on a single random Linear weight matrix."
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--bits", default="3,2")
    parser.add_argument("--ratio-p", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hessian-damp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-cos-3bit", type=float, default=0.99)
    parser.add_argument("--min-cos-2bit", type=float, default=0.94)
    args = parser.parse_args()

    if args.dim <= 0:
        raise SystemExit("[ERROR] --dim must be > 0")
    if args.samples <= 0:
        raise SystemExit("[ERROR] --samples must be > 0")
    if args.block_size <= 0:
        raise SystemExit("[ERROR] --block-size must be > 0")
    if args.ratio_p < 0.0 or args.ratio_p > 1.0:
        raise SystemExit("[ERROR] --ratio-p must be in [0, 1]")

    bits = parse_bits(args.bits)

    print(
        f"[INFO] Running SEPTQ unit test: dim={args.dim} samples={args.samples} "
        f"bits={bits} ratio_p={args.ratio_p} block_size={args.block_size}"
    )

    failed = False
    for bit in bits:
        result = run_case(
            dim=int(args.dim),
            samples=int(args.samples),
            bits=int(bit),
            ratio_p=float(args.ratio_p),
            block_size=int(args.block_size),
            hessian_damp=float(args.hessian_damp),
            seed=int(args.seed),
        )

        threshold = args.min_cos_3bit if bit == 3 else args.min_cos_2bit if bit == 2 else None
        if threshold is None:
            status = "INFO"
        else:
            status = "PASS" if result["cos"] >= float(threshold) else "FAIL"
            if status == "FAIL":
                failed = True

        print(
            f"[RESULT] bit={bit} cos={result['cos']:.6f} mse={result['mse']:.6e} "
            f"elapsed_sec={result['elapsed_sec']:.3f} status={status}"
        )

    if failed:
        raise SystemExit(1)

    print("[RESULT] SEPTQ unit test = PASS")


if __name__ == "__main__":
    main()
