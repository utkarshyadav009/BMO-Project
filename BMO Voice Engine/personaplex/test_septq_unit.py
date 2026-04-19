import argparse
import time

import torch

from apply_septq import find_affine_params_mse, quantize_vector_rtn_affine, septq_quantize_weight


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
        if bit not in {2, 3, 4, 5}:
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
        quant_min_range=1e-6,
        log_per_column_stats=False,
    )
    elapsed = time.perf_counter() - t0

    cos = cosine_similarity(weight, quantized)
    mse = float(torch.mean((weight - quantized) ** 2).item())

    scale, zero_point = find_affine_params_mse(weight, bits=bits)
    rtn = torch.empty_like(weight)
    for col in range(weight.shape[1]):
        rtn[:, col], _, _ = quantize_vector_rtn_affine(
            weight[:, col],
            bits,
            scale=scale,
            zero_point=zero_point,
        )
    rtn_cos = cosine_similarity(weight, rtn)
    rtn_mse = float(torch.mean((weight - rtn) ** 2).item())

    return {
        "bits": int(bits),
        "cos": float(cos),
        "mse": float(mse),
        "rtn_cos": float(rtn_cos),
        "rtn_mse": float(rtn_mse),
        "elapsed_sec": float(elapsed),
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEPTQ sanity check on a single random Linear weight matrix."
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--bits", default="2,3,4,5")
    parser.add_argument("--ratio-p", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hessian-damp", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--septq-rtn-margin", type=float, default=0.005)
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
    results_by_bit: dict[int, dict] = {}
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

        sep_ok = result["cos"] >= (result["rtn_cos"] - float(args.septq_rtn_margin))
        status = "PASS" if sep_ok else "FAIL"
        if not sep_ok:
            failed = True
        results_by_bit[int(bit)] = result

        print(
            f"[INFO] RTN: {result['rtn_cos']:.6f} SEPTQ: {result['cos']:.6f} "
            f"(margin={args.septq_rtn_margin:.4f})"
        )

        print(
            f"[RESULT] bit={bit} cos={result['cos']:.6f} mse={result['mse']:.6e} "
            f"rtn_cos={result['rtn_cos']:.6f} rtn_mse={result['rtn_mse']:.6e} "
            f"elapsed_sec={result['elapsed_sec']:.3f} status={status}"
        )

    required_bits = [2, 3, 4, 5]
    have_all_required = all(b in results_by_bit for b in required_bits)
    if have_all_required:
        cos2 = results_by_bit[2]["cos"]
        cos3 = results_by_bit[3]["cos"]
        cos4 = results_by_bit[4]["cos"]
        cos5 = results_by_bit[5]["cos"]
        mse2 = results_by_bit[2]["mse"]
        mse3 = results_by_bit[3]["mse"]
        mse4 = results_by_bit[4]["mse"]
        mse5 = results_by_bit[5]["mse"]

        cos_monotonic = (cos5 >= cos4) and (cos4 >= cos3) and (cos3 >= cos2)
        mse_monotonic = (mse2 >= mse3) and (mse3 >= mse4) and (mse4 >= mse5)

        print(
            f"[INFO] monotonic cos(5>=4>=3>=2)={cos_monotonic} "
            f"mse(2>=3>=4>=5)={mse_monotonic}"
        )

        if not cos_monotonic or not mse_monotonic:
            failed = True
    else:
        missing = [b for b in required_bits if b not in results_by_bit]
        print(f"[WARN] Monotonicity check skipped; missing bits: {missing}")

    if failed:
        raise SystemExit(1)

    print("[RESULT] SEPTQ unit test = PASS")


if __name__ == "__main__":
    main()
