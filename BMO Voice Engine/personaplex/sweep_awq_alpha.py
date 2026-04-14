import subprocess
import sys
import torch
import torch.nn.functional as F

import verify_int4_rollout_drift as V


def evaluate(bf16_ckpt: str, int4_ckpt: str, steps: int = 64, seed: int = 1234, device: str = "cuda"):
    bf16_logits, bf16_top1, _ = V.run_rollout(bf16_ckpt, steps, seed, device)
    int4_logits, int4_top1, _ = V.run_rollout(int4_ckpt, steps, seed, device)

    per_step = []
    for t in range(steps):
        a = bf16_logits[t]
        b = int4_logits[t]
        cos = F.cosine_similarity(a, b, dim=0).item()
        mse = F.mse_loss(a, b).item()
        top1_same = int(bf16_top1[t].item() == int4_top1[t].item())
        per_step.append((cos, mse, top1_same))

    mean_mse = sum(x[1] for x in per_step) / steps
    top1_agreement = sum(x[2] for x in per_step) / steps
    step63_cos = per_step[63][0]
    return step63_cos, mean_mse, top1_agreement


def main():
    bf16 = "v5_step1500.safetensors"
    scales = "bmo_awq_scales.pt"
    alphas = [0.5, 0.75, 1.0]

    rows = []
    for alpha in alphas:
        out_ckpt = f"bmo_awq_attention_a{int(alpha * 100):03d}.pt"
        cmd = [
            sys.executable,
            "apply_awq_scales.py",
            "--bf16",
            bf16,
            "--scales",
            scales,
            "--out",
            out_ckpt,
            "--alpha",
            str(alpha),
        ]
        print(f"\n[RUN] Export alpha={alpha} -> {out_ckpt}")
        subprocess.run(cmd, check=True)

        print(f"[RUN] Drift eval alpha={alpha}")
        step63_cos, mean_mse, top1 = evaluate(bf16, out_ckpt)
        rows.append((alpha, out_ckpt, step63_cos, mean_mse, top1))

    print("\n=== AWQ ALPHA SWEEP (64 steps) ===")
    print("alpha\tcheckpoint\tstep63_cos\tmean_mse\ttop1_agreement")
    for alpha, ckpt, c63, mmse, t1 in rows:
        print(f"{alpha}\t{ckpt}\t{c63:.6f}\t{mmse:.6f}\t{t1 * 100:.2f}%")


if __name__ == "__main__":
    main()
