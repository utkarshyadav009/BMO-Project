import argparse
import gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing this module applies the same runtime monkey patches used in test runs.
import test_rtx_edge  # noqa: F401
from moshi.models import loaders


@torch.no_grad()
def run_rollout(checkpoint_path: str, steps: int, seed: int, device: str = "cuda", lora_ckpt: str | None = None):
    print(f"\n[RUN] Loading model: {checkpoint_path}")
    model = loaders.get_moshi_lm(checkpoint_path, device=device, cpu_offload=False)

    if lora_ckpt is not None:
        apply_lora_ckpt(model, Path(lora_ckpt).resolve())

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    K = model.num_codebooks
    text_card = model.text_card
    audio_card = model.card

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Deterministic forced token stream: [T, K]
    text_tokens = torch.randint(0, text_card, (steps, 1), generator=g, dtype=torch.long)
    audio_tokens = torch.randint(0, audio_card + 1, (steps, K - 1), generator=g, dtype=torch.long)
    forced_tokens = torch.cat([text_tokens, audio_tokens], dim=1)

    logits_steps = []
    top1_steps = []

    with model.streaming(batch_size=1):
        for t in range(steps):
            seq_t = forced_tokens[t].view(1, K, 1).to(device)
            _, text_logits = model.forward_codes(seq_t)
            logits_t = text_logits[:, 0, 0, :].float().cpu().contiguous()
            logits_steps.append(logits_t)
            top1_steps.append(int(torch.argmax(logits_t, dim=-1).item()))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logits = torch.cat(logits_steps, dim=0)  # [T, V]
    top1 = torch.tensor(top1_steps, dtype=torch.long)
    return logits, top1, forced_tokens


class LoRAAdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int, alpha: float):
        super().__init__()
        self.base = base_layer

        for p in self.base.parameters():
            p.requires_grad = False

        if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            in_features = int(base_layer.in_features)
            out_features = int(base_layer.out_features)
        else:
            w = base_layer.weight
            out_features, in_features = int(w.shape[0]), int(w.shape[1])

        self.r = int(r)
        self.scaling = float(alpha) / float(r)

        base_param = next(self.base.parameters(), None)
        dev = base_param.device if base_param is not None else torch.device("cpu")
        dt = torch.bfloat16

        self.lora_A = nn.Parameter(torch.zeros(self.r, in_features, dtype=dt, device=dev), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.r, dtype=dt, device=dev), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lora = x if x.dtype == self.lora_A.dtype else x.to(self.lora_A.dtype)
        base_out = self.base(x_lora)
        delta = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
        if delta.dtype != base_out.dtype:
            delta = delta.to(base_out.dtype)
        return base_out + delta


def _get_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_ckpt(model: nn.Module, lora_ckpt_path: Path):
    ckpt = torch.load(str(lora_ckpt_path), map_location="cpu")
    rank = int(ckpt["rank"])
    alpha = float(ckpt["alpha"])
    wrapped_modules = list(ckpt["wrapped_modules"])
    lora_state = ckpt["lora_state_dict"]

    # Wrap target modules
    for mod_name in wrapped_modules:
        parent, attr = _get_parent_and_attr(model, mod_name)
        base_layer = getattr(parent, attr)
        setattr(parent, attr, LoRAAdapterLinear(base_layer, r=rank, alpha=alpha))

    # Load LoRA weights
    missing = []
    for mod_name in wrapped_modules:
        mod = dict(model.named_modules()).get(mod_name, None)
        if mod is None or not isinstance(mod, LoRAAdapterLinear):
            missing.append(mod_name)
            continue

        key_a = f"{mod_name}.lora_A"
        key_b = f"{mod_name}.lora_B"
        if key_a not in lora_state or key_b not in lora_state:
            missing.append(mod_name)
            continue

        with torch.no_grad():
            mod.lora_A.copy_(lora_state[key_a].to(device=mod.lora_A.device, dtype=mod.lora_A.dtype))
            mod.lora_B.copy_(lora_state[key_b].to(device=mod.lora_B.device, dtype=mod.lora_B.dtype))

    if missing:
        print(f"[WARN] LoRA missing/incomplete for {len(missing)} modules.")
    print(f"[INFO] Applied LoRA checkpoint: {lora_ckpt_path.name} (wrapped={len(wrapped_modules)})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", "--bf16-ckpt", dest="bf16", default="v5_step1500.safetensors")
    parser.add_argument("--int4", "--int4-ckpt", dest="int4", default="bmo_mixed_precision.pt")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--catastrophic-cos", type=float, default=0.95)
    parser.add_argument("--catastrophic-mse", type=float, default=0.1)
    parser.add_argument("--lora-ckpt", type=str, default=None, help="Optional LoRA checkpoint (train_lqec.py output)")
    parser.add_argument("--report-step", type=int, default=63, help="Print detailed metrics for this rollout step")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this drift probe")

    bf16_logits, bf16_top1, forced = run_rollout(args.bf16, args.steps, args.seed, args.device, lora_ckpt=None)
    int4_logits, int4_top1, _ = run_rollout(args.int4, args.steps, args.seed, args.device, lora_ckpt=args.lora_ckpt)

    assert bf16_logits.shape == int4_logits.shape

    per_step = []
    for t in range(args.steps):
        a = bf16_logits[t]
        b = int4_logits[t]
        cos = F.cosine_similarity(a, b, dim=0).item()
        mse = F.mse_loss(a, b).item()
        max_abs = torch.max(torch.abs(a - b)).item()
        top1_same = int(bf16_top1[t].item() == int4_top1[t].item())
        per_step.append((t, cos, mse, max_abs, top1_same, int(bf16_top1[t].item()), int(int4_top1[t].item())))

    mean_cos = sum(x[1] for x in per_step) / len(per_step)
    mean_mse = sum(x[2] for x in per_step) / len(per_step)
    worst_cos = min(per_step, key=lambda x: x[1])
    worst_mse = max(per_step, key=lambda x: x[2])
    top1_match_rate = sum(x[4] for x in per_step) / len(per_step)

    catastrophic_steps = [x for x in per_step if x[1] < args.catastrophic_cos or x[2] > args.catastrophic_mse]

    if 0 <= args.report_step < len(per_step):
        t, cos, mse, max_abs, top1_same, bf_tok, i4_tok = per_step[args.report_step]
        print("\n=== REPORT STEP ===")
        print(
            f"step={t:02d} cos={cos:.6f} mse={mse:.6f} max_abs={max_abs:.6f} "
            f"top1_same={top1_same} bf16_top1={bf_tok} int4_top1={i4_tok}"
        )

    print("\n=== ROLLOUT DRIFT SUMMARY ===")
    print(f"Steps: {args.steps}")
    print(f"Mean cosine: {mean_cos:.6f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Top-1 agreement: {top1_match_rate * 100:.2f}%")
    print(f"Worst cosine: step={worst_cos[0]} cos={worst_cos[1]:.6f} mse={worst_cos[2]:.6f} max_abs={worst_cos[3]:.6f}")
    print(f"Worst MSE: step={worst_mse[0]} mse={worst_mse[2]:.6f} cos={worst_mse[1]:.6f} max_abs={worst_mse[3]:.6f}")
    print(
        f"Catastrophic steps (cos<{args.catastrophic_cos} or mse>{args.catastrophic_mse}): {len(catastrophic_steps)}"
    )

    print("\n=== FIRST 20 STEPS ===")
    for t, cos, mse, max_abs, top1_same, bf_tok, i4_tok in per_step[:20]:
        print(
            f"step={t:02d} cos={cos:.6f} mse={mse:.6f} max_abs={max_abs:.6f} "
            f"top1_same={top1_same} bf16_top1={bf_tok} int4_top1={i4_tok}"
        )

    print("\n=== WORST 10 STEPS BY COSINE ===")
    for t, cos, mse, max_abs, top1_same, bf_tok, i4_tok in sorted(per_step, key=lambda x: x[1])[:10]:
        print(
            f"step={t:02d} cos={cos:.6f} mse={mse:.6f} max_abs={max_abs:.6f} "
            f"top1_same={top1_same} bf16_top1={bf_tok} int4_top1={i4_tok}"
        )


if __name__ == "__main__":
    main()
