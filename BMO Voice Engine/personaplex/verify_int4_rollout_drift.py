import argparse
import gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing this module applies the same runtime monkey patches used in test runs.
import test_rtx_edge  # noqa: F401
from moshi.models import loaders


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_forced_tokens(model, steps: int, seed: int) -> torch.Tensor:
    k = int(model.num_codebooks)
    text_card = int(model.text_card)
    audio_card = int(model.card)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    text_tokens = torch.randint(0, text_card, (steps, 1), generator=g, dtype=torch.long)
    audio_tokens = torch.randint(0, audio_card + 1, (steps, k - 1), generator=g, dtype=torch.long)
    return torch.cat([text_tokens, audio_tokens], dim=1)


def register_post_bridge_hook(model):
    cache = []

    out_norm = getattr(model, "out_norm", None)
    if out_norm is None:
        return cache, None

    def pre_hook(_module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            cache.append(x.detach().float().reshape(-1, x.shape[-1]))

    handle = out_norm.register_forward_pre_hook(pre_hook)
    return cache, handle


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


@torch.no_grad()
def run_pair_rollout(
    bf16_ckpt: str,
    int4_ckpt: str,
    *,
    steps: int,
    seed: int,
    device: str,
    teacher_forced: bool,
    lora_ckpt: str | None,
):
    print(f"\n[RUN] Loading model: {bf16_ckpt}")
    teacher = loaders.get_moshi_lm(bf16_ckpt, device=device, cpu_offload=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"\n[RUN] Loading model: {int4_ckpt}")
    student = loaders.get_moshi_lm(int4_ckpt, device=device, cpu_offload=False)
    if lora_ckpt is not None:
        apply_lora_ckpt(student, Path(lora_ckpt).resolve())
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    if int(teacher.num_codebooks) != int(student.num_codebooks):
        raise RuntimeError(
            f"num_codebooks mismatch: teacher={teacher.num_codebooks} student={student.num_codebooks}"
        )

    forced_tokens = build_forced_tokens(teacher, int(steps), int(seed))
    k = int(teacher.num_codebooks)

    t_bridge_cache, t_bridge_handle = register_post_bridge_hook(teacher)
    s_bridge_cache, s_bridge_handle = register_post_bridge_hook(student)

    student_prev_text = int(forced_tokens[0, 0].item())
    per_step = []

    try:
        with teacher.streaming(batch_size=1), student.streaming(batch_size=1):
            for t in range(int(steps)):
                teacher_codes = forced_tokens[t].clone()
                student_codes = forced_tokens[t].clone()
                if not bool(teacher_forced) and t > 0:
                    student_codes[0] = int(student_prev_text)

                teacher_in_text = int(teacher_codes[0].item())
                student_in_text = int(student_codes[0].item())

                teacher_seq = teacher_codes.view(1, k, 1).to(device)
                student_seq = student_codes.view(1, k, 1).to(device)

                _, teacher_text_logits = teacher.forward_codes(teacher_seq)
                _, student_text_logits = student.forward_codes(student_seq)

                teacher_logits = teacher_text_logits[:, 0, 0, :].float().cpu().contiguous().view(-1)
                student_logits = student_text_logits[:, 0, 0, :].float().cpu().contiguous().view(-1)

                teacher_top1 = int(torch.argmax(teacher_logits, dim=-1).item())
                student_top1 = int(torch.argmax(student_logits, dim=-1).item())
                student_prev_text = student_top1

                if not t_bridge_cache or not s_bridge_cache:
                    raise RuntimeError("Post-bridge caches are empty; out_norm hooks did not capture activations")

                teacher_post = t_bridge_cache.pop().cpu().contiguous()
                student_post = s_bridge_cache.pop().cpu().contiguous()
                n = min(int(teacher_post.shape[0]), int(student_post.shape[0]))
                if n <= 0:
                    raise RuntimeError("Invalid post-bridge activation shape captured by hooks")

                teacher_vec = teacher_post[:n].reshape(-1)
                student_vec = student_post[:n].reshape(-1)

                post_diff = teacher_vec - student_vec
                logits_diff = teacher_logits - student_logits

                cos = F.cosine_similarity(teacher_vec, student_vec, dim=0).item()
                mse = F.mse_loss(teacher_vec, student_vec).item()
                max_abs_post_bridge = torch.max(torch.abs(post_diff)).item()
                max_abs_logits = torch.max(torch.abs(logits_diff)).item()

                kl_div = F.kl_div(
                    F.log_softmax(student_logits.unsqueeze(0), dim=-1),
                    F.softmax(teacher_logits.unsqueeze(0), dim=-1),
                    reduction="batchmean",
                ).item()

                topk = 5
                t_topk = set(torch.topk(teacher_logits, k=topk).indices.tolist())
                s_topk = set(torch.topk(student_logits, k=topk).indices.tolist())
                overlap_pct = (100.0 * len(t_topk.intersection(s_topk))) / float(topk)

                per_step.append(
                    {
                        "step": t,
                        "cos": cos,
                        "mse": mse,
                        "max_abs_post_bridge": max_abs_post_bridge,
                        "max_abs_logits": max_abs_logits,
                        "kl_div": kl_div,
                        "top5_overlap_pct": overlap_pct,
                        "top1_same": int(teacher_top1 == student_top1),
                        "bf16_top1": teacher_top1,
                        "int4_top1": student_top1,
                        "teacher_in_text": teacher_in_text,
                        "student_in_text": student_in_text,
                    }
                )
    finally:
        if t_bridge_handle is not None:
            t_bridge_handle.remove()
        if s_bridge_handle is not None:
            s_bridge_handle.remove()

        del teacher
        del student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return per_step


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
    parser.add_argument(
        "--teacher-forced",
        type=parse_bool,
        default=False,
        help="If true, student uses teacher forced input tokens each step. If false, student self-feeds text token.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this drift probe")

    per_step = run_pair_rollout(
        args.bf16,
        args.int4,
        steps=int(args.steps),
        seed=int(args.seed),
        device=args.device,
        teacher_forced=bool(args.teacher_forced),
        lora_ckpt=args.lora_ckpt,
    )

    mean_cos = sum(x["cos"] for x in per_step) / len(per_step)
    mean_mse = sum(x["mse"] for x in per_step) / len(per_step)
    mean_kl = sum(x["kl_div"] for x in per_step) / len(per_step)
    mean_top5 = sum(x["top5_overlap_pct"] for x in per_step) / len(per_step)

    worst_cos = min(per_step, key=lambda x: x["cos"])
    worst_mse = max(per_step, key=lambda x: x["mse"])
    top1_match_rate = sum(x["top1_same"] for x in per_step) / len(per_step)

    catastrophic_steps = [
        x for x in per_step if x["cos"] < args.catastrophic_cos or x["mse"] > args.catastrophic_mse
    ]

    if 0 <= args.report_step < len(per_step):
        row = per_step[args.report_step]
        print("\n=== REPORT STEP ===")
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} "
            f"kl_div={row['kl_div']:.6f} top5_overlap_pct={row['top5_overlap_pct']:.2f} "
            f"top1_same={row['top1_same']} bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']} "
            f"teacher_in_text={row['teacher_in_text']} student_in_text={row['student_in_text']}"
        )

    print("\n=== ROLLOUT DRIFT SUMMARY ===")
    print(f"Steps: {args.steps}")
    print(f"Teacher forced mode: {bool(args.teacher_forced)}")
    print(f"Mean cosine: {mean_cos:.6f}")
    print(f"Mean MSE: {mean_mse:.6f}")
    print(f"Mean KL(student||teacher): {mean_kl:.6f}")
    print(f"Mean top-5 overlap: {mean_top5:.2f}%")
    print(f"Top-1 agreement: {top1_match_rate * 100:.2f}%")
    print(
        "Worst cosine: "
        f"step={worst_cos['step']} cos={worst_cos['cos']:.6f} mse={worst_cos['mse']:.6f} "
        f"max_abs_post_bridge={worst_cos['max_abs_post_bridge']:.6f} "
        f"max_abs_logits={worst_cos['max_abs_logits']:.6f}"
    )
    print(
        "Worst MSE: "
        f"step={worst_mse['step']} mse={worst_mse['mse']:.6f} cos={worst_mse['cos']:.6f} "
        f"max_abs_post_bridge={worst_mse['max_abs_post_bridge']:.6f} "
        f"max_abs_logits={worst_mse['max_abs_logits']:.6f}"
    )
    print(
        f"Catastrophic steps (cos<{args.catastrophic_cos} or mse>{args.catastrophic_mse}): {len(catastrophic_steps)}"
    )

    print("\n=== FIRST 20 STEPS ===")
    for row in per_step[:20]:
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} kl_div={row['kl_div']:.6f} "
            f"top5_overlap_pct={row['top5_overlap_pct']:.2f} top1_same={row['top1_same']} "
            f"bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']}"
        )

    print("\n=== WORST 10 STEPS BY COSINE ===")
    for row in sorted(per_step, key=lambda x: x["cos"])[:10]:
        print(
            f"step={row['step']:02d} cos={row['cos']:.6f} mse={row['mse']:.6f} "
            f"max_abs_post_bridge={row['max_abs_post_bridge']:.6f} "
            f"max_abs_logits={row['max_abs_logits']:.6f} kl_div={row['kl_div']:.6f} "
            f"top5_overlap_pct={row['top5_overlap_pct']:.2f} top1_same={row['top1_same']} "
            f"bf16_top1={row['bf16_top1']} int4_top1={row['int4_top1']}"
        )


if __name__ == "__main__":
    main()
