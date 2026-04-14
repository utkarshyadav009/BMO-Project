import argparse
import json
from pathlib import Path

import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F

# Apply runtime/loader patches used by existing INT4 experiments.
import test_rtx_edge  # noqa: F401
from moshi import offline
from moshi.models import loaders
from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn


class LoRAAdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Module, r: int = 64, alpha: float = 16.0, init_std: float = 0.02):
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

        self.r = r
        self.scaling = float(alpha) / float(r)
        base_param = next(self.base.parameters(), None)
        param_device = base_param.device if base_param is not None else torch.device("cpu")

        self.lora_A = nn.Parameter(
            torch.randn(r, in_features, dtype=torch.bfloat16, device=param_device) * init_std
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, r, dtype=torch.bfloat16, device=param_device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lora = x if x.dtype == self.lora_A.dtype else x.to(self.lora_A.dtype)
        base_out = self.base(x_lora)
        delta = (x_lora @ self.lora_A.T @ self.lora_B.T) * self.scaling
        if delta.dtype != base_out.dtype:
            delta = delta.to(base_out.dtype)
        return base_out + delta


def build_forced_tokens(
    model,
    mimi,
    tokenizer,
    input_wav: Path,
    text_prompt: str,
    steps: int,
    device: str,
):
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    sample_pcm = load_audio(str(input_wav), mimi.sample_rate)
    samples = _iterate_audio(sample_pcm, frame_size, max_len=steps, pad=True)
    encoded_iter = encode_from_sphn(mimi, samples, max_batch=1)

    text_ids = tokenizer.encode(offline.wrap_with_system_tags(text_prompt))

    K = model.num_codebooks
    audio_pad = int(model.card)
    text_pad = 0

    forced = torch.full((steps, K), audio_pad, dtype=torch.long)
    forced[:, 0] = text_pad

    for t in range(steps):
        if t < len(text_ids):
            forced[t, 0] = int(text_ids[t])

        try:
            frame_codes = next(encoded_iter)  # [1, K_audio, F]
            audio_codes = frame_codes[:, :, 0].to(dtype=torch.long).cpu()[0]
            n_audio = min(K - 1, int(audio_codes.numel()))
            forced[t, 1 : 1 + n_audio] = audio_codes[:n_audio]
        except StopIteration:
            pass

    return forced.to(device)


def inject_lora_on_quantized_temporal(student, r: int, alpha: float):
    wrapped = []
    for i, layer in enumerate(student.transformer.layers):
        # in_proj path is exported as attn.int4_in_proj for prequant checkpoints.
        if hasattr(layer.self_attn, "int4_in_proj"):
            layer.self_attn.int4_in_proj = LoRAAdapterLinear(layer.self_attn.int4_in_proj, r=r, alpha=alpha)
            wrapped.append(f"transformer.layers.{i}.self_attn.int4_in_proj")

        layer.self_attn.out_proj = LoRAAdapterLinear(layer.self_attn.out_proj, r=r, alpha=alpha)
        wrapped.append(f"transformer.layers.{i}.self_attn.out_proj")

        layer.gating.linear_in = LoRAAdapterLinear(layer.gating.linear_in, r=r, alpha=alpha)
        wrapped.append(f"transformer.layers.{i}.gating.linear_in")

        layer.gating.linear_out = LoRAAdapterLinear(layer.gating.linear_out, r=r, alpha=alpha)
        wrapped.append(f"transformer.layers.{i}.gating.linear_out")

    return wrapped


def lora_parameters(model):
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield p


def extract_lora_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapterLinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def main():
    parser = argparse.ArgumentParser(description="LQEC teacher-student distillation for temporal INT4 base")
    parser.add_argument("--teacher", default="v5_step1500.safetensors")
    parser.add_argument("--student", default="bmo_temporal_int4_base.pt")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="lqec_overfit_step50.pt")
    parser.add_argument("--log-json", default="lqec_overfit_log.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(__file__).resolve().parent
    teacher_path = (root / args.teacher).resolve()
    student_path = (root / args.student).resolve()
    mimi_weight = (root / args.mimi_weight).resolve()
    tokenizer_path = (root / args.tokenizer).resolve()
    input_wav = (root / args.input_wav).resolve()
    out_path = (root / args.out).resolve()
    log_json_path = (root / args.log_json).resolve()

    for p in [teacher_path, student_path, mimi_weight, tokenizer_path, input_wav]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    try:
        print(f"[INFO] Loading teacher BF16 on {args.device}: {teacher_path}")
        teacher = loaders.get_moshi_lm(str(teacher_path), device=args.device, dtype=torch.bfloat16, cpu_offload=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        print(f"[INFO] Loading student INT4 on {args.device}: {student_path}")
        student = loaders.get_moshi_lm(str(student_path), device=args.device, cpu_offload=False)
        student.train()
        for p in student.parameters():
            p.requires_grad = False

        print(f"[INFO] Loading Mimi for token extraction: {mimi_weight}")
        mimi = loaders.get_mimi(str(mimi_weight), args.device)
        mimi.eval()
        for p in mimi.parameters():
            p.requires_grad = False

        tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        wrapped = inject_lora_on_quantized_temporal(student, r=args.rank, alpha=args.alpha)
        print(f"[INFO] Injected LoRA wrappers: {len(wrapped)}")

        params = list(lora_parameters(student))
        if not params:
            raise RuntimeError("No LoRA parameters found after injection")

        optimizer = torch.optim.AdamW(params, lr=args.lr)

        forced_tokens = build_forced_tokens(
            model=student,
            mimi=mimi,
            tokenizer=tokenizer,
            input_wav=input_wav,
            text_prompt=args.text_prompt,
            steps=args.steps,
            device=args.device,
        )
        print(f"[INFO] Forced token stream shape: {tuple(forced_tokens.shape)}")  # (S, K)

        losses = []
        # Non-streaming loop for stable teacher-forced prefix distillation.
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            # forced_tokens prefix: [S, K] -> [1, K, S]
            seq_prefix = forced_tokens[: step + 1].clone().detach().to(torch.long).contiguous()
            seq_t = seq_prefix.transpose(0, 1).unsqueeze(0).contiguous()

            with torch.no_grad():
                _, teacher_logits = teacher.forward_codes(seq_t)

            _, student_logits = student.forward_codes(seq_t)

            loss = F.mse_loss(student_logits, teacher_logits.detach())
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            print(f"[STEP {step+1:03d}/{args.steps}] loss={loss_val:.6f}")

        save_obj = {
            "rank": args.rank,
            "alpha": args.alpha,
            "steps": args.steps,
            "lr": args.lr,
            "teacher": str(teacher_path),
            "student": str(student_path),
            "wrapped_modules": wrapped,
            "losses": losses,
            "lora_state_dict": extract_lora_state(student),
        }
        torch.save(save_obj, out_path)

        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "steps": args.steps,
                    "initial_loss": losses[0] if losses else None,
                    "final_loss": losses[-1] if losses else None,
                    "min_loss": min(losses) if losses else None,
                    "teacher": str(teacher_path),
                    "student": str(student_path),
                    "wrapped_count": len(wrapped),
                    "wrapped_modules": wrapped,
                },
                f,
                indent=2,
            )

        print(f"[INFO] Saved LoRA checkpoint: {out_path}")
        print(f"[INFO] Saved training log: {log_json_path}")
        if losses:
            print(f"[INFO] Loss: initial={losses[0]:.6f} final={losses[-1]:.6f} min={min(losses):.6f}")

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda error" in msg:
            print("[OOM] CUDA memory failure during LQEC run.")
            print(f"[OOM] Details: {e}")
            raise
        raise


if __name__ == "__main__":
    main()
