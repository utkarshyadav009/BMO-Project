import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from moshi.models import loaders
from moshi.models.lm import LMModel

from apply_slicegpt import (
    build_forced_tokens,
    build_real_forced_tokens,
    parse_bool,
)


def load_lm_for_tuning(checkpoint_path: str, *, device: torch.device):
    resolved = str(Path(checkpoint_path).resolve())

    # Native checkpoints can use the standard loader directly.
    if resolved.lower().endswith((".safetensors", ".sft", ".sfts")):
        return loaders.get_moshi_lm(resolved, device=device, cpu_offload=False)

    loaded_obj = torch.load(resolved, map_location="cpu")
    if not (isinstance(loaded_obj, dict) and "state_dict" in loaded_obj):
        return loaders.get_moshi_lm(resolved, device=device, cpu_offload=False)

    print("[INFO] Dense payload checkpoint detected; loading embedded state_dict/config_override")

    root = Path(__file__).resolve().parent
    config_path = (root / "bmo_config.json").resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict.pop("model_type", None)

    config_override = loaded_obj.get("config_override")
    if isinstance(config_override, dict):
        config_dict.update(config_override)

    model = LMModel(device="cpu", dtype=torch.bfloat16, **config_dict)
    model.eval()

    state_dict = loaded_obj["state_dict"]
    incompat = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompat.unexpected_keys:
        print(f"[WARN] Dense payload unexpected keys: {len(incompat.unexpected_keys)}")
    if incompat.missing_keys:
        print(f"[WARN] Dense payload missing keys: {len(incompat.missing_keys)}")

    model.to(device=device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Logit-aware bridge tuning for SliceGPT checkpoints")
    parser.add_argument("--teacher-ckpt", default="v5_step1500.safetensors")
    parser.add_argument("--student-ckpt", default="bmo_slicegpt_2816_hd_gain.pt")
    parser.add_argument("--out", default="bmo_slicegpt_2816_bridge_kl.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tune-bias", type=parse_bool, default=False)

    parser.add_argument("--fit-token-source", choices=["real", "random"], default="real")
    parser.add_argument("--fit-steps", type=int, default=20256)
    parser.add_argument("--high-density-ls", type=parse_bool, default=True)
    parser.add_argument("--fit-dataset-dir", default="bmo_dataset_clean")
    parser.add_argument("--fit-min-keep-rows", type=int, default=20000)
    parser.add_argument("--fit-max-audio-files", type=int, default=0)
    parser.add_argument("--fit-allow-audio-reuse", type=parse_bool, default=True)

    parser.add_argument("--fit-input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--fit-voice-prompt-wav", default="bmo_621.wav")
    parser.add_argument("--fit-text-prompt", default="Tell me a joke.")
    parser.add_argument("--fit-mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--fit-tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--fit-voice-ratio", type=float, default=0.25)
    return parser.parse_args()


@torch.no_grad()
def build_calibration_tokens(args, teacher_model, root: Path, device: torch.device) -> torch.Tensor:
    if args.fit_token_source == "random":
        return build_forced_tokens(
            teacher_model,
            steps=int(args.fit_steps),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
        )

    fit_steps = int(args.fit_steps)
    if bool(args.high_density_ls):
        fit_steps = max(fit_steps, int(args.fit_min_keep_rows) + 256)

    return build_real_forced_tokens(
        teacher_model,
        root=root,
        steps=int(fit_steps),
        batch_size=int(args.batch_size),
        extract_device=device,
        input_wav=args.fit_input_wav,
        voice_prompt_wav=args.fit_voice_prompt_wav,
        text_prompt=args.fit_text_prompt,
        mimi_weight=args.fit_mimi_weight,
        tokenizer_path=args.fit_tokenizer,
        voice_ratio=float(args.fit_voice_ratio),
        high_density_ls=bool(args.high_density_ls),
        dataset_dir=args.fit_dataset_dir,
        min_keep_rows=int(args.fit_min_keep_rows),
        max_audio_files=int(args.fit_max_audio_files),
        allow_audio_reuse=bool(args.fit_allow_audio_reuse),
    )


def get_trainable_bridge_params(student_model, tune_bias: bool):
    out_proj = getattr(getattr(student_model, "transformer", None), "output_proj", None)
    if out_proj is None or not hasattr(out_proj, "weight"):
        raise RuntimeError("Student model has no transformer.output_proj weight to tune")

    for p in student_model.parameters():
        p.requires_grad = False

    out_proj.weight.requires_grad = True
    trainables = [out_proj.weight]

    if tune_bias and hasattr(out_proj, "bias") and isinstance(out_proj.bias, torch.Tensor):
        out_proj.bias.requires_grad = True
        trainables.append(out_proj.bias)

    return out_proj, trainables


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    teacher_path = str((root / args.teacher_ckpt).resolve())
    student_path = str((root / args.student_ckpt).resolve())

    print(f"[INFO] Loading teacher: {teacher_path}")
    teacher = load_lm_for_tuning(teacher_path, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[INFO] Loading student: {student_path}")
    student = load_lm_for_tuning(student_path, device=device)
    student.eval()

    out_proj, trainables = get_trainable_bridge_params(student, tune_bias=bool(args.tune_bias))
    print(f"[INFO] Trainable bridge params: {sum(p.numel() for p in trainables):,}")

    forced_tokens = build_calibration_tokens(args, teacher, root, device)
    total_forced_steps = int(forced_tokens.shape[0])
    print(f"[INFO] Calibration stream shape: steps={forced_tokens.shape[0]} batch={forced_tokens.shape[1]} K={forced_tokens.shape[2]}")

    optimizer = torch.optim.AdamW(trainables, lr=float(args.lr), weight_decay=float(args.weight_decay))

    steps_to_run = min(int(args.train_steps), total_forced_steps)
    if steps_to_run <= 0:
        raise ValueError("No training steps requested")

    loss_ema = None
    with teacher.streaming(batch_size=int(args.batch_size)), student.streaming(batch_size=int(args.batch_size)):
        for step in range(steps_to_run):
            seq = forced_tokens[step].unsqueeze(-1).contiguous().to(device)

            with torch.no_grad():
                _, teacher_logits = teacher.forward_codes(seq)

            _, student_logits = student.forward_codes(seq)

            t_logits = teacher_logits[:, 0, 0, :].float()
            s_logits = student_logits[:, 0, 0, :].float()

            loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.clip_grad) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainables, float(args.clip_grad))
            optimizer.step()

            loss_val = float(loss.item())
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss_val

            if ((step + 1) % int(args.log_interval) == 0) or (step + 1 == steps_to_run):
                print(
                    f"[INFO] step={step + 1}/{steps_to_run} "
                    f"loss={loss_val:.6f} loss_ema={loss_ema:.6f}"
                )

    out_path = (root / args.out).resolve()
    src_payload = torch.load(student_path, map_location="cpu")

    with torch.no_grad():
        tuned_weight = out_proj.weight.detach().cpu().contiguous()
        tuned_bias = out_proj.bias.detach().cpu().contiguous() if hasattr(out_proj, "bias") and isinstance(out_proj.bias, torch.Tensor) else None

    if isinstance(src_payload, dict) and "state_dict" in src_payload:
        state_dict = src_payload["state_dict"]
        state_dict["transformer.output_proj.weight"] = tuned_weight.to(state_dict["transformer.output_proj.weight"].dtype)
        if tuned_bias is not None and "transformer.output_proj.bias" in state_dict:
            state_dict["transformer.output_proj.bias"] = tuned_bias.to(state_dict["transformer.output_proj.bias"].dtype)

        meta = src_payload.setdefault("slicegpt_meta", {})
        meta["bridge_logit_tuning"] = {
            "enabled": True,
            "teacher_ckpt": teacher_path,
            "student_ckpt": student_path,
            "train_steps": int(steps_to_run),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "token_source": args.fit_token_source,
            "high_density_ls": bool(args.high_density_ls),
            "fit_steps": int(forced_tokens.shape[0]),
            "fit_dataset_dir": args.fit_dataset_dir,
        }
        out_payload = src_payload
    else:
        # Fallback: export a plain dense payload.
        export_sd = {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}
        out_payload = {
            "state_dict": export_sd,
            "model_mode": "slicegpt_dense",
            "force_dense": True,
            "bridge_logit_tuning": {
                "enabled": True,
                "teacher_ckpt": teacher_path,
                "student_ckpt": student_path,
                "train_steps": int(steps_to_run),
                "lr": float(args.lr),
            },
        }

    torch.save(out_payload, out_path)
    print(f"[INFO] Saved tuned checkpoint: {out_path}")


if __name__ == "__main__":
    main()
