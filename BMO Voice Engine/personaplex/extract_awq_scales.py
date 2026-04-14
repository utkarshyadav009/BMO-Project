import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# Import applies the same stable runtime monkey patches used in prior probes.
import test_rtx_edge  # noqa: F401
from moshi import offline
from moshi.models import loaders

# Avoid extra first-layer probes from test_rtx_edge in this calibration script.
test_rtx_edge._attach_activation_probes = lambda model: None


def build_calibration_hook(name: str, awq_scales: Dict[str, Dict[str, torch.Tensor]]):
    def hook(module, input, output):
        if not input:
            return
        x = input[0]
        if not isinstance(x, torch.Tensor):
            return
        if x.dim() != 3:
            return

        # input[0]: [batch, seq_len, in_features]
        x_abs = x.detach().float().abs()
        x_flat = x_abs.view(-1, x_abs.shape[-1])

        channel_max = x_flat.max(dim=0)[0]
        p995 = torch.quantile(x_flat, 0.995, dim=0)
        p999 = torch.quantile(x_flat, 0.999, dim=0)

        if name in awq_scales:
            awq_scales[name]["max"] = torch.maximum(awq_scales[name]["max"], channel_max)
            awq_scales[name]["p995"] = torch.maximum(awq_scales[name]["p995"], p995)
            awq_scales[name]["p999"] = torch.maximum(awq_scales[name]["p999"], p999)
        else:
            awq_scales[name] = {
                "max": channel_max,
                "p995": p995,
                "p999": p999,
            }

    return hook


def attach_awq_hooks(model, awq_scales: Dict[str, Dict[str, torch.Tensor]]) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []
    for name, module in model.transformer.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(build_calibration_hook(name, awq_scales)))

        # Fused attention projection path: capture per-channel scale for in_proj_weight.
        if hasattr(module, "in_proj_weight") and isinstance(module.in_proj_weight, torch.Tensor):
            fused_name = f"{name}.in_proj_weight"
            handles.append(module.register_forward_hook(build_calibration_hook(fused_name, awq_scales)))
    print(f"[INFO] Attached AWQ hooks to {len(handles)} temporal transformer linear layers")
    return handles


def validate_scales(
    model, awq_scales: Dict[str, Dict[str, torch.Tensor]]
) -> tuple[int, int, List[str]]:
    expected: Dict[str, int] = {}
    for name, module in model.transformer.named_modules():
        if isinstance(module, nn.Linear):
            expected[name] = int(module.in_features)
        if hasattr(module, "in_proj_weight") and isinstance(module.in_proj_weight, torch.Tensor):
            expected[f"{name}.in_proj_weight"] = int(module.in_proj_weight.shape[1])

    matches = 0
    mismatches = 0
    mismatch_lines: List[str] = []

    for name, scale_dict in awq_scales.items():
        exp = expected.get(name)
        if exp is None:
            mismatches += 1
            mismatch_lines.append(f"{name}: missing module metadata")
            continue
        for stat_name in ["max", "p995", "p999"]:
            if stat_name not in scale_dict:
                mismatches += 1
                mismatch_lines.append(f"{name}: missing stat '{stat_name}'")
                continue
            got = int(scale_dict[stat_name].numel())
            if got == exp:
                matches += 1
            else:
                mismatches += 1
                mismatch_lines.append(f"{name}.{stat_name}: scale_len={got}, in_features={exp}")

    return matches, mismatches, mismatch_lines


def main():
    parser = argparse.ArgumentParser(description="Extract global AWQ activation scales from offline BF16 calibration pass")
    parser.add_argument("--moshi-weight", default="v5_step1500.safetensors")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--voice-prompt", default="bmo_621.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--output-scales", default="bmo_awq_scales.pt")
    parser.add_argument("--output-wav", default="awq_calibration_out.wav")
    parser.add_argument("--output-text", default="awq_calibration_text.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base = Path(__file__).resolve().parent
    moshi_weight = (base / args.moshi_weight).resolve()
    mimi_weight = (base / args.mimi_weight).resolve()
    tokenizer = (base / args.tokenizer).resolve()
    input_wav = (base / args.input_wav).resolve()
    voice_prompt = (base / args.voice_prompt).resolve()
    output_scales = (base / args.output_scales).resolve()
    output_wav = (base / args.output_wav).resolve()
    output_text = (base / args.output_text).resolve()

    for required in [moshi_weight, mimi_weight, tokenizer, input_wav, voice_prompt]:
        if not required.exists():
            raise FileNotFoundError(f"Required calibration asset not found: {required}")

    awq_scales: Dict[str, Dict[str, torch.Tensor]] = {}
    captured = {}

    original_get_moshi_lm = loaders.get_moshi_lm

    def calibration_get_moshi_lm(*l_args, **l_kwargs):
        model = original_get_moshi_lm(*l_args, **l_kwargs)
        print("[INFO] Attaching AWQ Calibration Hooks...")
        handles = attach_awq_hooks(model, awq_scales)
        captured["model"] = model
        captured["handles"] = handles
        return model

    loaders.get_moshi_lm = calibration_get_moshi_lm

    print("[INFO] Running Calibration Pass (offline harness)...")
    try:
        with torch.no_grad():
            offline.run_inference(
                input_wav=str(input_wav),
                output_wav=str(output_wav),
                output_text=str(output_text),
                text_prompt=args.text_prompt,
                voice_prompt_path=str(voice_prompt),
                tokenizer_path=str(tokenizer),
                moshi_weight=str(moshi_weight),
                mimi_weight=str(mimi_weight),
                hf_repo=loaders.DEFAULT_REPO,
                device=args.device,
                seed=args.seed,
                temp_audio=args.temp_audio,
                temp_text=args.temp_text,
                topk_audio=args.topk_audio,
                topk_text=args.topk_text,
                greedy=bool(args.greedy),
                save_voice_prompt_embeddings=False,
                cpu_offload=False,
            )
    finally:
        loaders.get_moshi_lm = original_get_moshi_lm
        for handle in captured.get("handles", []):
            handle.remove()

    print("[INFO] Saving AWQ Scales...")
    awq_scales_cpu = {
        k: {sk: sv.detach().float().cpu() for sk, sv in stats.items()}
        for k, stats in awq_scales.items()
    }
    torch.save(awq_scales_cpu, output_scales)
    print(f"[INFO] Extracted scales for {len(awq_scales_cpu)} layers")

    if "model" in captured:
        matches, mismatches, mismatch_lines = validate_scales(captured["model"], awq_scales_cpu)
        print(f"[INFO] Scale shape checks: matches={matches}, mismatches={mismatches}")
        if mismatch_lines:
            print("[WARN] Mismatch details (up to 10):")
            for line in mismatch_lines[:10]:
                print(f"  - {line}")

    if awq_scales_cpu:
        sample_keys = sorted(awq_scales_cpu.keys())[:8]
        print("[INFO] Sample extracted layers:")
        for key in sample_keys:
            stats = awq_scales_cpu[key]
            p995 = stats["p995"]
            p999 = stats["p999"]
            vmax = stats["max"]
            print(
                f"  - {key}: shape={tuple(p995.shape)}, "
                f"p995_max={p995.max().item():.4f}, p999_max={p999.max().item():.4f}, absmax={vmax.max().item():.4f}"
            )


if __name__ == "__main__":
    main()
