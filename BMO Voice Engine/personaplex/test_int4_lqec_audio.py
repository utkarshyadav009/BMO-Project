import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import bitsandbytes as bnb
import bitsandbytes.functional as bnb_functional
from bitsandbytes.nn.modules import Params4bit
import torch
import torch.nn as nn

from moshi.models.lm import LMModel
from moshi.offline import run_inference

from verify_int4_rollout_drift import apply_lora_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="INT4 plus optional LQEC audio generation via vanilla moshi.offline")
    parser.add_argument(
        "--int4-ckpt",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/bmo_temporal_int4_base.pt",
    )
    parser.add_argument("--lora-ckpt", default=None)
    parser.add_argument("--voice-prompt", default="bmo_621.wav")
    parser.add_argument(
        "--voice-prompt-dir",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/",
    )
    parser.add_argument(
        "--input-wav",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tellmeajoke_padded.wav",
    )
    parser.add_argument("--text-prompt", default="You are BMO.")
    parser.add_argument(
        "--mimi-weight",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        default="/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer_spm_32k_3.model",
    )
    parser.add_argument("--out-wav", default="/tmp/int4_test/output.wav")
    parser.add_argument("--out-text", default="/tmp/int4_test/output.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _extract_first_30_text_tokens(combined_output: str):
    # offline logs text as: text token '<piece>'
    tokens = re.findall(r"text token '([^']*)'", combined_output)
    return tokens[:30]


def _swap_selected_linears_to_4bit(
    module: nn.Module,
    quantized_weight_keys: set[str],
    prefix: str = "",
) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and f"{full_name}.weight" in quantized_weight_keys:
            setattr(
                module,
                child_name,
                bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                ),
            )
            replaced += 1
        else:
            replaced += _swap_selected_linears_to_4bit(child, quantized_weight_keys, full_name)
    return replaced


def _unwrap_payload(loaded_obj):
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
        return loaded_obj["state_dict"], loaded_obj.get("config_override")
    return loaded_obj, None


def _load_int4_prequant_model(ckpt_path, device, dtype):
    ckpt_path = str(Path(ckpt_path).resolve())
    root = Path(__file__).resolve().parent

    loaded_obj = torch.load(ckpt_path, map_location="cpu", mmap=True)
    state_dict, config_override = _unwrap_payload(loaded_obj)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint did not contain a valid state_dict dictionary")

    quant_suffix = ".quant_state.bitsandbytes__nf4"
    quant_bases = [k[:-len(quant_suffix)] for k in state_dict.keys() if k.endswith(quant_suffix)]
    if len(quant_bases) == 0:
        raise RuntimeError("No prequant metadata found in checkpoint")
    quantized_weight_keys = {b for b in quant_bases if b.endswith(".weight")}
    expected_linear4bit = len(quantized_weight_keys)

    config_path = (root / "bmo_config.json").resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("model_type", None)
    if isinstance(config_override, dict):
        cfg.update(config_override)

    with torch.device("meta"):
        model = LMModel(dtype=dtype, **cfg)

    n_replaced = _swap_selected_linears_to_4bit(model, quantized_weight_keys)
    if n_replaced == 0:
        raise RuntimeError("No quantized linear modules were replaced with Linear4bit")
    if n_replaced != expected_linear4bit:
        raise RuntimeError(
            f"Replaced {n_replaced} quantized linear modules, expected {expected_linear4bit} from checkpoint"
        )

    quant_base_set = set(quant_bases)
    dense_sd = {}
    for key, value in state_dict.items():
        if (
            key.endswith(".absmax")
            or key.endswith(".quant_map")
            or key.endswith(".nested_absmax")
            or key.endswith(".nested_quant_map")
            or key.endswith(".quant_state.bitsandbytes__nf4")
        ):
            continue
        if key in quant_base_set:
            continue
        dense_sd[key] = value

    model.load_state_dict(dense_sd, strict=False, assign=True)

    modules = dict(model.named_modules())
    loaded_linear4bit = 0

    for base in quant_bases:
        packed = state_dict[base]
        prefix = base + "."
        stats = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        if base.endswith(".weight"):
            module_name = base[:-len(".weight")]
            module = modules.get(module_name)
            if module is None or not isinstance(module, bnb.nn.Linear4bit):
                continue
            module.weight = Params4bit.from_prequantized(
                packed,
                stats,
                requires_grad=False,
                device="cpu",
                module=module,
            )
            loaded_linear4bit += 1
            continue

        if base.endswith(".in_proj_weight"):
            attn_name = base[:-len(".in_proj_weight")]
            attn_module = modules.get(attn_name)
            if attn_module is None or not hasattr(attn_module, "in_proj_weight"):
                continue

            qparam = Params4bit.from_prequantized(
                packed,
                stats,
                requires_grad=False,
                device="cpu",
                module=None,
            )
            if getattr(qparam, "quant_state", None) is None:
                raise RuntimeError(f"Missing quant_state for packed tensor: {base}")

            dense_weight = bnb_functional.dequantize_4bit(
                qparam.data,
                quant_state=qparam.quant_state,
                quant_type="nf4",
            )
            dense_weight = dense_weight.to(dtype=dtype, device="cpu").contiguous()
            setattr(attn_module, "in_proj_weight", nn.Parameter(dense_weight, requires_grad=False))

    if loaded_linear4bit == 0:
        raise RuntimeError("No Linear4bit modules loaded — checkpoint may not be prequant format")
    if loaded_linear4bit != expected_linear4bit:
        raise RuntimeError(
            f"Loaded {loaded_linear4bit} prequantized Linear4bit weights, expected {expected_linear4bit}"
        )

    # Keep packed INT4 buffers in their storage dtype; only move devices.
    model.to(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model._n_linear4bit_loaded = int(loaded_linear4bit)
    return model


def _apply_lora_if_requested(model, lora_ckpt_path):
    if lora_ckpt_path is None:
        return 0

    lora_path = Path(lora_ckpt_path).resolve()
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    ckpt = torch.load(str(lora_path), map_location="cpu")
    wrapped = list(ckpt.get("wrapped_modules", [])) if isinstance(ckpt, dict) else []
    apply_lora_ckpt(model, lora_path)
    return int(len(wrapped))


def main():
    args = parse_args()

    int4_ckpt = str(Path(args.int4_ckpt).resolve())
    lora_ckpt_value = str(Path(args.lora_ckpt).resolve()) if args.lora_ckpt else "none"
    output_wav = str(Path(args.out_wav).resolve())

    output_wav_bytes = 0
    n_linear4bit_loaded = 0
    n_lora_modules_wrapped = 0
    first_30_text_tokens = []
    passed = False

    error_message = None
    captured_output = ""

    try:
        if os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") == "1":
            raise RuntimeError("BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for test_int4_lqec_audio.py")

        if os.environ.get("MOSHI_ENABLE_INT4") == "1":
            raise RuntimeError("MOSHI_ENABLE_INT4=1 is not allowed for test_int4_lqec_audio.py")

        root = Path(__file__).resolve().parent

        out_wav_path = Path(args.out_wav).resolve()
        out_text_path = Path(args.out_text).resolve()
        out_wav_path.parent.mkdir(parents=True, exist_ok=True)
        out_text_path.parent.mkdir(parents=True, exist_ok=True)

        preloaded_model = _load_int4_prequant_model(args.int4_ckpt, device=str(args.device), dtype=torch.bfloat16)
        n_linear4bit_loaded = int(getattr(preloaded_model, "_n_linear4bit_loaded", 0))
        n_lora_modules_wrapped = _apply_lora_if_requested(preloaded_model, args.lora_ckpt)

        import moshi.models.loaders as _loaders

        _original_get_moshi_lm = _loaders.get_moshi_lm

        def _preloaded_get_moshi_lm(*_args, **_kwargs):
            return preloaded_model

        _loaders.get_moshi_lm = _preloaded_get_moshi_lm
        restore_error = None

        try:
            run_stdout = io.StringIO()
            run_stderr = io.StringIO()
            with redirect_stdout(run_stdout), redirect_stderr(run_stderr):
                run_inference(
                    input_wav=str(Path(args.input_wav).resolve()),
                    output_wav=str(out_wav_path),
                    output_text=str(out_text_path),
                    text_prompt=str(args.text_prompt),
                    voice_prompt_path=str((Path(args.voice_prompt_dir).resolve() / args.voice_prompt).resolve()),
                    tokenizer_path=str(Path(args.tokenizer).resolve()),
                    moshi_weight=str(Path(args.int4_ckpt).resolve()),
                    mimi_weight=str(Path(args.mimi_weight).resolve()),
                    hf_repo="nvidia/personaplex-7b-v1",
                    device=str(args.device),
                    seed=1234,
                    temp_audio=0.8,
                    temp_text=0.7,
                    topk_audio=250,
                    topk_text=25,
                    greedy=False,
                    save_voice_prompt_embeddings=False,
                    cpu_offload=False,
                )

            captured_output = run_stdout.getvalue() + "\n" + run_stderr.getvalue()
        finally:
            try:
                _loaders.get_moshi_lm = _original_get_moshi_lm
                if _loaders.get_moshi_lm is not _original_get_moshi_lm:
                    raise RuntimeError("Failed to restore moshi.models.loaders.get_moshi_lm")
            except Exception as exc:
                restore_error = exc

        if restore_error is not None:
            raise RuntimeError(str(restore_error))

        first_30_text_tokens = _extract_first_30_text_tokens(captured_output)

        out_wav_exists = out_wav_path.exists()
        if out_wav_exists:
            output_wav_bytes = int(out_wav_path.stat().st_size)

        if n_linear4bit_loaded == 0:
            raise RuntimeError("No Linear4bit modules loaded — checkpoint may not be prequant format")

        if output_wav_bytes <= 100_000:
            raise RuntimeError("Output WAV missing or too small (<100 KB)")

        if len(first_30_text_tokens) == 0:
            raise RuntimeError("Token stream is empty")

        passed = True

    except Exception as exc:
        error_message = str(exc)
        passed = False

    if error_message:
        print(f"[ERROR] {error_message}")
        if captured_output.strip():
            print("[ERROR] run_inference output (tail):")
            tail = captured_output.strip().splitlines()[-40:]
            for line in tail:
                print(line)

    print(f"[RESULT] int4_ckpt = {int4_ckpt}")
    print(f"[RESULT] lora_ckpt = {lora_ckpt_value}")
    print(f"[RESULT] n_linear4bit_loaded = {int(n_linear4bit_loaded)}")
    print(f"[RESULT] n_lora_modules_wrapped = {int(n_lora_modules_wrapped)}")
    print(f"[RESULT] output_wav = {output_wav}")
    print(f"[RESULT] output_wav_bytes = {int(output_wav_bytes)}")
    print(f"[RESULT] first_30_text_tokens = {first_30_text_tokens}")
    print(f"[RESULT] {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
