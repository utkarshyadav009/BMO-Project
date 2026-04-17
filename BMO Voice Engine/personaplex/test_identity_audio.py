import argparse
import math
import os
import sys
import time
import wave
from pathlib import Path

import torch

from moshi import offline
from moshi.models import loaders


def parse_dtype(name: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float32":
        return torch.float32
    raise argparse.ArgumentTypeError("--dtype must be one of: bfloat16, float32")


def resolve_local_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def read_wav_duration_seconds(wav_path: Path) -> float:
    if not wav_path.exists() or wav_path.stat().st_size <= 0:
        return 0.0
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
    if sample_rate <= 0:
        return 0.0
    return float(frame_count) / float(sample_rate)


def format_float(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.6f}"
    return "inf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline BMO audio from identity student checkpoint.")
    parser.add_argument("--ckpt", default="bmo_slicegpt_4096_identity_quarot_fix1.pt")
    parser.add_argument("--voice-prompt", default="bmo_621.wav")
    parser.add_argument("--input-wav", default="tellmeajoke_padded.wav")
    parser.add_argument("--text-prompt", default="Tell me a joke.")
    parser.add_argument("--mimi-weight", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    parser.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    parser.add_argument("--out", default="identity_student_output.wav")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ckpt_path = resolve_local_path(root, args.ckpt)
    voice_prompt_path = resolve_local_path(root, args.voice_prompt)
    input_wav_path = resolve_local_path(root, args.input_wav)
    mimi_weight_path = resolve_local_path(root, args.mimi_weight)
    tokenizer_path = resolve_local_path(root, args.tokenizer)
    output_wav_path = resolve_local_path(root, args.out)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path = output_wav_path.parent / f"{output_wav_path.stem}.json"

    generation_time_sec = 0.0
    output_duration_sec = 0.0
    passed = False

    start_time = time.perf_counter()
    try:
        if os.environ.get("BMO_ENABLE_RUNTIME_PATCHES") == "1":
            raise RuntimeError("BMO_ENABLE_RUNTIME_PATCHES=1 is not allowed for this audio test.")

        dtype = parse_dtype(args.dtype)

        for required in [ckpt_path, voice_prompt_path, input_wav_path, mimi_weight_path, tokenizer_path]:
            if not required.exists():
                raise FileNotFoundError(f"Required file not found: {required}")

        original_get_moshi_lm = loaders.get_moshi_lm

        def _loader_with_dtype(*l_args, **l_kwargs):
            l_kwargs.setdefault("dtype", dtype)
            return original_get_moshi_lm(*l_args, **l_kwargs)

        loaders.get_moshi_lm = _loader_with_dtype
        try:
            with torch.no_grad():
                offline.run_inference(
                    input_wav=str(input_wav_path),
                    output_wav=str(output_wav_path),
                    output_text=str(output_text_path),
                    text_prompt=args.text_prompt,
                    voice_prompt_path=str(voice_prompt_path),
                    tokenizer_path=str(tokenizer_path),
                    moshi_weight=str(ckpt_path),
                    mimi_weight=str(mimi_weight_path),
                    hf_repo=loaders.DEFAULT_REPO,
                    device=str(args.device),
                    seed=-1,
                    temp_audio=0.8,
                    temp_text=0.7,
                    topk_audio=250,
                    topk_text=25,
                    greedy=False,
                    save_voice_prompt_embeddings=False,
                    cpu_offload=False,
                )
        finally:
            loaders.get_moshi_lm = original_get_moshi_lm

        generation_time_sec = float(time.perf_counter() - start_time)
        output_duration_sec = read_wav_duration_seconds(output_wav_path)
        passed = (
            output_wav_path.exists()
            and output_wav_path.stat().st_size > 0
            and output_duration_sec > 1.0
        )
    except Exception as exc:
        generation_time_sec = float(time.perf_counter() - start_time)
        output_duration_sec = read_wav_duration_seconds(output_wav_path)
        print(f"[ERROR] {exc}")
        passed = False

    realtime_factor = (
        generation_time_sec / output_duration_sec
        if output_duration_sec > 0.0
        else float("inf")
    )

    print(f"[RESULT] checkpoint = {ckpt_path}")
    print(f"[RESULT] output_wav = {output_wav_path}")
    print(f"[RESULT] output_duration_sec = {format_float(output_duration_sec)}")
    print(f"[RESULT] generation_time_sec = {format_float(generation_time_sec)}")
    print(f"[RESULT] realtime_factor = {format_float(realtime_factor)}")
    print(f"[RESULT] {'PASS' if passed else 'FAIL'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
