#!/usr/bin/env python3
"""Run moshi.offline with native 2-bit turbo2bit weights (experiment wrapper)."""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("NO_CUDA_GRAPH", "1")
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import sphn
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from native2bit_loader import patch_moshi_loaders

DEFAULT_MODEL_DIR = EXPERIMENT_ROOT / "models" / "personaplex-7b-turbo2bit"


def ensure_voice_prompt(voice_dir: Path, hf_repo: str) -> Path:
    voice_dir.mkdir(parents=True, exist_ok=True)
    wav_prompt = voice_dir / "synthetic_voice_prompt.wav"
    if wav_prompt.is_file():
        return wav_prompt

    candidates = list(voice_dir.glob("*.pt")) + list(voice_dir.glob("*.PT"))
    if candidates:
        return candidates[0]

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        print(f"Downloading voices.tgz from {hf_repo}...")
        try:
            voices_tgz = hf_hub_download(hf_repo, "voices.tgz", token=token)
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(voice_dir.parent)
            candidates = list(voice_dir.glob("*.pt")) + list(voice_dir.parent.rglob("*.pt"))
            if candidates:
                return candidates[0]
        except Exception as exc:
            print(f"Could not download voices.tgz ({exc}); using synthetic WAV prompt.")

    # WAV prompts work without gated HF access (load_voice_prompt path in moshi).
    sample_rate = 24000
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = (0.12 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    sphn.write_wav(str(wav_prompt), audio, sample_rate)
    print(f"Created synthetic voice prompt WAV: {wav_prompt}")
    return wav_prompt


def ensure_input_wav(wav_path: Path, duration_s: float = 0.5, sample_rate: int = 24000) -> None:
    if wav_path.is_file():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sphn.write_wav(str(wav_path), audio, sample_rate)
    print(f"Created synthetic input WAV: {wav_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moshi-weight",
        type=Path,
        default=DEFAULT_MODEL_DIR / "model-turbo2bit.safetensors",
    )
    parser.add_argument(
        "--mimi-weight",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_MODEL_DIR / "tokenizer_spm_32k_3.model",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "input_user.wav",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "output_agent.wav",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "output_agent.json",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "voices",
    )
    parser.add_argument(
        "--voice-prompt",
        type=str,
        default="",
        help="Basename inside voice-prompt-dir; auto-picked if omitted.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="nvidia/personaplex-7b-v1",
        help="Repo for voices.tgz (requires accepted license + HF_TOKEN if gated).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    ensure_input_wav(args.input_wav)

    voice_pt = ensure_voice_prompt(args.voice_prompt_dir, args.hf_repo)
    if args.voice_prompt:
        voice_path = args.voice_prompt_dir / args.voice_prompt
    else:
        voice_path = voice_pt

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)

    patch_moshi_loaders()

    # BMO fork imports libbmo.so at module load; stub it for turbo2bit PyTorch-only runs.
    import sys
    import types

    if "moshi.bmo_engine" not in sys.modules:
        bmo_stub = types.ModuleType("moshi.bmo_engine")

        class _BMOEngineStub:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("BMOEngine stub — set BMO_USE_CPP only with libbmo.so")

        bmo_stub.BMOEngine = _BMOEngineStub
        sys.modules["moshi.bmo_engine"] = bmo_stub

    os.environ["BMO_USE_CPP"] = "0"
    from moshi.offline import run_inference

    print("Running native 2-bit offline inference...")
    run_inference(
        input_wav=str(args.input_wav),
        output_wav=str(args.output_wav),
        output_text=str(args.output_text),
        text_prompt=(
            "You are a wise and friendly teacher. Answer questions or provide advice "
            "in a clear and engaging way."
        ),
        voice_prompt_path=str(voice_path),
        tokenizer_path=str(args.tokenizer),
        moshi_weight=str(args.moshi_weight),
        mimi_weight=str(args.mimi_weight),
        hf_repo=args.hf_repo,
        temp_audio=0.8,
        temp_text=0.7,
        topk_audio=250,
        topk_text=25,
        greedy=args.greedy,
        device=args.device,
        cpu_offload=False,
        seed=args.seed,
        save_voice_prompt_embeddings=False,
    )
    print(f"Wrote {args.output_wav}")
    print(f"Wrote {args.output_text}")


if __name__ == "__main__":
    main()
