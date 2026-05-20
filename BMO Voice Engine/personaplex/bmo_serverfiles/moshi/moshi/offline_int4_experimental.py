# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline inference entrypoint for PersonaPlex that mirrors server.py behavior without a WebSocket server.

High-level flow:
- Load Mimi encoders/decoders, Moshi LM, and tokenizer (same as server.py)
- Warmup to initialize CUDA graphs and streaming state
- Prompt phase: load system text tokens and a voice prompt WAV (agent side)
- Streaming-like phase: feed user audio frames from a WAV file into the "input" channels,
  autoregressively sample text + agent audio channels each step, and decode audio frames
- Concatenate generated frames and write an output WAV matching the input duration

This script reuses helpers from lm.py (load_audio, _iterate_audio, encode_from_sphn) to
keep parity with voice-prompt feeding logic in the server.
"""

import argparse
import os
import tarfile
from pathlib import Path
import json
from typing import Optional, List
import time 

import numpy as np
import torch
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders, LMGen, MimiModel
from .models.lm import load_audio as lm_load_audio
from .models.lm import _iterate_audio as lm_iterate_audio
from .models.lm import encode_from_sphn as lm_encode_from_sphn

import bitsandbytes as bnb
import torch
import torch.nn as nn
from moshi.modules.gating import ActivationGating


SKIP_MODULES = {"text_emb", "audio_emb", "out_norm", "audio_heads", "text_heads", "extra_heads", "mimi"}


def patched_gating_forward(self, x: torch.Tensor):
    # Route through standard __call__ so bnb hooks trigger
    x = self.linear_in(x)
    B, T, _ = x.shape
    x = x.view(B, T, 2, -1)
    x = self.activation(x[..., 0, :]) * x[..., 1, :]
    x = self.linear_out(x)
    return x
    
ActivationGating.forward = patched_gating_forward

import moshi.modules.transformer as moshi_transformer
from moshi.modules.transformer import KVCacheResult

class TurboRingKVCache(nn.Module):
    def __init__(self, batch_size: int, num_heads: int, dim_per_head: int, capacity: int, device: torch.device = torch.device("cuda"), dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("cache", torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=torch.int8))
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(dim_per_head, dim_per_head, device=device))
            self.register_buffer("rotation_q", q.to(dtype))
        self.register_buffer("scales", torch.ones((2, batch_size, num_heads, capacity, 1), device=device, dtype=dtype))
        self.register_buffer("end_offset", torch.zeros(1, device=device, dtype=torch.long))

    def reset(self):
        self.end_offset.zero_()

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        B, H, T, D = k.shape
        dtype = k.dtype

        k_rot = k @ self.rotation_q
        v_rot = v @ self.rotation_q

        num_levels = 2 ** 2.5
        k_s = k_rot.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / num_levels
        v_s = v_rot.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / num_levels

        k_int = (k_rot / k_s).round().clamp(-3, 3).to(torch.int8)
        v_int = (v_rot / v_s).round().clamp(-3, 3).to(torch.int8)

        indexes = (torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset) % self.capacity
        
        self.cache[0].index_copy_(2, indexes, k_int)
        self.cache[1].index_copy_(2, indexes, v_int)
        self.scales[0].index_copy_(2, indexes, k_s.to(dtype))
        self.scales[1].index_copy_(2, indexes, v_s.to(dtype))
        
        self.end_offset.add_(T)

        keys = (self.cache[0].to(dtype) * self.scales[0]) @ self.rotation_q.T
        values = (self.cache[1].to(dtype) * self.scales[1]) @ self.rotation_q.T

        indexes_all = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        invalid = indexes_all >= self.end_offset
        delta = indexes_all - (self.end_offset % self.capacity)
        positions = torch.where(delta <= 0, self.end_offset + delta, self.end_offset + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)

# 🚨 OVERRIDE MOSHI'S NATIVE KV CACHE 🚨
moshi_transformer.RingKVCache = TurboRingKVCache

def quantize_mixed_precision_inplace(model):
    n_replaced = 0
    def _swap(module, prefix=""):
        nonlocal n_replaced
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            # Protect ONLY the Depth Transformer & core embeddings/heads
            if "depformer" in full_name or any(skip in full_name for skip in SKIP_MODULES):
                continue

            if isinstance(child, nn.Linear):
                int4_layer = bnb.nn.Linear4bit(
                    child.in_features, child.out_features, 
                    bias=child.bias is not None, 
                    compute_dtype=torch.bfloat16, quant_type="nf4"
                )
                int4_layer.weight = bnb.nn.Params4bit(
                    child.weight.data.clone(), requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    int4_layer.bias = nn.Parameter(child.bias.data.clone())
                    
                int4_layer.cuda()
                setattr(module, child_name, int4_layer)
                n_replaced += 1
                del child
            else:
                _swap(child, full_name)
    _swap(model)
    print(f"[INFO] Replaced {n_replaced} Temporal FFN layers with INT4.")



def log(level: str, msg: str):
    print(make_log(level, msg))


def seed_all(seed: int):
    """Seed torch, CUDA, numpy, and Python RNG for reproducible runs.

    Matches the seeding strategy in server.py.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def warmup(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, device: str, frame_size: int):
    """Run a short warmup loop to initialize CUDA graphs and streaming state.

    Replicates the same warmup behavior as server.py: zeros → encode → LMGen.step → decode.
    """
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c : c + 1])
            if tokens is None:
                continue
            # Decode agent audio channels to ensure decode graphs/states are primed
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def decode_tokens_to_pcm(mimi: MimiModel, other_mimi: MimiModel, lm_gen: LMGen, tokens: torch.Tensor) -> np.ndarray:
    """Decode a single step of model tokens to PCM using Mimi.

    tokens is shaped [B, dep_q+1, 1]; channels 1..dep_q are the agent audio codebooks.
    Returns a 1D float32 numpy array (mono) for the current frame.
    """
    pcm = mimi.decode(tokens[:, 1:9])
    _ = other_mimi.decode(tokens[:, 1:9])
    pcm = pcm.detach().cpu().numpy()[0, 0]
    return pcm


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    log("info", "retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        log("info", f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def run_inference(
    input_wav: str,
    output_wav: str,
    output_text: str,
    text_prompt: str,
    voice_prompt_path: str,
    tokenizer_path: Optional[str],
    moshi_weight: Optional[str],
    mimi_weight: Optional[str],
    hf_repo: str,
    device: str,
    seed: Optional[int],
    temp_audio: float,
    temp_text: float,
    topk_audio: int,
    topk_text: int,
    greedy: bool,
    save_voice_prompt_embeddings: bool,
    cpu_offload: bool = False,
):
    """Run offline inference using an input WAV as the user-side stream.

    - Loads/initializes models and tokenizer
    - Warms up execution
    - Loads system text tokens and voice prompt
    - Runs prompt phases (text + voice + silences) via LMGen.step_system_prompts
    - Streams the user WAV frames into the input channels and samples model outputs
    - Decodes and writes an output WAV of the same duration
    """
    if seed is not None and seed != -1:
        seed_all(seed)

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(hf_repo, "config.json")

    # 1) Load Mimi encoders/decoders (same as server.py)
    log("info", "loading mimi")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "mimi loaded")

    # 2) Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)  # type: ignore
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    # 3) Load Moshi LM and eval mode
    log("info", "loading moshi")
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    print("[INFO] Applying Mixed-Precision INT4 Quantization...")
    quantize_mixed_precision_inplace(lm)
    lm.eval()
    log("info", "moshi loaded")

    # 4) Construct LMGen like server.py's ServerState does
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),  # spacer after prompts
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        use_sampling=not greedy,
        temp=temp_audio,
        temp_text=temp_text,
        top_k=topk_audio,
        top_k_text=topk_text,
    )
    # Keep models in streaming mode similar to the server
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # 5) Warmup
    log("info", "warming up the model")
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # 6) Prompt configuration (text + voice)
    # System text tokens (k=0) and agent voice-prompt audio (k=1..dep_q) are forced
    if voice_prompt_path.endswith('.pt'):
        # Load pre-saved voice prompt embeddings
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
    else:
        lm_gen.load_voice_prompt(voice_prompt_path)
    lm_gen.text_prompt_tokens = (
        text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if len(text_prompt) > 0 else None
    )

    # 7) Reset streaming and run initial prompt phases
    #    - Voice prompt injection
    #    - Audio silence
    #    - Text prompt injection
    #    - Final audio silence
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    # Reset mimi streaming after voice prompt encoding
    mimi.reset_streaming()

    # 8) Load and iterate user audio frames for feeding into the input channels
    sample_rate = mimi.sample_rate
    user_audio = lm_load_audio(input_wav, sample_rate)  # (C, T) at model SR

    # 9) Encode user audio with Mimi (same iterator logic used for voice prompts),
    #    and step the model one frame at a time, collecting decoded PCM frames
    generated_frames: List[np.ndarray] = []
    generated_text_tokens: List[str] = []
    total_target_samples = user_audio.shape[-1]

    for user_encoded in lm_encode_from_sphn(
        mimi,
        lm_iterate_audio(
            user_audio, sample_interval_size=lm_gen._frame_size, pad=True
        ),
        max_batch=1,
    ):
        # user_encoded: [1, K, T]. Feed one step at a time (usually T==1)
        steps = user_encoded.shape[-1]
        for c in range(steps):
            step_in = user_encoded[:, :, c : c + 1]
            
            # Measure Latency
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # Feed user-side input channels; text + agent audio are sampled
            tokens = lm_gen.step(step_in)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t0) * 1000
            
            if tokens is None:
                continue
            # Decode current sampled agent frame to PCM
            pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
            generated_frames.append(pcm)
            # Decode text token
            text_token = tokens[0, 0, 0].item()
            if text_token not in (0, 3):
                #_text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = text_tokenizer.id_to_piece(text_token).replace("▁", " ")
                #_text = _text.replace("▁", " ")
                log("info", f"text token '{_text}'")
                print(f"{_text}", end="", flush=True) # Dump text live
                generated_text_tokens.append(_text)
            else:
                text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                log("info", f"text token '{text_token_map[text_token]}'")
                generated_text_tokens.append(text_token_map[text_token])
                # Print latency warning if we blow the 80ms realtime budget
            if step_ms > 80.0:
                print(f"\n[WARN] Frame {c} latency: {step_ms:.1f}ms (Budget: 80ms)")

    if len(generated_frames) == 0:
        log("error", "No audio frames were generated. Check input file and configuration.")
        return

    # 10) Concatenate frames and trim/pad to match input duration
    output_pcm = np.concatenate(generated_frames, axis=-1)
    if output_pcm.shape[-1] > total_target_samples:
        output_pcm = output_pcm[:total_target_samples]
    elif output_pcm.shape[-1] < total_target_samples:
        pad_len = total_target_samples - output_pcm.shape[-1]
        output_pcm = np.concatenate(
            [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
        )

    # 11) Write mono WAV at model sample rate
    sphn.write_wav(output_wav, output_pcm, sample_rate)
    log("info", f"Wrote output audio to {output_wav}")

    # 12) Write text tokens
    with open(output_text, "w") as file:
        json.dump(generated_text_tokens, file, ensure_ascii=False)
    log("info", f"Wrote output text to {output_text}")    


def main():
    """Parse CLI args and run offline inference."""
    parser = argparse.ArgumentParser(
        description="Offline inference from WAV input using Moshi server components."
    )
    parser.add_argument(
        "--input-wav", required=True, type=str, help="Path to input WAV file (user audio)"
    )
    parser.add_argument(
        "--output-wav", required=True, type=str, help="Path to output WAV file of agent audio to write"
    )
    parser.add_argument(
        "--output-text", required=True, type=str, help="Path to output JSON file of agent text to write"
    )
    parser.add_argument("--text-prompt", default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.", type=str, help="Text prompt")

    parser.add_argument(
        "--voice-prompt", required=True, type=str, help="Voice prompt filename (basename) inside --voice-prompt-dir (e.g. 'NATM1.pt')."
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from -voice-prompt arg will be joined with this directory path."
        )
    )

    # Model assets
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo to look into (defaults to pre-trained model repo)",
    )

    # Runtime / sampling controls (mirror UI semantics)
    parser.add_argument(
        "--temp-audio", type=float, default=0.8, help="Audio sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--temp-text", type=float, default=0.7, help="Text sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--topk-audio", type=int, default=250, help="Audio top-k sampling (default: 250)"
    )
    parser.add_argument(
        "--topk-text", type=int, default=25, help="Text top-k sampling (default: 25)"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling (greedy decoding)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'."
    )
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (-1 disables)")

    args = parser.parse_args()

    # If --voice-prompt-dir is omitted, voices.tgz is downloaded from HF and extracted.
    voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir does not exist: {voice_prompt_dir}")
    log("info", f"voice_prompt_dir = {voice_prompt_dir}")

    # Join basename with directory (DO NOT mutate args.voice_prompt)
    voice_prompt_path = os.path.join(voice_prompt_dir, args.voice_prompt)
    if not os.path.exists(voice_prompt_path):
        raise FileNotFoundError(
            f"Voice prompt '{args.voice_prompt}' not found in "
            f"'{voice_prompt_dir}' (resolved: {voice_prompt_path})"
        )

    # Normalize greedy flag behavior (True if present, False otherwise)
    greedy = bool(args.greedy)

    with torch.no_grad():
        run_inference(
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            output_text=args.output_text,
            text_prompt=args.text_prompt,
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=args.tokenizer,
            moshi_weight=args.moshi_weight,
            mimi_weight=args.mimi_weight,
            hf_repo=args.hf_repo,
            device=args.device,
            seed=args.seed,
            temp_audio=args.temp_audio,
            temp_text=args.temp_text,
            topk_audio=args.topk_audio,
            topk_text=args.topk_text,
            greedy=greedy,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
        )



def main_with_lora():
    """Wrapper that adds --lora-weights arg to main()."""
    import sys
    # extract --lora-weights before argparse sees it
    lora_path = None
    filtered_args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--lora-weights" and i+1 < len(sys.argv):
            lora_path = sys.argv[i+1]
            i += 2
        else:
            filtered_args.append(sys.argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + filtered_args

    if lora_path:
        # monkey-patch get_moshi_lm to inject LoRA after loading
        from moshi.models import loaders
        _original = loaders.get_moshi_lm
        def _patched(*args, **kwargs):
            # build with LoRA structure via lm_kwargs
            if "lm_kwargs" not in kwargs:
                kwargs["lm_kwargs"] = {}
            kwargs["lm_kwargs"]["lora"] = True
            kwargs["lm_kwargs"]["lora_rank"] = 128
            kwargs["lm_kwargs"]["lora_scaling"] = 1.0
            model = _original(*args, **kwargs)
            # load LoRA weights
            import safetensors.torch
            lora_state = safetensors.torch.load_file(lora_path)
            model.load_state_dict(lora_state, strict=False)
            print(f"[Info] Loaded {len(lora_state)} LoRA keys from {lora_path}")
            return model
        loaders.get_moshi_lm = _patched

    main()

if __name__ == "__main__":
    main_with_lora()
