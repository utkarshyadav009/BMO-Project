import os
import sys
import torch
import torchaudio
from pathlib import Path

REPO_DIR = Path("/home/jovyan/work/BMO-Project/personaplex_repo")
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "moshi"))

from moshi.offline import seed_all
from moshi.models import loaders, LMGen

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

MODEL_PATH = "/home/jovyan/work/BMO-Project/personaplex_repo/tile_region_experiment/qat_heavy_int2/qat_best.pt"
MIMI_PATH = "/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_PATH = "/home/jovyan/work/BMO-Project/personaplex_repo/tokenizer_spm_32k_3.model"
INPUT_WAV = "/home/jovyan/work/BMO-Project/personaplex_repo/tellmeajoke_padded.wav"
VOICE_PROMPT = "/home/jovyan/work/BMO-Project/personaplex_repo/bmo_621.wav"

OUT_DIR = Path("/home/jovyan/work/BMO-Project/personaplex_repo/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1783708826

def main():
    print(f"[INIT] Seeding with seed={SEED}...", flush=True)
    seed_all(SEED)

    print("[INIT] Loading Mimi model and Moshi LM...", flush=True)
    mimi = loaders.get_mimi(MIMI_PATH, "cuda")
    lm = loaders.get_moshi_lm(MODEL_PATH, device="cuda", dtype=torch.bfloat16)
    lm.eval()

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device="cuda",
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    print("[PROMPT] Loading voice prompt and setting prompts...", flush=True)
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)
    lm_gen.text_prompt_tokens = sp.encode("<system> Tell me a joke. <system>")
    lm_gen.load_voice_prompt(VOICE_PROMPT)
    mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()

    from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn
    user_audio = load_audio(INPUT_WAV, mimi.sample_rate)

    print("[GEN] Running streaming generation loop for user input...", flush=True)
    collected_tokens = []
    
    with torch.no_grad():
        for user_encoded in encode_from_sphn(
            mimi,
            _iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]
                tokens = lm_gen.step(step_in)
                if tokens is not None:
                    # tokens is [1, 17, 1]
                    collected_tokens.append(tokens.detach().cpu())

    if len(collected_tokens) == 0:
        raise RuntimeError("No tokens generated!")

    all_tokens = torch.cat(collected_tokens, dim=-1) # [1, 17, T]
    print(f"[GEN] Generation complete. Total frames collected: {all_tokens.shape[-1]}", flush=True)

    # Audio codebooks are channels 1..16 (index 1 to 17)
    audio_codes_16 = all_tokens[:, 1:17, :].cuda() # [1, 16, T]

    import sphn

    print("[DECODE] Decoding 3 ways with PyTorch Mimi decoder...", flush=True)
    
    # 1. All 16 codebooks
    with torch.no_grad():
        pcm_16 = mimi.decode(audio_codes_16[:, :16, :]).detach().cpu().numpy()[0, 0]
    out_16 = OUT_DIR / "dep_q_16_codebooks.wav"
    sphn.write_wav(str(out_16), pcm_16, mimi.sample_rate)
    print(f"[SAVED] All 16 codebooks -> {out_16}", flush=True)

    # 2. First 12 codebooks
    with torch.no_grad():
        pcm_12 = mimi.decode(audio_codes_16[:, :12, :]).detach().cpu().numpy()[0, 0]
    out_12 = OUT_DIR / "dep_q_12_codebooks.wav"
    sphn.write_wav(str(out_12), pcm_12, mimi.sample_rate)
    print(f"[SAVED] First 12 codebooks -> {out_12}", flush=True)

    # 3. First 8 codebooks
    with torch.no_grad():
        pcm_8 = mimi.decode(audio_codes_16[:, :8, :]).detach().cpu().numpy()[0, 0]
    out_8 = OUT_DIR / "dep_q_8_codebooks.wav"
    sphn.write_wav(str(out_8), pcm_8, mimi.sample_rate)
    print(f"[SAVED] First 8 codebooks -> {out_8}", flush=True)

    print("[COMPLETE] All 3 WAV files generated successfully!", flush=True)

if __name__ == "__main__":
    main()
