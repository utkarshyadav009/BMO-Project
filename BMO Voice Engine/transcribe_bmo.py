import os
import gc
import csv
import torch
import torchaudio
import whisperx
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. Your Hugging Face Token (REQUIRED for Diarization)
#    Get one at: https://huggingface.co/settings/tokens
#    Accept terms at:
#      https://huggingface.co/pyannote/segmentation-3.0
#      https://huggingface.co/pyannote/speaker-diarization-3.1
HF_TOKEN = "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"

# 2. Paths
INPUT_FOLDER   = r"D:\LocalWorkDir\2509362\BMO Episodes\Cleaned_Vocals"
OUTPUT_DATASET = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset"
REFERENCE_FILE = r"reference_bmo.wav"   # 5-10s of clean BMO speech, no music
LOG_FILE       = "processed_episodes.txt"

# 3. Tuning
BATCH_SIZE           = 16    # Lower to 8 or 4 if you get OOM errors
SIMILARITY_THRESHOLD = 0.50  # 0.0-1.0 — raise if getting too many false positives
MIN_CLIP_LENGTH      = 1.0   # Seconds — filters out breaths/noise
MAX_CLIP_LENGTH      = 12.0  # Seconds — keeps clips TTS-friendly
# =================================================


# ── Resume log helpers ────────────────────────────────────────────────────────

def load_processed_log() -> set:
    """Returns the set of filenames already successfully processed."""
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def mark_episode_done(filename: str):
    """Appends a filename to the resume log."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")


# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_existing_metadata(csv_path: str) -> set:
    """
    Returns the set of clip filenames already written to metadata.csv.
    Prevents duplicates if a resume somehow overlaps.
    """
    existing = set()
    if not os.path.exists(csv_path):
        return existing
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if row:
                existing.add(row[0].strip())
    return existing

def append_metadata(csv_path: str, rows: list):
    """
    Appends a list of 'filename|text' rows to the CSV.
    Creates the file if it doesn't exist.
    """
    if not rows:
        return
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        # Avoid leading blank line on very first write
        if os.path.getsize(csv_path) > 0:
            f.write("\n")
        f.write("\n".join(rows))


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_speaker_model() -> EncoderClassifier:
    print("-> Loading Speaker Verification Model (SpeechBrain ECAPA)...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

def get_embedding(classifier: EncoderClassifier, wav_tensor: torch.Tensor, input_sr: int) -> torch.Tensor:
    """
    Returns a normalised voice embedding for a waveform tensor.
    Always resamples to 16 kHz before encoding (SpeechBrain requirement).
    Keeps everything on GPU to avoid repeated CPU↔GPU transfers.
    """
    wav_tensor = wav_tensor.cuda()

    if input_sr != 16000:
        resampler = torchaudio.transforms.Resample(input_sr, 16000).cuda()
        wav_tensor = resampler(wav_tensor)

    with torch.no_grad():
        emb = classifier.encode_batch(wav_tensor).flatten()

    return emb

def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Preflight checks ──────────────────────────────────────────────────────
    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE":
        print("[ERROR] Please paste your HuggingFace token into HF_TOKEN.")
        return

    if not os.path.exists(REFERENCE_FILE):
        print(f"[ERROR] Reference file not found: {REFERENCE_FILE}")
        print("  Tip: Use the first 5-10s of the Distant Lands BMO special")
        print("       (cleaned vocals, no background music).")
        return

    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    csv_path          = os.path.join(OUTPUT_DATASET, "metadata.csv")
    processed_files   = load_processed_log()
    existing_clips    = load_existing_metadata(csv_path)

    print(f"Resume state: {len(processed_files)} episodes already done, "
          f"{len(existing_clips)} clips already in metadata.csv")

    device = "cuda"
    compute_type = "int8"   # Safe for 8 GB VRAM; swap to float16 if you have 12 GB+

    # ── Load models (once, outside the loop) ──────────────────────────────────
    print("\n=== PHASE 1: Loading Models ===")

    print("-> Loading WhisperX (large-v2) ...")
    whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    print("-> Loading Diarization Pipeline (pyannote) ...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    classifier = load_speaker_model()

    print(f"-> Learning BMO's voice from: {REFERENCE_FILE}")
    ref_wav, ref_sr = torchaudio.load(REFERENCE_FILE)
    bmo_emb = get_embedding(classifier, ref_wav, ref_sr)

    print("=== Models loaded ===\n")

    # ── Process episodes ──────────────────────────────────────────────────────
    audio_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav"))
    print(f"Found {len(audio_files)} vocal track(s) in input folder.\n")

    total_clips_saved  = 0
    total_clips_seen   = 0
    episode_stats      = []   # For the end-of-run summary

    for filename in tqdm(audio_files, desc="Episodes"):

        # ── Resume: skip completed episodes ───────────────────────────────
        if filename in processed_files:
            tqdm.write(f"  [SKIP] {filename} (already processed)")
            continue

        file_path  = os.path.join(INPUT_FOLDER, filename)
        episode_id = os.path.splitext(filename)[0]

        ep_clips_saved = 0
        ep_scores      = []
        new_metadata   = []

        try:
            # A. Transcribe ────────────────────────────────────────────────
            tqdm.write(f"\n  [→] Transcribing: {filename}")
            audio  = whisperx.load_audio(file_path)
            result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)

            # B. Align (word-level timestamps) ─────────────────────────────
            model_a, meta_a = whisperx.load_align_model(
                language_code=result["language"], device=device
            )
            result = whisperx.align(
                result["segments"], model_a, meta_a, audio, device,
                return_char_alignments=False,
            )
            del model_a, meta_a
            cleanup_vram()

            # C. Diarise ───────────────────────────────────────────────────
            diarize_segs = diarize_model(audio)
            result       = whisperx.assign_word_speakers(diarize_segs, result)

            # D. Load full waveform onto GPU once for efficient slicing ─────
            full_wav, sr = torchaudio.load(file_path)
            full_wav_gpu = full_wav.cuda()

            # E. Filter segments and verify speaker ────────────────────────
            for segment in result["segments"]:
                if "speaker" not in segment or not segment.get("text", "").strip():
                    continue

                start    = segment["start"]
                end      = segment["end"]
                duration = end - start
                text     = segment["text"].strip()

                total_clips_seen += 1

                # Duration filter
                if not (MIN_CLIP_LENGTH <= duration <= MAX_CLIP_LENGTH):
                    continue

                # Slice on GPU
                start_frame = int(start * sr)
                end_frame   = int(end   * sr)
                chunk_gpu   = full_wav_gpu[:, start_frame:end_frame]

                # Speaker verification
                chunk_emb = get_embedding(classifier, chunk_gpu, sr)
                score     = torch.nn.functional.cosine_similarity(
                    bmo_emb, chunk_emb, dim=0
                ).item()
                ep_scores.append(score)

                if score <= SIMILARITY_THRESHOLD:
                    continue

                # Duplicate guard
                clip_name = f"{episode_id}_{start:.2f}-{end:.2f}.wav"
                if clip_name in existing_clips:
                    continue

                # Save audio clip
                save_path = os.path.join(OUTPUT_DATASET, clip_name)
                torchaudio.save(save_path, chunk_gpu.cpu(), sr)

                new_metadata.append(f"{clip_name}|{text}")
                existing_clips.add(clip_name)
                ep_clips_saved  += 1
                total_clips_saved += 1

            # F. Flush metadata for this episode ───────────────────────────
            append_metadata(csv_path, new_metadata)

            # G. Mark episode done ─────────────────────────────────────────
            mark_episode_done(filename)

            # H. Per-episode score stats (helps you tune the threshold) ────
            if ep_scores:
                avg  = sum(ep_scores) / len(ep_scores)
                hi   = max(ep_scores)
                lo   = min(ep_scores)
                stat = (f"  [✓] {filename}: {ep_clips_saved} clips saved | "
                        f"scores avg={avg:.3f} hi={hi:.3f} lo={lo:.3f}")
            else:
                stat = f"  [✓] {filename}: 0 clips saved (no scored segments)"
            tqdm.write(stat)
            episode_stats.append(stat)

            del full_wav_gpu
            cleanup_vram()

        except Exception as e:
            tqdm.write(f"\n  [ERROR] {filename}: {e}")
            continue

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  RUN COMPLETE")
    print(f"  Total segments evaluated : {total_clips_seen}")
    print(f"  BMO clips saved          : {total_clips_saved}")
    print(f"  Dataset location         : {OUTPUT_DATASET}")
    print(f"  Metadata CSV             : {csv_path}")
    print("=" * 50)

    if total_clips_saved == 0:
        print("\n  ⚠  No clips were saved. Things to check:")
        print("     1. Is your reference_bmo.wav clean (no music/FX)?")
        print(f"     2. Try lowering SIMILARITY_THRESHOLD (currently {SIMILARITY_THRESHOLD})")
        print("     3. Check the score stats above — if avg scores are around 0.5-0.6,")
        print("        lowering the threshold to 0.55 might help.")


if __name__ == "__main__":
    main()