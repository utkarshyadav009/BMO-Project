import os
import gc
import csv
import torch
import torchaudio
import whisperx
from tqdm import tqdm

# ================= CONFIGURATION =================
HF_TOKEN = "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"

# 1. Paths
INPUT_FOLDER   = r"D:\LocalWorkDir\2509362\BMO Episodes\Cleaned_Vocals"
OUTPUT_DATASET = r"D:\LocalWorkDir\2509362\BMO Episodes\PlanB_Dataset"
LOG_FILE       = "planb_processed.txt"

# 2. Tuning
BATCH_SIZE      = 16    
MIN_CLIP_LENGTH = 1.0   
MAX_CLIP_LENGTH = 12.0  
# =================================================

def load_processed_log():
    if not os.path.exists(LOG_FILE): return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def mark_episode_done(filename):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def cleanup_vram():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE":
        print("[ERROR] Paste your HuggingFace token!")
        return

    os.makedirs(OUTPUT_DATASET, exist_ok=True)
    processed_files = load_processed_log()
    device = "cuda"
    compute_type = "int8"

    print("\n=== PLAN B: DRAGNET MODE (No Verification) ===")
    print("-> Loading WhisperX...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    
    print("-> Loading Diarization...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

    # Note: We do NOT load the Speaker Classifier. We blindly trust the diarization.

    audio_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav"))
    
    for filename in tqdm(audio_files, desc="Episodes"):
        if filename in processed_files: continue

        file_path  = os.path.join(INPUT_FOLDER, filename)
        episode_name = os.path.splitext(filename)[0]
        
        # Create a folder for this episode
        episode_dir = os.path.join(OUTPUT_DATASET, episode_name)
        os.makedirs(episode_dir, exist_ok=True)

        try:
            # 1. Transcribe
            audio = whisperx.load_audio(file_path)
            result = model.transcribe(audio, batch_size=BATCH_SIZE)
            
            # 2. Align
            model_a, meta_a = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, meta_a, audio, device, return_char_alignments=False)
            del model_a, meta_a
            cleanup_vram()

            # 3. Diarize
            diarize_segs = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segs, result)

            # 4. Slice EVERYTHING
            full_wav, sr = torchaudio.load(file_path)
            
            csv_rows = []
            
            for segment in result["segments"]:
                if "speaker" not in segment: continue
                
                speaker_id = segment["speaker"] # e.g. "SPEAKER_01"
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                
                if (end - start) < MIN_CLIP_LENGTH or (end - start) > MAX_CLIP_LENGTH:
                    continue

                # Create Speaker Folder (e.g. S01E08/SPEAKER_01)
                speaker_subdir = os.path.join(episode_dir, speaker_id)
                os.makedirs(speaker_subdir, exist_ok=True)

                # Slice
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                chunk = full_wav[:, start_frame:end_frame]
                
                # Save
                clip_name = f"{episode_name}_{speaker_id}_{start:.2f}.wav"
                save_path = os.path.join(speaker_subdir, clip_name)
                torchaudio.save(save_path, chunk, sr)
                
                csv_rows.append(f"{clip_name}|{text}")

            # Save minimal metadata for this episode
            with open(os.path.join(episode_dir, "metadata.csv"), "w", encoding="utf-8") as f:
                f.write("\n".join(csv_rows))

            mark_episode_done(filename)
            cleanup_vram()

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            continue

if __name__ == "__main__":
    main()