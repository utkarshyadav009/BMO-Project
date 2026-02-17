import os
os.environ["PATH"] = r"C:\Users\2509362\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin" + os.pathsep + os.environ["PATH"]
import shutil
import subprocess
import torch
from audio_separator.separator import Separator
from tqdm import tqdm

FFMPEG_PATH = r"C:\Users\2509362\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"


# ================= CONFIGURATION =================
# 1. Paths
SOURCE_FOLDER = r"D:\LocalWorkDir\2509362\BMO Episodes"
OUTPUT_FOLDER = r"D:\LocalWorkDir\2509362\BMO Episodes\extracted_Vocals"
TEMP_FOLDER = r"D:\LocalWorkDir\2509362\BMO Episodes\Temp_Work"

# 2. SEPARATION MODEL (Stage 1)
# The one you liked. Good for loudness and body.
MODEL_SEPARATION = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# ALTERNATIVE TO TRY IF ROBOTIC: "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# 3. DE-REVERB MODEL (Stage 2)
# Removes the "roomy" sound, making it cleaner for AI training.
MODEL_DEREVERB = "UVR-DeEcho-DeReverb.pth"
ENABLE_DEREVERB = True  # Set to False if you want to skip this

# 4. File Extensions to look for
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')
# =================================================

def setup_folders():
    for f in [OUTPUT_FOLDER, TEMP_FOLDER]:
        if not os.path.exists(f):
            os.makedirs(f)

def convert_to_wav(video_path):
    """Extracts audio from video to a temporary WAV file using ffmpeg."""
    filename = os.path.basename(video_path)
    wav_path = os.path.join(TEMP_FOLDER, os.path.splitext(filename)[0] + "_temp.wav")
    
    # Skip if temp file already exists
    if os.path.exists(wav_path): return wav_path

    cmd = [
        FFMPEG_PATH, "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", wav_path
    ]
    # Run silently
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def run_separator(input_path, model_name, output_stem_name):
    """Runs the Audio Separator on a specific file."""
    print(f"   -> Loading {model_name}...")
    
    # Initialize Separator
    separator = Separator(
        output_dir=TEMP_FOLDER,
        output_single_stem=output_stem_name, # "vocals" or "instrumental" (dereverb is diff)
        model_file_dir=os.path.join(TEMP_FOLDER, "models")
    )
    
    separator.load_model(model_filename=model_name)
    
    # Run
    output_files = separator.separate(input_path)
    
    # Return the full path of the result
    return os.path.join(TEMP_FOLDER, output_files[0])

def main():
    setup_folders()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: Running on CPU! This will be very slow.")

    # Find files
    # Find files
    files = sorted([f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(VIDEO_EXTS)], reverse=True)
    print(f"Found {len(files)} episodes to process.")

    for file in tqdm(files, desc="Processing Episodes"):
        try:
            full_path = os.path.join(SOURCE_FOLDER, file)
            base_name = os.path.splitext(file)[0]
            final_output = os.path.join(OUTPUT_FOLDER, f"{base_name}_Vocals.wav")
            
            # Skip if already done
            if os.path.exists(final_output):
                continue

            print(f"\n[START] {file}")

            # Step 1: Video -> WAV
            wav_path = convert_to_wav(full_path)

            # Step 2: Separation (Music Removal)
            # We want "vocals" output
            clean_vocals = run_separator(wav_path, MODEL_SEPARATION, "vocals")
            
            # Step 3: De-Reverb (Optional but Recommended)
            if ENABLE_DEREVERB:
                print("   -> Running De-Reverb...")
                # For DeReverb model, the "vocals" stem is technically "No Reverb"
                # Sometimes the output stem name varies, but usually it's the primary output
                dry_vocals = run_separator(clean_vocals, MODEL_DEREVERB, "No Reverb")
                
                # Move to final folder
                shutil.move(dry_vocals, final_output)
            else:
                # Just move the Step 2 result
                shutil.move(clean_vocals, final_output)

            # Cleanup Temp
            if os.path.exists(wav_path): os.remove(wav_path)
            # Note: We keep the intermediate files in Temp_Work just in case, 
            # or you can uncomment the line below to delete them:
            # shutil.rmtree(TEMP_FOLDER) 

            print(f"[DONE] Saved to: {final_output}")

        except Exception as e:
            print(f"\n[ERROR] Failed on {file}: {e}")

    print("\n🎉 All episodes processed!")

if __name__ == "__main__":
    main()