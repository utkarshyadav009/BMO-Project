import os
import shutil
import subprocess
import glob

# ================= CONFIGURATION =================
# 1. Path where your episodes are currently located
SOURCE_FOLDER = r"D:\LocalWorkDir\2509362\BMO_Episodes"

# 2. Path where you want the final Cleaned Vocals to go
OUTPUT_FOLDER = r"D:\LocalWorkDir\2509362\BMO_Episodes\Cleaned_Vocals"

# 3. SAFETY SWITCH: Set to True to DELETE the original video files after processing.
# (As you requested to save space)
DELETE_ORIGINAL_VIDEO = False 

# 4. Demucs Model (htdemucs is the high-quality Hybrid Transformer model)
DEMUCS_MODEL = "htdemucs"
# =================================================

def setup_folders():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def process_episode(video_path):
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    print(f"\n[PROCESSING] {filename}...")
    
    # 1. Run Demucs (AI Separation)
    # This command separates the audio into 4 stems: vocals, drums, bass, other
    # We use -n to specify the model and -o for output path
    cmd = [
        "python", "-m", "demucs", 
        "-n", DEMUCS_MODEL,
        "--two-stems", "vocals",  # <--- ADD THIS LINE (Speeds up processing)
        "-o", "temp_demucs_out",
        video_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to process {filename}. Is FFmpeg installed?")
        return

    # 2. Locate the extracted Vocals
    # Demucs output structure: temp_demucs_out/{model_name}/{filename}/vocals.wav
    # Note: Demucs sanitizes filenames (spaces to underscores), so we search loosely
    search_path = os.path.join("temp_demucs_out", DEMUCS_MODEL, "*", "vocals.wav")
    candidates = glob.glob(search_path)
    
    # Filter to find the one that was just created (approx match)
    # Since we process one by one, it's usually the only folder there if we clean up.
    if not candidates:
        print(f"[ERROR] Could not find vocals output for {filename}")
        return

    vocals_src = candidates[0]
    
    # 3. Move and Rename the Vocals
    final_name = f"{name_without_ext}_Vocals.wav"
    final_path = os.path.join(OUTPUT_FOLDER, final_name)
    
    shutil.move(vocals_src, final_path)
    print(f"[SUCCESS] Saved vocals to: {final_path}")

    # 4. Cleanup (Delete Temp Files)
    # Delete the specific demucs folder for this track
    track_folder = os.path.dirname(vocals_src)
    try:
        shutil.rmtree(track_folder) # Removes the folder with drums.wav, bass.wav, etc.
    except Exception as e:
        print(f"[WARNING] Could not delete temp folder: {e}")

    # 5. Delete Original Video (If enabled)
    if DELETE_ORIGINAL_VIDEO:
        try:
            os.remove(video_path)
            print(f"[CLEANUP] Deleted original video: {filename}")
        except OSError as e:
            print(f"[ERROR] Could not delete video {filename}: {e}")

def main():
    setup_folders()
    
    # Find all video files
    video_extensions = ['*.mkv', '*.mp4', '*.avi', '*.mov']
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))
    
    print(f"Found {len(files)} episodes to process.")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("Starting processing... (This may take time depending on GPU)")
    
    for file_path in files:
        process_episode(file_path)
        
    # Final cleanup of the main temp folder if empty
    if os.path.exists("temp_demucs_out"):
        try:
            shutil.rmtree("temp_demucs_out")
        except:
            pass
            
    print("\nAll done! Check your Cleaned_Vocals folder.")

if __name__ == "__main__":
    main()