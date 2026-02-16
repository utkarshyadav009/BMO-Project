import os
import subprocess
from audio_separator.separator import Separator

# ================= CONFIGURATION =================
# 1. Your Test Video File
TEST_FILE = r"D:\LocalWorkDir\2509362\BMO_Episodes\ansemble\S00E01_BMO.wav"

# 2. Output Folder
OUTPUT_FOLDER = r"D:\LocalWorkDir\2509362\BMO_Episodes\MDX_Test_Result"

# 3. Model Name (Using Kim Vocal 2 is often safer/better for anime)
MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# =================================================

def convert_to_wav(video_path):
    """
    Extracts audio from video to a temporary WAV file using ffmpeg.
    Returns the path to the new wav file.
    """
    wav_path = video_path.rsplit('.', 1)[0] + "_temp.wav"
    
    print(f"Extracting audio to: {wav_path}...")
    
    # ffmpeg command: -i input -vn (no video) -acodec pcm_s16le (standard wav) -ar 44100 (44.1kHz) -y (overwrite)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", wav_path
    ]
    
    # Run silently
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(wav_path):
        return wav_path
    else:
        raise FileNotFoundError("FFmpeg failed to create the wav file!")

def main():
    if not os.path.exists(TEST_FILE):
        print(f"[ERROR] Could not find file: {TEST_FILE}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. 
    if TEST_FILE.lower().endswith('.wav'):
        print(f"[INFO] {TEST_FILE} is already a WAV. Skipping conversion.")
        audio_file = TEST_FILE
    # 2. Convert MP4 -> WAV first
    else:
        try:
            audio_file = convert_to_wav(TEST_FILE)
        except Exception as e:
            print(f"[ERROR] {e}")
            return

    # 2. Configure Separator
    print(f"Loading Model: {MODEL_NAME}...")
    separator = Separator(
        output_dir=OUTPUT_FOLDER,
        output_single_stem="vocals",
        model_file_dir=os.path.join(OUTPUT_FOLDER, "models")
    )

    separator.load_model(model_filename=MODEL_NAME)

    # 3. Run Separation on the WAV file
    print(f"Processing audio...")
    output_files = separator.separate(audio_file)

    # 4. Cleanup Temp WAV
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print("Deleted temporary wav file.")

    print("\n" + "="*30)
    print(f"DONE! Output saved to: {OUTPUT_FOLDER}")
    print(f"Generated files: {output_files}")

if __name__ == "__main__":
    main()