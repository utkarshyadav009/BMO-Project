import os
import subprocess
import glob
import shutil

# ================= CONFIG =================

# 🔹 Path to ONE episode file you want to test
VIDEO_FILE = r"D:\LocalWorkDir\2509362\BMO_Episodes\S00E01_BMO.mp4"

# 🔹 Output folder
OUTPUT_FOLDER = r"D:\LocalWorkDir\2509362\BMO_Episodes\Test_Extraction"

# 🔹 Demucs model (try htdemucs first)
DEMUCS_MODEL = "htdemucs"

# ==========================================


def setup_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def extract_vocals(video_path):
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]

    print(f"\n[PROCESSING] {filename}")

    cmd = [
        "python", "-m", "demucs",
        "-n", DEMUCS_MODEL,
        "--two-stems", "vocals",
        "--segment", "7",
        "--overlap", "0.25",
        "-o", "temp_demucs_test",
        video_path
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] Demucs failed. Is it installed and working?")
        return

    # Locate extracted vocals
    search_path = os.path.join("temp_demucs_test", DEMUCS_MODEL, "*", "vocals.wav")
    candidates = glob.glob(search_path)

    if not candidates:
        print("[ERROR] Could not find extracted vocals.")
        return

    vocals_src = candidates[0]

    final_path = os.path.join(OUTPUT_FOLDER, f"{name_without_ext}_SpeechTest.wav")
    shutil.move(vocals_src, final_path)

    print(f"[SUCCESS] Saved to: {final_path}")

    # Cleanup temp folder
    shutil.rmtree("temp_demucs_test", ignore_errors=True)


if __name__ == "__main__":
    setup_folder()
    extract_vocals(VIDEO_FILE)
    print("\nDone. Compare this file with your original extraction.")
