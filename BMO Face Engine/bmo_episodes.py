import os
import shutil
import re

# ================= CONFIGURATION =================
# 1. Path to your downloaded seasons
SOURCE_ROOT = r"C:\Users\YourName\Downloads\Adventure Time" 

# 2. Path where you want to save the BMO episodes
DEST_ROOT = r"C:\ProjectBMO\dataset\real_data\raw_episodes"

# 3. YOUR LIST: Add the episodes here as tuples: (Season, Episode)
# Example: (1, 6) means Season 1, Episode 6.
bmo_episodes = [
    (1, 8),   # "Business Time" (First major appearance?)
    (2, 13),  # "The Pods"
    (3, 14),  # "Beautopia"
    (5, 17),  # "BMO Lost" (Crucial episode!)
    (5, 28),  # "Be More"
    (7, 14),  # "The More You Moe, The Moe You Know"
    # ... PASTE YOUR FULL LIST HERE ...
]

# =================================================

def setup_directories():
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)
        print(f"Created destination directory: {DEST_ROOT}")

def normalize_filename(filename):
    """
    Tries to extract season and episode numbers from common filename formats.
    Supports: S01E01, 1x01, Season 1 Episode 1
    """
    # Regex for S01E01 or s1e1
    match_sxe = re.search(r"[Ss](\d{1,2})[Ee](\d{1,2})", filename)
    if match_sxe:
        return int(match_sxe.group(1)), int(match_sxe.group(2))
    
    # Regex for 1x01
    match_x = re.search(r"(\d{1,2})x(\d{1,2})", filename)
    if match_x:
        return int(match_x.group(1)), int(match_x.group(2))

    return None, None

def find_and_copy_episodes():
    setup_directories()
    
    found_count = 0
    missing_episodes = list(bmo_episodes) # Track what we haven't found yet

    print(f"Scanning {SOURCE_ROOT} for {len(bmo_episodes)} target episodes...")

    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            # Skip non-video files (adjust extensions if needed)
            if not file.lower().endswith(('.mkv', '.mp4', '.avi')):
                continue

            season_num, episode_num = normalize_filename(file)

            if season_num is not None and (season_num, episode_num) in bmo_episodes:
                # We found a match!
                src_path = os.path.join(root, file)
                dest_path = os.path.join(DEST_ROOT, f"S{season_num:02d}E{episode_num:02d}_BMO.mkv") # Renaming for consistency
                
                print(f"Found S{season_num:02d}E{episode_num:02d}: {file} -> Copying...")
                shutil.copy2(src_path, dest_path)
                
                found_count += 1
                if (season_num, episode_num) in missing_episodes:
                    missing_episodes.remove((season_num, episode_num))

    print("-" * 30)
    print(f"Extraction Complete!")
    print(f"Total Found: {found_count}/{len(bmo_episodes)}")
    
    if missing_episodes:
        print("\nCould not find these episodes (check filenames):")
        for s, e in missing_episodes:
            print(f"- Season {s}, Episode {e}")

if __name__ == "__main__":
    find_and_copy_episodes()