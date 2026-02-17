import os
import shutil
import re

# ================= CONFIGURATION =================
# 1. Path to your downloaded seasons
SOURCE_ROOT = r"D:\Steam\Adventure Time" 

# 2. Path where you want to save the BMO episodes
DEST_ROOT = r"D:\Steam\BMO Episodes"

# 3. YOUR LIST: Add the episodes here as tuples: (Season, Episode)
# Example: (1, 6) means Season 1, Episode 6.
bmo_episodes = [
    # --- Season 1 ---
    (0,1),    # adventure time short all's well the rats swell
    (1, 8),   # "Business Time" (Debut)
    (1, 9),   # "My Two Favorite People"
    (1, 12),  # "Evicted!"
    (1, 15),  # "What Is Life?" (Creator of NEPTR)
    (1, 21),  # "Donny"
    (1, 23),  # "Rainy Day Daydream"

    # --- Season 2 ---
    (2, 6),   # "Slow Love"
    (2, 7),   # "Power Animal"
    (2, 12),  # "Her Parents"
    (2, 16),  # "Guardians of Sunshine" (Major BMO role inside the game)
    (2, 23),  # "Video Makers"

    # --- Season 3 ---
    (3, 1),   # "Conquest of Cuteness"
    (3, 5),   # "Too Young"
    (3, 6),   # "The Monster"
    (3, 9),   # "Fionna and Cake"
    (3, 10),  # "What Was Missing" (BMO as an instrument)
    (3, 12),  # "The Creeps"
    (3, 18),  # "The New Frontier"
    (3, 19),  # "Holly Jolly Secrets Part I"
    (3, 20),  # "Holly Jolly Secrets Part II"
    (3, 25),  # "Dad's Dungeon"
    (3, 26),  # "Incendium"

    # --- Season 4 ---
    (4, 2),   # "Five Short Graybles" (BMO Mirror segment)
    (4, 4),   # "Dream of Love"
    (4, 6),   # "Daddy's Little Monster"
    (4, 7),   # "In Your Footsteps"
    (4, 8),   # "Hug Wolf"
    (4, 11),  # "Beyond This Earthly Realm"
    (4, 12),  # "Gotcha!"
    (4, 14),  # "Card Wars"
    (4, 16),  # "Burning Low"
    (4, 17),  # "BMO Noire" (BMO Solo Noir Detective Episode)
    (4, 18),  # "King Worm"

    # --- Season 5 ---
    (5, 3),   # "Five More Short Graybles"
    (5, 5),   # "All the Little People"
    (5, 6),   # "Jake the Dad"
    (5, 7),   # "Davey"
    (5, 10),  # "Little Dude"
    (5, 11),  # "Bad Little Boy"
    (5, 15),  # "A Glitch Is a Glitch"
    (5, 16),  # "Puhoy"
    (5, 17),  # "BMO Lost" (Major BMO solo adventure)
    (5, 19),  # "James Baxter the Horse"
    (5, 20),  # "Shh!"
    (5, 23),  # "One Last Job"
    (5, 24),  # "Another Five More Short Graybles"
    (5, 25),  # "Candy Streets"
    (5, 27),  # "Jake Suit"
    (5, 28),  # "Be More" (BMO origin story/Factory visit)
    (5, 30),  # "Frost & Fire"
    (5, 31),  # "Too Old"
    (5, 32),  # "Earth & Water"
    (5, 33),  # "Time Sandwich"
    (5, 34),  # "The Vault"
    (5, 35),  # "Love Games"
    (5, 37),  # "Box Prince"
    (5, 39),  # "We Fixed a Truck"
    (5, 40),  # "Play Date"
    (5, 41),  # "The Pit"
    (5, 44),  # "Apple Wedding"

    # --- Season 6 ---
    (6, 4),   # "The Tower"
    (6, 5),   # "Sad Face"
    (6, 8),   # "Furniture & Meat"
    (6, 13),  # "Thanks for the Crabapples, Giuseppe!"
    (6, 16),  # "Joshua & Margaret Investigations"
    (6, 17),  # "Ghost Fly"
    (6, 18),  # "Everything's Jake"
    (6, 20),  # "Jake the Brick"
    (6, 27),  # "The Visitor"
    (6, 28),  # "The Mountain"
    (6, 29),  # "Dark Purple"
    (6, 33),  # "Jermaine"
    (6, 34),  # "Chips & Ice Cream"
    (6, 35),  # "Graybles 1000+"
    (6, 36),  # "Hoots"
    (6, 43),  # "The Comet"

    # --- Season 7 ---
    # "All's Well That Rats Swell" is a Short, not a numbered episode.
    (7, 5),   # "Football" (BMO switches places with 'Football')
    (7, 14),  # "The More You Moe, The Moe You Know (Part I)"
    (7, 15),  # "The More You Moe, The Moe You Know (Part II)"
    (7, 17),  # "Angel Face" (BMO as a cowboy/horse)
    (7, 18),  # "President Porpoise Is Missing!"
    (7, 19),  # "Blank-Eyed Girl"
    (7, 20),  # "Bad Jubies"
    (7, 23),  # "Crossover"
    (7, 24),  # "The Hall of Egress"
    (7, 25),  # "Flute Spell"

    # --- Season 8 ---
    (8, 2),   # "Don't Look"
    (8, 3),   # "Beyond the Grotto"
    (8, 5),   # "I Am a Sword"
    (8, 10),  # "The Music Hole"
    (8, 11),  # "Daddy-Daughter Card Wars"
    (8, 14),  # "Two Swords"
    (8, 17),  # "High Strangeness"
    (8, 18),  # "Horse and Ball" (BMO Centric)
    (8, 20),  # "Islands Part 1: The Invitation"
    (8, 21),  # "Islands Part 2: Whipple the Happy Dragon"
    (8, 22),  # "Islands Part 3: Mysterious Island"
    (8, 23),  # "Islands Part 4: Imaginary Resources" (BMO VR Episode)
    (8, 24),  # "Islands Part 5: Hide and Seek"
    (8, 25),  # "Islands Part 6: Min & Marty"
    (8, 26),  # "Islands Part 7: Helpers"
    (8, 27),  # "Islands Part 8: The Light Cloud"

    # --- Season 9 ---
    (9, 1),   # "Orb"
    (9, 2),   # "Elements Part 1: Skyhooks"
    (9, 8),   # "Elements Part 7: Hero Heart"
    (9, 9),   # "Elements Part 8: Skyhooks II"
    (9, 10),  # "Abstract"
    (9, 11),  # "Ketchup" (BMO retelling stories)
    (9, 12),  # "Fionna and Cake and Fionna"
    (9, 13),  # "Whispers"
    (9, 14),  # "Three Buckets"

    # --- Season 10 ---
    (10, 2),  # "Always BMO Closing" (BMO Salesman episode)
    (10, 3),  # "Son of Rap Bear"
    (10, 4),  # "Bonnibel Bubblegum"
    (10, 5),  # "Seventeen"
    (10, 9),  # "Blenanas"
    (10, 13)  # "Come Along With Me" (Series Finale - BMO is King of Ooo)
]
# =================================================

# =================================================

def extract_season_episode(filename):
    """
    Extracts season and episode from format: 
    'Adventure Time (2008) - S01E01 - Slumber Party Panic...'
    """
    # Regex looks for 'S' followed by digits, then 'E' followed by digits
    # It is case-insensitive to handle s01e01 or S01E01
    match = re.search(r"[S|s](\d+)[E|e](\d+)", filename)
    
    if match:
        season = int(match.group(1))  # Converts '01' to 1
        episode = int(match.group(2)) # Converts '01' to 1
        return season, episode
    return None, None

def find_and_copy_episodes():
    # Create the destination folder if it doesn't exist
    if not os.path.exists(DEST_ROOT):
        os.makedirs(DEST_ROOT)
        print(f"Created destination directory: {DEST_ROOT}")

    found_count = 0
    missing_episodes = list(bmo_episodes) # Track what we haven't found yet
    
    print(f"Scanning {SOURCE_ROOT}...")
    print(f"Looking for {len(bmo_episodes)} specific episodes...")

    # Walk through all folders in the source directory
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            # Check for video files only
            if file.lower().endswith(('.mkv', '.mp4', '.avi', '.mov')):
                
                season_num, episode_num = extract_season_episode(file)

                # If we found a valid SxxExx and it's in our target list
                if season_num is not None and (season_num, episode_num) in bmo_episodes:
                    
                    # Construct full file paths
                    src_path = os.path.join(root, file)
                    
                    # Rename the file for easier processing later (e.g., "S01E01_BMO.mkv")
                    new_filename = f"S{season_num:02d}E{episode_num:02d}_BMO{os.path.splitext(file)[1]}"
                    dest_path = os.path.join(DEST_ROOT, new_filename)
                    
                    print(f"[FOUND] S{season_num:02d}E{episode_num:02d} -> Copying to {new_filename}...")
                    
                    # Copy the file
                    try:
                        shutil.copy2(src_path, dest_path)
                        found_count += 1
                        
                        # Remove from missing list (handle duplicates in source just in case)
                        if (season_num, episode_num) in missing_episodes:
                            missing_episodes.remove((season_num, episode_num))
                            
                    except Exception as e:
                        print(f"Error copying {file}: {e}")

    print("-" * 30)
    print(f"Extraction Complete!")
    print(f"Total Found: {found_count}/{len(bmo_episodes)}")
    
    if missing_episodes:
        print("\n[WARNING] Could not find these episodes:")
        for s, e in missing_episodes:
            print(f"- Season {s}, Episode {e}")

if __name__ == "__main__":
    find_and_copy_episodes()