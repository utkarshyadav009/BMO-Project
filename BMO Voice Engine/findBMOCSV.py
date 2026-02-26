import os
import shutil
import csv

# === USER SETTINGS ===
csv_file = r"bmo_metadata.csv"
source_folder = r"D:\LocalWorkDir\2509362\BMO Episodes\PlanB_Dataset"
destination_folder = r"D:\LocalWorkDir\2509362\BMO Episodes\FinalFinalDataset"
# =====================

# Read filenames from CSV (pipe-delimited)
clips_to_find = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    next(reader)  # Skip header row
    for row in reader:
        if row:  # Avoid empty rows
            clips_to_find.append(row[0].strip())

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

found_clips = []

# Walk through source folder recursively
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file in clips_to_find:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)

            shutil.copy2(source_path, destination_path)
            found_clips.append(file)
            print(f"Copied: {file}")

# Report results
missing = set(clips_to_find) - set(found_clips)

print("\nDone.")
print(f"Found: {len(found_clips)}")
print(f"Missing: {len(missing)}")

if missing:
    print("\nMissing clips:")
    for clip in sorted(missing):
        print(clip)