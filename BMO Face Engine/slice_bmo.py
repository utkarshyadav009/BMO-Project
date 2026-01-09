import json
import os

# CONFIGURATION
INPUT_JSON = "assets/BMO_SpriteSheet_Data.json"
OUTPUT_JSON = "assets/BMO_Split_Data.json"

# The logic: We split the face at 50% height
# Top 50% = Eyes
# Bottom 50% = Mouth
SPLIT_RATIO = 0.55 # 55% for eyes (BMO's eyes are high up), 45% for mouth

def slice_atlas():
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Could not find {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    new_frames = []

    # Iterate through every face in the original list
    for texture in data["textures"]:
        for frame in texture["frames"]:
            original_name = frame["filename"]
            
            # Get original coordinates
            x = frame["frame"]["x"]
            y = frame["frame"]["y"]
            w = frame["frame"]["w"]
            h = frame["frame"]["h"]

            # Calculate the split height
            eye_height = int(h * SPLIT_RATIO)
            mouth_height = h - eye_height

            # --- CREATE EYE ENTRY ---
            eye_frame = frame.copy()
            eye_frame["filename"] = f"{original_name}_eyes"
            eye_frame["frame"] = {
                "x": x,
                "y": y,             # Start at top
                "w": w,
                "h": eye_height     # Only go down 55%
            }
            new_frames.append(eye_frame)

            # --- CREATE MOUTH ENTRY ---
            mouth_frame = frame.copy()
            mouth_frame["filename"] = f"{original_name}_mouth"
            mouth_frame["frame"] = {
                "x": x,
                "y": y + eye_height, # Start where eyes ended
                "w": w,
                "h": mouth_height    # Rest of the height
            }
            new_frames.append(mouth_frame)

    # Save the new JSON structure
    output_data = data.copy()
    output_data["textures"][0]["frames"] = new_frames

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Success! Created {OUTPUT_JSON} with {len(new_frames)} parts.")

if __name__ == "__main__":
    slice_atlas()