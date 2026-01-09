import json
import argparse
from copy import deepcopy

EYES_HEIGHT = 396  # your top slice height

def fix_one_frame(fr: dict) -> bool:
    if "filename" not in fr or "frame" not in fr:
        return False

    name = fr["filename"]
    is_mouth = name.endswith("_mouth")
    is_eyes = name.endswith("_eyes")

    # ensure spriteSourceSize exists
    if "spriteSourceSize" not in fr or not isinstance(fr["spriteSourceSize"], dict):
        fr["spriteSourceSize"] = {"x": 0, "y": 0, "w": fr["frame"]["w"], "h": fr["frame"]["h"]}
        return True

    sss = fr["spriteSourceSize"]
    changed = False

    # Always ensure w/h matches the actual cut
    fw = fr["frame"]["w"]
    fh = fr["frame"]["h"]
    if sss.get("w") != fw:
        sss["w"] = fw
        changed = True
    if sss.get("h") != fh:
        sss["h"] = fh
        changed = True

    # Fix offsets:
    # eyes live at y=0 in the original
    # mouth lives at y=EYES_HEIGHT in the original
    if is_eyes:
        if sss.get("y", 0) != 0:
            sss["y"] = 0
            changed = True
        if sss.get("x", 0) != 0:
            sss["x"] = 0
            changed = True

    if is_mouth:
        if sss.get("y", 0) != EYES_HEIGHT:
            sss["y"] = EYES_HEIGHT
            changed = True
        if sss.get("x", 0) != 0:
            sss["x"] = 0
            changed = True

    return changed

def fix_json(data: dict) -> tuple[dict, int]:
    out = deepcopy(data)
    changed = 0

    if "textures" in out and isinstance(out["textures"], list):
        for tex in out["textures"]:
            frames = tex.get("frames", [])
            for fr in frames:
                if isinstance(fr, dict) and fix_one_frame(fr):
                    changed += 1
    elif "frames" in out and isinstance(out["frames"], list):
        for fr in out["frames"]:
            if isinstance(fr, dict) and fix_one_frame(fr):
                changed += 1
    else:
        raise ValueError("Unknown JSON format")

    return out, changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="assets/BMO_Split_Data.json")
    ap.add_argument("-o", "--output", default=None, help="Output path (default: <input>.fixed.json)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input, write .bak backup")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or (in_path + ".fixed.json")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed, n = fix_json(data)

    if args.inplace:
        bak = in_path + ".bak"
        with open(bak, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(fixed, f, indent=4)
        print(f"[OK] Updated {n} frames. Overwrote {in_path}. Backup: {bak}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixed, f, indent=4)
        print(f"[OK] Updated {n} frames. Wrote {out_path}")

if __name__ == "__main__":
    main()
