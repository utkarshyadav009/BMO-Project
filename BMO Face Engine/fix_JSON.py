import json
import argparse
from copy import deepcopy

def fix_frame(fr: dict) -> bool:
    """
    Fix spriteSourceSize.w/h to match frame.w/h.
    Returns True if modified.
    """
    changed = False

    if "frame" not in fr:
        return False

    fw = fr["frame"].get("w", None)
    fh = fr["frame"].get("h", None)
    if fw is None or fh is None:
        return False

    # Ensure spriteSourceSize exists
    if "spriteSourceSize" not in fr or not isinstance(fr["spriteSourceSize"], dict):
        fr["spriteSourceSize"] = {"x": 0, "y": 0, "w": fw, "h": fh}
        return True

    sss = fr["spriteSourceSize"]

    # Preserve x/y if they exist, otherwise default to 0
    if "x" not in sss:
        sss["x"] = 0
        changed = True
    if "y" not in sss:
        sss["y"] = 0
        changed = True

    # Fix w/h
    if sss.get("w") != fw:
        sss["w"] = fw
        changed = True
    if sss.get("h") != fh:
        sss["h"] = fh
        changed = True

    return changed


def fix_json(data: dict) -> tuple[dict, int]:
    """
    Fixes either TexturePacker 'textures[].frames[]' format
    or 'frames[]' format.
    Returns (fixed_data, num_frames_changed).
    """
    out = deepcopy(data)
    changed_count = 0

    if isinstance(out, dict) and "textures" in out and isinstance(out["textures"], list):
        for tex in out["textures"]:
            if not isinstance(tex, dict):
                continue
            frames = tex.get("frames")
            if not isinstance(frames, list):
                continue
            for fr in frames:
                if isinstance(fr, dict) and fix_frame(fr):
                    changed_count += 1

    elif isinstance(out, dict) and "frames" in out and isinstance(out["frames"], list):
        for fr in out["frames"]:
            if isinstance(fr, dict) and fix_frame(fr):
                changed_count += 1

    else:
        raise ValueError("Unrecognized JSON format: expected 'textures[].frames[]' or 'frames[]'.")

    return out, changed_count


def main():
    ap = argparse.ArgumentParser(description="Fix spriteSourceSize.w/h to match frame.w/h (no re-cutting).")
    ap.add_argument("input_path", help="Input JSON path (e.g. assets/BMO_Split_Data.json)")
    ap.add_argument("-o", "--output", help="Output JSON path. Default: <input>.fixed.json")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input file (creates a .bak backup).")
    args = ap.parse_args()

    in_path = args.input_path
    out_path = args.output or (in_path + ".fixed.json")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed, n = fix_json(data)

    if args.inplace:
        bak_path = in_path + ".bak"
        with open(bak_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(fixed, f, indent=4)
        print(f"[OK] Fixed {n} frames. Overwrote: {in_path}")
        print(f"[OK] Backup saved as: {bak_path}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixed, f, indent=4)
        print(f"[OK] Fixed {n} frames. Wrote: {out_path}")


if __name__ == "__main__":
    main()
