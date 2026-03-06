#!/usr/bin/env python3
"""
BMO Non-Verbal Refiner — Pass 2
Seeds come from TWO sources combined:
  1. Manually found clips: D:\...\nonverbal\<category>\
  2. CLAP pass 1 confirmed: D:\...\BMO_Nonverbals_final\confirmed\<category>\

Combined profile is used to search v6 candidates again.
"""

import os
import re
import csv
import glob
import shutil
import pathlib
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Original manually found seeds
SEED_DIR_MANUAL    = r"D:\LocalWorkDir\u521785\BMO Episodes\nonverbal"

# Pass 1 confirmed + reviewed clips (your reviewed ones moved into confirmed/)
SEED_DIR_CONFIRMED = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_final\confirmed"

# v6 candidates to search through
CANDIDATES_DIR     = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_v6\confirmed"

# Output
OUTPUT_DIR         = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_pass2"

# Similarity thresholds
SIM_CONFIRMED = 0.78
SIM_REVIEW    = 0.62

CLAP_SR = 48000

# Maps manual seed folder name → v6 candidate folder name
CATEGORY_MAP = {
    "laugh":       "laugh",
    "cry":         "cry",
    "scream":      "scream",
    "angry grunt": "grunt",
}

# ─── CLAP ─────────────────────────────────────────────────────────────────────

def load_clap(device):
    print("🔊 Loading CLAP model...")
    model     = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    print("   ✅ CLAP ready")
    return model, processor

def embed_file(wav_path, model, processor, device):
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != CLAP_SR:
        t = torch.tensor(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, CLAP_SR)
        audio = t.squeeze().numpy()
    inputs = processor(audios=audio, sampling_rate=CLAP_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    emb = emb.squeeze().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    model, processor = load_clap(device)

    profiles = {}
    # Build set of filenames already confirmed in pass 1
    print("\n🔍 Loading pass 1 confirmed filenames to skip duplicates...")
    already_confirmed = set()
    for cat in profiles:
        p1_dir = os.path.join(SEED_DIR_CONFIRMED, cat)
        if os.path.isdir(p1_dir):
            for f in glob.glob(os.path.join(p1_dir, "*.wav")):
                already_confirmed.add(pathlib.Path(f).name)
    print(f"   {len(already_confirmed)} clips already confirmed in pass 1 — will skip")
    
    # Build combined seed profiles
    print("\n🧬 Building combined seed profiles (manual + pass1 confirmed)...")

    for seed_folder_name, cand_folder_name in CATEGORY_MAP.items():
        embeddings = []

        # Source 1: manual seeds
        manual_dir = os.path.join(SEED_DIR_MANUAL, seed_folder_name)
        manual_clips = glob.glob(os.path.join(manual_dir, "*.wav")) if os.path.isdir(manual_dir) else []

        # Source 2: pass 1 confirmed seeds
        # Try exact match first, then try the candidate folder name
        confirmed_dir = os.path.join(SEED_DIR_CONFIRMED, seed_folder_name)
        if not os.path.isdir(confirmed_dir):
            confirmed_dir = os.path.join(SEED_DIR_CONFIRMED, cand_folder_name)
        confirmed_clips = glob.glob(os.path.join(confirmed_dir, "*.wav")) if os.path.isdir(confirmed_dir) else []

        all_clips = manual_clips + confirmed_clips
        print(f"\n   [{seed_folder_name}] {len(manual_clips)} manual  +  {len(confirmed_clips)} confirmed  =  {len(all_clips)} total seeds")

        for clip_path in all_clips:
            try:
                emb = embed_file(clip_path, model, processor, device)
                embeddings.append(emb)
            except Exception as e:
                print(f"      ⚠️  {pathlib.Path(clip_path).name}: {e}")

        if embeddings:
            profile = np.mean(embeddings, axis=0)
            profile = profile / (np.linalg.norm(profile) + 1e-8)
            profiles[cand_folder_name] = profile
            print(f"      📊 Profile built from {len(embeddings)} clips")
        else:
            print(f"      ❌ No valid seeds for [{seed_folder_name}] — skipping")

    print(f"\n✅ Profiles ready: {list(profiles.keys())}")

    # Create output dirs
    for cat in profiles:
        os.makedirs(os.path.join(OUTPUT_DIR, "confirmed", cat), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "review",    cat), exist_ok=True)

    csv_rows = []
    total_confirmed = total_review = total_discarded = 0

    for cand_folder_name, profile in profiles.items():
        cand_dir   = os.path.join(CANDIDATES_DIR, cand_folder_name)
        if not os.path.isdir(cand_dir):
            print(f"\n⚠️  No candidate folder: {cand_dir}")
            continue

        candidates = sorted(glob.glob(os.path.join(cand_dir, "*.wav")))
        if not candidates:
            print(f"\n⚠️  No clips in {cand_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"🔍 [{cand_folder_name}]  {len(candidates)} candidates")
        print(f"{'='*60}")

        results = []
        for clip_path in candidates:
            try:
                emb = embed_file(clip_path, model, processor, device)
                sim = cosine_sim(emb, profile)
                results.append((sim, clip_path))
            except Exception as e:
                print(f"   ⚠️  {pathlib.Path(clip_path).name}: {e}")

        results.sort(key=lambda x: x[0], reverse=True)

        cat_confirmed = cat_review = cat_discarded = 0

        for sim, clip_path in results:
            fname = pathlib.Path(clip_path).name

            if fname in already_confirmed:
                continue
            if sim >= SIM_CONFIRMED:
                tier = "confirmed"; icon = "✅"
                cat_confirmed += 1; total_confirmed += 1
            elif sim >= SIM_REVIEW:
                tier = "review";    icon = "🔍"
                cat_review += 1;    total_review += 1
            else:
                tier = "discard";   icon = "❌"
                cat_discarded += 1; total_discarded += 1

            print(f"   {icon} sim={sim:.3f}  {fname}")

            if tier != "discard":
                out_path = os.path.join(OUTPUT_DIR, tier, cand_folder_name, fname)
                shutil.copy2(clip_path, out_path)

            ep_match = re.match(r'(S\d+E\d+)', fname, re.IGNORECASE)
            csv_rows.append({
                "episode":    ep_match.group(1).upper() if ep_match else "unknown",
                "category":   cand_folder_name,
                "similarity": round(sim, 4),
                "tier":       tier,
                "file":       fname,
            })

        print(f"   → ✅ {cat_confirmed}  🔍 {cat_review}  ❌ {cat_discarded}")

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "pass2_log.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["episode","category","similarity","tier","file"])
        writer.writeheader()
        writer.writerows(sorted(csv_rows, key=lambda x: -x["similarity"]))

    print(f"\n{'='*60}")
    print(f"🏁 PASS 2 DONE")
    print(f"   ✅ Confirmed : {total_confirmed}")
    print(f"   🔍 Review    : {total_review}")
    print(f"   ❌ Discarded : {total_discarded}")
    print(f"   📊 Log       : {csv_path}")

if __name__ == "__main__":
    main()