#!/usr/bin/env python3
"""
BMO Non-Verbal Refiner
Uses CLAP embeddings to match manually confirmed seed clips
against the v6 energy-filtered clips.

Seeds:  D:\LocalWorkDir\u521785\BMO Episodes\nonverbal\<category>\*.wav
Search: D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_v6\confirmed\<category>\*.wav

For each category:
  1. Embed all seed clips → compute mean profile vector
  2. Embed all v6 candidate clips
  3. Cosine similarity → rank by score
  4. Copy top matches to output folder
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

SEED_DIR       = r"D:\LocalWorkDir\u521785\BMO Episodes\nonverbal"
CANDIDATES_DIR = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_v6\confirmed"
OUTPUT_DIR     = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_final"

# Maps seed folder name → v6 candidate folder name
# Seed folders that have no mapping are skipped
CATEGORY_MAP = {
    "laugh":       "laugh",
    "cry":         "cry",
    "scream":      "scream",
    "angry grunt": "grunt",
}


# Similarity thresholds
SIM_CONFIRMED = 0.80   # ≥ this → high confidence
SIM_REVIEW    = 0.65   # ≥ this → worth reviewing
# below SIM_REVIEW → likely not BMO non-verbal

CLAP_SR = 48000   # CLAP expects 48kHz

# ─── CLAP SETUP ───────────────────────────────────────────────────────────────

def load_clap(device):
    print("🔊 Loading CLAP model...")
    model     = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    print("   ✅ CLAP ready")
    return model, processor

def embed_audio_clap(wav_path, model, processor, device, target_sr=CLAP_SR):
    """Load a WAV and return its CLAP embedding (normalised)."""
    audio, sr = sf.read(wav_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to CLAP SR
    if sr != target_sr:
        t = torch.tensor(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        audio = t.squeeze().numpy()

    inputs = processor(audios=audio, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = model.get_audio_features(**inputs)

    emb = emb.squeeze().cpu().numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    model, processor = load_clap(device)

    # Discover categories from seed folder structure
    seed_categories = []
    for seed_name, cand_name in CATEGORY_MAP.items():
        seed_subdir = os.path.join(SEED_DIR, seed_name)
        seeds = glob.glob(os.path.join(seed_subdir, "*.wav"))
        if seeds:
            seed_categories.append((seed_name, cand_name, seed_subdir, seeds))
        else:
            print(f"⚠️  No seeds found in {seed_subdir}")

    if not seed_categories:
        print(f"❌ No seed categories found in {SEED_DIR}")
        print("   Expected subfolders like: laugh/, cry/, scream/ etc.")
        return

    print(f"\n📂 Found {len(seed_categories)} seed categories:")
    for cat, cand_name, _, seeds in seed_categories:
        print(f"   {cat}: {len(seeds)} seed clips")

    # Build seed profiles
    print("\n🧬 Building CLAP seed profiles...")
    profiles = {}
    for cat, cand_name, seed_dir, seed_clips in seed_categories:
        embeddings = []
        for clip_path in seed_clips:
            try:
                emb = embed_audio_clap(clip_path, model, processor, device)
                embeddings.append(emb)
                print(f"   ✅ [{cat}] {pathlib.Path(clip_path).name}")
            except Exception as e:
                print(f"   ⚠️  [{cat}] failed {pathlib.Path(clip_path).name}: {e}")

        if embeddings:
            profile = np.mean(embeddings, axis=0)
            profile = profile / (np.linalg.norm(profile) + 1e-8)
            profiles[cat] = profile
            print(f"   📊 [{cat}] profile built from {len(embeddings)} clips")
        else:
            print(f"   ❌ [{cat}] no valid seeds — skipping")

    print(f"\n✅ Profiles ready: {list(profiles.keys())}")

    # Create output dirs
    for cat in profiles:
        os.makedirs(os.path.join(OUTPUT_DIR, "confirmed", cat), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "review",    cat), exist_ok=True)

    # Search through v6 candidates
    csv_rows = []
    total_confirmed = total_review = total_discarded = 0

    for cat, profile in profiles.items():
        # Find candidate clips for this category
        cand_dir  = os.path.join(CANDIDATES_DIR, cat)
        if not os.path.isdir(cand_dir):
            # Also try matching v6 categories to seed categories loosely
            print(f"\n⚠️  No candidate folder found for category '{cat}' in {CANDIDATES_DIR}")
            print(f"   Expected: {cand_dir}")
            continue

        candidates = sorted(glob.glob(os.path.join(cand_dir, "*.wav")))
        if not candidates:
            print(f"\n⚠️  No candidate clips in {cand_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"🔍 [{cat}]  {len(candidates)} candidates vs {len([c for c in seed_categories if c[0]==cat][0][2])} seeds")
        print(f"{'='*60}")

        results = []
        for clip_path in candidates:
            try:
                emb = embed_audio_clap(clip_path, model, processor, device)
                sim = cosine_sim(emb, profile)
                results.append((sim, clip_path))
            except Exception as e:
                print(f"   ⚠️  Error on {pathlib.Path(clip_path).name}: {e}")

        # Sort by similarity descending
        results.sort(key=lambda x: x[0], reverse=True)

        cat_confirmed = cat_review = cat_discarded = 0

        for sim, clip_path in results:
            fname = pathlib.Path(clip_path).name

            if sim >= SIM_CONFIRMED:
                tier = "confirmed"
                icon = "✅"
                cat_confirmed += 1
                total_confirmed += 1
            elif sim >= SIM_REVIEW:
                tier = "review"
                icon = "🔍"
                cat_review += 1
                total_review += 1
            else:
                tier = "discard"
                icon = "❌"
                cat_discarded += 1
                total_discarded += 1

            print(f"   {icon} sim={sim:.3f}  {fname}")

            if tier != "discard":
                out_path = os.path.join(OUTPUT_DIR, tier, cat, fname)
                shutil.copy2(clip_path, out_path)

            # Extract episode from filename
            ep_match = re.match(r'(S\d+E\d+)', fname, re.IGNORECASE)
            ep_code  = ep_match.group(1).upper() if ep_match else "unknown"

            csv_rows.append({
                "episode":  ep_code,
                "category": cat,
                "similarity": round(sim, 4),
                "tier":     tier,
                "file":     fname,
            })

        print(f"   → {cat_confirmed} confirmed  |  {cat_review} review  |  {cat_discarded} discarded")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "similarity_log.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["episode","category","similarity","tier","file"])
        writer.writeheader()
        writer.writerows(sorted(csv_rows, key=lambda x: -x["similarity"]))

    print(f"\n{'='*60}")
    print(f"🏁 DONE")
    print(f"   ✅ Confirmed : {total_confirmed}")
    print(f"   🔍 Review    : {total_review}")
    print(f"   ❌ Discarded : {total_discarded}")
    print(f"   📊 Log       : {csv_path}")

if __name__ == "__main__":
    main()