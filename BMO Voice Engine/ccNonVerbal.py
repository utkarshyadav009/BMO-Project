#!/usr/bin/env python3
"""
BMO Non-Verbal Hunter v6
Strategy: Parse CC/SRT files for non-verbal cues → get exact timestamps →
cut from Demucs-cleaned WAVs → ECAPA verify it's BMO → save confirmed clips.
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
from datetime import timedelta
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Where your Demucs-cleaned BMO-only WAVs live (one per episode)
CLEANED_VOCALS_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\Cleaned_Vocals"

# Where the SRT files are (same folder as MKVs)
SRT_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes"

# Where to save results
OUTPUT_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\BMO_Nonverbals_v6"

# Your existing confirmed BMO speech clips (954 clips) — used to build voiceprint
BMO_SPEECH_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\wavs"

# ECAPA similarity thresholds
SIM_CONFIRMED = 0.75   # ≥ this → confirmed BMO
SIM_REVIEW    = 0.55   # ≥ this → review folder
# below SIM_REVIEW → discard

# How many BMO speech clips to use for voiceprint (more = slower but better)
MAX_VOICEPRINT_CLIPS = 100

# Padding around CC timestamp when cutting audio (seconds)
PAD_START = 0.05
PAD_END   = 0.15

TARGET_SR = 16000

# ─── CATEGORY KEYWORD MAP ─────────────────────────────────────────────────────
# Maps regex patterns to category names
# Order matters — first match wins
CATEGORIES = [
    ("laugh",       r"laugh|giggl|chuckl|cackl|titter|snicker|lol"),
    ("cry",         r"cr(y|ies|ying)|sob|whimper|weep|sniffl|wail"),
    ("gasp",        r"gasp|sharp breath|inhale"),
    ("scream",      r"scream|shriek|screech|yell|wail|howl"),
    ("grunt",       r"grunt|groan|strain|effort|uggh|ugh"),
    ("sigh",        r"sigh|exhale|breath"),
    ("pant",        r"pant|huff"),
    ("other",       r"exclaim|yelp|squeak|squeal|moan"),
]

# Patterns to SKIP — these are not BMO non-verbals we want
SKIP_PATTERNS = [
    r"^J\s",          # OCR'd music note ♪ at start
    r"\sJ$",          # OCR'd music note ♪ at end
    r"^J$",           # lone J (music note)
    r"singing",
    r"music",
    r"theme",
    r"narrator",
    r"subtitl",
    r"caption",
    r"translate",
    r"♪",
    r"♫",
]

# ─── SRT PARSING ──────────────────────────────────────────────────────────────

def parse_timestamp(ts_str):
    """Convert SRT timestamp HH:MM:SS,mmm to seconds."""
    ts_str = ts_str.strip().replace(',', '.')
    parts = ts_str.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s

def extract_nonverbal_cues(srt_path):
    """
    Parse an SRT file and return list of:
    (start_sec, end_sec, raw_text, category)
    for lines that contain non-verbal sound cues.
    """
    with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split into subtitle blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    cues = []

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        # Find timestamp line
        ts_line = None
        text_lines = []
        for i, line in enumerate(lines):
            if re.match(r'\d+:\d+:\d+[,\.]\d+\s*-->\s*\d+:\d+:\d+[,\.]\d+', line):
                ts_line = line
                text_lines = lines[i+1:]
                break

        if not ts_line or not text_lines:
            continue

        # Parse timestamps
        try:
            ts_parts = re.split(r'\s*-->\s*', ts_line)
            start = parse_timestamp(ts_parts[0])
            end   = parse_timestamp(ts_parts[1])
        except Exception:
            continue

        # Check each text line for non-verbal cues
        for text in text_lines:
            # Skip music/singing lines
            skip = False
            for pat in SKIP_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            # Extract content from brackets or parens
            # Match [ ... ] or ( ... ) 
            bracket_match = re.search(r'[\[\(]\s*(.+?)\s*[\]\)]', text)
            if not bracket_match:
                continue

            inner = bracket_match.group(1).strip()
            if not inner or len(inner) < 3:
                continue

            # Check against skip patterns again on inner text
            skip = False
            for pat in SKIP_PATTERNS:
                if re.search(pat, inner, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            # Match to a category
            category = None
            for cat_name, pat in CATEGORIES:
                if re.search(pat, inner, re.IGNORECASE):
                    category = cat_name
                    break

            if category is None:
                continue

            cues.append((start, end, inner, category))

    return cues

# ─── ECAPA VOICEPRINT ─────────────────────────────────────────────────────────

def build_bmo_voiceprint(speech_dir, classifier, max_clips=MAX_VOICEPRINT_CLIPS):
    """Build a mean ECAPA embedding from confirmed BMO speech clips."""
    print(f"\n🎙️  Building BMO voiceprint from {speech_dir}...")
    clips = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))[:max_clips]
    if not clips:
        raise FileNotFoundError(f"No WAV files found in {speech_dir}")

    embeddings = []
    for clip_path in clips:
        try:
            wav, sr = torchaudio.load(clip_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            emb = classifier.encode_batch(wav)
            embeddings.append(emb.squeeze().cpu().numpy())
        except Exception:
            continue

    if not embeddings:
        raise RuntimeError("Could not embed any BMO speech clips")

    voiceprint = np.mean(embeddings, axis=0)
    voiceprint = voiceprint / np.linalg.norm(voiceprint)
    print(f"   ✅ Voiceprint built from {len(embeddings)} clips")
    return voiceprint

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def embed_audio(wav_tensor, classifier):
    emb = classifier.encode_batch(wav_tensor)
    v = emb.squeeze().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)

# ─── AUDIO EXTRACTION ─────────────────────────────────────────────────────────

def load_audio_segment(wav_path, start_sec, end_sec, pad_start=PAD_START, pad_end=PAD_END):
    """Load a segment from a WAV file, with padding."""
    info = sf.info(wav_path)
    sr = info.samplerate
    total_samples = info.frames

    start_sample = max(0, int((start_sec - pad_start) * sr))
    end_sample   = min(total_samples, int((end_sec + pad_end) * sr))

    if end_sample <= start_sample:
        return None, sr

    audio, sr = sf.read(wav_path, start=start_sample, stop=end_sample, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # Load ECAPA
    print("\n🔊 Loading ECAPA speaker encoder...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain_ecapa"),
        local_strategy=LocalStrategy.COPY
    )

    # Build BMO voiceprint
    voiceprint = build_bmo_voiceprint(BMO_SPEECH_DIR, classifier)

    # Find all SRT files
    srt_files = sorted(glob.glob(os.path.join(SRT_DIR, "*.srt")))
    print(f"\n📁 Found {len(srt_files)} SRT files")

    # Create output dirs
    categories_all = [c[0] for c in CATEGORIES]
    for cat in categories_all:
        for tier in ["confirmed", "review"]:
            os.makedirs(os.path.join(OUTPUT_DIR, tier, cat), exist_ok=True)

    # CSV log
    csv_path = os.path.join(OUTPUT_DIR, "extraction_log.csv")
    csv_rows = []

    total_confirmed = 0
    total_review    = 0
    total_discarded = 0

    for srt_path in srt_files:
        episode_name = pathlib.Path(srt_path).stem  # e.g. S01E08_BMO_Vocals

        # Find matching cleaned WAV
        # SRT might be named S01E08_BMO.srt, WAV might be S01E08_BMO_Vocals.wav
        # Try to match by episode code
        ep_code = re.match(r'(S\d+E\d+)', episode_name, re.IGNORECASE)
        if not ep_code:
            continue
        ep_code = ep_code.group(1).upper()

        wav_candidates = glob.glob(os.path.join(CLEANED_VOCALS_DIR, f"{ep_code}*.wav"))
        if not wav_candidates:
            print(f"\n⚠️  No WAV found for {ep_code}, skipping")
            continue
        wav_path = wav_candidates[0]

        # Parse SRT for non-verbal cues
        cues = extract_nonverbal_cues(srt_path)
        if not cues:
            continue

        print(f"\n{'='*60}")
        print(f"📺 {ep_code}  |  {len(cues)} non-verbal cues found")
        print(f"{'='*60}")

        ep_confirmed = ep_review = ep_discarded = 0

        for start, end, text, category in cues:
            dur = end - start
            if dur < 0.2:   # too short
                continue
            if dur > 8.0:   # too long — probably a scene description, not a sound
                continue

            # Load audio segment
            audio, sr = load_audio_segment(wav_path, start, end)
            if audio is None or len(audio) < sr * 0.1:
                continue

            # Resample to 16kHz for ECAPA
            if sr != 16000:
                wav_tensor = torch.tensor(audio).unsqueeze(0)
                wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
            else:
                wav_tensor = torch.tensor(audio).unsqueeze(0)

            # ECAPA embed and compare
            try:
                emb = embed_audio(wav_tensor, classifier)
                sim = cosine_sim(emb, voiceprint)
            except Exception as e:
                print(f"   ⚠️  Embed error: {e}")
                continue

            # Route by similarity
            if sim >= SIM_CONFIRMED:
                tier = "confirmed"
                icon = "✅"
                ep_confirmed += 1
                total_confirmed += 1
            elif sim >= SIM_REVIEW:
                tier = "review"
                icon = "🔍"
                ep_review += 1
                total_review += 1
            else:
                ep_discarded += 1
                total_discarded += 1
                print(f"   ❌ [{category:12s}] sim={sim:.3f}  {start:.2f}s→{end:.2f}s  \"{text}\"")
                continue

            print(f"   {icon} [{category:12s}] sim={sim:.3f}  {start:.2f}s→{end:.2f}s  \"{text}\"")

            # Save clip
            safe_text = re.sub(r'[^\w]', '_', text)[:30]
            out_name = f"{ep_code}_{start:.2f}_{category}_{safe_text}.wav"
            out_path = os.path.join(OUTPUT_DIR, tier, category, out_name)

            # Save at 16kHz
            audio_16k = wav_tensor.squeeze().numpy()
            sf.write(out_path, audio_16k, 16000)

            csv_rows.append({
                "episode": ep_code,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "category": category,
                "raw_text": text,
                "similarity": round(sim, 4),
                "tier": tier,
                "file": out_name,
            })

        print(f"   → {ep_confirmed} confirmed  |  {ep_review} review  |  {ep_discarded} discarded")

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["episode","start","end","duration","category","raw_text","similarity","tier","file"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"🏁 DONE")
    print(f"   ✅ Confirmed : {total_confirmed}")
    print(f"   🔍 Review    : {total_review}")
    print(f"   ❌ Discarded : {total_discarded}")
    print(f"   📊 Log saved : {csv_path}")

if __name__ == "__main__":
    main()