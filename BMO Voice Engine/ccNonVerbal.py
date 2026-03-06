#!/usr/bin/env python3
"""
BMO Non-Verbal Hunter v6 (energy-only + episode whitelist)
Strategy:
  1. Build whitelist of episodes where BMO is confirmed to speak
     (from FinalFinalDataset filenames)
  2. Only process SRTs for those episodes
  3. Parse CC → find non-verbal cues → cut from Demucs WAV → check energy → save
"""

import os
import re
import csv
import glob
import pathlib
import numpy as np
import soundfile as sf

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Demucs-separated BMO-only vocal WAVs (one per episode)
CLEANED_VOCALS_DIR = r"D:\LocalWorkDir\u521785\BMO Episodes\Extracted_Vocals"

# SRT files
SRT_DIR = r"D:\LocalWorkDir\u521785\BMO Episodes\srtfiles"

# Confirmed BMO speech clips — used only to build episode whitelist
BMO_SPEECH_DIR = r"D:\LocalWorkDir\u521785\BMO Episodes\FinalFinalDataset"

# Output folder
OUTPUT_DIR = r"D:\LocalWorkDir\u521785\BMO Episodes\BMO_Nonverbals_v6"

# Minimum RMS energy to consider a clip non-silent
# 0.005 = quite quiet, 0.01 = moderate, 0.02 = louder
ENERGY_THRESHOLD = 0.005

# Padding around CC timestamp when cutting (seconds)
PAD_START = 0.1
PAD_END   = 0.2

# Duration limits
MIN_DUR = 0.2
MAX_DUR = 8.0

TARGET_SR = 16000

# ─── CATEGORY KEYWORD MAP ─────────────────────────────────────────────────────
CATEGORIES = [
    ("laugh",   r"laugh|giggl|chuckl|cackl|titter|snicker|hehe|haha"),
    ("cry",     r"cr(y|ies|ying)|sob|whimper|weep|sniffl|wail"),
    ("gasp",    r"gasp|sharp.?breath|inhale|exhale.?deep"),
    ("scream",  r"scream|shriek|screech|yell|howl"),
    ("grunt",   r"grunt|groan|strain|cough"),
    ("sigh",    r"sigh|exhale|breath"),
    ("pant",    r"pant|huff"),
    ("other",   r"exclaim|yelp|squeal|squeak"),
]

# Lines to skip entirely
SKIP_PATTERNS = [
    r"^J[\s\W]",
    r"[\s\W]J$",
    r"^J$",
    r"singing", r"music", r"theme",
    r"narrator", r"subtitl", r"caption", r"translate",
    r"♪", r"♫",
    r"tires?\s+screech", r"wind\s+howl",
    r"siren", r"alarm",
    r"mouse\s+squeak", r"chair\s+squeak",
    r"crowd", r"audience",
    r"all\s+(laugh|cry|scream|gasp|groan|cheer)",
    r"both\s+(laugh|cry|scream|gasp|groan)",
]

# ─── BUILD EPISODE WHITELIST ──────────────────────────────────────────────────

def build_episode_whitelist(speech_dir):
    """Extract unique episode codes from FinalFinalDataset filenames."""
    clips = glob.glob(os.path.join(speech_dir, "*.wav"))
    episodes = set()
    for f in clips:
        m = re.match(r'(S\d+E\d+)', pathlib.Path(f).name, re.IGNORECASE)
        if m:
            episodes.add(m.group(1).upper())
    return episodes

# ─── SRT PARSING ──────────────────────────────────────────────────────────────

def parse_timestamp(ts_str):
    ts_str = ts_str.strip().replace(',', '.')
    parts = ts_str.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s

def extract_nonverbal_cues(srt_path):
    with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    cues = []

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        ts_line = None
        text_lines = []
        for i, line in enumerate(lines):
            if re.match(r'\d+:\d+:\d+[,\.]\d+\s*-->\s*\d+:\d+:\d+[,\.]\d+', line):
                ts_line = line
                text_lines = lines[i+1:]
                break

        if not ts_line or not text_lines:
            continue

        try:
            ts_parts = re.split(r'\s*-->\s*', ts_line)
            start = parse_timestamp(ts_parts[0])
            end   = parse_timestamp(ts_parts[1])
        except Exception:
            continue

        dur = end - start
        if dur < MIN_DUR or dur > MAX_DUR:
            continue

        for text in text_lines:
            skip = False
            for pat in SKIP_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            bracket_match = re.search(r'[\[\(]\s*(.+?)\s*[\]\)]', text)
            if not bracket_match:
                continue

            inner = bracket_match.group(1).strip()
            if not inner or len(inner) < 3:
                continue

            skip = False
            for pat in SKIP_PATTERNS:
                if re.search(pat, inner, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue

            category = None
            for cat_name, pat in CATEGORIES:
                if re.search(pat, inner, re.IGNORECASE):
                    category = cat_name
                    break
            if category is None:
                continue

            cues.append((start, end, inner, category))

    return cues

# ─── AUDIO ────────────────────────────────────────────────────────────────────

def load_segment(wav_path, start_sec, end_sec):
    info = sf.info(wav_path)
    sr   = info.samplerate
    total = info.frames

    s0 = max(0, int((start_sec - PAD_START) * sr))
    s1 = min(total, int((end_sec + PAD_END) * sr))
    if s1 <= s0:
        return None, TARGET_SR

    audio, _ = sf.read(wav_path, start=s0, stop=s1, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        import torch, torchaudio
        t = torch.tensor(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, TARGET_SR)
        audio = t.squeeze().numpy()

    return audio, TARGET_SR

def rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("🎮 BMO Non-Verbal Hunter v6 (whitelist + energy)")

    # Build episode whitelist from confirmed speech clips
    bmo_episodes = build_episode_whitelist(BMO_SPEECH_DIR)
    print(f"\n📋 BMO confirmed in {len(bmo_episodes)} episodes:")
    print("   " + ", ".join(sorted(bmo_episodes)))

    # Find SRTs and filter to whitelist only
    all_srts = sorted(glob.glob(os.path.join(SRT_DIR, "*.srt")))
    srt_files = []
    for srt_path in all_srts:
        ep_match = re.match(r'(S\d+E\d+)', pathlib.Path(srt_path).name, re.IGNORECASE)
        if ep_match and ep_match.group(1).upper() in bmo_episodes:
            srt_files.append(srt_path)

    print(f"\n📁 Processing {len(srt_files)} / {len(all_srts)} SRT files (BMO episodes only)")

    # Create output dirs
    for cat_name, _ in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_DIR, "confirmed", cat_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "silent",    cat_name), exist_ok=True)

    csv_rows = []
    total_saved  = 0
    total_silent = 0
    total_no_wav = 0

    for srt_path in srt_files:
        ep_code = re.match(r'(S\d+E\d+)', pathlib.Path(srt_path).name, re.IGNORECASE).group(1).upper()

        # Find matching Demucs WAV
        wav_candidates = glob.glob(os.path.join(CLEANED_VOCALS_DIR, f"{ep_code}*.wav"))
        if not wav_candidates:
            print(f"\n⚠️  No WAV found for {ep_code}, skipping")
            total_no_wav += 1
            continue
        wav_path = wav_candidates[0]

        cues = extract_nonverbal_cues(srt_path)
        if not cues:
            print(f"\n{ep_code}  — no non-verbal cues in SRT")
            continue

        print(f"\n{'='*60}")
        print(f"📺 {ep_code}  |  {len(cues)} cues  |  {pathlib.Path(wav_path).name}")
        print(f"{'='*60}")

        ep_saved = ep_silent = 0

        for start, end, text, category in cues:
            audio, sr = load_segment(wav_path, start, end)
            if audio is None or len(audio) < int(sr * 0.1):
                continue

            energy    = rms(audio)
            has_sound = energy >= ENERGY_THRESHOLD
            tier      = "confirmed" if has_sound else "silent"
            icon      = "✅" if has_sound else "🔇"

            if has_sound:
                ep_saved += 1
                total_saved += 1
            else:
                ep_silent += 1
                total_silent += 1

            safe_text = re.sub(r'[^\w]', '_', text)[:25]
            out_name  = f"{ep_code}_{start:.2f}_{category}_{safe_text}.wav"
            out_path  = os.path.join(OUTPUT_DIR, tier, category, out_name)
            sf.write(out_path, audio, sr)

            print(f"   {icon} [{category:8s}] rms={energy:.4f}  {start:.2f}s→{end:.2f}s  \"{text}\"")

            csv_rows.append({
                "episode":  ep_code,
                "start":    round(start, 3),
                "end":      round(end, 3),
                "duration": round(end - start, 3),
                "category": category,
                "text":     text,
                "rms":      round(energy, 5),
                "tier":     tier,
                "file":     out_name,
            })

        print(f"   → {ep_saved} saved  |  {ep_silent} silent")

    # CSV log
    csv_path = os.path.join(OUTPUT_DIR, "extraction_log.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["episode","start","end","duration","category","text","rms","tier","file"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"🏁 DONE")
    print(f"   ✅ Saved with audio : {total_saved}")
    print(f"   🔇 Silent (skipped) : {total_silent}")
    print(f"   ⚠️  No WAV found     : {total_no_wav}")
    print(f"   📊 Log              : {csv_path}")

if __name__ == "__main__":
    main()