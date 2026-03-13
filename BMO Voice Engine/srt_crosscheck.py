#!/usr/bin/env python3
"""
SRT Cross-Check
Compares your labeled metadata.csv transcriptions against the SRT closed captions.
Flags clips where the SRT text is very different from your label — likely mislabels.

Output:
  - crosscheck_report.csv  — all clips with SRT match, similarity score, flags
  - crosscheck_mismatches.csv — only the suspicious ones
"""

import os
import re
import csv
import glob
import pathlib
import difflib
import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────

METADATA_CSV = r"C:\Users\2509362\Downloads\BMO_SpeechDataset\BMO_SpeechDataset\metadata.csv"
SRT_DIR      = r"C:\Users\2509362\OneDrive - Abertay University\Documents\GitHub\BMO-Project\BMO Voice Engine\srtfiles"   # folder where *.srt files live
OUTPUT_DIR   = r"C:\Users\2509362\Downloads\BMO_SpeechDataset\BMO_SpeechDataset"

# Similarity threshold — below this is flagged as suspicious
# 0.0 = completely different, 1.0 = identical
# 0.3 means "shares less than 30% of characters/words in common"
SIMILARITY_THRESHOLD = 0.35

# ─── SRT PARSING ──────────────────────────────────────────────────────────────

def parse_timestamp(ts_str):
    ts_str = ts_str.strip().replace(',', '.')
    parts  = ts_str.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s

def load_srt(srt_path):
    """Returns list of (start_sec, end_sec, text) for every subtitle block."""
    with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []

    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        ts_line    = None
        text_lines = []
        for i, line in enumerate(lines):
            if re.match(r'\d+:\d+:\d+[,\.]\d+\s*-->\s*\d+:\d+:\d+[,\.]\d+', line):
                ts_line    = line
                text_lines = lines[i+1:]
                break

        if not ts_line or not text_lines:
            continue

        try:
            ts_parts = re.split(r'\s*-->\s*', ts_line)
            start    = parse_timestamp(ts_parts[0])
            end      = parse_timestamp(ts_parts[1])
        except Exception:
            continue

        text = ' '.join(text_lines)
        entries.append((start, end, text))

    return entries

def srt_text_at(srt_entries, timestamp_sec, window=1.5):
    """Find SRT text at a given timestamp (±window seconds)."""
    best_text = ""
    best_overlap = 0.0

    for start, end, text in srt_entries:
        # Check overlap between [timestamp-window, timestamp+window] and [start, end]
        overlap_start = max(timestamp_sec - window, start)
        overlap_end   = min(timestamp_sec + window, end)
        overlap       = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_text    = text

    return best_text, best_overlap

# ─── TEXT SIMILARITY ──────────────────────────────────────────────────────────

def normalise(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # remove tags like [laugh], (gasps)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def similarity(a, b):
    """SequenceMatcher ratio — 0.0 to 1.0."""
    a = normalise(a)
    b = normalise(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("📋 BMO SRT Cross-Check")

    # Load metadata
    try:
        meta_df = pd.read_csv(METADATA_CSV, sep="|", on_bad_lines='skip', dtype=str)
    except Exception as e:
        print(f"❌ Cannot read metadata: {e}")
        return

    print(f"   {len(meta_df)} clips in metadata")

    # Index SRT files by episode code
    srt_files = glob.glob(os.path.join(SRT_DIR, "**", "*.srt"), recursive=True)
    srt_index = {}
    for srt_path in srt_files:
        ep_match = re.match(r'(S\d+E\d+)', pathlib.Path(srt_path).name, re.IGNORECASE)
        if ep_match:
            srt_index[ep_match.group(1).upper()] = srt_path
    print(f"   {len(srt_index)} SRT files indexed")

    # Cache loaded SRT entries per episode
    srt_cache = {}

    rows = []

    for _, row in meta_df.iterrows():
        filename = str(row.get('filename', '')).strip()
        label    = str(row.get('text', '')).strip()
        tone     = str(row.get('tone', 'neutral')).strip()

        # Parse episode code and timestamp from filename
        # Format: S04E17_123.45_category_text.wav  or  S04E17_BMO_Vocals_SPEAKER_03_123.45.wav
        ep_match = re.match(r'(S\d+E\d+)', filename, re.IGNORECASE)
        if not ep_match:
            rows.append({
                "filename": filename, "label": label, "tone": tone,
                "srt_text": "", "similarity": -1, "flag": "NO_EPISODE"
            })
            continue

        ep_code = ep_match.group(1).upper()

        # Extract timestamp — look for a float in the filename
        ts_match = re.search(r'_(\d+\.\d+)', filename)
        if not ts_match:
            rows.append({
                "filename": filename, "label": label, "tone": tone,
                "srt_text": "", "similarity": -1, "flag": "NO_TIMESTAMP"
            })
            continue

        timestamp = float(ts_match.group(1))

        # Load SRT if not cached
        if ep_code not in srt_cache:
            srt_path = srt_index.get(ep_code)
            if srt_path:
                srt_cache[ep_code] = load_srt(srt_path)
            else:
                srt_cache[ep_code] = []

        srt_entries = srt_cache[ep_code]
        srt_text, overlap = srt_text_at(srt_entries, timestamp, window=2.0)

        if not srt_text:
            flag = "NO_SRT_MATCH"
            sim  = -1.0
        else:
            sim  = similarity(label, srt_text)
            if sim < SIMILARITY_THRESHOLD:
                flag = "MISMATCH"
            elif sim > 0.85:
                flag = "OK"
            else:
                flag = "REVIEW"

        rows.append({
            "filename":   filename,
            "label":      label,
            "tone":       tone,
            "srt_text":   srt_text,
            "similarity": round(sim, 3),
            "timestamp":  round(timestamp, 2),
            "episode":    ep_code,
            "flag":       flag
        })

    # Save full report
    report_path = os.path.join(OUTPUT_DIR, "crosscheck_report.csv")
    fieldnames  = ["episode", "timestamp", "filename", "label", "srt_text", "similarity", "tone", "flag"]
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x.get('similarity', 1.0)))
    print(f"\n📊 Full report: {report_path}")

    # Save mismatches only
    mismatches = [r for r in rows if r.get('flag') == "MISMATCH"]
    mismatch_path = os.path.join(OUTPUT_DIR, "crosscheck_mismatches.csv")
    with open(mismatch_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatches)

    # Summary
    flags = {}
    for r in rows:
        flags[r['flag']] = flags.get(r['flag'], 0) + 1

    print(f"\n{'='*50}")
    print(f"🏁 CROSS-CHECK SUMMARY")
    for flag, count in sorted(flags.items()):
        icon = {"OK": "✅", "REVIEW": "🔍", "MISMATCH": "❌", "NO_SRT_MATCH": "⚠️", "NO_EPISODE": "❓", "NO_TIMESTAMP": "❓"}.get(flag, "•")
        print(f"   {icon} {flag}: {count}")
    print(f"\n   ❌ Mismatches saved to: {mismatch_path}")
    print(f"   Review these — they may have wrong transcription or wrong timestamp")


if __name__ == "__main__":
    main()
