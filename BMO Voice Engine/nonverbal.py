import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import soundfile as sf
import whisperx
from pathlib import Path
from transformers import ClapModel, ClapProcessor
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ================= CONFIGURATION =================

# Full Demucs-cleaned episode WAVs (118 files)
CLEANED_VOCALS_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\Cleaned_Vocals"

# Your manually-found seed clips — one folder per category
# Each folder can have any number of .wav files
SEED_ROOT = r"D:\LocalWorkDir\2509362\BMO Episodes\nonverbal"
SEED_CATEGORIES = {
    "laugh":       os.path.join(SEED_ROOT, "laugh"),
    "cry":         os.path.join(SEED_ROOT, "cry"),
    "scream":      os.path.join(SEED_ROOT, "scream"),
    "angry_grunt": os.path.join(SEED_ROOT, "angry grunt"),
}

# Output
OUTPUT_DIR  = r"D:\LocalWorkDir\2509362\BMO Episodes\BMO_Nonverbals_v5"
HF_CACHE    = r"D:\LocalWorkDir\2509362\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE

# ── WHISPERX ─────────────────────────────────────────────────
WHISPER_MODEL = "base"

# ── GAP DETECTION ─────────────────────────────────────────────
GAP_MIN_SEC   = 0.3    # Minimum gap duration to consider
GAP_MAX_SEC   = 5.0    # Maximum gap duration to consider
PAD_SEC       = 0.10   # Padding added either side when saving

# ── CLAP SIMILARITY THRESHOLDS ────────────────────────────────
# How similar a gap must be to the seed profile to be kept.
# Tune these after your first run — start permissive, tighten later.
SIMILARITY_CONFIRMED = 0.80   # Auto-confirmed
SIMILARITY_REVIEW    = 0.70   # Needs manual review

# CLAP sample rate
CLAP_SR = 48000

TARGET_SR = 16000
# =================================================

# Output subdirs — one per category + review
output_dirs = {}
for cat in SEED_CATEGORIES:
    output_dirs[cat] = os.path.join(OUTPUT_DIR, "confirmed", cat)
    os.makedirs(output_dirs[cat], exist_ok=True)
DIR_REVIEW  = os.path.join(OUTPUT_DIR, "needs_review")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "nonverbal_results.csv")
os.makedirs(DIR_REVIEW, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  MODEL SETUP
# ─────────────────────────────────────────────────────────────

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = device.type
print(f"🖥️  Device: {device}\n")

print(f"🎙️  Loading WhisperX ({WHISPER_MODEL})...")
whisper_model = whisperx.load_model(
    WHISPER_MODEL, device=device_str,
    compute_type="float16" if device_str == "cuda" else "int8"
)

print("🎵 Loading CLAP model...")
clap_model     = ClapModel.from_pretrained(
    "laion/clap-htsat-unfused", cache_dir=HF_CACHE
).to(device).eval()
clap_processor = ClapProcessor.from_pretrained(
    "laion/clap-htsat-unfused", cache_dir=HF_CACHE
)
print("✅ Models ready\n")


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────

def load_for_clap(path):
    """Load audio resampled to 48kHz mono for CLAP."""
    try:
        sig, sr = torchaudio.load(str(path))
        if sr != CLAP_SR:
            sig = T.Resample(sr, CLAP_SR)(sig)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return sig.squeeze().numpy()  # 1D numpy float32
    except Exception as e:
        print(f"  ⚠️  load_for_clap failed {Path(path).name}: {e}")
        return None


def load_for_save(path, offset_sec=0.0, duration_sec=None):
    """Load audio at 16kHz mono for saving output clips."""
    try:
        info = torchaudio.info(str(path))
        sr   = info.sample_rate
        frame_offset = int(offset_sec * sr)
        num_frames   = int(duration_sec * sr) if duration_sec else -1
        sig, sr = torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames)
        if sr != TARGET_SR:
            sig = T.Resample(sr, TARGET_SR)(sig)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return sig.squeeze().numpy()
    except:
        return None


def get_clap_embedding(audio_np):
    """Get CLAP audio embedding for a 1D float32 numpy array at 48kHz."""
    if audio_np is None or len(audio_np) < CLAP_SR * 0.2:
        return None
    try:
        inputs = clap_processor(
            audios=audio_np, sampling_rate=CLAP_SR, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = clap_model.get_audio_features(**inputs)
        emb_np = emb.squeeze().cpu().numpy()
        norm   = np.linalg.norm(emb_np)
        return emb_np / (norm + 1e-8)
    except Exception as e:
        print(f"  ⚠️  CLAP embedding failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  BUILD SEED PROFILES (one per category)
# ─────────────────────────────────────────────────────────────

def build_seed_profiles(seed_root_map):
    """
    For each category, embed all seed clips with CLAP and
    compute a mean normalised profile vector.
    Returns dict: category → mean_embedding (numpy 1D)
    """
    profiles = {}
    print("🌱 Building CLAP seed profiles...\n")

    for category, folder in seed_root_map.items():
        wavs = list(Path(folder).glob("*.wav"))
        if not wavs:
            print(f"  ⚠️  No .wav files found in {folder} — skipping {category}")
            continue

        embeddings = []
        for wav in wavs:
            audio_np = load_for_clap(wav)
            emb      = get_clap_embedding(audio_np)
            if emb is not None:
                embeddings.append(emb)
                print(f"  ✅ [{category}] embedded {wav.name}")
            else:
                print(f"  ❌ [{category}] failed  {wav.name}")

        if not embeddings:
            print(f"  ⚠️  No valid embeddings for {category}")
            continue

        stacked = np.stack(embeddings)
        mean    = stacked.mean(axis=0)
        mean   /= (np.linalg.norm(mean) + 1e-8)
        profiles[category] = mean

        print(f"  📊 [{category}] profile built from {len(embeddings)}/{len(wavs)} clips\n")

    return profiles


# ─────────────────────────────────────────────────────────────
#  WHISPERX GAP EXTRACTION
# ─────────────────────────────────────────────────────────────

def get_word_timestamps(episode_path):
    audio  = whisperx.load_audio(str(episode_path))
    result = whisper_model.transcribe(audio, language="en", batch_size=4)

    align_model, metadata = whisperx.load_align_model(
        language_code="en", device=device_str
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata,
        audio, device_str, return_char_alignments=False
    )

    words = []
    for seg in aligned.get("word_segments", []):
        s = seg.get("start")
        e = seg.get("end")
        if s is not None and e is not None:
            words.append((seg.get("word", ""), float(s), float(e)))

    return words


def find_gaps(word_timestamps, total_duration):
    if not word_timestamps:
        return []

    gaps = []

    first_start = word_timestamps[0][1]
    if first_start >= GAP_MIN_SEC:
        gaps.append((0.0, first_start))

    for i in range(len(word_timestamps) - 1):
        _, _, end_cur   = word_timestamps[i]
        _, start_next, _ = word_timestamps[i + 1]
        gap = start_next - end_cur
        if GAP_MIN_SEC <= gap <= GAP_MAX_SEC:
            gaps.append((end_cur, start_next))

    last_end  = word_timestamps[-1][2]
    remaining = total_duration - last_end
    if GAP_MIN_SEC <= remaining <= GAP_MAX_SEC:
        gaps.append((last_end, total_duration))

    return gaps


# ─────────────────────────────────────────────────────────────
#  SCORE A GAP AGAINST ALL SEED PROFILES
# ─────────────────────────────────────────────────────────────

def score_gap(gap_emb, seed_profiles):
    """
    Compare gap embedding against all category seed profiles.
    Returns (best_category, best_score, all_scores_dict)
    """
    scores = {}
    for cat, profile in seed_profiles.items():
        sim = float(sklearn_cosine(
            gap_emb.reshape(1, -1),
            profile.reshape(1, -1)
        )[0][0])
        scores[cat] = round(sim, 4)

    best_cat   = max(scores, key=scores.get)
    best_score = scores[best_cat]
    return best_cat, best_score, scores


# ─────────────────────────────────────────────────────────────
#  PROCESS ONE EPISODE
# ─────────────────────────────────────────────────────────────

def process_episode(ep_path, seed_profiles, ep_name):
    try:
        info    = torchaudio.info(str(ep_path))
        total_s = info.num_frames / info.sample_rate
    except Exception as e:
        print(f"  ⚠️  Cannot read {ep_name}: {e}")
        return []

    print(f"  🎙️  Transcribing...")
    words = get_word_timestamps(str(ep_path))
    if not words:
        print(f"  ⚠️  No words found — skipping")
        return []

    gaps = find_gaps(words, total_s)
    print(f"  📐 {len(words)} words  |  {len(gaps)} gaps ({GAP_MIN_SEC}s–{GAP_MAX_SEC}s)")

    if not gaps:
        return []

    results         = []
    confirmed_count = 0
    review_count    = 0

    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start

        # Load gap at CLAP sample rate for embedding
        try:
            info      = torchaudio.info(str(ep_path))
            sr_raw    = info.sample_rate
            pad_start = max(0.0, gap_start - PAD_SEC)
            pad_end   = min(total_s, gap_end + PAD_SEC)
            load_dur  = pad_end - pad_start

            sig_raw, sr = torchaudio.load(
                str(ep_path),
                frame_offset=int(pad_start * sr_raw),
                num_frames=int(load_dur * sr_raw)
            )
            # Resample to CLAP SR
            if sr != CLAP_SR:
                sig_clap = T.Resample(sr, CLAP_SR)(sig_raw)
            else:
                sig_clap = sig_raw
            if sig_clap.shape[0] > 1:
                sig_clap = sig_clap.mean(dim=0, keepdim=True)
            audio_np_clap = sig_clap.squeeze().numpy()
        except Exception:
            continue

        # Skip near-silence
        if np.sqrt(np.mean(audio_np_clap ** 2)) < 0.003:
            continue

        # Get CLAP embedding
        gap_emb = get_clap_embedding(audio_np_clap)
        if gap_emb is None:
            continue

        # Score against all seed profiles
        best_cat, best_score, all_scores = score_gap(gap_emb, seed_profiles)

        if best_score < SIMILARITY_REVIEW:
            continue  # Not similar enough to any seed category

        # Determine output
        confirmed = best_score >= SIMILARITY_CONFIRMED
        out_dir   = output_dirs.get(best_cat, DIR_REVIEW) if confirmed else DIR_REVIEW

        out_name = f"{ep_name}_{best_cat}_{gap_start:.2f}.wav"
        out_path = os.path.join(out_dir, out_name)

        # Save at 16kHz
        audio_np_16k = load_for_save(
            str(ep_path),
            offset_sec=pad_start,
            duration_sec=load_dur
        )
        if audio_np_16k is not None:
            sf.write(out_path, audio_np_16k, TARGET_SR)

        status = "✅" if confirmed else "🔍"
        emoji  = {"laugh":"😂","cry":"😢","scream":"😱","angry_grunt":"😤"}.get(best_cat, "❓")
        scores_str = "  ".join(f"{c}={s:.2f}" for c, s in all_scores.items())
        print(f"    {status} {emoji} [{best_cat:12s}] best={best_score:.3f}  "
              f"dur={gap_dur:.2f}s  {gap_start:.2f}s→{gap_end:.2f}s")
        print(f"         scores: {scores_str}")

        row = {
            "episode":    ep_name,
            "output_file": out_name,
            "label":       best_cat,
            "gap_start":   round(gap_start, 3),
            "gap_end":     round(gap_end, 3),
            "duration":    round(gap_dur, 3),
            "best_score":  best_score,
            "confirmed":   confirmed,
        }
        row.update({f"score_{c}": s for c, s in all_scores.items()})
        results.append(row)

        if confirmed: confirmed_count += 1
        else:         review_count    += 1

    print(f"  → {confirmed_count} confirmed  |  {review_count} review")
    return results


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # 1. Build seed profiles from manually-found clips
    seed_profiles = build_seed_profiles(SEED_CATEGORIES)
    if not seed_profiles:
        print("❌ No seed profiles built — check your seed folders!")
        return
    print(f"✅ Profiles ready for: {list(seed_profiles.keys())}\n")

    # 2. Find all episode WAVs
    episode_wavs = sorted(Path(CLEANED_VOCALS_DIR).glob("*.wav"))
    print(f"📁 Found {len(episode_wavs)} episode WAVs\n")

    # 3. Resume support
    all_results   = []
    done_episodes = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            prev          = pd.read_csv(OUTPUT_CSV)
            done_episodes = set(prev["episode"].tolist())
            all_results   = prev.to_dict("records")
            print(f"📋 Resuming — {len(done_episodes)} episodes already done\n")
        except Exception:
            pass

    total_confirmed = 0
    total_review    = 0

    for ep_path in episode_wavs:
        ep_name = ep_path.stem
        if ep_name in done_episodes:
            print(f"  ⏭️  {ep_name} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"📺 {ep_name}")
        print(f"{'='*60}")

        results          = process_episode(ep_path, seed_profiles, ep_name)
        all_results.extend(results)

        ep_confirmed     = sum(1 for r in results if r["confirmed"])
        ep_review        = sum(1 for r in results if not r["confirmed"])
        total_confirmed += ep_confirmed
        total_review    += ep_review

        done_episodes.add(ep_name)
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"🎉 Done!")
    for cat, d in output_dirs.items():
        emoji = {"laugh":"😂","cry":"😢","scream":"😱","angry_grunt":"😤"}.get(cat,"❓")
        print(f"  {emoji} {cat:12s} → {d}")
    print(f"  🔍 review      → {DIR_REVIEW}")
    print(f"  ✅ confirmed   : {total_confirmed}")
    print(f"  🔍 review      : {total_review}")
    print(f"  💾 CSV         : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()