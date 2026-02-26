import os
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import soundfile as sf
import whisperx
from pathlib import Path
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ================= CONFIGURATION =================

# Full Demucs-cleaned episode WAVs (the 118 files)
CLEANED_VOCALS_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\Cleaned_Vocals"

# Confirmed clean BMO speech clips for voiceprint
BMO_SEED_DIR       = r"D:\LocalWorkDir\2509362\BMO Episodes\FinalFinalDataset"

# Output
OUTPUT_DIR         = r"D:\LocalWorkDir\2509362\BMO Episodes\BMO_Nonverbals_v4"

# SpeechBrain cache
MODEL_CACHE        = r"D:\LocalWorkDir\2509362\BMO-Project\BMO Voice Engine\tmpdir_local"

# HuggingFace cache (avoids symlink issues on Windows)
HF_CACHE           = r"D:\LocalWorkDir\2509362\hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE

# ── WHISPERX ─────────────────────────────────────────────────
WHISPER_MODEL      = "base"

# ── GAP DETECTION ─────────────────────────────────────────────
GAP_MIN_SEC        = 0.4    # Ignore gaps shorter than this (too brief)
GAP_MAX_SEC        = 4.0    # Ignore gaps longer than this (probably silence/music)
PAD_SEC            = 0.15   # Padding added either side of each gap when saving

# ── SPEAKER VERIFICATION ──────────────────────────────────────
BMO_CONFIRMED      = 0.50   # Gap scores above this → confirmed BMO non-verbal
BMO_REVIEW         = 0.33   # Gap scores above this → needs review
SEED_MAX_FILES     = 9999
TOP_N_SEED         = 150

# ── AUDIOSET LABELS WE CARE ABOUT ────────────────────────────
# These are the exact class names in the AST AudioSet model
TARGET_LABELS = {
    "Laughter":  "laugh",
    "Crying, sobbing": "cry",
    "Whimper":   "cry",
    "Gasp":      "gasp",
    "Breathing": "gasp",
    "Humming":   "hum",
    "Singing":   "hum",
    "Music":     None,    # None = skip these entirely
    "Silence":   None,
}
# Minimum AST confidence to trust the label (otherwise → "unknown")
AST_MIN_CONFIDENCE = 0.10

TARGET_SR = 16000
# =================================================

# Output subdirs
DIR_LAUGH    = os.path.join(OUTPUT_DIR, "confirmed", "laugh")
DIR_CRY      = os.path.join(OUTPUT_DIR, "confirmed", "cry")
DIR_GASP     = os.path.join(OUTPUT_DIR, "confirmed", "gasp")
DIR_HUM      = os.path.join(OUTPUT_DIR, "confirmed", "hum")
DIR_UNKNOWN  = os.path.join(OUTPUT_DIR, "confirmed", "unknown")
DIR_REVIEW   = os.path.join(OUTPUT_DIR, "needs_review")
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "nonverbal_results.csv")

for d in [DIR_LAUGH, DIR_CRY, DIR_GASP, DIR_HUM, DIR_UNKNOWN, DIR_REVIEW]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  MODEL SETUP
# ─────────────────────────────────────────────────────────────

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = device.type
print(f"🖥️  Device: {device}\n")

print("🧠 Loading ECAPA-TDNN Speaker Model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=MODEL_CACHE,
    local_strategy=LocalStrategy.COPY
)
verification = verification.to(device)

print(f"🎙️  Loading WhisperX ({WHISPER_MODEL})...")
whisper_model = whisperx.load_model(
    WHISPER_MODEL,
    device=device_str,
    compute_type="float16" if device_str == "cuda" else "int8"
)

print("🎵 Loading AST AudioSet classifier...")
ast_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    cache_dir=HF_CACHE
)
ast_model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    cache_dir=HF_CACHE
).to(device).eval()

# Build label index → name map from AST model
ast_id2label = ast_model.config.id2label
print(f"   ✅ AST loaded — {len(ast_id2label)} AudioSet classes\n")

print("✅ All models ready\n")


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────

def load_mono_16k(path):
    try:
        signal, sr = torchaudio.load(str(path))
        if sr != TARGET_SR:
            signal = T.Resample(sr, TARGET_SR)(signal)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        return signal  # [1, N]
    except:
        return None


def get_embedding(signal):
    if signal is None or signal.shape[1] < TARGET_SR * 0.2:
        return None
    signal = signal.to(device)
    with torch.no_grad():
        feats = verification.mods.compute_features(signal)
        feats = verification.mods.mean_var_norm(feats, torch.ones(1).to(device))
        emb   = verification.mods.embedding_model(feats).squeeze().cpu()
    norm = emb.norm().clamp(min=1e-8)
    return emb / norm


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=1
    ).item()


# ─────────────────────────────────────────────────────────────
#  VOICEPRINT BUILDER
# ─────────────────────────────────────────────────────────────

def build_voiceprint(seed_dir):
    wavs = list(Path(seed_dir).glob("*.wav"))
    print(f"🔍 Building BMO voiceprint from {len(wavs)} seed clips...")
    embeddings = []
    for i, path in enumerate(wavs):
        emb = get_embedding(load_mono_16k(path))
        if emb is not None:
            embeddings.append(emb)
        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{len(wavs)}", end="\r")

    stacked    = torch.stack(embeddings)
    rough_mean = stacked.mean(dim=0);  rough_mean /= rough_mean.norm()
    scores     = torch.nn.functional.cosine_similarity(
                     stacked, rough_mean.unsqueeze(0).expand_as(stacked), dim=1)
    top_idx    = scores.topk(min(TOP_N_SEED, len(embeddings))).indices
    master     = stacked[top_idx].median(dim=0).values;  master /= master.norm()
    print(f"\n   ✅ Voiceprint coherence: {scores[top_idx].mean().item():.3f}\n")
    return master


# ─────────────────────────────────────────────────────────────
#  STEP 1 — WHISPERX: GET WORD-LEVEL TIMESTAMPS
# ─────────────────────────────────────────────────────────────

def get_word_timestamps(episode_path):
    """
    Run WhisperX on the full episode WAV and return a list of
    (word, start_sec, end_sec) for every recognised word.
    """
    print(f"  🎙️  Transcribing {Path(episode_path).name}...")
    audio  = whisperx.load_audio(str(episode_path))
    result = whisper_model.transcribe(audio, language="en", batch_size=4)

    # Force alignment for word-level timestamps
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
        w = seg.get("word", "")
        if s is not None and e is not None:
            words.append((w, float(s), float(e)))

    print(f"     → {len(words)} words aligned")
    return words


# ─────────────────────────────────────────────────────────────
#  STEP 2 — INVERT: FIND GAPS BETWEEN WORDS
# ─────────────────────────────────────────────────────────────

def find_gaps(word_timestamps, total_duration):
    """
    Given word timestamps, return list of (gap_start, gap_end)
    for all silent/wordless periods within our target duration range.
    """
    if not word_timestamps:
        return []

    gaps = []

    # Gap before first word
    first_start = word_timestamps[0][1]
    if first_start >= GAP_MIN_SEC:
        gaps.append((0.0, first_start))

    # Gaps between words
    for i in range(len(word_timestamps) - 1):
        _, _, end_current  = word_timestamps[i]
        _, start_next, _   = word_timestamps[i + 1]
        gap = start_next - end_current
        if GAP_MIN_SEC <= gap <= GAP_MAX_SEC:
            gaps.append((end_current, start_next))

    # Gap after last word
    last_end = word_timestamps[-1][2]
    remaining = total_duration - last_end
    if GAP_MIN_SEC <= remaining <= GAP_MAX_SEC:
        gaps.append((last_end, total_duration))

    return gaps


# ─────────────────────────────────────────────────────────────
#  STEP 3 — AST AUDIOSET CLASSIFIER
# ─────────────────────────────────────────────────────────────

def classify_with_ast(signal_np):
    """
    Run the AST model on a mono float32 numpy array.
    Returns (label, confidence) where label is one of
    laugh/cry/gasp/hum/unknown, or None if it matches a skip class.
    """
    inputs = ast_extractor(
        signal_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = ast_model(**inputs).logits

    probs    = torch.sigmoid(logits).squeeze().cpu().numpy()
    top_idx  = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_name = ast_id2label.get(top_idx, "unknown")

    # Check against our target labels
    if top_name in TARGET_LABELS:
        mapped = TARGET_LABELS[top_name]
        if mapped is None:
            return None, top_prob, top_name  # skip class (music/silence)
        if top_prob >= AST_MIN_CONFIDENCE:
            return mapped, top_prob, top_name

    # Not in our target list — scan top-5 for anything relevant
    top5_idx = np.argsort(probs)[::-1][:5]
    for idx in top5_idx:
        name = ast_id2label.get(int(idx), "")
        prob = float(probs[idx])
        if name in TARGET_LABELS and TARGET_LABELS[name] is not None:
            if prob >= AST_MIN_CONFIDENCE:
                return TARGET_LABELS[name], prob, name

    return "unknown", top_prob, top_name


# ─────────────────────────────────────────────────────────────
#  MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────

def process_episode(ep_path, master_emb, ep_name):
    """Full pipeline for one episode WAV."""
    # Get duration and load the FULL episode into RAM once
    try:
        full_sig, sr_raw = torchaudio.load(str(ep_path))
        total_s = full_sig.shape[1] / sr_raw
        
        # Resample and mixdown the full episode upfront
        if sr_raw != TARGET_SR:
            full_sig = T.Resample(sr_raw, TARGET_SR)(full_sig)
        if full_sig.shape[0] > 1:
            full_sig = full_sig.mean(dim=0, keepdim=True)
            
    except Exception as e:
        print(f"  ⚠️  Cannot read {ep_name}: {e}")
        return []

    # Step 1: WhisperX word timestamps
    words = get_word_timestamps(str(ep_path))
    if not words:
        print(f"  ⚠️  No words found in {ep_name} — skipping")
        return []

    # Step 2: Find gaps
    gaps = find_gaps(words, total_s)
    print(f"  📐 {len(gaps)} wordless gaps found  ({GAP_MIN_SEC}s–{GAP_MAX_SEC}s)")

    if not gaps:
        return []

    results = []
    confirmed_count = 0
    review_count    = 0

    for gap_start, gap_end in gaps:
        gap_dur = gap_end - gap_start

        # Load with padding
        load_start = max(0.0, gap_start - PAD_SEC)
        load_dur   = (gap_end - gap_start) + PAD_SEC * 2

        # Step 2.5: INSTANT TENSOR SLICING (No disk I/O)
        start_frame = int(load_start * TARGET_SR)
        end_frame   = int((load_start + load_dur) * TARGET_SR)
        
        # Safely bound the slice to the actual audio length
        end_frame = min(end_frame, full_sig.shape[1])
        
        sig = full_sig[:, start_frame:end_frame]
        
        # If the slice is empty for some reason, skip
        if sig.shape[1] == 0:
            continue

        sig_np = sig.squeeze().numpy()

        # Skip near-silence
        if np.sqrt(np.mean(sig_np ** 2)) < 0.003:
            continue

        # Step 3: Speaker verification — is this BMO?
        emb = get_embedding(sig)
        if emb is None:
            continue
        spk_score = cosine_sim(master_emb, emb)

        if spk_score < BMO_REVIEW:
            continue  # Not BMO

        # Step 4: AST classification
        label, ast_conf, ast_raw = classify_with_ast(sig_np)

        if label is None:
            continue  # Music/silence — skip

        # Step 5: Route to output folder
        confirmed = spk_score >= BMO_CONFIRMED
        if confirmed:
            out_dir = {
                "laugh":   DIR_LAUGH,
                "cry":     DIR_CRY,
                "gasp":    DIR_GASP,
                "hum":     DIR_HUM,
                "unknown": DIR_UNKNOWN,
            }.get(label, DIR_UNKNOWN)
        else:
            out_dir = DIR_REVIEW

        out_name = f"{ep_name}_{label}_{gap_start:.2f}.wav"
        out_path = os.path.join(out_dir, out_name)
        sf.write(out_path, sig_np, TARGET_SR)

        status = "✅" if confirmed else "🔍"
        emoji  = {"laugh":"😂","cry":"😢","gasp":"😱","hum":"🎵"}.get(label, "❓")
        print(f"    {status} {emoji} [{label.upper():7s}] "
              f"spk={spk_score:.3f}  ast={ast_conf:.2f} ({ast_raw[:20]})  "
              f"{gap_start:.2f}s→{gap_end:.2f}s  dur={gap_dur:.2f}s")

        results.append({
            "episode":       ep_name,
            "output_file":   out_name,
            "label":         label,
            "ast_raw_label": ast_raw,
            "gap_start":     round(gap_start, 3),
            "gap_end":       round(gap_end, 3),
            "duration":      round(gap_dur, 3),
            "speaker_score": round(spk_score, 4),
            "ast_confidence":round(ast_conf, 4),
            "confirmed":     confirmed,
        })

        if confirmed: confirmed_count += 1
        else:         review_count    += 1

    print(f"  → {confirmed_count} confirmed  |  {review_count} review")
    return results


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    master_emb = build_voiceprint(BMO_SEED_DIR)

    episode_wavs = sorted(Path(CLEANED_VOCALS_DIR).glob("*.wav"))
    print(f"📁 Found {len(episode_wavs)} episode WAVs\n")

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

        results          = process_episode(ep_path, master_emb, ep_name)
        all_results.extend(results)

        ep_confirmed     = sum(1 for r in results if r["confirmed"])
        ep_review        = sum(1 for r in results if not r["confirmed"])
        total_confirmed += ep_confirmed
        total_review    += ep_review

        done_episodes.add(ep_name)
        pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"🎉 Done!")
    print(f"  ✅ Confirmed : {total_confirmed}")
    print(f"     😂 laugh  → {DIR_LAUGH}")
    print(f"     😢 cry    → {DIR_CRY}")
    print(f"     😱 gasp   → {DIR_GASP}")
    print(f"     🎵 hum    → {DIR_HUM}")
    print(f"  🔍 Review    : {total_review}  → {DIR_REVIEW}")
    print(f"  💾 CSV       : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()