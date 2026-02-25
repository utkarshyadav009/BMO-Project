import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

# Directories
EPISODES_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes"  # your raw episodes
OUTPUT_LAUGH   = r"D:\LocalWorkDir\2509362\BMO Episodes\Nonverbal\laugh_candidates"
OUTPUT_CRY     = r"D:\LocalWorkDir\2509362\BMO Episodes\Nonverbal\cry_candidates"
OUTPUT_GENERAL = r"D:\LocalWorkDir\2509362\BMO Episodes\Nonverbal\general_candidates"

os.makedirs(OUTPUT_LAUGH, exist_ok=True)
os.makedirs(OUTPUT_CRY, exist_ok=True)
os.makedirs(OUTPUT_GENERAL, exist_ok=True)

TARGET_SR    = 16000
FRAME_HOP    = 512
FRAME_LEN    = 2048

def extract_features(y, sr):
    """Extract acoustic features useful for detecting non-verbal sounds."""
    rms      = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=FRAME_HOP)[0]
    zcr      = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=FRAME_HOP)[0]
    spectral = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=FRAME_HOP)[0]
    
    # Pitch detection — relevant for cry (sustained high pitch)
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=150, fmax=1000, sr=sr,
                                      frame_length=FRAME_LEN, hop_length=FRAME_HOP)
    except:
        f0 = np.zeros(len(rms))
        voiced = np.zeros(len(rms), dtype=bool)
    
    return rms, zcr, spectral, f0, voiced

def score_segment(rms, zcr, spectral, f0, voiced):
    """
    Returns (laugh_score, cry_score) for a segment.
    
    Laughter signature:
      - Rhythmic energy bursts (high variance in RMS)
      - High zero crossing rate (breathy, aspirated)
      - Mid-high spectral centroid
      - Short voiced bursts with gaps
    
    Crying signature:
      - Sustained high pitch (F0 > 300Hz for BMO's voice)
      - Tremolo (pitch variance)
      - Lower ZCR than laughter
      - Long voiced segments
    """
    energy_var    = np.var(rms)
    mean_zcr      = np.mean(zcr)
    mean_spectral = np.mean(spectral)
    
    valid_f0      = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
    mean_f0       = np.mean(valid_f0) if len(valid_f0) > 0 else 0
    f0_var        = np.var(valid_f0)  if len(valid_f0) > 0 else 0
    voiced_ratio  = np.mean(voiced)   if voiced is not None else 0
    
    # Laugh score — rhythmic bursts, high ZCR, mid energy variance
    laugh_score = (
        min(energy_var * 5000, 1.0) * 0.35 +
        min(mean_zcr * 5, 1.0)      * 0.35 +
        min(mean_spectral / 3000, 1.0) * 0.30
    )
    
    # Cry score — high sustained pitch, pitch tremolo, long voiced segments
    cry_score = (
        min(mean_f0 / 500, 1.0)      * 0.40 +
        min(f0_var / 2000, 1.0)      * 0.30 +
        min(voiced_ratio * 2, 1.0)   * 0.30
    )
    
    return laugh_score, cry_score

def scan_episode(episode_path, window_sec=2.0, hop_sec=0.5,
                 laugh_threshold=0.45, cry_threshold=0.40):
    """
    Slide a window across the episode and score each segment.
    Returns list of candidate segments with timestamps and scores.
    """
    try:
        y, sr = librosa.load(episode_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  ⚠️  Could not load {Path(episode_path).name}: {e}")
        return []
    
    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec * sr)
    candidates     = []
    
    for start in range(0, len(y) - window_samples, hop_samples):
        segment = y[start:start + window_samples]
        
        # Skip near-silence
        if np.sqrt(np.mean(segment**2)) < 0.005:
            continue
        
        rms, zcr, spec, f0, voiced = extract_features(segment, sr)
        laugh_score, cry_score = score_segment(rms, zcr, spec, f0, voiced)
        
        if laugh_score >= laugh_threshold or cry_score >= cry_threshold:
            candidates.append({
                "start_sec":   start / sr,
                "end_sec":     (start + window_samples) / sr,
                "laugh_score": round(laugh_score, 3),
                "cry_score":   round(cry_score, 3),
            })
    
    return candidates

def merge_overlapping(candidates, gap_sec=0.3):
    """Merge candidate windows that are close together into single segments."""
    if not candidates:
        return []
    
    merged = [candidates[0].copy()]
    for c in candidates[1:]:
        if c["start_sec"] - merged[-1]["end_sec"] <= gap_sec:
            merged[-1]["end_sec"]     = max(merged[-1]["end_sec"], c["end_sec"])
            merged[-1]["laugh_score"] = max(merged[-1]["laugh_score"], c["laugh_score"])
            merged[-1]["cry_score"]   = max(merged[-1]["cry_score"],   c["cry_score"])
        else:
            merged.append(c.copy())
    return merged

def save_segment(y, sr, start_sec, end_sec, output_path,
                 pad_sec=0.2):
    """Save a segment with a little padding on each side."""
    start_sample = max(0, int((start_sec - pad_sec) * sr))
    end_sample   = min(len(y), int((end_sec + pad_sec) * sr))
    segment      = y[start_sample:end_sample]
    sf.write(str(output_path), segment, sr)

def main():
    # Find all episode wav files
    episode_files = list(Path(EPISODES_DIR).rglob("*.wav"))
    # Filter out files already in your processed dataset folders
    episode_files = [f for f in episode_files
                     if "Final_Dataset" not in str(f)
                     and "PlanB_Dataset" not in str(f)]
    
    print(f"🎬 Found {len(episode_files)} episode audio files")
    print(f"🔍 Scanning for laughing, crying, non-verbal sounds...\n")
    
    laugh_count = 0
    cry_count   = 0
    
    for ep_path in episode_files:
        ep_name = ep_path.stem
        print(f"  📺 {ep_name}")
        
        y, sr = librosa.load(str(ep_path), sr=TARGET_SR, mono=True)
        candidates = scan_episode(str(ep_path))
        merged     = merge_overlapping(candidates)
        
        for seg in merged:
            is_laugh = seg["laugh_score"] >= 0.45
            is_cry   = seg["cry_score"]   >= 0.40
            
            # Label by dominant type
            if is_laugh and seg["laugh_score"] >= seg["cry_score"]:
                label   = "laugh"
                out_dir = OUTPUT_LAUGH
                laugh_count += 1
            elif is_cry:
                label   = "cry"
                out_dir = OUTPUT_CRY
                cry_count += 1
            else:
                label   = "general"
                out_dir = OUTPUT_GENERAL
            
            timestamp = f"{int(seg['start_sec'])}"
            filename  = f"{ep_name}_{label}_{timestamp}.wav"
            out_path  = Path(out_dir) / filename
            
            save_segment(y, sr, seg["start_sec"], seg["end_sec"], out_path)
            print(f"    {'😂' if label == 'laugh' else '😢'} "
                  f"[L:{seg['laugh_score']:.2f} C:{seg['cry_score']:.2f}] "
                  f"{seg['start_sec']:.1f}s → {seg['end_sec']:.1f}s")
    
    print(f"\n{'='*50}")
    print(f"✅ Done!")
    print(f"  😂 Laugh candidates : {laugh_count}  →  {OUTPUT_LAUGH}")
    print(f"  😢 Cry candidates   : {cry_count}   →  {OUTPUT_CRY}")

if __name__ == "__main__":
    main()