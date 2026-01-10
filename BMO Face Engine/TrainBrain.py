import os
import csv
import pickle
import numpy as np
import soundfile as sf
import python_speech_features as psf
from scipy.signal import resample_poly

# --- CONFIG ---
DIALOGUE_WAV = "calibration_audio/dialogue.wav"
DIALOGUE_TSV = "calibration_audio/dialogue.tsv"
OUTPUT_BRAIN = "bmo_brain.pkl"
TARGET_RATE = 48000
N_MFCC = 12

# Your sprite names (A-H) match the Rhubarb standard (A-H) perfectly.
RHUBARB_MAP = {

    "A": "mouth_phoneme_A",  # Closed (MBP)
    "B": "mouth_phoneme_B",  # Consonants/Slightly Open (K, S, T)
    "C": "mouth_phoneme_C",  # Wide (Ee)
    "D": "mouth_phoneme_D",  # Open (Ah)
    "E": "mouth_phoneme_E",  # Round (Oh)
    "F": "mouth_phoneme_F",  # Pucker (W/Q)
    "G": "mouth_phoneme_G",  # Tuck (F/V)
    "H": "mouth_phoneme_H",  # Tongue Up (L)    
    "X": "mouth_phoneme_X",  # Silence
}

# 2. MANUAL FILE MAPPING (For your specific voice)
# This teaches the brain YOUR version of these sounds
MANUAL_FILES = {
    "A.wav": "mouth_phoneme_D",  # Your "Ah"
    "E.wav": "mouth_phoneme_C",  # Your "Ee"
    "O.wav": "mouth_phoneme_E",  # Your "Oh"
    "B.wav": "mouth_phoneme_B",  # Your "Mm/B"
    "F.wav": "mouth_phoneme_G",  # Your "Ff"
}

def load_audio(path):
    print(f"Loading {path}...")
    try:
        audio, rate = sf.read(path, always_2d=False)
    except: return None, 0

    if audio.ndim > 1: audio = audio[:, 0]
    
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0: audio = np.clip(audio, -1.0, 1.0)

    if rate != TARGET_RATE:
        print(f"Resampling {rate} -> {TARGET_RATE}...")
        audio = resample_poly(audio, TARGET_RATE, rate).astype(np.float32)
        rate = TARGET_RATE
    return np.ascontiguousarray(audio, dtype=np.float32), rate

def parse_rhubarb_tsv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [r for r in reader if len(r) > 0]

    segments = []
    try: is_start_end = len(rows) > 0 and len(rows[0]) >= 3 and float(rows[0][1])
    except: is_start_end = False

    if is_start_end:
        for r in rows:
            if len(r) >= 3: segments.append((float(r[0]), float(r[1]), r[2]))
    else:
        times = []
        for r in rows:
            try: times.append((float(r[0]), r[1]))
            except: continue
        for i in range(len(times)-1):
            segments.append((times[i][0], times[i+1][0], times[i][1]))
    return segments

def train():
    buckets = {v: [] for v in set(RHUBARB_MAP.values())}
    
    # --- PHASE 1: TRAIN ON DIALOGUE (General) ---
    if os.path.exists(DIALOGUE_WAV) and os.path.exists(DIALOGUE_TSV):
        print("--- Phase 1: Rhubarb Dialogue ---")
        audio, rate = load_audio(DIALOGUE_WAV)
        segments = parse_rhubarb_tsv(DIALOGUE_TSV)
        min_samples = int(0.03 * rate)

        for start, end, label in segments:
            viseme = RHUBARB_MAP.get(label.strip(), "mouth_phoneme_X")
            s = max(0, int(start * rate))
            e = min(len(audio), int(end * rate))
            chunk = audio[s:e]
            
            if len(chunk) < min_samples: continue
            mfccs = psf.mfcc(chunk, samplerate=rate, winlen=0.025, winstep=0.010, numcep=N_MFCC, nfilt=26, nfft=2048)
            if mfccs.size > 0: buckets[viseme].append(mfccs.astype(np.float32))

    # --- PHASE 2: TRAIN ON YOUR VOICE (Specific) ---
    print("\n--- Phase 2: Manual Calibration Files ---")
    for filename, viseme in MANUAL_FILES.items():
        path = os.path.join("calibration_audio", filename)
        if not os.path.exists(path):
            print(f"⚠️ Skipping {filename} (Not found)")
            continue
            
        audio, rate = load_audio(path)
        if audio is None: continue
        
        # Treat the WHOLE file as one sample of that viseme
        mfccs = psf.mfcc(audio, samplerate=rate, winlen=0.025, winstep=0.010, numcep=N_MFCC, nfilt=26, nfft=2048)
        if mfccs.size > 0:
            print(f"✅ Added {filename} to {viseme} ({len(mfccs)} frames)")
            buckets[viseme].append(mfccs.astype(np.float32))

    # --- COMPILE BRAIN ---
    brain = {}
    print("\n--- Final Brain Stats ---")
    for viseme, arrs in buckets.items():
        if not arrs:
            print(f"⚠️  {viseme}: No data")
            brain[viseme] = {"mean": np.zeros(N_MFCC), "var": np.ones(N_MFCC), "count": 0}
            continue

        X = np.vstack(arrs)
        mean = np.mean(X, axis=0).astype(np.float32)
        var = (np.var(X, axis=0).astype(np.float32) + 1e-5) 
        
        brain[viseme] = {"mean": mean, "var": var, "count": int(X.shape[0])}
        print(f"🧠 {viseme}: {X.shape[0]} total frames")

    with open(OUTPUT_BRAIN, "wb") as f:
        pickle.dump(brain, f)
    print(f"\n💾 Saved Hybrid Brain to {OUTPUT_BRAIN}")

if __name__ == "__main__":
    train()