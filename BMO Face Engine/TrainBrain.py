import os
import csv
import pickle
import numpy as np
import soundfile as sf
import python_speech_features as psf
from scipy.signal import resample_poly

# --- CONFIG ---
WAV_FILE = "calibration_audio/dialogue.wav"
TSV_FILE = "calibration_audio/dialogue.tsv"
OUTPUT_BRAIN = "bmo_brain.pkl"

TARGET_RATE = 48000
N_MFCC = 12

# Rhubarb -> your sprite names
# --- MAPPING ---
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

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def load_audio(path: str):
    print(f"Loading {path}...")
    audio, rate = sf.read(path, always_2d=False)

    if audio.ndim > 1:
        audio = audio[:, 0]

    # Ensure float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

    # Resample correctly
    if rate != TARGET_RATE:
        print(f"Resampling {rate} -> {TARGET_RATE}...")
        audio = resample_poly(audio, TARGET_RATE, rate).astype(np.float32)
        rate = TARGET_RATE

    return np.ascontiguousarray(audio, dtype=np.float32), rate

def parse_rhubarb_tsv(path: str):
    """
    Supports:
      1) time<TAB>label
      2) start<TAB>end<TAB>label
    Skips headers/comments/non-numeric rows.
    Returns: list[(start,end,label)]
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for r in reader:
            if not r:
                continue
            r = [c.strip() for c in r]
            rows.append(r)

    # Find first numeric row to detect schema
    first_num = None
    for r in rows:
        if len(r) >= 2 and is_float(r[0]):
            first_num = r
            break
    if first_num is None:
        raise ValueError("TSV has no numeric timestamp rows")

    start_end_format = (len(first_num) >= 3 and is_float(first_num[1]))

    segments = []
    if start_end_format:
        # start, end, label
        for r in rows:
            if len(r) < 3:
                continue
            if not (is_float(r[0]) and is_float(r[1])):
                continue
            segments.append((float(r[0]), float(r[1]), r[2]))
    else:
        # time, label -> infer end from next time
        times = []
        for r in rows:
            if len(r) < 2 or not is_float(r[0]):
                continue
            times.append((float(r[0]), r[1]))
        for i in range(len(times) - 1):
            start_t, label = times[i]
            end_t = times[i + 1][0]
            segments.append((start_t, end_t, label))

    return segments

def train():
    if not (os.path.exists(WAV_FILE) and os.path.exists(TSV_FILE)):
        print("❌ Missing dialogue.wav or dialogue.tsv")
        return

    audio, rate = load_audio(WAV_FILE)
    segments = parse_rhubarb_tsv(TSV_FILE)

    print(f"Audio: {len(audio)/rate:.2f}s @ {rate} Hz")
    print(f"Segments: {len(segments)}")

    # Collect MFCC frames per target viseme
    buckets = {v: [] for v in set(RHUBARB_MAP.values())}

    min_samples = int(0.03 * rate)  # ~30ms

    for start, end, label in segments:
        vis = RHUBARB_MAP.get(label.strip(), "mouth_phoneme_X")

        s = max(0, int(start * rate))
        e = min(len(audio), int(end * rate))
        if e <= s:
            continue

        chunk = audio[s:e]
        if len(chunk) < min_samples:
            continue

        mfccs = psf.mfcc(
            chunk,
            samplerate=rate,
            winlen=0.025,
            winstep=0.010,
            numcep=N_MFCC,
            nfilt=26,
            nfft=2048,
        )
        if mfccs.size == 0:
            continue

        # IMPORTANT: no per-chunk mean subtraction here
        buckets[vis].append(mfccs.astype(np.float32))

    brain = {}
    print("\n--- Training Results ---")
    for viseme, arrs in buckets.items():
        if not arrs:
            print(f"⚠️  {viseme}: 0 frames")
            brain[viseme] = {"mean": np.zeros(N_MFCC, np.float32),
                             "var": np.ones(N_MFCC, np.float32),
                             "count": 0}
            continue

        X = np.vstack(arrs)
        mean = np.mean(X, axis=0).astype(np.float32)
        var = (np.var(X, axis=0).astype(np.float32) + 1e-5)

        brain[viseme] = {"mean": mean, "var": var, "count": int(X.shape[0])}
        print(f"✅ {viseme}: {X.shape[0]} frames")

    with open(OUTPUT_BRAIN, "wb") as f:
        pickle.dump(brain, f)

    print(f"\n🧠 Saved brain to {OUTPUT_BRAIN}")

if __name__ == "__main__":
    train()
