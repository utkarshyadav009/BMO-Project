import os
import time
import struct
import pickle
from collections import deque, Counter

import zmq
import numpy as np
import soundfile as sf
import python_speech_features as psf
from scipy.signal import resample_poly

TARGET_RATE = 48000
N_MFCC = 12
ZMQ_PORT = 5555
PROFILE_FILE = "bmo_brain.pkl"

CHUNK_MS = 20
SILENCE_RMS = 0.006
STABILITY_FRAMES = 2

def load_and_prep_audio(filepath, target_rate=TARGET_RATE):
    audio, rate = sf.read(filepath, always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # float32 [-1,1]
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = np.clip(audio, -1.0, 1.0)

    if rate != target_rate:
        audio = resample_poly(audio, target_rate, rate).astype(np.float32)
        rate = target_rate

    return np.ascontiguousarray(audio, dtype=np.float32), rate

class SmartVisemeEngine:
    def __init__(self):
        self.brain = {}
        self.ready = False
        self.history = deque(maxlen=STABILITY_FRAMES)
        self.last = "mouth_phoneme_X"

        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE, "rb") as f:
                self.brain = pickle.load(f)
            self.ready = True
            # quick sanity print
            nz = [(k, v.get("count", 0)) for k, v in self.brain.items() if v.get("count", 0) > 0]
            nz.sort(key=lambda x: x[1], reverse=True)
            print("Brain nonzero counts:", nz)
        else:
            print("❌ No brain found. Run TrainBrain.py first.")

    def classify(self, chunk: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        if rms < SILENCE_RMS:
            self.last = "mouth_phoneme_X"
            self.history.clear()
            return self.last

        if not self.ready:
            return "mouth_phoneme_B"

        mfccs = psf.mfcc(
            chunk,
            samplerate=TARGET_RATE,
            winlen=0.025,
            winstep=0.010,
            numcep=N_MFCC,
            nfilt=26,
            nfft=2048,
        )
        if mfccs.size == 0:
            return self.last

        feat = np.mean(mfccs, axis=0).astype(np.float32)

        best = self.last
        best_dist = float("inf")

        for viseme, data in self.brain.items():
            if data.get("count", 0) == 0:
                continue
            diff = feat - data["mean"]
            dist = float(np.sum((diff * diff) / data["var"]))  # diagonal Mahalanobis
            if dist < best_dist:
                best_dist = dist
                best = viseme

        self.history.append(best)
        winner, count = Counter(self.history).most_common(1)[0]
        if count >= STABILITY_FRAMES:
            self.last = winner

        return self.last

def stream_file(socket, engine, filepath):
    audio, rate = load_and_prep_audio(filepath)
    chunk_size = int(rate * (CHUNK_MS / 1000.0))

    t0 = time.perf_counter()
    sent = 0

    for i in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunk = audio[i:i+chunk_size]
        viseme = engine.classify(chunk)

        vb = viseme.encode("utf-8")
        header = struct.pack("<I", len(vb))
        socket.send(header + vb + chunk.tobytes())

        # stable pacing
        sent += chunk_size
        target = t0 + sent / rate
        while True:
            dt = target - time.perf_counter()
            if dt <= 0:
                break
            time.sleep(min(dt, 0.001))

if __name__ == "__main__":
    engine = SmartVisemeEngine()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.bind(f"tcp://*:{ZMQ_PORT}")

    path = "calibration_audio/dialogue.wav"
    files = [path] if os.path.exists(path) else [
        "calibration_audio/A.wav",
        "calibration_audio/E.wav",
        "calibration_audio/O.wav",
    ]

    print("🎙️ STARTING STREAM...")
    try:
        while True:
            for f in files:
                if os.path.exists(f):
                    stream_file(sock, engine, f)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Done.")
