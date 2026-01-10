import os
import time
import struct
import pickle
import zmq
import numpy as np
import sounddevice as sd
import python_speech_features as psf
from collections import deque, Counter

# --- CONFIG ---
TARGET_RATE = 48000
N_MFCC = 12
ZMQ_PORT = 5555
PROFILE_FILE = "bmo_brain.pkl"
LOG_FILENAME = "viseme_debug_log.txt"

# --- THE FEEDBACK KILLER ---
MUTE_AUDIO_OUTPUT = True  # Set to True to stop the speakers (Fixes feedback loop)

# Microphone Settings
DEVICE_INDEX = None 
BLOCK_MS = 20        

# Tuning
SILENCE_RMS = 0.005     
STABILITY_FRAMES = 2
SMOOTHING_ALPHA = 0.5   # Moderate smoothing

with open(LOG_FILENAME, "w", encoding="utf-8") as f:
    f.write("--- BMO LIVE MIC DEBUG LOG (MUTED) ---\n")

def log(msg):
    print(msg)
    try:
        with open(LOG_FILENAME, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except: pass

class LiveVisemeEngine:
    def __init__(self):
        self.brain = {}
        self.ready = False
        self.history = deque(maxlen=STABILITY_FRAMES)
        self.last = "mouth_phoneme_X"
        self.debug_last = {}
        self.avg_features = None  

        if os.path.exists(PROFILE_FILE):
            with open(PROFILE_FILE, "rb") as f:
                self.brain = pickle.load(f)
            self.ready = True
            log("🧠 Hybrid Brain Loaded.")
        else:
            log("❌ No brain found. Run TrainBrain.py first.")

    def classify(self, chunk):
        # 1. Silence Gate
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        
        if rms < SILENCE_RMS:
            self.last = "mouth_phoneme_X"
            self.history.clear()
            self.avg_features = None 
            self.debug_last = {"rms": rms, "best": "X", "stable": "X", "top4": []}
            return self.last

        if not self.ready: return "mouth_phoneme_B"

        # 2. Extract Features
        mfccs = psf.mfcc(chunk, samplerate=TARGET_RATE, winlen=0.025, winstep=0.010,
                         numcep=N_MFCC, nfilt=26, nfft=2048)

        if mfccs.size == 0: return self.last

        # --- FEATURE SMOOTHING ---
        raw_feat = np.mean(mfccs, axis=0).astype(np.float32)
        
        if self.avg_features is None:
            self.avg_features = raw_feat
        else:
            self.avg_features = (self.avg_features * SMOOTHING_ALPHA) + (raw_feat * (1.0 - SMOOTHING_ALPHA))
            
        feat = self.avg_features

        # 3. Mahalanobis Distance + LOUDNESS BIAS
        candidates = []
        
        # Punish these mouths if loud to prevent closing early
        closed_mouths = ['mouth_phoneme_B', 'mouth_phoneme_G', 'mouth_phoneme_H', 'mouth_phoneme_X']
        is_loud = rms > 0.04 

        for viseme, data in self.brain.items():
            if data.get('count', 0) == 0: continue
            
            diff = feat - data['mean']
            dist = float(np.sum((diff * diff) / data['var']))
            
            # Bias Logic
            if is_loud and viseme in closed_mouths:
                dist *= 1.5  # 50% penalty for closing mouth while loud
            
            candidates.append((viseme, dist))

        if not candidates: return self.last

        candidates.sort(key=lambda x: x[1])
        top4 = candidates[:4]
        best = top4[0][0]

        # 4. Stability
        self.history.append(best)
        winner, count = Counter(self.history).most_common(1)[0]
        if count >= STABILITY_FRAMES:
            self.last = winner

        self.debug_last = {
            "rms": rms, "best": best, "top4": top4, "stable": self.last
        }
        return self.last

# --- GLOBAL VARS FOR CALLBACK ---
engine = LiveVisemeEngine()
ctx = zmq.Context()
socket = ctx.socket(zmq.PUSH)
socket.bind(f"tcp://*:{ZMQ_PORT}")
packet_id = 0

def audio_callback(indata, frames, time_info, status):
    global packet_id
    if status: print(status)
    
    # 1. Get Mic Audio
    chunk = indata[:, 0]
    
    # 2. Classify (Using Real Audio)
    viseme = engine.classify(chunk)
    
    # 3. Prepare Audio Payload
    if MUTE_AUDIO_OUTPUT:
        # SEND SILENCE to Speakers (Prevents Feedback Loop)
        audio_payload = np.zeros_like(chunk)
    else:
        # Send Real Audio (Causes Feedback if speakers are on)
        audio_payload = chunk

    # 4. Send Packet
    vb = viseme.encode("utf-8")
    header = struct.pack("<I", len(vb))
    socket.send(header + vb + audio_payload.tobytes())

    # 5. Logging
    packet_id += 1
    if packet_id % 25 == 0:
        d = engine.debug_last
        top_str = ", ".join([f"{n[-1]}:{d:.1f}" for n, d in d.get('top4', [])])
        log(f"[MIC] rms={d.get('rms',0):.4f} raw={d.get('best')[-1]} stable={d.get('stable')[-1]} tops=[{top_str}]")

if __name__ == "__main__":
    log("\n🎙️  LIVE MICROPHONE (MUTED OUTPUT) ACTIVE...")
    block_size = int(TARGET_RATE * (BLOCK_MS / 1000.0))
    
    try:
        with sd.InputStream(device=DEVICE_INDEX, channels=1, callback=audio_callback,
                            blocksize=block_size, samplerate=TARGET_RATE, dtype='float32'):
            while True:
                sd.sleep(1000) 
    except KeyboardInterrupt:
        log("Done.")
    except Exception as e:
        log(f"❌ Mic Error: {e}")