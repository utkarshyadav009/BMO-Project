import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np

# ================= CONFIGURATION =================
# 1. Folder containing the long, Demucs-cleaned WAV files
INPUT_FOLDER = r"D:\Steam\BMO Episodes\Cleaned_Vocals"

# 2. Output folder for the final training clips
OUTPUT_FOLDER = r"D:\Steam\BMO Episodes\Final_Dataset"

# 3. A single, perfect 10-second clip of BMO talking (The "Gold Standard")
REFERENCE_FILE = r"C:\ProjectBMO\reference_bmo.wav" 

# 4. Sensitivity (0.0 to 1.0). Higher = Stricter (Only perfect BMO matches)
# Start at 0.75. If you get too few clips, lower to 0.65.
SIMILARITY_THRESHOLD = 0.75 

# 5. Length of clips (in milliseconds)
MIN_CLIP_LEN = 2000  # 2 seconds
MAX_CLIP_LEN = 12000 # 12 seconds
# =================================================

def load_encoder():
    print("Loading Speaker Recognition Model...")
    # This downloads a pre-trained speaker encoder
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    return classifier

def get_embedding(classifier, wav_file):
    """Computes the 'voice fingerprint' of a wav file."""
    signal, fs = torchaudio.load(wav_file)
    
    # Resample if needed (Model expects 16k)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(fs, 16000)
        signal = resampler(signal)
    
    # Compute embedding
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, dim=2)
        return embeddings.squeeze()

def slicer_pipeline():
    # 1. Setup
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    # 2. Load the Judge (AI Model)
    classifier = load_encoder()
    
    # 3. Learn what BMO sounds like
    if not os.path.exists(REFERENCE_FILE):
        print(f"ERROR: You must create a reference file at {REFERENCE_FILE}")
        return
        
    print(f"Computing BMO's voice fingerprint from {REFERENCE_FILE}...")
    bmo_fingerprint = get_embedding(classifier, REFERENCE_FILE)

    # 4. Process all cleaned episodes
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav")]
    print(f"Found {len(files)} cleaned episodes to slice.")

    total_clips_saved = 0

    for file in files:
        file_path = os.path.join(INPUT_FOLDER, file)
        print(f"Processing: {file}...")
        
        # Load audio (pydub for easy slicing)
        audio = AudioSegment.from_wav(file_path)
        
        # A. Split on Silence (VAD)
        # This chops the 11-minute file into sentences based on pauses
        chunks = split_on_silence(
            audio, 
            min_silence_len=500, # 0.5 seconds of silence marks a break
            silence_thresh=-40,  # dB cutoff for silence
            keep_silence=200     # Keep a tiny bit of silence for naturalness
        )

        for i, chunk in enumerate(chunks):
            # Check length constraints
            if len(chunk) < MIN_CLIP_LEN or len(chunk) > MAX_CLIP_LEN:
                continue
                
            # Export temp file for AI analysis
            temp_name = "temp_check.wav"
            chunk.export(temp_name, format="wav")
            
            try:
                # B. Check Identity (Is this BMO?)
                clip_embedding = get_embedding(classifier, temp_name)
                
                # Calculate Similarity Score (Cosine Similarity)
                score = torch.dot(clip_embedding, bmo_fingerprint).item()
                
                if score > SIMILARITY_THRESHOLD:
                    # It's a match! Save it.
                    final_name = f"{file.replace('.wav','')}_clip_{i:04d}.wav"
                    chunk.export(os.path.join(OUTPUT_FOLDER, final_name), format="wav")
                    total_clips_saved += 1
                    print(f"  -> Saved Clip {i} (Conf: {score:.2f})")
                else:
                    pass # It's probably Finn or Jake
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")

    print("="*30)
    print(f"COMPLETE. Saved {total_clips_saved} confirmed BMO clips.")
    if os.path.exists("temp_check.wav"):
        os.remove("temp_check.wav")

if __name__ == "__main__":
    slicer_pipeline()