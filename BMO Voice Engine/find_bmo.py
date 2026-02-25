import os
import shutil
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy # Add this import at the top!

# --- CONFIGURATION ---
CLEAN_BMO_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\wavs"
UNLABELED_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\PlanB_Dataset"
OUTPUT_CSV      = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\bmo_ai_candidates.csv"

# Clips above HIGH_THRESHOLD go straight to auto-confirmed folder
# Clips between LOW and HIGH go to review folder
HIGH_THRESHOLD  = 0.72
LOW_THRESHOLD   = 0.50

# Output folders (will be created automatically)
CONFIRMED_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\auto_confirmed"
REVIEW_DIR      = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\needs_review"

# How many top BMO clips to use for the master voiceprint
# Using fewer but higher-quality clips is better than averaging everything
TOP_N_FOR_MASTER = 100

# --- SETUP ---
os.makedirs(CONFIRMED_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

print("🧠 Loading ECAPA-TDNN Speaker Model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir",
    # This force-kills the symlink requirement
    local_strategy=LocalStrategy.COPY 
)

TARGET_SR = 16000

def load_and_preprocess(filepath):
    """Load audio, resample to 16kHz, convert to mono."""
    try:
        signal, fs = torchaudio.load(filepath)
        if fs != TARGET_SR:
            signal = torchaudio.transforms.Resample(fs, TARGET_SR)(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        # Skip very short clips (< 0.5s) — they produce unreliable embeddings
        if signal.shape[1] < TARGET_SR * 0.5:
            return None
        return signal
    except Exception as e:
        print(f"  ⚠️  Could not load {Path(filepath).name}: {e}")
        return None

def get_embedding(filepath):
    """Returns a normalized embedding vector, or None on failure."""
    signal = load_and_preprocess(filepath)
    if signal is None:
        return None
    with torch.no_grad():
        emb = verification.encode_batch(signal)
    # Normalize so cosine sim is stable
    emb = emb.squeeze()
    return emb / emb.norm()

def build_master_voiceprint(clean_dir, top_n=TOP_N_FOR_MASTER):
    """
    Build a robust master voiceprint using the median of embeddings
    rather than a simple mean — more resistant to outlier clips.
    """
    wav_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
    print(f"🔍 Building master voiceprint from {len(wav_files)} clean BMO clips...")
    
    embeddings = []
    for filename in wav_files:
        path = os.path.join(clean_dir, filename)
        emb = get_embedding(path)
        if emb is not None:
            embeddings.append(emb)
    
    if not embeddings:
        raise RuntimeError("❌ No valid embeddings from clean BMO directory!")
    
    stacked = torch.stack(embeddings)  # (N, D)
    
    # Step 1: Get initial mean estimate
    initial_mean = stacked.mean(dim=0)
    initial_mean = initial_mean / initial_mean.norm()
    
    # Step 2: Score every clip against the initial mean
    scores = torch.nn.functional.cosine_similarity(
        stacked, initial_mean.unsqueeze(0).expand_as(stacked), dim=1
    )
    
    # Step 3: Keep only the top N most representative clips
    top_indices = scores.topk(min(top_n, len(embeddings))).indices
    top_embeddings = stacked[top_indices]
    
    # Step 4: Final master is median of top clips (more robust than mean)
    master = top_embeddings.median(dim=0).values
    master = master / master.norm()
    
    print(f"  ✅ Master voiceprint built from top {len(top_indices)} clips")
    print(f"  📊 Internal coherence score (mean similarity): {scores[top_indices].mean():.3f}")
    return master

def main():
    # Build the voiceprint
    master_emb = build_master_voiceprint(CLEAN_BMO_DIR)
    
    # Get the set of filenames already in the clean folder (to skip)
    already_have = set(os.listdir(CLEAN_BMO_DIR))
    
    # Load previously processed files from CSV if it exists
    already_processed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            prev = pd.read_csv(OUTPUT_CSV)
            already_processed = set(prev["filename"].tolist())
            print(f"📋 Skipping {len(already_processed)} previously processed files")
        except Exception:
            pass
    
    print(f"\n🕵️  Scanning: {UNLABELED_DIR}")
    
    results = []
    count = 0
    skipped = 0
    
    wav_files = []
    for root, _, files in os.walk(UNLABELED_DIR):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(wav_files)} .wav files to process\n")
    
    for filepath in wav_files:
        filename = Path(filepath).name
        
        # Skip if already in clean folder or already processed
        if filename in already_have or filename in already_processed:
            skipped += 1
            continue
        
        test_emb = get_embedding(filepath)
        
        if test_emb is not None:
            score = torch.nn.functional.cosine_similarity(
                master_emb.unsqueeze(0),
                test_emb.unsqueeze(0),
                dim=1
            ).item()
            
            row = {"filename": filename, "path": filepath, "score": round(score, 4)}
            results.append(row)
            
            if score >= HIGH_THRESHOLD:
                shutil.copy2(filepath, os.path.join(CONFIRMED_DIR, filename))
                print(f"  ✅ AUTO  [{score:.3f}] {filename}")
            elif score >= LOW_THRESHOLD:
                shutil.copy2(filepath, os.path.join(REVIEW_DIR, filename))
                print(f"  🔍 REVIEW [{score:.3f}] {filename}")
        
        count += 1
        if count % 200 == 0:
            print(f"  ... processed {count}/{len(wav_files)} files")
            # Save progress incrementally so you don't lose work if it crashes
            _save_csv(results, OUTPUT_CSV)
    
    # Final save
    _save_csv(results, OUTPUT_CSV)
    
    confirmed = sum(1 for r in results if r["score"] >= HIGH_THRESHOLD)
    review    = sum(1 for r in results if LOW_THRESHOLD <= r["score"] < HIGH_THRESHOLD)
    
    print(f"\n{'='*50}")
    print(f"🎉 Done! Processed {count} files, skipped {skipped}")
    print(f"  ✅ Auto-confirmed : {confirmed}  →  {CONFIRMED_DIR}")
    print(f"  🔍 Needs review   : {review}   →  {REVIEW_DIR}")
    print(f"  💾 Full scores    : {OUTPUT_CSV}")

def _save_csv(results, path):
    if results:
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        df.to_csv(path, index=False)

if __name__ == "__main__":
    main()