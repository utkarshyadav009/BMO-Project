import os
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"

import shutil
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# --- CONFIGURATION ---
CLEAN_BMO_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\wavs"
UNLABELED_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\PlanB_Dataset"
OUTPUT_CSV      = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\bmo_pass2_candidates.csv"
CONFIRMED_DIR   = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\auto_confirmed"
REVIEW_DIR      = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset\needs_review_pass2"

# Thresholds — can tune these based on pass 1 results
HIGH_THRESHOLD  = 0.65
LOW_THRESHOLD   = 0.45

TOP_N_FOR_MASTER = 200  # Higher than pass 1 since we have more good clips now
BATCH_SIZE       = 16   # 3070 can handle this, increase to 64 if no OOM

os.makedirs(CONFIRMED_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)

TARGET_SR = 16000

print("🧠 Loading ECAPA-TDNN Speaker Model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=r"D:\LocalWorkDir\2509362\BMO-Project\BMO Voice Engine\tmpdir_local",
    # This is the magic line that kills the Symlink error:
    local_strategy=LocalStrategy.COPY
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
verification = verification.to(device)  # ← KEY FIX


def load_and_preprocess(filepath):
    try:
        signal, fs = torchaudio.load(filepath)
        if fs != TARGET_SR:
            signal = torchaudio.transforms.Resample(fs, TARGET_SR)(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if signal.shape[1] < TARGET_SR * 0.5:
            return None
        return signal
    except Exception as e:
        print(f"  ⚠️  Could not load {Path(filepath).name}: {e}")
        return None

def get_embedding_single(filepath):
    """Single file embedding used for voiceprint building."""
    signal = load_and_preprocess(filepath)
    if signal is None:
        return None
    
    # Force the signal to the correct device immediately
    signal = signal.to(device)
    
    with torch.no_grad():
        # We manually call the embedding model to bypass the "lengths" error
        # and ensure device consistency
        feats = verification.mods.compute_features(signal)
        feats = verification.mods.mean_var_norm(feats, torch.ones(1).to(device))
        emb = verification.mods.embedding_model(feats)
    
    emb = emb.squeeze().cpu()
    norm = emb.norm().clamp(min=1e-8)
    return emb / norm

def get_embeddings_batch(filepaths):
    """Process files one at a time to ensure device and argument stability."""
    results = []
    for filepath in filepaths:
        signal = load_and_preprocess(filepath)
        if signal is None:
            continue
        
        signal = signal.to(device)
        
        with torch.no_grad():
            # Manual forward pass to avoid the "lengths" TypeError
            feats = verification.mods.compute_features(signal)
            feats = verification.mods.mean_var_norm(feats, torch.ones(1).to(device))
            emb = verification.mods.embedding_model(feats)
            
        emb = emb.squeeze().cpu()
        norm = emb.norm().clamp(min=1e-8)
        results.append((filepath, emb / norm))
    return results

def build_master_voiceprint(seed_dirs, top_n=TOP_N_FOR_MASTER):
    """Build voiceprint from multiple source directories combined."""
    all_wavs = []
    for d in seed_dirs:
        if os.path.exists(d):
            found = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav")]
            all_wavs.extend(found)
            print(f"  📂 {d}: {len(found)} clips")

    print(f"\n🔍 Building master voiceprint from {len(all_wavs)} total clips...")

    embeddings = []
    for i, path in enumerate(all_wavs):
        emb = get_embedding_single(path)
        if emb is not None:
            embeddings.append(emb)
        if (i + 1) % 100 == 0:
            print(f"  ... voiceprint progress {i + 1}/{len(all_wavs)}", end="\r")

    print()

    if not embeddings:
        raise RuntimeError("❌ No valid embeddings found! Check your seed directories.")

    stacked = torch.stack(embeddings)

    initial_mean = stacked.mean(dim=0)
    initial_mean = initial_mean / initial_mean.norm()

    scores = torch.nn.functional.cosine_similarity(
        stacked,
        initial_mean.unsqueeze(0).expand_as(stacked),
        dim=1
    )

    top_k = min(top_n, len(embeddings))
    top_indices = scores.topk(top_k).indices
    top_embeddings = stacked[top_indices]

    master = top_embeddings.median(dim=0).values
    master = master / master.norm()

    coherence = scores[top_indices].mean().item()
    print(f"  ✅ Voiceprint built from top {top_k} clips")
    print(f"  📊 Coherence score: {coherence:.3f}")

    if coherence < 0.65:
        print(f"  ⚠️  Low coherence — possible non-BMO clips in seed folders!")
    else:
        print(f"  💪 Strong voiceprint — ready for pass 2!")

    suggested_high = round(coherence * 0.90, 2)
    suggested_low  = round(coherence * 0.63, 2)
    print(f"\n  💡 Suggested thresholds based on coherence:")
    print(f"     HIGH: {suggested_high}  |  LOW: {suggested_low}")
    print(f"     Currently using HIGH: {HIGH_THRESHOLD}  |  LOW: {LOW_THRESHOLD}")

    return master

def _save_csv(results, path):
    if results:
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        df.to_csv(path, index=False)

def main():
    master_emb = build_master_voiceprint(
        seed_dirs=[CLEAN_BMO_DIR, CONFIRMED_DIR]
    )

    # Build skip set
    already_have = set()
    for d in [CLEAN_BMO_DIR, CONFIRMED_DIR]:
        if os.path.exists(d):
            already_have.update(os.listdir(d))

    already_processed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            prev = pd.read_csv(OUTPUT_CSV)
            if "filename" in prev.columns and len(prev) > 0:
                already_processed = set(prev["filename"].tolist())
                print(f"\n📋 Resuming — skipping {len(already_processed)} already scored files")
        except Exception:
            pass

    # Collect all wavs
    all_wavs = []
    for root, _, files in os.walk(UNLABELED_DIR):
        for file in files:
            if file.endswith(".wav"):
                all_wavs.append(os.path.join(root, file))

    # Filter to unprocessed only
    to_process = []
    skipped = 0
    for filepath in all_wavs:
        filename = Path(filepath).name
        if filename in already_have or filename in already_processed:
            skipped += 1
        else:
            to_process.append(filepath)

    print(f"\n📁 {len(all_wavs)} total wavs | {skipped} skipped | {len(to_process)} to process\n")

    results = []
    count = 0
    confirmed_count = 0
    review_count = 0

    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[batch_start : batch_start + BATCH_SIZE]
        batch_results = get_embeddings_batch(batch)

        for filepath, test_emb in batch_results:
            filename = Path(filepath).name

            score = torch.nn.functional.cosine_similarity(
                master_emb.unsqueeze(0),
                test_emb.unsqueeze(0),
                dim=1
            ).item()

            row = {"filename": filename, "path": filepath, "score": round(score, 4)}
            results.append(row)

            relative = Path(filepath).relative_to(UNLABELED_DIR).parent

            if score >= HIGH_THRESHOLD:
                shutil.copy2(filepath, os.path.join(CONFIRMED_DIR, filename))
                confirmed_count += 1
                print(f"  ✅ AUTO   [{score:.3f}] {relative} / {filename}")
            elif score >= LOW_THRESHOLD:
                shutil.copy2(filepath, os.path.join(REVIEW_DIR, filename))
                review_count += 1
                print(f"  🔍 REVIEW [{score:.3f}] {relative} / {filename}")

        count += len(batch)
        if count % 100 == 0:
            print(f"  ... {count}/{len(to_process)} | ✅ {confirmed_count} confirmed | 🔍 {review_count} review")
            _save_csv(results, OUTPUT_CSV)

    _save_csv(results, OUTPUT_CSV)

    print(f"\n{'='*55}")
    print(f"🎉 Pass 2 complete!")
    print(f"  Processed : {count}")
    print(f"  Skipped   : {skipped}")
    print(f"  ✅ Auto-confirmed : {confirmed_count}  →  {CONFIRMED_DIR}")
    print(f"  🔍 Needs review   : {review_count}   →  {REVIEW_DIR}")
    print(f"  💾 Scores saved   : {OUTPUT_CSV}")
    print(f"\n  🔁 Run again after reviewing needs_review_pass2 to do pass 3!")

if __name__ == "__main__":
    main()