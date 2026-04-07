# BMO Project — Embodied AI Companion

> *"BMO, we need you to be our friend forever."*

BMO is an attempt to build a genuinely **alive** AI companion — not a chatbot in a box, but an agent with internal drives, temporal awareness, and a persistent identity that deepens over time. It runs entirely locally on an **NVIDIA Jetson Orin Nano 8GB**, inspired by the Adventure Time character of the same name.

This repo contains two primary subsystems built in parallel:

| Subsystem | Language | What it does |
|---|---|---|
| **BMO Face Engine** | C++ / GLSL / Raylib | Renders BMO's expressive procedural face in real-time |
| **BMO Voice Engine** | Python | Builds and processes the voice dataset for TTS/STS fine-tuning |

---

## Table of Contents

- [Project Philosophy](#project-philosophy)
- [Hardware](#hardware)
- [Cognitive Architecture](#cognitive-architecture)
- [BMO Face Engine](#bmo-face-engine)
- [BMO Voice Engine](#bmo-voice-engine)
- [Speech Architecture](#speech-architecture)
- [Memory System](#memory-system)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)

---

## Project Philosophy

Most "AI companions" are passive — they wait for input and return output. BMO is different. The goal is **synthetic aliveness**, defined by three criteria:

1. **Temporal Grounding** — BMO perceives time passing during silence, enabling natural turn-taking and the pressure of a conversational lull.
2. **Active Agency** — BMO has internal homeostatic drives (boredom, social hunger, energy) and will *initiate* interaction when those drives cross a threshold.
3. **Identity Consolidation** — BMO evolves. Memories from past conversations are consolidated during idle periods and shape future personality, so BMO genuinely *knows* you over time.

This is achieved not by prompting harder, but by adopting a **hybrid System 1 / System 2 cognitive architecture** grounded in **Active Inference**.

---

## Hardware

BMO targets the **NVIDIA Jetson Orin Nano 8GB** in **25W "Super Mode"**.

| Component | Choice | Notes |
|---|---|---|
| **Compute** | Jetson Orin Nano 8GB | Ampere GPU, 1024 CUDA cores |
| **OS** | JetPack 6.2 (headless) | Ubuntu 22.04 base, no desktop environment |
| **Storage** | NVMe M.2 SSD (256GB+) | microSD is too slow for model weight swapping |
| **Display** | 4–5" HDMI Capacitive Touch LCD | Waveshare IPS recommended |
| **Adapter** | Active DisplayPort → HDMI | The Orin Nano has **no DSI** — DP only |
| **Camera** | IMX219 (CSI) or USB webcam | CSI saves USB bandwidth |
| **Audio** | USB Conference Speakerphone | Handles echo cancellation (AEC) natively |
| **Touch** | Adafruit MPR121 + copper foil | Wired to I2C GPIO pins 3/5 |
| **Enclosure** | 3D printed PETG | PLA will deform from heat — use PETG or ABS |
| **Cooling** | 40mm 5V PWM fan (Noctua NF-A4x10) | Active chimney airflow required in 25W mode |

### Memory Budget (25W Mode)

| Component | Allocation |
|---|---|
| OS (headless) | ~1.5 GB |
| LLM (Gemma 2 2B, INT4 AWQ) | ~1.8 GB |
| Depth Transformer (FP16) | ~0.6 GB |
| Mimi Audio Codec | ~0.15 GB |
| KV Cache (PagedAttention INT8) | ~1.2 GB |
| Vision / Face Engine | ~0.3 GB |
| OS / IPC overhead | ~2.0 GB |
| Safety headroom | ~0.45 GB |

---

## Cognitive Architecture

BMO uses a **dual-loop** architecture inspired by Kahneman's System 1 / System 2 framework, grounded in **Active Inference** (Karl Friston's Free Energy Principle).

### System 1 — The Presence Engine
- Runs on CPU at 10–50ms cycle time
- Handles: Voice Activity Detection (Silero VAD), syllable-rate detection, coupled oscillator for turn-taking rhythm, affect recognition (Valence/Arousal), backchannel reflexes ("mhm", "yeah")
- Always running — gives BMO a sense of *being present* even when not speaking

### System 2 — The Narrative Engine
- Runs on GPU (event-driven) at 1–5s cycle time
- Handles: semantic understanding, LLM inference, Active Inference POMDP policy selection, memory retrieval, narrative construction
- Only wakes when System 1 detects something requiring high-level reasoning

### The Thalamus (System Interface)
- If the user interrupts, System 1 sends a `HALT` signal to abort GPU generation immediately
- If a high-arousal event is detected (shout, loud noise), System 1 forces a System 2 wake-up
- Predictive cooling: when a complex question is detected, fans ramp to 100% *before* inference starts

### Active Inference & Personality
BMO's personality is not a system prompt. It is encoded as a **C-Matrix** of prior preferences in a POMDP generative model (implemented with `pymdp`). For example:

```python
C_affect[2] = +4.0  # Strong preference for observing user happiness
C_affect[0] = -5.0  # Strong aversion to observing user anger
```

This creates a constant "gravitational pull" toward behaviours that satisfy BMO's preferences, making personality robust even as conversations drift.

---

## BMO Face Engine

Located in `BMO Face Engine/`. Built with **C++17**, **Raylib 5.5**, and custom **GLSL fragment shaders**. Renders directly to the framebuffer via DRM (no desktop environment required).

### Architecture

```
BMO Face Engine/
├── src/
│   ├── SFacePoserFinal.cpp       # Main editor / driver
│   ├── ShaderParametricFace.cpp  # Unified face system (eyes + mouth)
│   ├── ShaderParametricEyes.cpp  # Eye shader engine + physics
│   ├── ShaderParametricMouth.cpp # Mouth shader engine + geometry
│   ├── FaceData.h                # EyeParams / MouthParams / FaceState structs
│   ├── AffectiveEngine.h         # Manifold-based mood → face solver
│   ├── eyes_es.fs                # Eye GLSL shader (12 shape types)
│   ├── brow_es.fs                # Eyebrow GLSL shader (Bezier curve)
│   ├── mouth_es.fs               # Mouth GLSL shader (SDF polygon)
│   ├── tears_es.fs               # Tears & blush GLSL shader
│   └── pixelizer_es.fs           # Optional pixel-art post-process shader
└── CMakeLists.txt
```

### Key Features

**Parametric Face System**
- All facial features are mathematically generated — no pre-baked animations
- Eyes support 13 shape types: dot, line, arc, cross, star, heart, spiral, chevron, shuriken, kawaii, shocked, teary, and colon-eyes (`::`)
- Mouths are generated from a 16-point Catmull-Rom spline with independent controls for: openness, width, curve, squeeze (top/bottom), asymmetry, squareness, teeth gap, tongue position, and stress lines

**Affective Engine**
- Maps a 5-dimensional **AppraisalVector** `(valence, arousal, control, novelty, obstruct)` to a `FaceState` via nearest-neighbour lookup over a hand-crafted database
- Spring-damper physics ensure organic transitions between expressions
- "Frankenstein Guard" prevents blending incompatible face shapes (e.g. star eyes + spiral eyes)

**Shader Features**
- All shapes rendered as **Signed Distance Functions (SDFs)** for resolution-independent, perfectly smooth anti-aliasing at any scale
- Hot-reload: shaders are watched on disk and recompiled live without restarting
- Pixelizer post-process shader for optional retro BMO aesthetic

**Tools**
- `SFacePoserFinal.cpp` — Full GUI editor to sculpt and save facial presets to `face_database.txt`
- Face presets are saved as plain-text tuples, hot-reloadable at runtime with `KEY_R`
- Debug mode with exploded-view layer inspection

### Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Raylib and RayGui are fetched automatically via CMake FetchContent.

### Face Database Format

```
faces["face_happy_standard"] = { 
    <eye_params...>, 
    <mouth_params...> 
};
```

---

## BMO Voice Engine

Located in `BMO Voice Engine/`. A Python pipeline for constructing a high-quality speech dataset from Adventure Time episodes, and curating it for voice model fine-tuning.

### Pipeline Overview

```
Raw Episodes (.mkv/.mp4)
        ↓
[1] bmo_episodes.py          # Copy BMO episodes from library
        ↓
[2] extract_bmo_vocals.py    # Demucs / BS-Roformer vocal separation
        ↓
[3] transcribe_bmo.py        # WhisperX transcription + speaker diarization
        ↓
[4] find_bmo.py              # ECAPA-TDNN speaker verification (filter for BMO's voice)
        ↓
[5] label_bmo_emote.py       # Streamlit labeling tool — correct transcripts + assign tone labels
        ↓
[6] srt_crosscheck.py        # Validate labels against official closed captions
        ↓  
[Non-verbal branch]
[7] ccNonVerbal.py           # Extract non-verbal cues (laughs, cries, gasps) from CC timestamps
[8] process_nv_bmo.py        # CLAP embeddings + AST AudioSet classification
[9] bmoNonVerbalRefiner.py   # Refine with seed-based CLAP similarity scoring
```

### Tone Labeling

Each clip is labelled with a **tone** that maps directly to an `AppraisalVector` used by the Affective Engine at inference time:

| Tone | Valence | Arousal | Control | Novelty | Obstruct |
|---|---|---|---|---|---|
| happy | +0.8 | 0.4 | 0.8 | 0.0 | 0.0 |
| excited | +1.0 | 0.9 | 0.7 | 0.8 | 0.0 |
| sad | −0.6 | 0.2 | 0.2 | 0.0 | 0.5 |
| angry | −0.7 | 0.8 | 0.9 | 0.0 | 0.9 |
| scared | −0.5 | 0.8 | 0.1 | 0.9 | 0.3 |
| *...and 7 more* | | | | | |

**Metadata format:** `filename|transcript|tone`

### Non-Verbal Categories

The pipeline extracts and classifies: `laugh`, `cry`, `gasp`, `scream`, `grunt`, `sigh`, `hum`. These are used to train deterministic non-verbal triggers (e.g. BMO humming while processing).

### Key Dependencies

```
whisperx, speechbrain, demucs, audio-separator
transformers (CLAP, AST), torchaudio, soundfile
streamlit (labeling UI), pandas
```

---

## Speech Architecture

BMO's speech system is a heavily modified **Moshi**-style architecture adapted for the Jetson's 8GB memory envelope.

### Core Components

| Component | Implementation | Notes |
|---|---|---|
| **Temporal Transformer** | Gemma 2 2B (INT4 AWQ) | Replaces the original 7B Helium backbone |
| **Depth Transformer** | ~300M params (FP16) | Models inter-codebook acoustic dependencies |
| **Audio Codec** | Mimi (Split-RVQ, 12.5Hz) | Semantic token (CB0) + 7 acoustic codebooks |
| **Memory Manager** | TensorRT-LLM PagedAttention | C++ runtime, eliminates Python GIL overhead |

### Voice Identity: SFT over Zero-Shot

BMO's voice is locked via **Supervised Fine-Tuning** on the curated dataset (30–50 minutes), not zero-shot cloning. This means:

- Zero inference overhead at runtime (identity baked into LoRA weights)
- No KV cache consumed by voice prompts
- Consistent identity across emotional extremes, not just neutral baseline
- LoRA adapters merged via **TIES-Merging** to combine acoustic identity + semantic personality

### Emotional Steering

Post-SFT, **PersonaPlex** is repurposed as a dynamic emotional anchor system. Combined with **Activation Steering** (injecting directional vectors into the model's residual stream), this enables:

- Frame-by-frame prosody modulation (subtle valence shifts)
- Discontinuous state transitions (e.g. sudden laughter or crying)

### Soliloquy Prevention

A custom `LogitsProcessor` in TensorRT-LLM monitors the text-to-audio token ratio. If the Inner Monologue races ahead without acoustic output, a penalty collapses text token probability and forces audio generation — preventing the model from getting lost in its own thoughts.

---

## Memory System

BMO uses a tiered **MaRS** (Memory and Retrieval System) architecture:

| Layer | Storage | Lifetime |
|---|---|---|
| **Sensory Buffer** | Circular buffer (RAM) | Milliseconds |
| **Working Memory** | LLM context window (4096 tokens) | Current session |
| **Episodic Memory** | LanceDB (disk-based vector store) | Days / Years |
| **Semantic Memory** | Knowledge graph | Permanent |

### Sleep Consolidation

When idle for >15 minutes, BMO runs a consolidation cycle:
1. Clusters recent conversation embeddings by topic (agglomerative clustering)
2. Wakes System 2 to generate a "Gist" summary per cluster
3. Tags each Gist with the average affective state during that conversation
4. Embeds and stores in LanceDB
5. Discards the raw transcript — only meaning is kept

This mirrors biological memory consolidation and ensures BMO's long-term memory grows in meaning, not just volume.

---

## Getting Started

### Prerequisites

- CMake 3.11+
- C++17 compiler
- Python 3.10+
- CUDA-capable GPU (for voice pipeline; Jetson for deployment)

### Face Engine (Development, any platform)

```bash
cd "BMO Face Engine"
mkdir build && cd build
cmake ..
cmake --build .
# Assets must be in build/assets/
```

### Voice Pipeline

```bash
cd "BMO Voice Engine"
pip install whisperx speechbrain demucs audio-separator \
            transformers torchaudio soundfile streamlit pandas

# 1. Extract and separate vocals
python bmo_episodes.py
python extract_bmo_vocals.py

# 2. Transcribe and identify BMO's voice
python transcribe_bmo.py
python find_bmo.py

# 3. Label with the Streamlit UI
streamlit run label_bmo_emote.py

# 4. Cross-check against SRT captions
python srt_crosscheck.py
```

### Jetson Deployment

```bash
# Enable 25W Super Mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Disable desktop environment
sudo systemctl set-default multi-user.target
```

---

## Roadmap

- [x] Parametric face engine (shader-based SDF rendering)
- [x] Affective engine (AppraisalVector → FaceState manifold)
- [x] Voice dataset pipeline (separation, transcription, speaker verification)
- [x] Tone labeling tool with AppraisalVector mapping
- [x] Non-verbal sound extraction and classification
- [ ] Active Inference controller (pymdp integration)
- [ ] LanceDB episodic memory + sleep consolidation
- [ ] Moshi-based STS pipeline on Jetson
- [ ] SFT voice fine-tuning (LoRA + TIES merge)
- [ ] ZeroMQ IPC bridge (Face Engine ↔ Cognitive Core)
- [ ] MPR121 touch sensor integration
- [ ] Full Jetson enclosure (PETG, active cooling)
- [ ] "The Aliveness Tests" (Interruption, Silence Pressure, Inside Joke)

---

## Project Structure

```
BMO-Project/
├── BMO Face Engine/
│   ├── src/                  # C++ source + GLSL shaders
│   ├── assets/               # Sprite atlases, audio
│   ├── CMakeLists.txt
│   └── face_database.txt     # Hand-crafted facial expression presets
├── BMO Voice Engine/
│   ├── bmo_episodes.py       # Episode collection
│   ├── extract_bmo_vocals.py # Vocal separation
│   ├── transcribe_bmo.py     # Transcription + diarization
│   ├── find_bmo.py           # Speaker verification
│   ├── label_bmo_emote.py    # Streamlit labeling UI
│   ├── srt_crosscheck.py     # Caption cross-validation
│   ├── ccNonVerbal.py        # Non-verbal extraction
│   ├── process_nv_bmo.py     # CLAP/AST classification
│   └── bmoNonVerbalRefiner.py
└── README.md
```

---

*Built with equal parts computer science and Adventure Time lore.*
