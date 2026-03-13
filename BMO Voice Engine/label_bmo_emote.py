#!/usr/bin/env python3
"""
BMO Voice Factory — with Tone Labeling
Metadata format: filename|text|tone
Tone maps directly to AffectiveEngine AppraisalVector (valence, arousal, control, novelty, obstruct)
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import shutil
import re
import numpy as np
import wave
import json

# ================= CONFIGURATION =================
INPUT_DIR  = r"/home/bmo/BMO-LabelData/Final_Dataset/bmo_wavs"
OUTPUT_DIR = r"/home/bmo/BMO-LabelData/Final_Dataset/BMO_SpeechDataset"

METADATA_FILE  = os.path.join(INPUT_DIR, "final_bmo_metadata.csv")
FINAL_METADATA = os.path.join(OUTPUT_DIR, "metadata.csv")
TRASH_FILE     = os.path.join(OUTPUT_DIR, "trash.txt")
# =================================================

# ── COMPONENT ────────────────────────────────────────────────────────────────
_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav_trim_component")
_wav_trim_component = components.declare_component("wav_trim", path=_COMPONENT_DIR)

st.set_page_config(layout="wide", page_title="BMO Voice Factory", page_icon="🕹️")

# ── TONE LABEL DEFINITIONS ───────────────────────────────────────────────────
# Each tone maps to an AppraisalVector (valence, arousal, control, novelty, obstruct)
# These feed directly into AffectiveEngine::LoadFromDB equivalent at runtime
TONE_LABELS = {
    "happy":      {"v":  0.8, "a": 0.4, "c": 0.8, "n": 0.0, "o": 0.0, "emoji": "😊"},
    "excited":    {"v":  1.0, "a": 0.9, "c": 0.7, "n": 0.8, "o": 0.0, "emoji": "🤩"},
    "playful":    {"v":  0.6, "a": 0.6, "c": 0.7, "n": 0.3, "o": 0.0, "emoji": "😜"},
    "earnest":    {"v":  0.5, "a": 0.3, "c": 0.9, "n": 0.0, "o": 0.0, "emoji": "🫡"},
    "smug":       {"v":  0.3, "a": 0.3, "c": 1.0, "n": 0.0, "o": 0.0, "emoji": "😏"},
    "whispering": {"v":  0.1, "a": 0.1, "c": 0.6, "n": 0.1, "o": 0.0, "emoji": "🤫"},
    "neutral":    {"v":  0.0, "a": 0.2, "c": 0.5, "n": 0.0, "o": 0.0, "emoji": "😐"},
    "tired":      {"v": -0.2, "a": 0.0, "c": 0.3, "n": 0.0, "o": 0.2, "emoji": "😴"},
    "confused":   {"v": -0.1, "a": 0.4, "c": 0.2, "n": 0.7, "o": 0.3, "emoji": "😕"},
    "sad":        {"v": -0.6, "a": 0.2, "c": 0.2, "n": 0.0, "o": 0.5, "emoji": "😢"},
    "scared":     {"v": -0.5, "a": 0.8, "c": 0.1, "n": 0.9, "o": 0.3, "emoji": "😱"},
    "angry":      {"v": -0.7, "a": 0.8, "c": 0.9, "n": 0.0, "o": 0.9, "emoji": "😠"},
}

TONE_COLORS = {
    "happy":      "#ffec47",
    "excited":    "#ff8c00",
    "playful":    "#a8ff78",
    "earnest":    "#78d6ff",
    "smug":       "#c678ff",
    "whispering": "#b0c4de",
    "neutral":    "#d9d9d9",
    "tired":      "#9e9e9e",
    "confused":   "#ffa07a",
    "sad":        "#87ceeb",
    "scared":     "#dda0dd",
    "angry":      "#ff4444",
}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'history'         not in st.session_state: st.session_state.history = []
if 'active_file'     not in st.session_state: st.session_state.active_file = None
if 'show_wav_editor' not in st.session_state: st.session_state.show_wav_editor = False
if 'crop_applied'    not in st.session_state: st.session_state.crop_applied = False
if 'crop_start'      not in st.session_state: st.session_state.crop_start = 0.0
if 'crop_end'        not in st.session_state: st.session_state.crop_end = 0.0
if 'active_tone'     not in st.session_state: st.session_state.active_tone = "neutral"

# ── AUDIO BACKENDS ────────────────────────────────────────────────────────────
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def read_audio(path):
    if _HAS_SOUNDFILE:
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        return data, sr, data.shape[1]
    if _HAS_LIBROSA:
        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1: data = data[:, np.newaxis]
        else: data = data.T
        return data.astype(np.float32), sr, data.shape[1]
    with wave.open(path, 'rb') as wf:
        nc, sw, sr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(nf)
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    flat = np.frombuffer(raw, dtype=dtype_map.get(sw, np.int16)).astype(np.float32)
    flat /= float(2 ** (8 * sw - 1))
    return flat.reshape(-1, nc), sr, nc


def write_audio(path, data, samplerate, n_channels):
    if _HAS_SOUNDFILE:
        out = data if data.ndim == 2 else data[:, np.newaxis]
        sf.write(path, out, samplerate, subtype='PCM_16')
        return
    mono = data[:, 0] if data.ndim == 2 else data
    out = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(samplerate)
        wf.writeframes(out.tobytes())


def crop_wav_file(src_path, dst_path, start_sec, end_sec):
    data, sr, nc = read_audio(src_path)
    s = max(0, int(start_sec * sr))
    e = min(len(data), int(end_sec * sr))
    write_audio(dst_path, data[s:e], sr, nc)


def get_wav_duration(path):
    if _HAS_SOUNDFILE:
        try: return sf.info(path).duration
        except: pass
    if _HAS_LIBROSA:
        try: return librosa.get_duration(path=path)
        except: pass
    try:
        with wave.open(path, 'rb') as wf: return wf.getnframes() / wf.getframerate()
    except: pass
    return 0.0


def get_waveform_thumbnail(path, max_points=300):
    try:
        data, sr, nc = read_audio(path)
        mono = data.mean(axis=1) if data.ndim == 2 else data
        chunk = max(1, len(mono) // max_points)
        n = (len(mono) // chunk) * chunk
        arr = np.abs(mono[:n].reshape(-1, chunk)).max(axis=1)
        peak = arr.max()
        if peak > 0: arr /= peak
        return arr.tolist()
    except:
        return [0.0] * max_points


def get_temp_crop_path(filename):
    return os.path.join(OUTPUT_DIR, "wavs", f"__tmp_crop_{filename}")


# ── BACKEND HELPERS ───────────────────────────────────────────────────────────

@st.cache_resource
def index_files(root_dir):
    file_map = {}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".wav"):
                file_map[f] = os.path.join(root, f)
    return file_map


def init_environment():
    os.makedirs(os.path.join(OUTPUT_DIR, "wavs"), exist_ok=True)
    if not os.path.exists(FINAL_METADATA):
        with open(FINAL_METADATA, "w", encoding="utf-8") as f:
            f.write("filename|text|tone\n")
    else:
        # Migrate old 2-column metadata to 3-column if needed
        with open(FINAL_METADATA, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line == "filename|text":
            with open(FINAL_METADATA, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(FINAL_METADATA, "w", encoding="utf-8") as f:
                f.write("filename|text|tone\n")
                for line in lines[1:]:
                    f.write(line.rstrip("\n") + "|neutral\n")
    if not os.path.exists(TRASH_FILE):
        open(TRASH_FILE, "w", encoding="utf-8").close()


def get_processed_files():
    processed = set()
    if os.path.exists(FINAL_METADATA):
        try:
            saved_df = pd.read_csv(FINAL_METADATA, sep="|", on_bad_lines='skip', dtype=str)
            processed.update(saved_df['filename'].tolist())
        except: pass
    if os.path.exists(TRASH_FILE):
        with open(TRASH_FILE, "r", encoding="utf-8") as f:
            for line in f: processed.add(line.strip())
    return processed


def get_existing_tone(filename):
    """Look up already-saved tone for a file (for undo/re-edit)."""
    if os.path.exists(FINAL_METADATA):
        try:
            saved_df = pd.read_csv(FINAL_METADATA, sep="|", on_bad_lines='skip', dtype=str)
            row = saved_df[saved_df['filename'] == filename]
            if not row.empty and 'tone' in row.columns:
                tone = str(row.iloc[0]['tone']).strip()
                if tone in TONE_LABELS:
                    return tone
        except: pass
    return "neutral"


def get_episode_info(filename):
    match = re.search(r'(S\d+E\d+)', filename)
    return match.group(1) if match else "Unknown Ep"


def clear_temp_crop(filename):
    tmp = get_temp_crop_path(filename)
    if os.path.exists(tmp): os.remove(tmp)


# ── ACTION HANDLERS ───────────────────────────────────────────────────────────

def perform_save(filename, original_path, corrected_text, tone, action):
    st.session_state.history.append({'filename': filename, 'action': action, 'tone': tone})
    if len(st.session_state.history) > 20: st.session_state.history.pop(0)

    if action == 'save':
        dst = os.path.join(OUTPUT_DIR, "wavs", filename)
        tmp_crop = get_temp_crop_path(filename)
        src = tmp_crop if os.path.exists(tmp_crop) else original_path
        try:
            shutil.copy2(src, dst)
            clean_text = corrected_text.replace("\n", " ").replace("\r", "").replace("|", "").strip()
            safe_tone  = tone if tone in TONE_LABELS else "neutral"
            with open(FINAL_METADATA, "a", encoding="utf-8", newline='') as f:
                f.write(f"{filename}|{clean_text}|{safe_tone}\n")
            st.toast(f"💾 SAVED [{safe_tone}]", icon="✅")
        except Exception as ex:
            st.error(f"Save failed: {ex}"); return
        finally:
            clear_temp_crop(filename)
    elif action == 'trash':
        with open(TRASH_FILE, "a", encoding="utf-8") as f: f.write(f"{filename}\n")
        clear_temp_crop(filename)
        st.toast("🗑️ TRASHED", icon="🚮")

    st.session_state.active_file     = None
    st.session_state.show_wav_editor = False
    st.session_state.crop_applied    = False
    st.session_state.crop_start      = 0.0
    st.session_state.crop_end        = 0.0
    st.session_state.active_tone     = "neutral"
    if 'user_text' in st.session_state: del st.session_state['user_text']
    st.rerun()


def perform_undo():
    if not st.session_state.history:
        st.toast("Nothing to undo!", icon="⚠️"); return

    last_item = st.session_state.history.pop()
    filename, action = last_item['filename'], last_item['action']

    if action == 'save':
        wav_path = os.path.join(OUTPUT_DIR, "wavs", filename)
        if os.path.exists(wav_path): os.remove(wav_path)
        if os.path.exists(FINAL_METADATA):
            with open(FINAL_METADATA, "r", encoding="utf-8") as f: lines = f.readlines()
            with open(FINAL_METADATA, "w", encoding="utf-8") as f:
                [f.write(l) for l in lines if not l.startswith(f"{filename}|")]
        st.toast(f"⏪ Undid Save: {filename}", icon="↩️")
    elif action == 'trash':
        if os.path.exists(TRASH_FILE):
            with open(TRASH_FILE, "r", encoding="utf-8") as f: lines = f.readlines()
            with open(TRASH_FILE, "w", encoding="utf-8") as f:
                [f.write(l) for l in lines if l.strip() != filename]
        st.toast(f"⏪ Undid Trash: {filename}", icon="↩️")

    st.session_state.show_wav_editor = False
    st.session_state.crop_applied    = False
    st.session_state.crop_start      = 0.0
    st.session_state.crop_end        = 0.0
    st.session_state.active_file     = filename
    st.session_state.active_tone     = last_item.get('tone', 'neutral')
    if 'user_text' in st.session_state: del st.session_state['user_text']
    st.rerun()


def bulk_trash_folder(folder_path):
    files_to_trash = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    with open(TRASH_FILE, "a", encoding="utf-8") as f:
        for fname in files_to_trash: f.write(f"{fname}\n")
    st.session_state.active_file     = None
    st.session_state.show_wav_editor = False
    st.toast(f"💥 Trashed {len(files_to_trash)} files!", icon="🚮")
    st.rerun()


# ── TRANSCRIPT TAG BUTTONS ────────────────────────────────────────────────────

def render_tag_buttons(filename):
    st.markdown(
        "<p style='color:#0f382a;font-family:Courier New;font-size:12px;"
        "font-weight:bold;margin-bottom:4px;'>⚡ Quick Tags</p>",
        unsafe_allow_html=True
    )
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    with b1:
        if st.button("😂 [laugh]", key=f"tag_laugh_{filename}", use_container_width=True):
            current = st.session_state.get('user_text', '')
            if not current.startswith('[laugh]'):
                st.session_state.user_text = '[laugh] ' + current.lstrip()
            st.rerun()
    with b2:
        if st.button("😢 [cry]", key=f"tag_cry_{filename}", use_container_width=True):
            current = st.session_state.get('user_text', '')
            if not current.startswith('[cry]'):
                st.session_state.user_text = '[cry] ' + current.lstrip()
            st.rerun()
    with b3:
        if st.button("🎵 [sing]", key=f"tag_sing_{filename}", use_container_width=True):
            current = st.session_state.get('user_text', '')
            if not current.startswith('[sing]'):
                st.session_state.user_text = '[sing] ' + current.lstrip()
            st.rerun()
    with b4:
        if st.button("🔠 ALL CAPS", key=f"tag_caps_{filename}", use_container_width=True):
            st.session_state.user_text = st.session_state.get('user_text', '').upper()
            st.rerun()
    with b5:
        if st.button("🔡 lowercase", key=f"tag_lower_{filename}", use_container_width=True):
            st.session_state.user_text = st.session_state.get('user_text', '').lower()
            st.rerun()
    with b6:
        if st.button("🧹 Clear Tags", key=f"tag_clear_{filename}", use_container_width=True):
            current = st.session_state.get('user_text', '')
            cleaned = re.sub(r'^\s*\[(laugh|cry|sing)\]\s*', '', current, flags=re.IGNORECASE)
            st.session_state.user_text = cleaned
            st.rerun()


# ── TONE BUTTONS ──────────────────────────────────────────────────────────────

def render_tone_buttons(filename):
    current_tone = st.session_state.get('active_tone', 'neutral')
    tone_info    = TONE_LABELS.get(current_tone, TONE_LABELS['neutral'])
    tone_color   = TONE_COLORS.get(current_tone, '#d9d9d9')

    # Show current tone badge with its AppraisalVector
    st.markdown(
        f"<div style='background:{tone_color};color:#0f382a;border-radius:10px;"
        f"padding:6px 14px;font-family:Courier New;font-size:13px;font-weight:700;"
        f"margin-bottom:8px;display:inline-block;'>"
        f"{tone_info['emoji']} TONE: {current_tone.upper()}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"V:{tone_info['v']:+.1f} "
        f"A:{tone_info['a']:.1f} "
        f"C:{tone_info['c']:.1f} "
        f"N:{tone_info['n']:.1f} "
        f"O:{tone_info['o']:.1f}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#0f382a;font-family:Courier New;font-size:12px;"
        "font-weight:bold;margin-bottom:4px;'>🎭 Tone Label (→ AppraisalVector)</p>",
        unsafe_allow_html=True
    )

    # Row 1: positive tones
    cols1 = st.columns(6)
    positive_tones = ["happy", "excited", "playful", "earnest", "smug", "whispering"]
    for col, tone in zip(cols1, positive_tones):
        info  = TONE_LABELS[tone]
        color = TONE_COLORS[tone]
        is_active = (current_tone == tone)
        border = "4px solid #0f382a" if is_active else f"2px solid {color}"
        with col:
            if st.button(
                f"{info['emoji']} {tone}",
                key=f"tone_{tone}_{filename}",
                use_container_width=True
            ):
                st.session_state.active_tone = tone
                st.rerun()

    # Row 2: negative/neutral tones
    cols2 = st.columns(6)
    negative_tones = ["neutral", "tired", "confused", "sad", "scared", "angry"]
    for col, tone in zip(cols2, negative_tones):
        info  = TONE_LABELS[tone]
        color = TONE_COLORS[tone]
        is_active = (current_tone == tone)
        with col:
            if st.button(
                f"{info['emoji']} {tone}",
                key=f"tone_{tone}_{filename}",
                use_container_width=True
            ):
                st.session_state.active_tone = tone
                st.rerun()


# ── WAV EDITOR ────────────────────────────────────────────────────────────────

def render_wav_editor(filename, audio_path):
    duration = get_wav_duration(audio_path)
    if duration == 0:
        backend = "soundfile" if _HAS_SOUNDFILE else "librosa" if _HAS_LIBROSA else "stdlib wave"
        st.error(f"❌ Cannot read WAV (backend: {backend}).")
        return

    waveform_data = get_waveform_thumbnail(audio_path, max_points=300)
    tmp_crop      = get_temp_crop_path(filename)
    has_staged    = os.path.exists(tmp_crop)

    crop_start = st.session_state.get('crop_start', 0.0)
    crop_end   = st.session_state.get('crop_end', duration)
    if crop_end <= 0.0: crop_end = duration

    result = _wav_trim_component(
        waveform=waveform_data,
        duration=duration,
        initStart=crop_start,
        initEnd=crop_end,
        hasStaged=has_staged,
        key=f"wav_trim_{filename}",
        default=None,
    )

    if result is not None:
        st.session_state[f"fs_{filename}"] = round(float(result["start"]), 2)
        st.session_state[f"fe_{filename}"] = round(float(result["end"]),   2)

    st.markdown(
        "<p style='color:#aeddd1;font-family:Courier New;font-size:12px;margin:6px 0 2px;'>"
        "Fine-tune (seconds) ↓</p>", unsafe_allow_html=True
    )

    if f"fs_{filename}" not in st.session_state:
        st.session_state[f"fs_{filename}"] = float(round(crop_start, 2))
    if f"fe_{filename}" not in st.session_state:
        st.session_state[f"fe_{filename}"] = float(round(crop_end, 2))

    fi1, fi2, fi3, fi4 = st.columns([2, 2, 2, 1])
    with fi1:
        new_start = st.number_input(
            "Start", min_value=0.0, max_value=round(duration - 0.01, 2),
            step=0.01, format="%.2f",
            key=f"fs_{filename}", label_visibility="collapsed"
        )
    with fi2:
        new_end = st.number_input(
            "End", min_value=0.01, max_value=round(duration, 2),
            step=0.01, format="%.2f",
            key=f"fe_{filename}", label_visibility="collapsed"
        )
    with fi3:
        sel_dur = max(0.0, new_end - new_start)
        st.markdown(
            f"<div style='background:#1e5c45;color:#d9ffea;border-radius:9px;"
            f"padding:8px 8px;font-family:Courier New;font-size:12px;font-weight:700;"
            f"text-align:center;'>{new_start:.2f}→{new_end:.2f}s &nbsp;|&nbsp; {sel_dur:.2f}s</div>",
            unsafe_allow_html=True
        )
    with fi4:
        if st.button("✅ Apply Trim", key=f"do_trim_{filename}", use_container_width=True):
            cur_start = st.session_state.get(f"fs_{filename}", crop_start)
            cur_end   = st.session_state.get(f"fe_{filename}", crop_end)
            if cur_end > cur_start:
                try:
                    crop_wav_file(audio_path, tmp_crop, cur_start, cur_end)
                    st.session_state.crop_start   = cur_start
                    st.session_state.crop_end     = cur_end
                    st.session_state.crop_applied = True
                    st.toast(f"✂️ Trimmed to {cur_end - cur_start:.2f}s!", icon="✂️")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Trim failed: {ex}")

    if has_staged:
        st.markdown(
            "<div style='background:#f20553;color:#fff;border-radius:10px;"
            "padding:7px 14px;font-family:Courier New;font-size:13px;font-weight:700;"
            "margin-top:8px;text-align:center;'>"
            "✂️ TRIM STAGED — Preview ↓ &nbsp;·&nbsp; Hit 💾 SAVE AS BMO to commit</div>",
            unsafe_allow_html=True
        )
        st.audio(tmp_crop, autoplay=False)

    rc1, rc2 = st.columns([1, 1])
    with rc1:
        if has_staged:
            if st.button("🔄 Reset Trim", key=f"rst_{filename}", use_container_width=True):
                clear_temp_crop(filename)
                st.session_state.crop_applied = False
                st.session_state.crop_start   = 0.0
                st.session_state.crop_end     = duration
                st.toast("Trim reset.", icon="🔄")
                st.rerun()
    with rc2:
        if st.button("❌ Close Editor", key=f"cls_{filename}", use_container_width=True):
            st.session_state.show_wav_editor = False
            st.rerun()


# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #63bda4; }
.stTextArea textarea {
    background-color: #d9ffea; color: #0f382a;
    font-family: 'Courier New', monospace; font-weight: bold;
    font-size: 20px; border: 4px solid #0f382a; border-radius: 10px;
}
.bmo-box {
    background-color: #0f382a; color: #d9ffea; padding: 15px;
    border-radius: 10px; font-family: 'Courier New', monospace;
    margin-bottom: 20px; border: 2px solid #d9ffea;
}
.meta-tag {
    background-color: #ffec47; color: #5c5400; padding: 5px 12px;
    border-radius: 15px; font-weight: bold; border: 2px solid #bdae2a;
    display: inline-block; margin-right: 8px; margin-bottom: 8px;
    font-family: 'Courier New', monospace; font-size: 13px;
}
section[data-testid="stSidebar"] { background-color: #558c7a; }
div.stButton > button[kind="primary"] {
    background-color: #f20553; color: white; border: 3px solid #8a0330;
    border-radius: 50px; font-size: 22px; height: 3.2em;
}
div.stButton > button[kind="secondary"] {
    background-color: #ffec47; color: #5c5400; border: 3px solid #bdae2a;
    border-radius: 10px; font-weight: bold; font-size: 16px; height: 2.8em;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background-color: #1e5c45; color: #d9ffea; border: 2px solid #63bda4;
    border-radius: 20px; font-family: 'Courier New', monospace;
    font-size: 13px; font-weight: 800; padding: 4px 6px; height: 2.2em;
    transition: background-color 0.15s;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {
    background-color: #2d8c60; border-color: #ffec47;
}
@media (max-width: 640px) { .block-container { padding: 0.5rem !important; } }
</style>
""", unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    init_environment()
    file_map        = index_files(INPUT_DIR)
    processed_files = get_processed_files()

    try:
        df = pd.read_csv(METADATA_FILE, sep="|", header=None, names=["filename", "text"], dtype=str)
    except:
        st.error(f"Metadata missing! {METADATA_FILE}"); return

    total_files = len(df)
    done_count  = len(processed_files.intersection(set(df['filename'])))

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.image("https://static.wikia.nocookie.net/adventuretimewithfinnandjake/images/8/81/BMO.png", width=140)
        st.markdown(f"""
        <div class="bmo-box" style="text-align:center">
        <h3>📊 PROGRESS</h3><b>Done:</b> {done_count} / {total_files}
        </div>""", unsafe_allow_html=True)
        st.progress(min(done_count / total_files, 1.0) if total_files else 0)
        st.write("---")

        if st.session_state.history:
            if st.button("↩️ UNDO LAST", use_container_width=True): perform_undo()
        else:
            st.button("↩️ UNDO", disabled=True, use_container_width=True)

        # Tone distribution summary
        st.write("---")
        st.markdown("**🎭 Tone Distribution**")
        if os.path.exists(FINAL_METADATA):
            try:
                saved_df = pd.read_csv(FINAL_METADATA, sep="|", on_bad_lines='skip', dtype=str)
                if 'tone' in saved_df.columns:
                    tone_counts = saved_df['tone'].value_counts()
                    for tone, count in tone_counts.items():
                        if tone in TONE_LABELS:
                            emoji = TONE_LABELS[tone]['emoji']
                            st.markdown(f"`{emoji} {tone}`: {count}")
            except: pass

        st.title("📖 Rulebook")
        with st.expander("🎭 Tone Labels"):
            for tone, info in TONE_LABELS.items():
                st.markdown(
                    f"**{info['emoji']} {tone}** — "
                    f"V:{info['v']:+.1f} A:{info['a']:.1f} C:{info['c']:.1f} "
                    f"N:{info['n']:.1f} O:{info['o']:.1f}"
                )
        with st.expander("🏷️ Tags"):
            st.markdown(
                "* **Laughing:** `[laugh]` before transcript\n"
                "* **Crying:** `[cry]`\n"
                "* **Singing:** `[sing]`"
            )
        with st.expander("🗣️ Normal Speech"):
            st.markdown("* Punctuation: `. , ? !`\n* Numbers: `two` not `2`\n* `OK` not `okay`")
        with st.expander("😱 Screaming"):
            st.markdown("* ALL CAPS for screaming: `FINN NO!`")
        with st.expander("🚫 Trash"):
            st.markdown("* Other character talking at start\n* Dead air / SFX\n* Unintelligible")

    # ── TITLE ──────────────────────────────────────────────────
    st.title("🤖 BMO Voice Factory")

    # ── FILE SELECTION ─────────────────────────────────────────
    current_row = None

    if st.session_state.active_file:
        target_file = st.session_state.active_file
        row_match   = df[df['filename'] == target_file]
        fpath       = file_map.get(target_file)
        if not row_match.empty and fpath:
            current_row = row_match.iloc[0]
            current_row['path'] = fpath
        else:
            st.session_state.active_file = None

    if current_row is None:
        subset = df.sample(n=min(200, len(df)))
        for _, row in subset.iterrows():
            fname = row['filename']
            if fname not in processed_files:
                fpath = file_map.get(fname)
                if fpath:
                    current_row = row
                    current_row['path'] = fpath
                    st.session_state.active_file = fname
                    break

    if current_row is None:
        st.balloons()
        st.success("🎉 MISSION COMPLETE!")
        if st.button("Refresh"): st.rerun()
        return

    # ── RENDER ─────────────────────────────────────────────────
    filename   = current_row['filename']
    audio_path = current_row['path']
    episode    = get_episode_info(filename)
    folder     = os.path.basename(os.path.dirname(audio_path))
    tmp_crop   = get_temp_crop_path(filename)
    playback   = tmp_crop if os.path.exists(tmp_crop) else audio_path

    # Init text
    if 'user_text' not in st.session_state:
        raw = current_row['text']
        st.session_state.user_text = str(raw) if isinstance(raw, str) else ""

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.write("---")
        current_tone  = st.session_state.get('active_tone', 'neutral')
        tone_info     = TONE_LABELS.get(current_tone, TONE_LABELS['neutral'])
        tone_color    = TONE_COLORS.get(current_tone, '#d9d9d9')

        badge = (f'<div class="meta-tag">📺 {episode}</div>'
                 f'<div class="meta-tag">📂 {folder}</div>'
                 f'<div class="meta-tag" style="background:{tone_color};color:#0f382a;border-color:#000;">'
                 f'{tone_info["emoji"]} {current_tone}</div>')
        if os.path.exists(tmp_crop):
            badge += ('<div class="meta-tag" style="background:#f20553;color:white;'
                      'border-color:#8a0330;">✂️ TRIMMED</div>')
        st.markdown(badge, unsafe_allow_html=True)
        st.audio(playback, autoplay=True)
        st.write("---")

        if st.button("🔴 SAVE AS BMO", type="primary", use_container_width=True):
            perform_save(
                filename, audio_path,
                st.session_state.user_text,
                st.session_state.get('active_tone', 'neutral'),
                'save'
            )

        if st.button("🟡 Manual Edit / Trash", type="secondary", use_container_width=True):
            perform_save(filename, audio_path, "", "neutral", 'trash')

        st.write("")
        editor_open = st.session_state.show_wav_editor
        edit_label  = "🔵 CLOSE EDITOR" if editor_open else "🔵 EDIT WAV"
        if st.button(edit_label, use_container_width=True):
            st.session_state.show_wav_editor = not editor_open
            if not editor_open:
                dur = get_wav_duration(audio_path)
                st.session_state.crop_start = 0.0
                st.session_state.crop_end   = dur
            st.rerun()

    with col2:
        st.markdown("### 📺 TRANSCRIPT")
        render_tag_buttons(filename)
        st.text_area("hidden", key="user_text", height=200, label_visibility="collapsed")
        st.caption("⬆️ Edit above. Tag buttons prepend to transcript start.")
        st.write("---")
        render_tone_buttons(filename)

    # ── WAV EDITOR (full-width below) ──────────────────────────
    if st.session_state.show_wav_editor:
        st.write("---")
        render_wav_editor(filename, audio_path)


if __name__ == "__main__":
    main()
