import streamlit as st
import pandas as pd
import os
import shutil

# ================= CONFIGURATION =================
# 📂 WINDOWS PATHS
INPUT_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\PlanB_Dataset"
OUTPUT_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes\Final_Dataset"

# 🔴 SET THIS TO YOUR EPISODE VIDEO FOLDER:
VIDEO_DIR = r"D:\LocalWorkDir\2509362\BMO Episodes" 

METADATA_FILE = os.path.join(INPUT_DIR, "metadata.csv")
FINAL_METADATA = os.path.join(OUTPUT_DIR, "metadata.csv")
# =================================================

st.set_page_config(layout="wide", page_title="BMO Voice Factory", page_icon="🕹️")

# --- 🎨 BMO PALETTE & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #63bda4; }
    
    /* Screen Text Area */
    .stTextArea textarea {
        background-color: #d9ffea; 
        color: #0f382a; 
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 20px;
        border: 4px solid #0f382a; 
        border-radius: 10px;
    }
    
    /* Info Box */
    .bmo-box {
        background-color: #0f382a;
        color: #d9ffea;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
        border: 2px solid #d9ffea;
    }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff !important; text-shadow: 2px 2px 0px #0f382a; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #558c7a; }
    
    /* Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #f20553;
        color: white;
        border: 3px solid #8a0330;
        border-radius: 50px;
        font-size: 24px;
        height: 3.5em;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ffec47;
        color: #5c5400;
        border: 3px solid #bdae2a;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

def init_environment():
    os.makedirs(os.path.join(OUTPUT_DIR, "wavs"), exist_ok=True)
    if not os.path.exists(FINAL_METADATA):
        with open(FINAL_METADATA, "w", encoding="utf-8") as f:
            f.write("filename|text\n")

def get_audio_path(filename):
    for root, dirs, files in os.walk(INPUT_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

def get_video_reference(filename):
    try:
        parts = filename.split('_')
        episode_id = parts[0]  # S01E08
        timestamp_str = parts[-1].replace(".wav", "")
        start_time = float(timestamp_str)
        
        video_path = None
        for file in os.listdir(VIDEO_DIR):
            if episode_id in file and file.endswith(('.mkv', '.mp4', '.avi')):
                video_path = os.path.join(VIDEO_DIR, file)
                break
                
        return video_path, int(start_time)
    except:
        return None, 0

def save_and_move(filename, corrected_text, is_bmo):
    src = get_audio_path(filename)
    if not src:
        st.error("❌ File not found!")
        return

    if is_bmo:
        dst = os.path.join(OUTPUT_DIR, "wavs", filename)
        shutil.copy2(src, dst)
        clean_text = corrected_text.replace("\n", " ").replace("\r", "").replace("|", "").strip()
        with open(FINAL_METADATA, "a", encoding="utf-8", newline='') as f:
            f.write(f"{filename}|{clean_text}\n")
        st.toast(f"💾 SAVED: {clean_text[:20]}...", icon="✅")
    else:
        st.toast("🗑️ TRASHED", icon="🚮")

    try:
        os.rename(src, src + ".done")
    except:
        pass 
    
    # Reset text box
    if 'user_text' in st.session_state:
        del st.session_state['user_text']

def main():
    init_environment()
    
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/6/6b/BMO_Adventure_Time.png", width=140)
        st.title("📖 Rulebook")
        st.markdown("""
        <div class="bmo-box">
        <b>MISSION:</b><br>
        1. Watch Video (Context).<br>
        2. Listen to Audio (Quality).<br>
        3. Edit Text.<br>
        4. Save (Red Button).
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🗣️ NORMAL"): st.markdown("* Use punctuation.\n* `two` not `2`.")
        with st.expander("😱 SCREAMING"): st.markdown("* ALL CAPS: `FINN NO!`")
        with st.expander("🎵 SINGING"): st.markdown("* Lyrics only.")
        with st.expander("🚫 TRASH"): st.markdown("* Finn/Jake talking.\n* Music/Mumbling.")

    st.title("🤖 BMO Voice Factory")
    
    if not os.path.exists(METADATA_FILE):
        st.error(f"🚨 Metadata missing! {METADATA_FILE}")
        return
        
    df = pd.read_csv(METADATA_FILE, sep="|", header=None, names=["filename", "text"])
    
    next_row = None
    for index, row in df.iterrows():
        path = get_audio_path(row['filename'])
        if path and not path.endswith(".done"):
            next_row = row
            break
            
    if next_row is None:
        st.balloons()
        st.success("🎉 MISSION COMPLETE!")
        return

    filename = next_row['filename']
    original_text = next_row['text']
    audio_path = get_audio_path(filename)
    
    # State Management
    if 'current_file' not in st.session_state or st.session_state.current_file != filename:
        st.session_state.current_file = filename
        st.session_state.user_text = original_text

    # --- LAYOUT ---
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 🎬 REFERENCE (Context)")
        
        # 1. VIDEO PLAYER (Context)
        video_path, start_time = get_video_reference(filename)
        if video_path and os.path.exists(video_path):
            st.video(video_path, start_time=start_time)
            st.caption(f"Jumped to: {start_time}s")
        else:
            st.warning("⚠️ Video not found (Check VIDEO_DIR path)")

        st.write("---")

        # 2. AUDIO PLAYER (The Actual File)
        st.markdown("### 🎧 ISOLATED CLIP (What gets Saved)")
        if audio_path:
             st.audio(audio_path, format="audio/wav", start_time=0)
        else:
             st.error("Audio file missing on disk!")
        
        st.write("---")
        
        # 3. CONTROLS
        if st.button("🔴 SAVE AS BMO", type="primary", use_container_width=True):
            save_and_move(filename, st.session_state.user_text, True)
            st.rerun()
            
        if st.button("🟡 TRASH / SKIP", type="secondary", use_container_width=True):
            save_and_move(filename, "", False)
            st.rerun()

    with col2:
        st.markdown("### 📺 TRANSCRIPT")
        st.text_area(
            label="screen_hidden",
            value=st.session_state.user_text,
            height=300,
            key="user_text",
            label_visibility="collapsed"
        )
        st.caption("⬆️ Edit text above. Make sure it matches the audio!")

if __name__ == "__main__":
    main()