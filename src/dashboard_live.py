"""
Smart Traffic Monitoring Dashboard - Streamlit UI
Three-screen flow: Welcome → Mode Selection → Processing
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

try:
    from vdolinks import YOUTUBE_LIVE_STREAMS
except ImportError:
    YOUTUBE_LIVE_STREAMS = {"fresno": "YouTube Live - Fresno"}

# Configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MAIN_SCRIPT = BASE_DIR / "main.py"
MAIN2_SCRIPT = BASE_DIR / "main2.py"
STATE_FILE = PROJECT_DIR / "data" / "dashboard_state.json"
VIDEO_DIR = PROJECT_DIR / "videos" / "input"
OUTPUT_DIR = PROJECT_DIR / "videos" / "output"
LOG_DIR = PROJECT_DIR / "logs"

SUPPORTED_VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

# Streamlit page config
st.set_page_config(
    page_title="🚗 Traffic Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Based Smart Traffic Monitoring System"}
)

# Custom CSS
st.markdown("""
<style>
    .welcome-box { text-align: center; padding: 40px; }
    .mode-card { 
        border: 2px solid #667eea; 
        padding: 30px; 
        border-radius: 10px; 
        background: #f8f9fa;
    }
    .status-running { color: #51cf66; font-weight: bold; }
    .status-stopped { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def list_videos():
    """List available demo videos."""
    if not VIDEO_DIR.exists():
        return []
    return sorted([
        str(p) for p in VIDEO_DIR.glob("*")
        if p.suffix.lower() in SUPPORTED_VIDEO_EXT
    ])


# ==================== SESSION STATE INITIALIZATION ====================
if "screen" not in st.session_state:
    st.session_state.screen = "welcome"
if "mode" not in st.session_state:
    st.session_state.mode = None
if "backend_process" not in st.session_state:
    st.session_state.backend_process = None
if "live_source_type" not in st.session_state:
    st.session_state.live_source_type = "video"
if "live_source_value" not in st.session_state:
    st.session_state.live_source_value = None
if "live_youtube_stream" not in st.session_state:
    st.session_state.live_youtube_stream = "fresno"


# ==================== WELCOME SCREEN ====================
if st.session_state.screen == "welcome":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        # 🚗 Smart Traffic Monitoring System
        
        ## Welcome!
        
        This AI-powered system provides **real-time traffic monitoring**, **vehicle tracking**, 
        and **violation detection** using advanced computer vision technology.
        
        ### ✨ Key Features:
        - 🎥 **Vehicle Tracking**: Real-time detection and tracking of all vehicles
        - ⚡ **Violation Detection**: Overspeed, illegal parking, red-light jumping, lane violations
        - 📊 **Analytics**: Congestion levels, traffic flow analysis, and statistics
        - 🎯 **High Accuracy**: Powered by YOLOv8 deep learning model
        
        ---
        """)
        
        if st.button("🚀 Get Started", use_container_width=True, key="welcome_start"):
            st.session_state.screen = "mode_selection"
            st.rerun()

# ==================== MODE SELECTION SCREEN ====================
elif st.session_state.screen == "mode_selection":
    st.title("📋 Select Execution Mode")
    st.markdown("Choose how you'd like to run the traffic monitoring system:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("""
            ### 🎬 Local Demo Mode
            
            Process recorded traffic video files from your local storage.
            
            **✓ Features:**
            - Uses pre-recorded video files
            - Perfect for testing and demonstrations
            - Controlled playback (pause/resume)
            - Consistent output for analysis
            
            **📁 Input Source:**
            - `./videos/input/`
            """)
            
            if st.button("▶️ Start Local Demo", use_container_width=True, key="demo_mode_btn"):
                st.session_state.mode = "demo"
                st.session_state.screen = "processing"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown("""
            ### 🌐 Live Stream Mode
            
            Process live traffic streams in real-time.
            
            **✓ Features:**
            - YouTube 24/7 live streams
            - Webcam input support
            - RTSP stream support
            - Real-time processing and analysis
            
            **📡 Input Sources:**
            - YouTube live streams
            - Webcam/IP cameras
            - RTSP streams
            """)
            
            if st.button("▶️ Start Live Stream", use_container_width=True, key="live_mode_btn"):
                st.session_state.mode = "live"
                st.session_state.screen = "processing"
                st.rerun()
    
    st.markdown("---")
    if st.button("← Back to Welcome", use_container_width=True):
        st.session_state.screen = "welcome"
        st.rerun()

# ==================== PROCESSING SCREEN ====================
elif st.session_state.screen == "processing":
    
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        if st.session_state.mode == "demo":
            st.subheader("📁 Local Demo Settings")
            videos = list_videos() or ["No videos found"]
            st.selectbox("Select video", videos, key="video_select")
        
        elif st.session_state.mode == "live":
            st.subheader("🌐 Live Stream Settings")
            source_type = st.radio("Source Type", ["YouTube", "Webcam", "RTSP"], key="source_type")
            
            if source_type == "YouTube":
                st.session_state.live_source_type = "video"
                selected_stream = st.selectbox(
                    "YouTube Stream", 
                    list(YOUTUBE_LIVE_STREAMS.keys()), 
                    key="yt_select"
                )
                st.session_state.live_youtube_stream = selected_stream
                st.session_state.live_source_value = None
            elif source_type == "Webcam":
                st.session_state.live_source_type = "camera"
                camera_idx = st.number_input("Camera Index", 0, 5, 0, key="cam_idx")
                st.session_state.live_source_value = str(camera_idx)
            else:
                st.session_state.live_source_type = "rtsp"
                rtsp_url = st.text_input("RTSP URL", "rtsp://", key="rtsp_url")
                st.session_state.live_source_value = rtsp_url
        
        st.markdown("---")
        
        process = st.session_state.backend_process
        running = process is not None and process.poll() is None
        
        col_start, col_stop = st.columns(2)
        with col_start:
            start_btn = st.button("▶️ Start", use_container_width=True, key="start_exec")
        with col_stop:
            stop_btn = st.button("⏹️ Stop", use_container_width=True, key="stop_exec")
        
        col_back, col_clear = st.columns(2)
        with col_back:
            if st.button("← Back", use_container_width=True, key="back_btn"):
                if running:
                    process.terminate()
                    st.session_state.backend_process = None
                st.session_state.screen = "mode_selection"
                st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_btn"):
                if STATE_FILE.exists():
                    os.remove(STATE_FILE)
                st.success("State cleared")
        
        st.markdown("---")
        st.subheader("📊 Status")
        if running:
            st.success(f"✅ Processing Active (PID: {process.pid})")
        else:
            st.error("⏸️ Processing Stopped")
    
    st.title(f"{'🎬 Local Demo' if st.session_state.mode == 'demo' else '🌐 Live Stream'} - Processing")
    
    process = st.session_state.backend_process
    running = process is not None and process.poll() is None
    
    if start_btn:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            if st.session_state.mode == "demo":
                cmd = [sys.executable, str(MAIN_SCRIPT)]
                st.info("🚀 Starting Local Demo Processing...")
            else:
                cmd = [sys.executable, str(MAIN2_SCRIPT)]
                cmd += [
                    "--source-type", st.session_state.live_source_type
                ]
                # Add YouTube stream selection
                if st.session_state.live_source_type == "video":
                    cmd += ["--live-stream", st.session_state.live_youtube_stream]
                # Add source value for camera/RTSP
                elif st.session_state.live_source_value:
                    cmd += ["--source", st.session_state.live_source_value]
                st.info("🚀 Starting Live Stream Processing...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            st.session_state.backend_process = process
            st.success("✅ Backend started!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    if stop_btn:
        if process and process.poll() is None:
            process.terminate()
            st.session_state.backend_process = None
            st.success("✅ Backend stopped")
            time.sleep(0.5)
            st.rerun()
    
    if running:
        st.info(f"⏱️ Processing... (PID: {process.pid})")
        st.markdown("""
        ---
        ### 📺 Output
        Processing frames in real-time. Check the `./videos/output/` folder for processed videos.
        
        **Violations displayed in RED** | Normal vehicles in GREEN | Parking violations in ORANGE
        """)
    else:
        st.warning("⏸️ Click 'Start' to begin processing")
