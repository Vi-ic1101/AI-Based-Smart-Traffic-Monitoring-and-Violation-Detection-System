"""
Live Traffic Monitoring Dashboard - Streamlit UI
Displays video feed with overlays, metrics, and real-time violations
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import streamlit as st
import numpy as np
from PIL import Image
from vdolinks import YOUTUBE_LIVE_STREAMS

# Configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MAIN_SCRIPT = BASE_DIR / "main.py"
STATE_FILE = PROJECT_DIR / "data" / "dashboard_state.json"
VIDEO_DIR = PROJECT_DIR / "videos" / "input"
OUTPUT_DIR = PROJECT_DIR / "videos" / "output"
LIVE_FRAME_PATH = OUTPUT_DIR / "live_frame.jpg"
VIOLATIONS_CSV = BASE_DIR / "violations.csv"
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "backend.log"

SUPPORTED_VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

# Streamlit page config
st.set_page_config(
    page_title="Traffic Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Based Smart Traffic Monitoring System"}
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .violation-alert {
        background: #ff6b6b;
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    .violation-item {
        background: #f8f9fa;
        padding: 12px;
        border-left: 4px solid #ff6b6b;
        margin: 8px 0;
        border-radius: 4px;
    }
    .status-running {
        color: #51cf66;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff6b6b;
        font-weight: bold;
    }
    .congestion-low {
        color: #51cf66;
    }
    .congestion-high {
        color: #ff6b6b;
    }
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


def load_state():
    """Load current dashboard state."""
    if not STATE_FILE.exists():
        return {
            "entries": {},
            "stats": {
                "total_vehicles": 0, "overspeed": 0, "parking": 0,
                "red_light_violations": 0, "lane_violations": 0, "history": []
            },
            "recent_violations": [],
            "congestion_data": {}
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "entries": {}, "stats": {"total_vehicles": 0, "overspeed": 0,
                                      "parking": 0, "history": []},
            "recent_violations": [], "congestion_data": {}
        }


def get_congestion_color(level):
    """Get color for congestion level."""
    level_lower = level.lower() if level else "free_flow"
    colors = {
        "free_flow": "#51cf66",  # Green
        "low_congestion": "#75b0ff",  # Light blue
        "moderate_congestion": "#ffd43b",  # Yellow
        "high_congestion": "#ff922b",  # Orange
        "severe_congestion": "#ff6b6b"  # Red
    }
    return colors.get(level_lower, "#999999")


def get_violation_icon(violation_type):
    """Get emoji icon for violation type."""
    icons = {
        "overspeed": "⚡",
        "illegal_parking": "🅿️",
        "red_light_jumping": "🚦",
        "improper_lane_change": "🛣️",
        "wrong_way_driving": "🔄"
    }
    return icons.get(violation_type, "⚠️")


def load_live_frame():
    """Load the latest live frame from the processing backend."""
    if LIVE_FRAME_PATH.exists():
        return Image.open(LIVE_FRAME_PATH)
    return None


def validate_source(source_type, source_value):
    """Validate input source."""
    if source_type == "camera":
        try:
            cap = cv2.VideoCapture(int(source_value))
            opened = cap.isOpened()
            cap.release()
            return opened, "" if opened else "Camera not accessible"
        except (ValueError, TypeError):
            return False, "Invalid camera index"

    elif source_type == "rtsp":
        return bool(source_value and source_value.startswith("rtsp")), \
               "Please enter valid RTSP URL"

    elif source_type == "video":
        return os.path.exists(source_value), "Video file not found"

    return False, "Unknown source type"


def format_violation_record(record):
    """Format violation record for display."""
    v_type = record.get("violation", "unknown").replace("_", " ").title()
    timestamp = record.get("timestamp", "N/A")
    track_id = record.get("id", "?")
    vehicle_type = record.get("vehicle_type", "Unknown").title()
    
    details = f"ID: {track_id} | {vehicle_type}"
    
    if record.get("speed_kmh"):
        details += f" | {record['speed_kmh']:.1f} km/h"
    
    if record.get("duration_seconds"):
        details += f" | {record['duration_seconds']:.1f}s"
    
    return v_type, timestamp, details


# ==================== MAIN DASHBOARD ====================

st.title("🚗 Smart Traffic Monitoring Dashboard")
st.markdown("Real-time vehicle tracking, violation detection, and traffic analytics")

# Initialize session state
if "backend_process" not in st.session_state:
    st.session_state.backend_process = None
if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = 0

process = st.session_state.backend_process
running = process is not None and process.poll() is None

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    tab_control, tab_info = st.tabs(["Controls", "Info"])
    
    with tab_control:
        st.subheader("Source Configuration")
        
        source_mode = st.radio("Mode", ["Demo Video", "Live Camera", "Stream", "YouTube Live"])
        
        source_type = "video"
        source_value = ""
        live_stream_key = None
        
        if source_mode == "Demo Video":
            videos = list_videos() or ["No videos found"]
            selected = st.selectbox("Select video", videos)
            custom_path = st.text_input("Or custom path")
            source_value = custom_path if custom_path else (
                selected if selected != "No videos found" else ""
            )
        elif source_mode == "Live Camera":
            source_type = "camera"
            source_value = str(st.number_input("Camera Index", 0, 5, 0))
        elif source_mode == "Stream":
            source_type = "rtsp"
            source_value = st.text_input("RTSP URL", "rtsp://")
        else:  # YouTube Live
            source_type = "video"
            live_stream_key = st.selectbox("Select YouTube live stream", list(YOUTUBE_LIVE_STREAMS.keys()))
        
        st.markdown("---")
        st.subheader("Backend Options")
        evidence_enabled = st.checkbox("Enable Evidence Capture", False)
        auto_refresh = st.checkbox("Auto-Refresh", True)
        
        st.markdown("---")
        st.subheader("Actions")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_btn = st.button("▶️ Start", use_container_width=True)
        with col_btn2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        
        col_btn3, col_btn4 = st.columns(2)
        with col_btn3:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col_btn4:
            refresh_btn = st.button("🔄 Refresh", use_container_width=True)
    
    with tab_info:
        st.subheader("System Status")
        
        if running:
            st.success(f"✅ Backend Running (PID: {process.pid})")
            st.info(f"📹 Source: {source_type.upper()}")
            if auto_refresh:
                st.info("🔄 Auto-refreshing every 1 second")
        else:
            st.error("❌ Backend Stopped")
            st.info("Click 'Start' to begin monitoring")
        
        st.divider()
        st.subheader("Quick Links")
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.write("[View Logs](#backend-feed)")
        with col_link2:
            st.write("[Violations]()")

# ==================== BACKEND CONTROL LOGIC ====================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if start_btn:
    if source_mode == "YouTube Live":
        valid, error_msg = True, ""
    else:
        valid, error_msg = validate_source(source_type, source_value)

    if not valid:
        st.sidebar.error(f"❌ {error_msg}")
    elif running:
        st.sidebar.warning("⚠️ Already running")
    else:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            
            cmd = [sys.executable, str(MAIN_SCRIPT)]
            if source_mode == "Demo Video":
                cmd += ["--mode", "demo", "--source-type", "video", "--source", source_value]
            elif source_mode == "Live Camera":
                cmd += ["--mode", "real-time", "--source-type", "camera", "--source", source_value]
            elif source_mode == "Stream":
                cmd += ["--mode", "real-time", "--source-type", "rtsp", "--source", source_value]
            else:
                cmd += ["--mode", "real-time", "--source-type", "video", "--live-stream", live_stream_key]

            cmd += ["--state-file", str(STATE_FILE), "--log-file", str(LOG_FILE), "--headless"]
            if evidence_enabled:
                cmd.append("--evidence-enabled")

            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.session_state.backend_process = process
            running = True
            st.sidebar.success(f"✅ Backend started!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

if stop_btn and running:
    process.terminate()
    st.session_state.backend_process = None
    st.sidebar.success("✅ Backend stopped")
    time.sleep(0.5)
    st.rerun()

if clear_btn:
    if STATE_FILE.exists():
        os.remove(STATE_FILE)
    st.sidebar.success("✅ State cleared")
    st.rerun()

if refresh_btn or (auto_refresh and running):
    st.rerun()

# Auto-refresh during processing
if running and auto_refresh:
    time.sleep(1)
    st.rerun()

# ==================== MAIN DISPLAY ====================

st.markdown("---")

# Load current state
state = load_state()
stats = state.get("stats", {})
congestion = state.get("congestion_data", {})
violations = state.get("recent_violations", [])[:5]  # Show 5 most recent

# Top metrics row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric("🚗 Total Vehicles", stats.get("total_vehicles", 0))

with col_m2:
    congestion_level = congestion.get("congestion_level", "free_flow")
    congestion_pct = congestion.get("congestion_percentage", 0)
    color = get_congestion_color(congestion_level)
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background: {color}22; border-radius: 8px;">
        <small>Congestion</small><br>
        <strong style="font-size: 24px; color: {color}">
            {congestion_pct:.0f}%
        </strong>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    avg_speed = congestion.get("avg_speed_kmh", 0)
    st.metric("🏎️ Avg Speed", f"{avg_speed:.1f} km/h")

with col_m4:
    total_violations = (stats.get("overspeed", 0) +
                        stats.get("parking", 0) +
                        stats.get("red_light_violations", 0) +
                        stats.get("lane_violations", 0))
    st.metric("⚠️ Total Violations", total_violations)

st.markdown("---")

# Main content area: Video feed and violations
col_video, col_sidebar = st.columns([3, 1])

with col_video:
    st.subheader("📹 Live Feed")
    
    # Display live frame if available
    live_frame = load_live_frame()
    
    if live_frame:
        st.image(live_frame, use_column_width=True)
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    elif running:
        st.info("⏳ Waiting for video stream... (first frame coming soon)")
    else:
        st.warning("📴 Backend not running - click 'Start' to begin")

with col_sidebar:
    st.subheader("📋 Recent Violations")
    
    if violations:
        for i, v in enumerate(violations, 1):
            v_type, timestamp, details = format_violation_record(v)
            icon = get_violation_icon(v.get("violation", "unknown"))
            
            st.markdown(f"""
            <div class="violation-item">
                <strong>{icon} {v_type}</strong><br>
                <small>{timestamp}</small><br>
                <small>{details}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No violations detected yet")

st.markdown("---")

# Violation statistics
st.subheader("📊 Violation Statistics")

col_v1, col_v2, col_v3, col_v4 = st.columns(4)

with col_v1:
    st.metric("⚡ Overspeed", stats.get("overspeed", 0))

with col_v2:
    st.metric("🅿️ Illegal Parking", stats.get("parking", 0))

with col_v3:
    st.metric("🚦 Red-Light", stats.get("red_light_violations", 0))

with col_v4:
    st.metric("🛣️ Lane Violations", stats.get("lane_violations", 0))

# Vehicle count history chart
st.subheader("📈 Vehicle Count Trend")
history = stats.get("history", [])

if history:
    chart_data = {
        "Time": [h.get("time", "?") for h in history],
        "Count": [h.get("count", 0) for h in history]
    }
    st.line_chart(chart_data, x="Time", y="Count", use_container_width=True)
else:
    st.info("No history data yet")

st.markdown("---")

# Backend feed section
st.subheader("🔴 Backend Feed")

col_feed_refresh = st.columns([0.9, 0.1])
with col_feed_refresh[1]:
    if st.button("🔄", key="refresh_logs"):
        pass

if LOG_FILE.exists():
    try:
        with open(LOG_FILE, "r") as f:
            logs = f.read().splitlines()[-20:]  # Last 20 lines
        st.code("\n".join(logs), language="log")
    except Exception as e:
        st.error(f"Error reading logs: {e}")
else:
    st.info("No logs yet")
