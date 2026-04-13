import json
import os
import subprocess
import sys
from pathlib import Path
# import cv2
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
APP_SCRIPT = BASE_DIR / "main.py"
STATE_PATH = BASE_DIR.parent / "data" / "dashboard_state.json"
VIDEO_DIR = BASE_DIR.parent / "videos" / "input"

def list_videos():
    if not VIDEO_DIR.exists():
        return []
    return sorted([str(path) for path in VIDEO_DIR.glob("*") if path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])

def load_state(path):
    if not os.path.exists(path):
        return {"entries": {}, "stats": {"total_vehicles": 0, "overspeed": 0, "parking": 0, "history": []}, "recent_violations": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("Smart Traffic Monitoring Dashboard")

st.sidebar.header("Demo source")
source_type = st.sidebar.selectbox("Source type", ["Video file", "Camera", "RTSP stream"])
if source_type == "Video file":
    video_files = list_videos()
    source = st.sidebar.selectbox("Choose video", video_files or ["No videos found"])
    source_input = st.sidebar.text_input("Or enter video path", "")
    source_value = source_input.strip() if source_input.strip() else (source if source != "No videos found" else "")
elif source_type == "Camera":
    source_value = str(st.sidebar.number_input("Camera index", min_value=0, max_value=5, value=0, step=1))
else:
    source_value = st.sidebar.text_input("Stream URL", "rtsp://")

state_file = st.sidebar.text_input("State file", str(STATE_PATH))
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)

if "process" not in st.session_state:
    st.session_state.process = None

start = st.sidebar.button("Start pipeline")
stop = st.sidebar.button("Stop pipeline")
clear = st.sidebar.button("Clear state file")

if clear and os.path.exists(state_file):
    os.remove(state_file)

if start:
    if not source_value:
        st.sidebar.error("Select a valid source first.")
    elif st.session_state.process is None or st.session_state.process.poll() is not None:
        cmd = [
            sys.executable,
            str(APP_SCRIPT),
            "--source-type",
            source_type.lower(),
            "--source",
            source_value,
            "--state-file",
            state_file
        ]
        st.session_state.process = subprocess.Popen(cmd)
        st.sidebar.success(f"Started pipeline ({source_type})")
    else:
        st.sidebar.info("Pipeline is already running.")

if stop:
    if st.session_state.process is not None and st.session_state.process.poll() is None:
        st.session_state.process.terminate()
        st.session_state.process = None
        st.sidebar.success("Pipeline stopped")

st.sidebar.markdown("---")
if st.session_state.process is not None and st.session_state.process.poll() is None:
    st.sidebar.success(f"Pipeline running (PID {st.session_state.process.pid})")
else:
    st.sidebar.info("Pipeline stopped")

state = load_state(state_file)
stats = state.get("stats", {})
entries = state.get("entries", {})
recent_violations = state.get("recent_violations", [])

col1, col2, col3 = st.columns(3)
col1.metric("Total vehicles", stats.get("total_vehicles", 0))
col2.metric("Overspeed", stats.get("overspeed", 0))
col3.metric("Parking", stats.get("parking", 0))

st.subheader("Traffic count history")
history = stats.get("history", [])
if history:
    st.line_chart({"count": [item["count"] for item in history]}, width="stretch")
else:
    st.info("No history data yet.")

show_only_violations = st.sidebar.checkbox("Show only violations", value=False)
table_rows = []
for track_id, data in entries.items():
    if show_only_violations and data["status"] == "Normal":
        continue
    table_rows.append({
        "Track ID": track_id,
        "Type": data["type"],
        "Speed (km/h)": data["speed"],
        "Status": data["status"],
        "Last update": data["updated"]
    })

st.subheader("Live vehicle table")
st.dataframe(table_rows, width="stretch")

st.subheader("Recent violations")
for violation in recent_violations[:8]:
    st.markdown(f"**{violation['violation'].replace('_', ' ').title()}** — ID {violation['id']} — {violation.get('speed_kmh', '')} km/h — {violation['time']}")
    if violation.get("frame_path") and os.path.exists(violation["frame_path"]):
        st.image(violation["frame_path"], caption="Full frame", width="stretch")
    if violation.get("crop_path") and os.path.exists(violation["crop_path"]):
        st.image(violation["crop_path"], caption="Vehicle crop", width="stretch")
    st.markdown("---")

st.sidebar.markdown("## Instructions")
st.sidebar.write("1. Choose source type.")
st.sidebar.write("2. Start pipeline.")
st.sidebar.write("3. Refresh dashboard to see live state.")
