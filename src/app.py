import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
APP_SCRIPT = BASE_DIR / "main.py"
DEFAULT_STATE_FILE = BASE_DIR.parent / "data" / "dashboard_state.json"
VIDEO_DIR = BASE_DIR.parent / "videos" / "input"
OUTPUT_DIR = BASE_DIR.parent / "videos" / "output"
LOG_DIR = BASE_DIR.parent / "logs"
LOG_FILE = LOG_DIR / "backend.log"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "output.mp4"

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


def list_demo_videos():
    if not VIDEO_DIR.exists():
        return []
    return sorted(
        [str(path) for path in VIDEO_DIR.glob("*") if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
    )


def tail_log(path, lines=50):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().splitlines()
    return "\n".join(content[-lines:])


def validate_source(source_type, source_value):
    if source_type == "camera":
        try:
            camera_index = int(source_value)
        except (TypeError, ValueError):
            return False, "Camera index must be a valid integer."
        cap = cv2.VideoCapture(camera_index)
        opened = cap.isOpened()
        cap.release()
        if not opened:
            return False, f"Unable to open camera index {camera_index}. Check that the camera is connected and available."
        return True, ""

    if source_type == "rtsp":
        if not source_value or source_value.strip() in {"", "rtsp://", "http://", "https://"}:
            return False, "Please enter a valid RTSP/stream URL."
        return True, ""

    if source_type == "video":
        if not source_value:
            return False, "Please select a demo video or enter a video file path."
        if not os.path.exists(source_value):
            return False, f"Video file not found: {source_value}"
        return True, ""

    return False, "Unsupported source type."


def load_state(path):
    if not os.path.exists(path):
        return {
            "entries": {},
            "stats": {"total_vehicles": 0, "overspeed": 0, "parking": 0, "history": []},
            "recent_violations": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {
            "entries": {},
            "stats": {"total_vehicles": 0, "overspeed": 0, "parking": 0, "history": []},
            "recent_violations": [],
        }


st.set_page_config(page_title="Smart Traffic Monitoring", layout="wide")
st.title("AI-Based Smart Traffic Monitoring and Violation Detection")
st.markdown(
    "Use this dashboard to launch vehicle monitoring in real time or run a demo on sample videos. "
    "The backend core starts when you click `Start` and writes live state to a JSON file for display."
)

with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio("Choose mode", ["Monitoring", "Demo"], index=0)

    source_type = "video"
    source_value = ""
    mode_description = ""

    if mode == "Monitoring":
        monitoring_mode = st.selectbox(
            "Real-time source", ["Camera", "RTSP stream", "Stream API"], index=0
        )
        if monitoring_mode == "Camera":
            source_type = "camera"
            source_value = str(
                st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
            )
            mode_description = "Use a connected camera for live traffic monitoring."
        elif monitoring_mode == "RTSP stream":
            source_type = "rtsp"
            source_value = st.text_input("RTSP stream URL", "rtsp://")
            mode_description = "Use an RTSP camera or stream source for live traffic analysis."
        else:
            source_type = "rtsp"
            source_value = st.text_input(
                "Stream API URL",
                "rtsp://",
            )
            mode_description = "Use a stream API URL as a real-time traffic source."
    else:
        demo_videos = list_demo_videos()
        source_type = "video"
        source_choice = st.selectbox(
            "Demo video", demo_videos or ["No videos found"], index=0
        )
        custom_demo = st.text_input("Or enter custom video path", "")
        source_value = custom_demo.strip() if custom_demo.strip() else (
            source_choice if source_choice != "No videos found" else ""
        )
        mode_description = "Play a recorded traffic video to demo the violation detection pipeline."

    state_file = st.text_input("State file", str(DEFAULT_STATE_FILE))
    st.markdown("---")
    st.write(mode_description)

    st.markdown("## Controls")
    evidence_enabled = st.checkbox("Enable evidence capture", value=False)
    show_output_monitor = st.checkbox("Show output monitor", value=True)
    show_backend_logs = st.checkbox("Show backend logs", value=False)
    start_button = st.button("Start")
    stop_button = st.button("Stop")
    clear_button = st.button("Clear state")
    refresh_button = st.button("Refresh dashboard")

    st.markdown("---")
    st.write("Current file paths and settings are used to run the backend and refresh the dashboard.")

if "backend_process" not in st.session_state:
    st.session_state.backend_process = None

process = st.session_state.backend_process
running = process is not None and process.poll() is None

if clear_button:
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
            st.sidebar.success("State file cleared.")
        except OSError as exc:
            st.sidebar.error(f"Unable to clear state file: {exc}")
    else:
        st.sidebar.info("State file does not exist.")

if stop_button:
    if running:
        process.terminate()
        st.session_state.backend_process = None
        running = False
        st.sidebar.success("Backend stopped.")
    else:
        st.sidebar.info("No backend process is currently running.")

if start_button:
    valid_source, source_error = validate_source(source_type, source_value)
    if not source_value:
        st.sidebar.error("Please select or enter a valid source before starting.")
    elif not valid_source:
        st.sidebar.error(source_error)
    elif running:
        st.sidebar.info("Backend is already running.")
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(APP_SCRIPT),
            "--source-type",
            source_type,
            "--source",
            source_value,
            "--state-file",
            state_file,
            "--log-file",
            str(LOG_FILE),
        ]
        if evidence_enabled:
            cmd.append("--evidence-enabled")

        try:
            process = subprocess.Popen(cmd)
            st.session_state.backend_process = process
            running = True
            st.sidebar.success(f"Backend started (PID {process.pid}).")
        except OSError as exc:
            st.sidebar.error(f"Unable to start backend: {exc}")

if process is not None and not running:
    st.session_state.backend_process = None

status_text = "Running" if running else "Stopped"
st.sidebar.markdown(f"**Backend status:** {status_text}")
if running:
    st.sidebar.markdown(f"PID: {process.pid}")

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.write("1. Choose Monitoring or Demo mode.")
st.sidebar.write("2. Configure the source and state file.")
st.sidebar.write("3. Click Start to launch the backend core.")
st.sidebar.write("4. Watch live metrics and violations below.")

st.subheader("Live Traffic Dashboard")

if refresh_button:
    st.sidebar.success("Dashboard refreshed.")

state = load_state(state_file)
stats = state.get("stats", {})
entries = state.get("entries", {})
recent_violations = state.get("recent_violations", [])

col1, col2, col3 = st.columns(3)
col1.metric("Total vehicles", stats.get("total_vehicles", 0))
col2.metric("Overspeed", stats.get("overspeed", 0))
col3.metric("Parking", stats.get("parking", 0))

st.markdown("#### Count history")
history = stats.get("history", [])
if history:
    st.line_chart({"count": [item.get("count", 0) for item in history]})
else:
    st.info("No count history available yet.")

show_violations_only = st.checkbox("Show only violations", value=False)
rows = []
for track_id, data in entries.items():
    if show_violations_only and data.get("status") == "Normal":
        continue
    rows.append(
        {
            "Track ID": track_id,
            "Type": data.get("type", ""),
            "Speed (km/h)": data.get("speed", 0),
            "Status": data.get("status", ""),
            "Last update": data.get("updated", ""),
        }
    )

st.markdown("#### Vehicle tracking table")
if rows:
    st.dataframe(rows, use_container_width=True)
else:
    st.info("No vehicle entries available in the current state file.")

if show_output_monitor:
    st.markdown("#### Output monitor")
    if OUTPUT_VIDEO_PATH.exists():
        st.video(str(OUTPUT_VIDEO_PATH))
    else:
        st.info("No backend output video yet. Start the backend to generate it.")

if show_backend_logs:
    st.markdown("#### Backend logs")
    if LOG_FILE.exists():
        st.code(tail_log(LOG_FILE, 100), language="text")
    else:
        st.info("No backend log file available yet.")

st.markdown("#### Recent violations")
if recent_violations:
    for violation in recent_violations[:8]:
        violation_type = violation.get("violation", "unknown").replace("_", " ").title()
        time_stamp = violation.get("time", "")
        vehicle_id = violation.get("id", "")
        speed = violation.get("speed_kmh")
        st.markdown(f"**{violation_type}** — ID {vehicle_id} — {time_stamp}")
        if speed is not None:
            st.write(f"Speed: {speed} km/h")

        if evidence_enabled:
            frame_path = violation.get("frame_path")
            crop_path = violation.get("crop_path")
            if frame_path and os.path.exists(frame_path):
                st.image(frame_path, caption="Full frame", width=400)
            if crop_path and os.path.exists(crop_path):
                st.image(crop_path, caption="Vehicle crop", width=300)
        st.markdown("---")
else:
    st.info("No recent violations to display.")
