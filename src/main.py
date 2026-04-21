import cv2
import importlib
import os
import sys
import time
import argparse
from tracker import VehicleTracker
from counter import VehicleCounter
from visualizer import Visualizer
from violations import ViolationDetector
from red_light_detector import RedLightDetector
from lane_violation_detector import LaneViolationDetector
from congestion_analyzer import CongestionAnalyzer

try:
    from .vdolinks import YOUTUBE_LIVE_STREAMS, DEFAULT_YOUTUBE_STREAM, DEFAULT_YOUTUBE_STREAM_KEY
except ImportError:
    from vdolinks import YOUTUBE_LIVE_STREAMS, DEFAULT_YOUTUBE_STREAM, DEFAULT_YOUTUBE_STREAM_KEY


def resolve_youtube_stream(source_key):
    if not source_key:
        return DEFAULT_YOUTUBE_STREAM, DEFAULT_YOUTUBE_STREAM_KEY
    if source_key in YOUTUBE_LIVE_STREAMS:
        return YOUTUBE_LIVE_STREAMS[source_key], source_key
    if source_key.startswith("http://") or source_key.startswith("https://"):
        return source_key, source_key
    return DEFAULT_YOUTUBE_STREAM, DEFAULT_YOUTUBE_STREAM_KEY


def get_youtube_direct_url(youtube_url):
    try:
        yt_dlp = importlib.import_module("yt_dlp")
        YoutubeDL = getattr(yt_dlp, "YoutubeDL")
    except (ImportError, ModuleNotFoundError):
        try:
            youtube_dl = importlib.import_module("youtube_dl")
            YoutubeDL = getattr(youtube_dl, "YoutubeDL")
            print(" yt_dlp not installed, using youtube_dl fallback.")
        except (ImportError, ModuleNotFoundError):
            print(" yt_dlp / youtube_dl not installed, using the raw YouTube URL.")
            return youtube_url

    ydl_opts = {
        "format": "best",
        "quiet": True,
        "skip_download": True,
        "nocheckcertificate": True,
        "ignoreerrors": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if info is None:
                return youtube_url

            if "formats" in info:
                formats = sorted(
                    (f for f in info["formats"] if f.get("url")),
                    key=lambda item: item.get("height", 0) or 0,
                    reverse=True,
                )
                for fmt in formats:
                    url = fmt.get("url")
                    if url:
                        return url

            return info.get("url", youtube_url)
    except Exception as exc:
        print(f" Failed to resolve YouTube stream URL: {exc}")
        return youtube_url

# Paths
default_video_path = "./videos/input/Road traffic video for object recognition.mp4"
output_path = "./videos/output/output.mp4"
model_path = "./models/yolov8n.pt"

# Config
line_y = 570
frame_width = 1150
frame_height = line_y + 270

vehicle_classes = [2, 3, 5, 7]
class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
color_map = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 0, 255), 7: (255, 255, 0)}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["demo", "real-time"], default="demo",
                    help="Operation mode: demo uses local videos, real-time uses live streams.")
parser.add_argument("--source-type", choices=["video", "camera", "rtsp"], default="video",
                    help="Source type: 'video' for local demo, 'camera' for webcam, or 'rtsp' for stream URL")
parser.add_argument("--source", default=default_video_path,
                    help="Source path/ID: file path, camera index (0,1,2...), or RTSP URL")
parser.add_argument("--live-stream", default=None,
                    help="YouTube live stream key or URL when using real-time mode")
parser.add_argument("--max-frames", type=int, default=0,
                    help="Max frames to process (0=unlimited, useful for live streams)")
args = parser.parse_args()

source = args.source
if args.mode == "real-time":
    print(f"Operation Mode: REAL-TIME")
    if args.source_type == "video":
        live_source, resolved_key = resolve_youtube_stream(args.live_stream)
        source = get_youtube_direct_url(live_source)
        print(f"Using YouTube live stream: {resolved_key}")
    elif args.source_type == "camera":
        print(f"Using live camera: {args.source}")
    elif args.source_type == "rtsp":
        print(f"Using RTSP/live stream URL: {args.source}")
else:
    print(f"Operation Mode: DEMO")
    if args.source_type != "video":
        print(" Demo mode supports only local video files. Switching source type to 'video'.")
        args.source_type = "video"
    source = args.source

print(f"Input Source Type: {args.source_type}")
print(f"Input Source: {source}")
if args.mode == "demo":
    print(f"Processing recorded demo video...")
elif args.mode == "real-time":
    print(f"Processing real-time source...")

# Initialize video capture based on source type
if args.mode == "demo":
    if not os.path.exists(source):
        print(f" Video file not found: {source}")
        sys.exit(1)
    cap = cv2.VideoCapture(source)
    print(f" Opened video file: {source}")
elif args.source_type == "camera":
    try:
        camera_id = int(args.source)
        cap = cv2.VideoCapture(camera_id)
        print(f" Connected to camera {camera_id}")
    except ValueError:
        print(f" Invalid camera ID: {args.source}")
        sys.exit(1)
elif args.source_type == "rtsp":
    cap = cv2.VideoCapture(source)
    print(f" Connecting to RTSP stream...")
else:  # video file or real-time YouTube source
    if args.mode == "demo" and not os.path.exists(source):
        print(f" Video file not found: {source}")
        sys.exit(1)
    cap = cv2.VideoCapture(source)
    print(f" Opened video source: {source}")

if not cap.isOpened():
    raise RuntimeError(f"Unable to open source: {args.source_type} -> {source}")

tracker = VehicleTracker(model_path)
counter = VehicleCounter(line_y)
visualizer = Visualizer(line_y, frame_width, frame_height)

# Initialize violation detectors
red_light_detector = RedLightDetector(frame_width, frame_height)
lane_detector = LaneViolationDetector(frame_width, frame_height, num_lanes=3)
congestion_analyzer = CongestionAnalyzer(frame_width, frame_height)

# Adjust FPS and output settings based on operation mode
if args.mode == "demo" and args.source_type == "video":
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    print(f"Video FPS: {fps}")
else:
    # For live streams and real-time sources, use a standard frame rate
    fps = 30
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for live streams to minimize latency
    print(f"Live stream FPS target: {fps}")

# For real-time sources, generate a timestamped output file
if args.mode == "real-time":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"./videos/output/output_{timestamp}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output will be saved to: {output_path}")

violation_detector = ViolationDetector(
    fps,
    speed_threshold_kmh=80,
    meters_per_pixel=0.05,
    evidence_enabled=False
)
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

paused = False
frame = None
prev_vehicle_positions = {}  # Track positions for direction vectors
frame_count = 0
max_frames = args.max_frames if args.max_frames > 0 else float('inf')

print(f"Processing {args.source_type.upper()} at {fps} FPS...")
print("=" * 80)
print("Violations Tracked: Overspeed | Parking | Red-Light | Lane Violations")
print("Controls: SPACE to pause/resume | ESC to stop")
if args.mode == "real-time":
    print(f"Press 'q' to quit at any time (live stream)")
print("=" * 80)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            if args.mode == "demo":
                print(" End of video reached.")
                break
            else:
                print(" Failed to read frame from live stream. Reconnecting...")
                time.sleep(2)
                continue
        
        # Check if max frames reached (useful for testing live streams)
        if frame_count >= max_frames:
            print(f" Reached max frame limit ({max_frames} frames).")
            break

    if frame is None:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    visualizer.draw_line(frame)
    results = tracker.track(frame)

    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
    else:
        ids = []

    active_ids = set(ids)
    boxes = results.boxes
    vehicle_positions = []
    vehicle_speeds = []

    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            track_id = ids[i] if len(ids) > i else -1
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Calculate direction vector for lane detection
            direction_vector = None
            if track_id in prev_vehicle_positions:
                prev_cx, prev_cy = prev_vehicle_positions[track_id]
                direction_vector = (cx - prev_cx, cy - prev_cy)
            prev_vehicle_positions[track_id] = (cx, cy)

            # OVERSPEED DETECTION
            is_violation, speed, violation_record = violation_detector.update(
                track_id,
                cx,
                cy,
                class_names[cls],
                frame=frame,
                bbox=(x1, y1, x2, y2)
            )

            # ILLEGAL PARKING DETECTION
            is_parked, parked_record = violation_detector.check_illegal_parking(
                track_id,
                cx,
                cy,
                vehicle_type=class_names[cls],
                frame=frame,
                bbox=(x1, y1, x2, y2)
            )

            # RED-LIGHT JUMPING DETECTION
            red_light_violation, red_light_record = red_light_detector.update(
                frame, track_id, cx, cy, vehicle_type=class_names[cls]
            )

            # LANE VIOLATION DETECTION
            lane_violation, lane_record = lane_detector.update(
                track_id, cx, cy, vehicle_type=class_names[cls],
                direction_vector=direction_vector
            )

            # Determine primary violation status (priority order)
            if red_light_violation:
                status = "Red-Light"
                record = red_light_record
            elif is_violation:
                status = "Overspeed"
                record = violation_record
            elif lane_violation:
                status = "Lane Viol"
                record = lane_record
            elif is_parked:
                status = "Parked"
                record = parked_record
            else:
                status = "Normal"
                record = None

            # Color based on violation type
            if red_light_violation or is_violation or lane_violation:
                color = (0, 0, 255)  # Red for serious violations
            elif is_parked:
                color = (0, 165, 255)  # Orange for parking
            else:
                color = (0, 255, 0)  # Green for normal

            label = f"{class_names[cls]} ID:{track_id} {conf:.2f} Speed:{int(speed)} {status}"
            visualizer.draw_center(frame, cx, cy)
            counter.update(track_id, cy, cls, class_names)
            visualizer.draw_box(frame, x1, y1, x2, y2, color, label)

            # Collect for congestion analysis
            vehicle_positions.append((cx, cy))
            vehicle_speeds.append(speed)

    violation_detector.cleanup_inactive_tracks(active_ids)
    
    # Cleanup for other detectors
    for vid in list(prev_vehicle_positions.keys()):
        if vid not in active_ids:
            del prev_vehicle_positions[vid]
    
    # Update congestion analysis
    congestion_data = congestion_analyzer.update(vehicle_positions, vehicle_speeds)
    
    # Draw visualizations
    red_light_detector.draw_light_indicator(frame)
    congestion_analyzer.draw_congestion_indicator(frame, congestion_data)
    visualizer.draw_ui(frame, counter.get_count(), paused)
    
    # Write to output video file
    out.write(frame)
    
    cv2.imshow("Vehicle Detection & Counting", frame)
    
    # Progress tracking
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[{frame_count} frames] Vehicles: {counter.get_count()} | Congestion: {congestion_data.get('level', 'N/A')}")
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        print("\n⏹️  Stopping processing...")
        break
    elif key == 32:  # SPACE to pause/resume
        paused = not paused
        status = "PAUSED" if paused else "RESUMED"
        print(f"[{frame_count}] {status}")

cap.release()
if args.source_type == "camera":
    print(" Camera disconnected.")
elif args.source_type == "rtsp":
    print(" RTSP stream disconnected.")
else:
    print(" Video file closed.")

out.release()
cv2.destroyAllWindows()

print("=" * 80)
print(f" PROCESSING COMPLETE!")
print(f"   Frames Processed: {frame_count}")
print(f"   Total Vehicles Detected: {counter.get_count()}")
print(f"   Output Video: {output_path}")
violation_detector.export_csv()
print(f"   Violations CSV: ./violations.csv")
if args.source_type != "video":
    print(f"   Source Type: {args.source_type.upper()} (Live Stream)")
print("=" * 80)