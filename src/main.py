import cv2
import os
import sys
import time
from tracker import VehicleTracker
from counter import VehicleCounter
from visualizer import Visualizer
from violations import ViolationDetector
from red_light_detector import RedLightDetector
from lane_violation_detector import LaneViolationDetector
from congestion_analyzer import CongestionAnalyzer
from scene_preprocessor import ScenePreprocessor

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

print("=" * 80)
print("LOCAL DEMO - TRAFFIC MONITORING")
print("=" * 80)
print(f"📹 Input: {default_video_path}")
print(f"💾 Output: {output_path}")

if not os.path.exists(default_video_path):
    print(f"❌ Video file not found: {default_video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(default_video_path)
print(f"✅ Opened video file")

if not cap.isOpened():
    print(f"❌ Unable to open source: {default_video_path}")
    sys.exit(1)

tracker = VehicleTracker(model_path)
counter = VehicleCounter(line_y)
visualizer = Visualizer(line_y, frame_width, frame_height)

# Initialize violation detectors
red_light_detector = RedLightDetector(frame_width, frame_height)
lane_detector = LaneViolationDetector(frame_width, frame_height, num_lanes=3)
congestion_analyzer = CongestionAnalyzer(frame_width, frame_height)

# Get FPS from video
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
print(f"⏱️  Video FPS: {fps}")

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
prev_vehicle_positions = {}
frame_count = 0

print("=" * 80)
print("Violations Tracked: Overspeed | Parking | Red-Light | Lane Violations")
print("Controls: SPACE to pause/resume | ESC to stop")
print("=" * 80)

# ==================== SCENE PREPROCESSING ====================
print("\n📋 PREPROCESSING SCENE...")

# Read first frame for analysis
ret, first_frame = cap.read()
if not ret:
    print("❌ Failed to read first frame")
    sys.exit(1)

first_frame = cv2.resize(first_frame, (frame_width, frame_height))

# Initialize scene preprocessor
preprocessor = ScenePreprocessor(model_path)
traffic_lights, pedestrian_zones = preprocessor.analyze_scene(first_frame)

# Configure visualizer with detected regions
visualizer.set_traffic_lights(traffic_lights)
visualizer.set_pedestrian_zones(pedestrian_zones)

# Configure red light detector with detected traffic lights
if traffic_lights:
    red_light_detector.set_traffic_light_regions(traffic_lights)

# Configure counter to exclude pedestrian crossing zones
if pedestrian_zones:
    counter.set_pedestrian_zones(pedestrian_zones)

print("\n✅ PREPROCESSING COMPLETE\n")
print("=" * 80)

# Reset frame counter after preprocessing
frame_count = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video reached.")
            break

    if frame is None:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))
    visualizer.draw_line(frame)
    
    # Draw detected traffic lights and pedestrian zones
    visualizer.draw_traffic_light_regions(frame)
    visualizer.draw_pedestrian_zones(frame)
    
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
            
            direction_vector = None
            if track_id in prev_vehicle_positions:
                prev_cx, prev_cy = prev_vehicle_positions[track_id]
                direction_vector = (cx - prev_cx, cy - prev_cy)
            prev_vehicle_positions[track_id] = (cx, cy)

            # OVERSPEED DETECTION
            is_violation, speed, violation_record = violation_detector.update(
                track_id, cx, cy, class_names[cls],
                frame=frame, bbox=(x1, y1, x2, y2)
            )

            # ILLEGAL PARKING DETECTION
            is_parked, parked_record = violation_detector.check_illegal_parking(
                track_id, cx, cy, vehicle_type=class_names[cls],
                frame=frame, bbox=(x1, y1, x2, y2)
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
            elif is_violation:
                status = "Overspeed"
            elif lane_violation:
                status = "Lane Viol"
            elif is_parked:
                status = "Parked"
            else:
                status = "Normal"

            # Color based on violation type - RED FOR ALL VIOLATIONS
            if red_light_violation or is_violation or lane_violation:
                color = (0, 0, 255)  # RED for serious violations
            elif is_parked:
                color = (0, 165, 255)  # Orange for parking
            else:
                color = (0, 255, 0)  # Green for normal

            label = f"{class_names[cls]} ID:{track_id} {conf:.2f} Speed:{int(speed)} {status}"
            visualizer.draw_center(frame, cx, cy)
            counter.update(track_id, cy, cls, class_names, cx=cx)
            visualizer.draw_box(frame, x1, y1, x2, y2, color, label)

            vehicle_positions.append((cx, cy))
            vehicle_speeds.append(speed)

    violation_detector.cleanup_inactive_tracks(active_ids)
    
    for vid in list(prev_vehicle_positions.keys()):
        if vid not in active_ids:
            del prev_vehicle_positions[vid]
    
    congestion_data = congestion_analyzer.update(vehicle_positions, vehicle_speeds)
    
    red_light_detector.draw_light_indicator(frame)
    congestion_analyzer.draw_congestion_indicator(frame, congestion_data)
    visualizer.draw_ui(frame, counter.get_count(), paused)
    
    out.write(frame)
    cv2.imshow("Local Demo - Traffic Monitoring", frame)
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[{frame_count:06d} frames] Vehicles: {counter.get_count()} | Congestion: {congestion_data.get('level', 'N/A')}")
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("\n⏹️  Stopping processing...")
        break
    elif key == 32:  # SPACE
        paused = not paused
        status = "PAUSED" if paused else "RESUMED"
        print(f"[{frame_count}] {status}")

cap.release()
out.release()
cv2.destroyAllWindows()

print("=" * 80)
print(f"✅ DEMO PROCESSING COMPLETE!")
print(f"   Frames Processed: {frame_count}")
print(f"   Total Vehicles Detected: {counter.get_count()}")
print(f"   Output Video: {output_path}")
violation_detector.export_csv()
print(f"   Violations CSV: ./violations.csv")
print("=" * 80)
