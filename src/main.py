import cv2
from tracker import VehicleTracker
from counter import VehicleCounter
from visualizer import Visualizer
from violations import ViolationDetector
from dashboard import Dashboard

# Paths
video_path = "./videos/input/Road traffic video for object recognition.mp4"
output_path = "./videos/output/output.mp4"
model_path = "./models/yolov8n.pt"

# Config
line_y = 570
frame_width = 1150
frame_height = line_y + 270

vehicle_classes = [2, 3, 5, 7]

class_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

color_map = {
    2: (0, 255, 0),
    3: (255, 0, 0),
    5: (0, 0, 255),
    7: (255, 255, 0)
}

# Initialize modules
tracker = VehicleTracker(model_path)
counter = VehicleCounter(line_y)
visualizer = Visualizer(line_y, frame_width, frame_height)
dashboard = Dashboard()

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 20

violation_detector = ViolationDetector(
    fps,
    speed_threshold_kmh=80,
    meters_per_pixel=0.05  # set this to your real camera calibration
)
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)


paused = False
frame = None

print(f"Processing video at {fps} FPS...")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
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

    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = ids[i] if len(ids) > i else -1

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            is_violation, speed = violation_detector.update(
                track_id,
                cx,
                cy,
                class_names[cls]
            )

            is_parked = violation_detector.check_illegal_parking(
                track_id,
                cx,
                cy
            )

            color = color_map.get(cls, (255, 255, 255))
            if is_violation:
                color = (0, 0, 255)
            else:
                color = (0, 165, 255)

            status = "Overspeed" if is_violation else "Parked" if is_parked else "Normal"

            label = f"{class_names[cls]} ID:{track_id} {conf:.2f} Speed:{int(speed)} {status}"

            visualizer.draw_center(frame, cx, cy)
            counter.update(track_id, cy, cls, class_names)
            visualizer.draw_box(frame, x1, y1, x2, y2, color, label)

            dashboard.update(track_id, class_names[cls], speed, status)

    violation_detector.cleanup_inactive_tracks(active_ids)
    dashboard.cleanup(active_ids)

    visualizer.draw_ui(frame, counter.get_count(), paused)
    cv2.imshow("Vehicle Detection & Counting", frame)
    dashboard.show_window()

    key = cv2.waitKey(1) & 0xFF
    dashboard.process_key(key)

    if key == 27:
        break
    elif key == 32:
        paused = not paused

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nFinal count: {counter.get_count()} vehicles")
violation_detector.export_csv()
print(f"Output saved to: {output_path}")