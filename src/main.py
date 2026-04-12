from ultralytics import YOLO
import cv2

# Paths
video_path = "./videos/input/traffic_video.mp4"
output_path = "./videos/output/output.mp4"    
model_path = "./models/yolov8n.pt"

# Load YOLO model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:  # Fallback if FPS can't be read
    fps = 20

# Configuration
line_y = 570
frame_width = 1150
frame_height = line_y + 270

# Output video
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Class names and colors
class_names = {
    2: "car",
    3: "motorcycle",            
    5: "bus",
    7: "truck"  # Changed from "heavy" for clarity
}

# Vehicle classes (COCO dataset)
vehicle_classes = [2, 3, 5, 7]

color_map = {
    2: (0, 255, 0),    # Green - car
    3: (255, 0, 0),    # Blue - motorcycle     
    5: (0, 0, 255),    # Red - bus
    7: (255, 255, 0)   # Cyan - truck
}

# Tracking variables
count_vehicle = 0
counted_ids = set()
prev_y = {}
paused = False
frame = None

print(f"Processing video at {fps} FPS...")
print("Controls: SPACE = Pause/Resume | ESC = Exit")

while True:
    # Read frame only if not paused
    if not paused:
        ret, frame = cap.read()        
        if not ret:
            print("End of video or error reading frame")
            break
    
    if frame is None:
        continue
    
    # Resize for performance
    frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Draw counting line
    cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 255), 2)
    
    # Run detection and tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4)[0]
    
    # Get tracking IDs
    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
    else:
        ids = []
    
    boxes = results.boxes
    
    # Process each detected object
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Only process vehicle classes
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = ids[i] if len(ids) > i else -1
            
            # Get color and label
            color = color_map.get(cls, (255, 255, 255))  # Default to white if class not found
            label = f"{class_names[cls]} ID:{track_id} {conf:.2f}"
            
            # Calculate center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
            # Check if vehicle crosses the counting line (downward crossing)
            if track_id != -1:  # Valid tracking ID
                if track_id in prev_y:
                    prev_cy = prev_y[track_id]
                    # Detect crossing from above to below the line
                    if prev_cy < line_y and cy >= line_y:
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            count_vehicle += 1
                            print(f"Vehicle counted: {class_names[cls]} (ID: {track_id}) - Total: {count_vehicle}")
                
                # Update previous position
                prev_y[track_id] = cy
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with matching color
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color,  # Use the same color as bounding box
                2
            )
    
    # Draw info panel with background
    # Vehicle count
    count_text = f"Count: {count_vehicle}"
    (text_width, text_height), baseline = cv2.getTextSize(
        count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    )
    cv2.rectangle(frame, (45, 20), (55 + text_width, 60), (0, 0, 0), -1)
    cv2.putText(
        frame, 
        count_text, 
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    # Status indicator
    status_text = "PAUSED" if paused else "PLAYING"
    status_color = (0, 165, 255) if paused else (0, 255, 0)  # Orange if paused, green if playing
    (status_width, status_height), _ = cv2.getTextSize(
        status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.rectangle(frame, (45, 70), (55 + status_width, 105), (0, 0, 0), -1)
    cv2.putText(
        frame,
        status_text,  
        (50, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        status_color, 
        2
    )
    
    # Instructions at bottom
    instructions = "[SPACE] Pause/Resume | [ESC] Exit"
    (inst_width, inst_height), _ = cv2.getTextSize(
        instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(frame, (45, frame_height - 40), (55 + inst_width, frame_height - 10), (0, 0, 0), -1)
    cv2.putText(
        frame,
        instructions,
        (50, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
    # Show video
    cv2.imshow("Vehicle Detection & Counting", frame)
    
    # Save output
    out.write(frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("Exiting...")
        break
    elif key == 32:  # SPACE key
        paused = not paused
        print(f"{'Paused' if paused else 'Resumed'}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nFinal count: {count_vehicle} vehicles")
print(f"Output saved to: {output_path}")
