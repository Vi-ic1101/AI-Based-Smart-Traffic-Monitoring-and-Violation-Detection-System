from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic video.mp4")

# Output video
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20,
                      (1150,1030))

class_names={
    2: "car",
    3: "motorcycle",            
    5: "bus",
    7: "heavy"
}

# Vehicle classes (COCO dataset)
vehicle_classes = [2, 3, 5, 7]

color_map={

    2: (0,255,0), #car
    3: (255,0,0), #motorcycle     
    5: (0,0,255),#bus
    7: (255,255,0)#truck

}

line_y=570
offset=10
countVehicle=0

counted_ids=set()

prev_y={}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (1150, line_y+330))

    cv2.line(frame,(0,line_y),(1150,line_y),(255,0,255),2)

    # Run detection
    results = model.track(frame, persist=True,tracker="bytetrack.yaml", conf=0.4)[0]
    if results.boxes.id is not None:
         ids=results.boxes.id.cpu().numpy().astype(int)
    else:
         ids=[]
    boxes=results.boxes

    for i,box in enumerate(boxes):

        cls = int(box.cls[0])
        conf=float(box.conf[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id=ids[i] if len(ids) > i else -1

            if cls in [5, 7]:
                    label = f"Heavy ID:{track_id} {conf:.2f}"
                    color = (0,0,255)  # Red for heavy vehicles
            else:
                    label = f"{class_names[cls]} ID:{track_id} {conf:.2f}"
                    color = color_map[cls]
            # Draw box
            # Center of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

            # Check if vehicle crosses line
            if track_id in prev_y:
                prev_cy=prev_y[track_id]
                if prev_cy < line_y and cy >= line_y:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        countVehicle += 1
            prev_y[track_id]=cy
                    
            # color=color_map[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Count: {countVehicle}", (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show video
    cv2.imshow("Vehicle Detection", frame)

    # Save output
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()