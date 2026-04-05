from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

# Output video
out = cv2.VideoWriter("output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20,
                      (640, 480))

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    frame = cv2.resize(frame, (1150, 1030))

    # Run detection
    results = model.track(frame, persist=True)[0]
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
            # color=color_map[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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