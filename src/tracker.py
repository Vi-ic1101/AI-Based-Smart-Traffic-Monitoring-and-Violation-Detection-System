from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4
        )[0]
        return results