from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4)[0]

        if results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
        else:
            ids = []

        return results.boxes, ids