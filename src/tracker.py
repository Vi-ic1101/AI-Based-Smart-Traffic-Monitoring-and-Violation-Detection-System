from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path, device='gpu'):
        # Load YOLO model on specified device (GPU for faster inference)
        self.model = YOLO(model_path)
        self.device = device
        # Set device for inference
        try:
            self.model.to(device)
            print(f"   ✅ YOLO Model loaded on {device.upper()} device")
        except Exception as e:
            print(f"   ⚠️  Failed to load on {device}: {e}")
            print(f"   ↻ Falling back to CPU")
            self.device = 'cpu'
            self.model.to('cpu')

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4
        )[0]
        return results