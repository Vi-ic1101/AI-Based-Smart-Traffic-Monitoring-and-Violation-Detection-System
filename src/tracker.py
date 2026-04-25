import torch
from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path, device=None):
        # Load YOLO model
        self.model = YOLO(model_path)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Set device for inference
        try:
            self.model.to(self.device)
            print(f"   ✅ YOLO Model loaded on {self.device.upper()} device")
        except Exception as e:
            print(f"   ⚠️  Failed to load on {self.device}: {e}")
            print(f"   ↻ Falling back to CPU")
            self.device = 'cpu'
            self.model.to('cpu')

    def track(self, frame, classes=None):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            classes=classes
        )[0]
        return results