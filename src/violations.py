import os
import time
import math
import csv
import json
import cv2

class ViolationDetector:
    def export_csv(self, filename="violations.csv"):
        if not self.violations:
            print("No violations to save.")
            return

        keys = self.violations[0].keys()
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.violations)

        print(f"✅ Violations saved to {filename}")

    def export_json(self, filename="violations.json"):
        if not self.violations:
            print("No violations to save.")
            return

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.violations, file, indent=2, ensure_ascii=False)

        print(f"✅ Violations saved to {filename}")

    def __init__(self, fps, speed_threshold_kmh=80, meters_per_pixel=0.05, evidence_dir="./evidence", evidence_enabled=True):
        self.fps = fps
        self.speed_threshold_kmh = speed_threshold_kmh
        self.meters_per_pixel = meters_per_pixel
        self.evidence_enabled = evidence_enabled

        self.prev_positions = {}
        self.prev_time = {}
        self.prev_speed = {}
        self.violations = []

        self.stationary_start = {}
        self.parking_threshold = 5
        self.smoothing_alpha = 0.25
        self.min_movement_pixels = 2

        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        self.recent_violations = []

    def _save_evidence(self, track_id, vehicle_type, violation_type, frame, bbox):
        if not self.evidence_enabled or frame is None or bbox is None:
            return None, None

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_ID{track_id}_{violation_type}"

        frame_path = os.path.join(self.evidence_dir, base_name + ".jpg")
        crop_path = os.path.join(self.evidence_dir, base_name + "_crop.jpg")

        cv2.imwrite(frame_path, frame)
        crop = frame[y1:y2, x1:x2]
        if crop.size:
            cv2.imwrite(crop_path, crop)
        else:
            crop_path = ""

        return frame_path.replace("\\", "/"), crop_path.replace("\\", "/")

    def _record_violation(self, violation):
        self.violations.append(violation)
        self.recent_violations.insert(0, violation)
        if len(self.recent_violations) > 10:
            self.recent_violations.pop()

    def check_illegal_parking(self, track_id, cx, cy, vehicle_type=None, frame=None, bbox=None):
        current_time = time.time()
        if track_id in self.prev_positions:
            prev_x, prev_y = self.prev_positions[track_id]
            distance = math.hypot(cx - prev_x, cy - prev_y)

            if distance < 5:
                if track_id not in self.stationary_start:
                    self.stationary_start[track_id] = current_time
                else:
                    duration = current_time - self.stationary_start[track_id]
                    if duration > self.parking_threshold:
                        frame_path, crop_path = self._save_evidence(
                            track_id,
                            vehicle_type or "unknown",
                            "illegal_parking",
                            frame,
                            bbox
                        )
                        violation = {
                            "id": track_id,
                            "type": vehicle_type or "unknown",
                            "violation": "illegal_parking",
                            "time": time.strftime("%H:%M:%S"),
                            "frame_path": frame_path,
                            "crop_path": crop_path
                        }
                        self._record_violation(violation)
                        print(f"Illegal Parking: ID {track_id} saved to {frame_path}")
                        return True, violation
            else:
                self.stationary_start.pop(track_id, None)

        return False, None

    def calculate_speed(self, track_id, cx, cy):
        current_time = time.time()

        if track_id in self.prev_positions and track_id in self.prev_time:
            prev_x, prev_y = self.prev_positions[track_id]
            prev_t = self.prev_time[track_id]

            distance_pixels = math.hypot(cx - prev_x, cy - prev_y)
            time_diff = current_time - prev_t

            if time_diff <= 0 or distance_pixels < self.min_movement_pixels:
                smoothed_speed = self.prev_speed.get(track_id, 0.0)
            else:
                distance_meters = distance_pixels * self.meters_per_pixel
                speed_m_s = distance_meters / time_diff
                speed_kmh = speed_m_s * 3.6
                previous_speed = self.prev_speed.get(track_id, speed_kmh)
                smoothed_speed = (
                    self.smoothing_alpha * speed_kmh +
                    (1 - self.smoothing_alpha) * previous_speed
                )

            self.prev_speed[track_id] = smoothed_speed
            return smoothed_speed

        return 0.0

    def update(self, track_id, cx, cy, vehicle_type, frame=None, bbox=None):
        if track_id == -1:
            return False, 0.0, None

        speed_kmh = self.calculate_speed(track_id, cx, cy)

        self.prev_positions[track_id] = (cx, cy)
        self.prev_time[track_id] = time.time()

        if speed_kmh > self.speed_threshold_kmh:
            frame_path, crop_path = self._save_evidence(
                track_id,
                vehicle_type,
                "overspeed",
                frame,
                bbox
            )
            violation = {
                "id": track_id,
                "type": vehicle_type,
                "violation": "overspeed",
                "speed_kmh": round(speed_kmh, 2),
                "time": time.strftime("%H:%M:%S"),
                "frame_path": frame_path,
                "crop_path": crop_path
            }
            self._record_violation(violation)
            print(f"Overspeed detected: ID {track_id} | Speed: {speed_kmh:.2f} km/h")
            return True, speed_kmh, violation

        return False, speed_kmh, None

    def cleanup_inactive_tracks(self, active_ids):
        stale_ids = set(self.prev_positions) - set(active_ids)
        for track_id in stale_ids:
            self.prev_positions.pop(track_id, None)
            self.prev_time.pop(track_id, None)
            self.prev_speed.pop(track_id, None)
            self.stationary_start.pop(track_id, None)

    def get_logs(self):
        return self.violations

    def get_recent_violations(self, limit=5):
        return self.recent_violations[:limit]