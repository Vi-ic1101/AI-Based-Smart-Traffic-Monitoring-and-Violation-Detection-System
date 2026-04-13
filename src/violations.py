import time
import math
import csv

class ViolationDetector:
    def export_csv(self, filename="violations.csv"):
        if not self.violations:
            print("No violations to save.")
            return

        keys = self.violations[0].keys()

        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.violations)

        print(f"✅ Violations saved to {filename}")

    def __init__(self, fps, speed_threshold_kmh=80, meters_per_pixel=0.05):
        self.fps = fps
        self.speed_threshold_kmh = speed_threshold_kmh
        self.meters_per_pixel = meters_per_pixel

        self.prev_positions = {}
        self.prev_time = {}
        self.prev_speed = {}         # track_id → last smoothed speed
        self.violations = []

        self.stationary_start = {}
        self.parking_threshold = 5
        self.smoothing_alpha = 0.25  # smaller = smoother
        self.min_movement_pixels = 2

    def check_illegal_parking(self, track_id, cx, cy):
        current_time = time.time()

        if track_id in self.prev_positions:
            prev_x, prev_y = self.prev_positions[track_id]

            # Movement distance
            distance = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)

            # If barely moved → considered stationary
            if distance < 5:
                if track_id not in self.stationary_start:
                    self.stationary_start[track_id] = current_time
                else:
                    duration = current_time - self.stationary_start[track_id]

                    if duration > self.parking_threshold:
                        violation = {
                            "id": track_id,
                            "type": "unknown",
                            "violation": "illegal_parking",
                            "time": time.strftime("%H:%M:%S")
                        }
                        self.violations.append(violation)
                        print(f"Illegal Parking: ID {track_id}")
                        return True
            else:
                self.stationary_start.pop(track_id, None)

        return False

    def calculate_speed(self, track_id, cx, cy):
        current_time = time.time()

        if track_id in self.prev_positions and track_id in self.prev_time:
            prev_x, prev_y = self.prev_positions[track_id]
            prev_t = self.prev_time[track_id]

            distance_pixels = math.hypot(cx - prev_x, cy - prev_y)
            time_diff = current_time - prev_t
            if time_diff <= 0 or distance_pixels < self.min_movement_pixels:
                smoothed_speed = 0.0
            else:
                distance_meters = distance_pixels * self.meters_per_pixel
                speed_m_s = distance_meters / time_diff
                speed_kmh = speed_m_s * 3.6
                previous_speed = self.prev_speed.get(track_id, speed_kmh)
                smoothed_speed = (
                    self.smoothing_alpha * speed_kmh
                    + (1 - self.smoothing_alpha) * previous_speed
                )

            self.prev_speed[track_id] = smoothed_speed
            return smoothed_speed

        return 0.0

    def update(self, track_id, cx, cy, vehicle_type):
        speed_kmh = self.calculate_speed(track_id, cx, cy)

        self.prev_positions[track_id] = (cx, cy)
        self.prev_time[track_id] = time.time()

        if speed_kmh > self.speed_threshold_kmh:
            violation = {
                "id": track_id,
                "type": vehicle_type,
                "violation": "overspeed",
                "speed_kmh": round(speed_kmh, 2),
                "time": time.strftime("%H:%M:%S")
            }

            self.violations.append(violation)
            print(f"⚠️ Overspeed detected: ID {track_id} | Speed: {speed_kmh:.2f} km/h")

            return True, speed_kmh

        return False, speed_kmh

    def cleanup_inactive_tracks(self, active_ids):
        stale_ids = set(self.prev_positions) - set(active_ids)
        for track_id in stale_ids:
            self.prev_positions.pop(track_id, None)
            self.prev_time.pop(track_id, None)
            self.prev_speed.pop(track_id, None)
            self.stationary_start.pop(track_id, None)

    def get_logs(self):
        return self.violations