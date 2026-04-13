import os
import time
import json
import cv2
import numpy as np

class DashboardData:
    def __init__(self, state_path="./data/dashboard_state.json"):
        self.state_path = os.path.abspath(state_path)
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

        self.entries = {}
        self.stats = {
            "total_vehicles": 0,
            "overspeed": 0,
            "parking": 0,
            "start_time": time.time(),
            "history": [],
            "last_history_time": 0
        }
        self.recent_violations = []
        self.save_state()

    def update(self, track_id, vehicle_type, speed_kmh, status, violation_record=None):
        track_id = int(track_id)
        track_key = str(track_id)

        self.entries[track_key] = {
            "type": vehicle_type,
            "speed": round(speed_kmh, 1),
            "status": status,
            "updated": time.strftime("%H:%M:%S")
        }

        if violation_record is not None:
            violation_record["id"] = int(violation_record["id"])
            if violation_record["violation"] == "overspeed":
                self.stats["overspeed"] += 1
            elif violation_record["violation"] == "illegal_parking":
                self.stats["parking"] += 1

            self.recent_violations.insert(0, violation_record)
            if len(self.recent_violations) > 12:
                self.recent_violations.pop()

        self.save_state()

    def cleanup(self, active_ids):
        active_keys = {str(int(track_id)) for track_id in active_ids}
        stale_ids = [tid for tid in self.entries if tid not in active_keys]
        for tid in stale_ids:
            self.entries.pop(tid, None)
        self.save_state()

    def set_total_vehicles(self, total):
        self.stats["total_vehicles"] = total
        now = time.time()
        if now - self.stats.get("last_history_time", 0) > 1.0:
            timestamp = time.strftime("%H:%M:%S")
            self.stats["history"].append({"time": timestamp, "count": total})
            if len(self.stats["history"]) > 20:
                self.stats["history"].pop(0)
            self.stats["last_history_time"] = now
        self.save_state()

    def clear(self):
        self.entries.clear()
        self.recent_violations.clear()
        self.stats["overspeed"] = 0
        self.stats["parking"] = 0
        self.stats["history"].clear()
        self.save_state()

    def save_state(self):
        payload = {
            "entries": self.entries,
            "stats": self.stats,
            "recent_violations": self.recent_violations,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def get_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "entries": {},
            "stats": {
                "total_vehicles": 0,
                "overspeed": 0,
                "parking": 0,
                "history": []
            },
            "recent_violations": []
        }

class Dashboard:
    def __init__(self, max_entries=6, width=420, height=320):
        self.max_entries = max_entries
        self.width = width
        self.height = height
        self.entries = {}
        self.show_panel = True
        self.panel_name = "Vehicle Dashboard"
        self.show_violations_only = False
        self.control_text = [
            "Controls:",
            "D - toggle dashboard window",
            "V - filter violations only",
            "+ / - - change rows",
            "C - clear entries",
            "R - reset stats"
        ]

        self.stats = {
            "total_vehicles": 0,
            "overspeed": 0,
            "parking": 0,
            "start_time": time.time(),
            "history": []
        }
        self.recent_violations = []
        self.last_history_time = time.time()

        cv2.namedWindow(self.panel_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.panel_name, self.width, self.height)

    def update(self, track_id, vehicle_type, speed_kmh, status, violation_record=None):
        self.entries[track_id] = {
            "type": vehicle_type,
            "speed": round(speed_kmh, 1),
            "status": status,
            "updated": time.time()
        }

        if violation_record is not None:
            if violation_record["violation"] == "overspeed":
                self.stats["overspeed"] += 1
            elif violation_record["violation"] == "illegal_parking":
                self.stats["parking"] += 1

            self.recent_violations.insert(0, violation_record)
            if len(self.recent_violations) > 8:
                self.recent_violations.pop()

    def set_total_vehicles(self, total):
        self.stats["total_vehicles"] = total
        now = time.time()
        if now - self.last_history_time > 1.0:
            timestamp = time.strftime("%H:%M:%S")
            self.stats["history"].append((timestamp, total))
            if len(self.stats["history"]) > 10:
                self.stats["history"].pop(0)
            self.last_history_time = now

    def cleanup(self, active_ids):
        stale_ids = [tid for tid in self.entries if tid not in active_ids]
        for tid in stale_ids:
            self.entries.pop(tid, None)

    def clear(self):
        self.entries.clear()
        self.recent_violations.clear()

    def reset_stats(self):
        self.stats = {
            "total_vehicles": 0,
            "overspeed": 0,
            "parking": 0,
            "start_time": time.time(),
            "history": []
        }
        self.recent_violations.clear()

    def toggle(self):
        self.show_panel = not self.show_panel
        if not self.show_panel:
            cv2.destroyWindow(self.panel_name)

    def increase_rows(self):
        self.max_entries = min(self.max_entries + 1, 12)

    def decrease_rows(self):
        self.max_entries = max(self.max_entries - 1, 1)

    def toggle_violations(self):
        self.show_violations_only = not self.show_violations_only

    def _draw_bar_graph(self, canvas, x, y):
        total = self.stats["total_vehicles"]
        overspeed = self.stats["overspeed"]
        parking = self.stats["parking"]
        max_val = max(total, overspeed, parking, 1)
        bar_width = 80
        spacing = 20

        labels = [
            ("Total", total, (0, 255, 0)),
            ("Speed", overspeed, (0, 0, 255)),
            ("Park", parking, (255, 165, 0))
        ]
        for index, (label, value, color) in enumerate(labels):
            x_pos = x + index * (bar_width + spacing)
            height = int((value / max_val) * 80)
            cv2.rectangle(canvas, (x_pos, y + 80 - height), (x_pos + bar_width, y + 80), color, -1)
            cv2.putText(canvas, str(value), (x_pos + 5, y + 75 - height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (x_pos + 2, y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_count_history(self, canvas, x, y):
        history = self.stats["history"]
        if not history:
            return

        max_val = max(count for _, count in history)
        for i, (_, count) in enumerate(history):
            x0 = x + i * 30
            height = int((count / max_val) * 40) if max_val > 0 else 0
            cv2.rectangle(canvas, (x0, y + 40 - height), (x0 + 15, y + 40), (150, 200, 255), -1)
            cv2.putText(canvas, str(count), (x0, y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2.putText(canvas, "Vehicle Dashboard", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, f"Total Vehicles: {self.stats['total_vehicles']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Overspeed: {self.stats['overspeed']}", (220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Parking: {self.stats['parking']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)

        self._draw_bar_graph(canvas, 10, 80)
        self._draw_count_history(canvas, 280, 90)

        header_y = 170
        cv2.putText(canvas, "ID  Type     Speed  Status", (10, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        entries = list(self.entries.items())
        if self.show_violations_only:
            entries = [(tid, data) for tid, data in entries if data["status"] != "Normal"]

        for idx, (track_id, data) in enumerate(entries[: self.max_entries]):
            y = header_y + 22 + idx * 20
            text = f"{track_id:<3} {data['type']:<8} {data['speed']:>4}   {data['status']}"
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        violation_y = 245
        cv2.putText(canvas, "Recent Violations:", (10, violation_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        for idx, violation in enumerate(self.recent_violations[:3]):
            text = f"{violation['violation']} ID:{violation['id']} {violation.get('speed_kmh', '')}"
            y = violation_y + 18 + idx * 18
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        for idx, line in enumerate(self.control_text):
            cv2.putText(canvas, line, (220, self.height - 80 + idx * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        return canvas

    def show_window(self):
        if self.show_panel:
            cv2.imshow(self.panel_name, self.render())

    def process_key(self, key):
        if key == ord("d"):
            self.toggle()
        elif key == ord("v"):
            self.toggle_violations()
        elif key == ord("+") or key == ord("="):
            self.increase_rows()
        elif key == ord("-"):
            self.decrease_rows()
        elif key == ord("c"):
            self.clear()
        elif key == ord("r"):
            self.reset_stats()