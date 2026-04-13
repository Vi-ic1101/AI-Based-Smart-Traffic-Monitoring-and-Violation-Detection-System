import cv2
import time
import numpy as np

class Dashboard:
    def __init__(self, max_entries=6, width=380, height=260):
        self.max_entries = max_entries
        self.width = width
        self.height = height
        self.entries = {}
        self.show_panel = True
        self.panel_name = "Vehicle Dashboard"
        self.control_text = [
            "Controls:",
            "D - toggle dashboard",
            "+ / - - change rows",
            "C - clear dashboard"
        ]
        cv2.namedWindow(self.panel_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.panel_name, self.width, self.height)

    def update(self, track_id, vehicle_type, speed_kmh, status):
        self.entries[track_id] = {
            "type": vehicle_type,
            "speed": round(speed_kmh, 1),
            "status": status,
            "updated": time.time()
        }

    def cleanup(self, active_ids):
        stale_ids = [tid for tid in self.entries if tid not in active_ids]
        for tid in stale_ids:
            self.entries.pop(tid, None)

    def clear(self):
        self.entries.clear()

    def toggle(self):
        self.show_panel = not self.show_panel
        if not self.show_panel:
            cv2.destroyWindow(self.panel_name)

    def increase_rows(self):
        self.max_entries = min(self.max_entries + 1, 12)

    def decrease_rows(self):
        self.max_entries = max(self.max_entries - 1, 1)

    def render(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        header_y = 30
        row_y = 55

        cv2.putText(canvas, "Vehicle Dashboard", (10, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, "ID  Type       Speed  Status", (10, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for idx, (track_id, data) in enumerate(list(self.entries.items())[:self.max_entries]):
            y = row_y + (idx + 1) * 25
            text = f"{track_id:<3} {data['type']:<9} {data['speed']:>5}   {data['status']}"
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        for idx, line in enumerate(self.control_text):
            cv2.putText(canvas, line, (10, self.height - 80 + idx * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        return canvas

    def show_window(self):
        if self.show_panel:
            cv2.imshow(self.panel_name, self.render())

    def process_key(self, key):
        if key == ord("d"):
            self.toggle()
        elif key == ord("+") or key == ord("="):
            self.increase_rows()
        elif key == ord("-"):
            self.decrease_rows()
        elif key == ord("c"):
            self.clear()