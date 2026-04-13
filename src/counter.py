class VehicleCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.count_vehicle = 0
        self.counted_ids = set()
        self.prev_y = {}

    def update(self, track_id, cy, cls, class_names):
        if track_id != -1:
            if track_id in self.prev_y:
                prev_cy = self.prev_y[track_id]

                if prev_cy < self.line_y and cy >= self.line_y:
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        self.count_vehicle += 1
                        print(f"Vehicle counted: {class_names[cls]} (ID: {track_id}) - Total: {self.count_vehicle}")

            self.prev_y[track_id] = cy

    def get_count(self):
        return self.count_vehicle