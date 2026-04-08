class VehicleCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.count = 0
        self.counted_ids = set()
        self.prev_y = {}

    def update(self, track_id, cy):
        if track_id in self.prev_y:
            prev_cy = self.prev_y[track_id]

            if (prev_cy < self.line_y and cy >= self.line_y) or \
               (prev_cy > self.line_y and cy <= self.line_y):

                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.count += 1

        self.prev_y[track_id] = cy
        return self.count