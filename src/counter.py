class VehicleCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.count_vehicle = 0
        self.counted_ids = set()
        self.prev_y = {}
        self.pedestrian_zones = []  # Pedestrian (zebra) crossing zones
        self.crossed_zones = {}  # Track which vehicles have crossed which zones
    
    def set_pedestrian_zones(self, zones):
        """Set pedestrian crossing zones - vehicles are ONLY counted when crossing these zones"""
        self.pedestrian_zones = zones
        print(f"   ✅ Vehicle Counter: Configured for {len(zones)} zebra crossing zones")
    
    def is_in_zone(self, cx, cy, zone):
        """Check if point is within a zone"""
        return zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]
    
    def is_crossing_zone(self, track_id, cx, cy, prev_cx, prev_cy, zone):
        """Check if vehicle is crossing through a zone (entering from outside)"""
        # Was outside zone, now inside
        was_outside = not self.is_in_zone(prev_cx, prev_cy, zone)
        is_inside = self.is_in_zone(cx, cy, zone)
        
        return was_outside and is_inside

    def update(self, track_id, cy, cls, class_names, cx=None):
        """
        Update vehicle counter
        ONLY counts vehicles that cross zebra crossing zones
        """
        if track_id != -1 and cx is not None:
            # Only count vehicles crossing zebra crossing zones
            if self.pedestrian_zones:
                if track_id in self.prev_y:
                    prev_cx, prev_cy = self.prev_y.get(track_id, (cx, cy))
                else:
                    prev_cx, prev_cy = (cx, cy)
                
                # Check if vehicle is crossing any zebra zone
                for zone_idx, zone in enumerate(self.pedestrian_zones):
                    zone_key = f"{zone_idx}_{track_id}"
                    
                    # Check if vehicle is crossing into this zone
                    if self.is_crossing_zone(track_id, cx, cy, prev_cx, prev_cy, zone):
                        if zone_key not in self.crossed_zones:
                            # First time crossing this zone
                            self.crossed_zones[zone_key] = True
                            self.counted_ids.add(track_id)
                            self.count_vehicle += 1
                            print(f"✅ Vehicle counted at crossing: {class_names[cls]} (ID: {track_id}) - Total: {self.count_vehicle}")
                
            # Always update position
            self.prev_y[track_id] = (cx, cy)

    def get_count(self):
        return self.count_vehicle