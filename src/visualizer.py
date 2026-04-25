import cv2

class Visualizer:
    def __init__(self, line_y, frame_width, frame_height):
        self.line_y = line_y
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.traffic_lights = []  # Detected traffic light regions
        self.pedestrian_zones = []  # Detected pedestrian crossing zones

    def set_traffic_lights(self, traffic_lights):
        """Set detected traffic light positions for visualization"""
        self.traffic_lights = traffic_lights
    
    def set_pedestrian_zones(self, pedestrian_zones):
        """Set detected pedestrian crossing zones for visualization"""
        self.pedestrian_zones = pedestrian_zones

    def draw_line(self, frame):
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), (255, 0, 255), 2)

    def draw_traffic_light_regions(self, frame):
        """Draw detected traffic light positions on frame"""
        if not self.traffic_lights:
            return
        
        for tl in self.traffic_lights:
            x, y, radius = tl["x"], tl["y"], tl["radius"]
            color_name = tl["color"]
            
            # Color mapping
            color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "YELLOW": (0, 255, 255)}
            color = color_map.get(color_name, (255, 255, 255))
            
            # Draw circle for traffic light
            cv2.circle(frame, (x, y), radius + 5, color, 3)
            
            # Draw label
            cv2.putText(frame, f"TL:{color_name}", (x - 30, y - radius - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_pedestrian_zones(self, frame):
        """Draw detected pedestrian crossing zones on frame"""
        if not self.pedestrian_zones:
            return
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        for zone in self.pedestrian_zones:
            x1, y1, x2, y2 = zone["x1"], zone["y1"], zone["x2"], zone["y2"]
            
            # Draw semi-transparent rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Label
            cv2.putText(frame, "CROSSING", (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    

    def draw_box(self, frame, x1, y1, x2, y2, color, label):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_center(self, frame, cx, cy):
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    def draw_ui(self, frame, count, paused):
        count_text = f"Count: {count}"
        (w, h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (45, 20), (55 + w, 60), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        status_text = "PAUSED" if paused else "PLAYING"
        status_color = (0, 165, 255) if paused else (0, 255, 0)
        (w, h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (45, 70), (55 + w, 105), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        instructions = "[SPACE] Pause/Resume | [ESC] Exit"
        (w, h), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (45, self.frame_height - 40),
                      (55 + w, self.frame_height - 10), (0, 0, 0), -1)
        cv2.putText(frame, instructions, (50, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)