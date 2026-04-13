import cv2

class Visualizer:
    def __init__(self, line_y, frame_width, frame_height):
        self.line_y = line_y
        self.frame_width = frame_width
        self.frame_height = frame_height

    def draw_line(self, frame):
        cv2.line(frame, (0, self.line_y), (self.frame_width, self.line_y), (255, 0, 255), 2)

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