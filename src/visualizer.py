import cv2

class Visualizer:
    def __init__(self, line_y):
        self.line_y = line_y

    def draw_line(self, frame):
        cv2.line(frame, (0, self.line_y), (1150, self.line_y), (255,0,255), 2)

    def draw_box(self, frame, x1, y1, x2, y2, color):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def draw_center(self, frame, cx, cy):
        cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

    def draw_text(self, frame, text, x, y, color=(0,255,0)):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_ui(self, frame, count):
        cv2.putText(frame, f"Count: {count}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, "Press SPACE to Pause/Resume", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)