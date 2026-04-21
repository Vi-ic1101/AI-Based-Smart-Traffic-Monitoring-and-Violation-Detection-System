import cv2
import os
import json
import numpy as np

class TrafficSceneAnalyzer:
    def __init__(self, frame_width=1150, frame_height=840, sample_interval=30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sample_interval = sample_interval

    def _normalize_frame(self, frame):
        if frame is None:
            return None
        if self.frame_width and self.frame_height:
            return cv2.resize(frame, (self.frame_width, self.frame_height))
        return frame

    def detect_traffic_lights(self, frame):
        frame = self._normalize_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = {
            'red': self._color_mask(hsv, [(0, 100, 100), (10, 255, 255)], [(160, 100, 100), (179, 255, 255)]),
            'yellow': self._color_mask(hsv, [(18, 100, 100), (30, 255, 255)]),
            'green': self._color_mask(hsv, [(40, 50, 50), (90, 255, 255)])
        }

        traffic_lights = []
        for label, mask in masks.items():
            contours = self._find_contours(mask)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 15 or h < 15 or w > 150 or h > 150:
                    continue
                traffic_lights.append({
                    'label': label,
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'area': int(w * h)
                })

        traffic_lights = sorted(traffic_lights, key=lambda x: x['area'], reverse=True)
        return traffic_lights[:6]

    def _color_mask(self, hsv, range1, range2=None):
        lower1 = np.array(range1[0], dtype=np.uint8)
        upper1 = np.array(range1[1], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        if range2 is not None:
            lower2 = np.array(range2[0], dtype=np.uint8)
            upper2 = np.array(range2[1], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _find_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_road_structure(self, frame):
        frame = self._normalize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=20)
        if lines is None:
            return {'intersection_type': 'unknown', 'line_count': 0, 'major_directions': []}

        horiz = 0
        vert = 0
        diag = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2((y2 - y1), (x2 - x1)))
            if angle < np.pi / 8:
                horiz += 1
            elif angle > 3 * np.pi / 8:
                vert += 1
            else:
                diag += 1

        intersection_type = 'unknown'
        if horiz >= 2 and vert >= 2:
            intersection_type = '4-way intersection'
        elif horiz >= 1 and vert >= 1 and diag >= 1:
            intersection_type = 'T/Y intersection'
        elif horiz >= 2 or vert >= 2:
            intersection_type = '3-way intersection'

        return {
            'intersection_type': intersection_type,
            'line_count': int(len(lines)),
            'horizontal_lines': int(horiz),
            'vertical_lines': int(vert),
            'diagonal_lines': int(diag)
        }

    def map_traffic_lights(self, frame, light_regions, road_structure):
        frame = self._normalize_frame(frame)
        overlay = frame.copy()
        for item in light_regions:
            x1, y1, x2, y2 = item['bbox']
            color = (0, 0, 255) if item['label'] == 'red' else (0, 255, 255) if item['label'] == 'yellow' else (0, 255, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, item['label'].upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        if '4-way' in road_structure['intersection_type']:
            cv2.line(overlay, (center_x - 80, center_y), (center_x + 80, center_y), (255, 255, 255), 2)
            cv2.line(overlay, (center_x, center_y - 80), (center_x, center_y + 80), (255, 255, 255), 2)
        elif '3-way' in road_structure['intersection_type'] or 'T/Y' in road_structure['intersection_type']:
            cv2.line(overlay, (center_x - 80, center_y), (center_x + 80, center_y), (255, 255, 255), 2)
            cv2.line(overlay, (center_x, center_y), (center_x, center_y + 80), (255, 255, 255), 2)

        cv2.putText(overlay, f"Road Structure: {road_structure['intersection_type']}",
                    (10, self.frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    def initialize_violation_tracking(self, light_regions, road_structure):
        return {
            'traffic_lights': light_regions,
            'road_structure': road_structure,
            'rule_set': {
                '4-way intersection': ['straight', 'right_turn', 'left_turn'],
                'T/Y intersection': ['straight', 'turn']
            }
        }

    def analyze_video(self, source_path, max_samples=5):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source_path}")

        samples = 0
        light_accum = []
        structure_accum = []
        frame_id = 0

        while samples < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if frame_id % self.sample_interval != 0:
                continue

            frame = self._normalize_frame(frame)
            lights = self.detect_traffic_lights(frame)
            structure = self.detect_road_structure(frame)
            light_accum.append(lights)
            structure_accum.append(structure)
            samples += 1

        cap.release()

        final_lights = light_accum[0] if light_accum else []
        final_structure = structure_accum[0] if structure_accum else {'intersection_type': 'unknown', 'line_count': 0}
        return final_lights, final_structure

    def save_analysis(self, filename, analysis):
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scene analysis module for traffic video.')
    parser.add_argument('--source', default='./videos/output/output.mp4',
                        help='Video file or stream source for scene analysis')
    parser.add_argument('--output', default='./src/scene_analysis.json',
                        help='Path to save scene analysis results')
    parser.add_argument('--preview', action='store_true', help='Show preview with traffic light mapping')
    args = parser.parse_args()

    analyzer = TrafficSceneAnalyzer()
    lights, structure = analyzer.analyze_video(args.source)
    config = analyzer.initialize_violation_tracking(lights, structure)
    result = {
        'source': args.source,
        'traffic_lights': lights,
        'road_structure': structure,
        'violation_tracking': config
    }
    analyzer.save_analysis(args.output, result)

    print(f"Scene analysis complete. Results saved to: {args.output}")
    if args.preview and lights:
        cap = cv2.VideoCapture(args.source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                overlay = analyzer.map_traffic_lights(frame, lights, structure)
                cv2.imshow('Scene Analysis Preview', overlay)
                cv2.waitKey(0)
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
