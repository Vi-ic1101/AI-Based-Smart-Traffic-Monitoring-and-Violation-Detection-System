"""
Scene Preprocessor: Analyzes frame before main processing loop
Detects:
1. Traffic light locations and positions (re-analyzed every 10 seconds)
2. Pedestrian crossing zones (zebra crossing patterns ONLY)
"""

import cv2
import numpy as np
import importlib
from tracker import VehicleTracker


class ScenePreprocessor:
    """Preprocesses scene to detect static elements (traffic lights, crossings)"""
    
    def __init__(self, model_path="./models/yolov8n.pt"):
        """Initialize with YOLO model for vehicle detection"""
        self.tracker = VehicleTracker(model_path)
        self.traffic_lights = []  # List of traffic lights detected
        self.pedestrian_zones = []  # List of bounding boxes for zebra crossings
        self.analyzed = False
        
        # Traffic light analysis scheduling
        self.light_analysis_interval = 300  # Re-analyze every 300 frames (~10 sec at 30fps)
    
    def detect_traffic_lights(self, frame):
        """
        Detect traffic lights using HSV color ranges
        ONLY in typical traffic light regions (top-center, sides)
        Only detect actively lit lights (red, green, yellow)
        Returns: list of (x, y, radius, color) tuples
        """
        print("🚦 Analyzing traffic lights (state-change zones only)...")
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_h, frame_w = frame.shape[:2]
        traffic_lights = []
        
        # Define typical traffic light regions (where lights usually are)
        roi_regions = [
            # Top-center region
            {"x1": int(frame_w * 0.3), "y1": 0, "x2": int(frame_w * 0.7), "y2": int(frame_h * 0.2)},
            # Top-left region
            {"x1": 0, "y1": 0, "x2": int(frame_w * 0.25), "y2": int(frame_h * 0.25)},
            # Top-right region
            {"x1": int(frame_w * 0.75), "y1": 0, "x2": frame_w, "y2": int(frame_h * 0.25)},
            # Mid-left region (pole-mounted)
            {"x1": 0, "y1": int(frame_h * 0.2), "x2": int(frame_w * 0.15), "y2": int(frame_h * 0.5)},
            # Mid-right region (pole-mounted)
            {"x1": int(frame_w * 0.85), "y1": int(frame_h * 0.2), "x2": frame_w, "y2": int(frame_h * 0.5)},
        ]
        
        # Define strict HSV ranges for traffic light colors
        # RED - traffic light specific (saturated red)
        lower_red1 = np.array([0, 100, 120])
        upper_red1 = np.array([8, 255, 255])
        lower_red2 = np.array([172, 100, 120])
        upper_red2 = np.array([180, 255, 255])
        
        # GREEN - traffic light specific (bright green)
        lower_green = np.array([40, 100, 120])
        upper_green = np.array([80, 255, 255])
        
        # YELLOW - traffic light specific (bright yellow)
        lower_yellow = np.array([20, 100, 120])
        upper_yellow = np.array([30, 255, 255])
        
        # Scan each typical traffic light region
        for roi_def in roi_regions:
            x1, y1, x2, y2 = roi_def["x1"], roi_def["y1"], roi_def["x2"], roi_def["y2"]
            roi_hsv = hsv[y1:y2, x1:x2]
            
            if roi_hsv.size == 0:
                continue
            
            # Create masks for each color
            mask_red = cv2.inRange(roi_hsv, lower_red1, upper_red1) | cv2.inRange(roi_hsv, lower_red2, upper_red2)
            mask_green = cv2.inRange(roi_hsv, lower_green, upper_green)
            mask_yellow = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)
            
            # Process each color
            for color_name, mask in [("RED", mask_red), ("GREEN", mask_green), ("YELLOW", mask_yellow)]:
                if cv2.countNonZero(mask) < 100:  # Need sufficient colored pixels
                    continue
                
                # Morphological operations to clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours in this ROI
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 80 or area > 3000:  # Traffic light size range
                        continue
                    
                    # Fit circle to contour
                    (x_roi, y_roi), radius = cv2.minEnclosingCircle(contour)
                    
                    if radius < 4 or radius > 45:  # Traffic light typical radius
                        continue
                    
                    # Check circularity
                    if radius > 0:
                        area_circle = np.pi * radius ** 2
                        circularity = area / area_circle
                        if circularity < 0.65:  # Must be fairly circular
                            continue
                    
                    # Convert back to frame coordinates
                    x = int(x_roi) + x1
                    y = int(y_roi) + y1
                    
                    # Verify brightness (lights are bright)
                    roi_check = hsv[max(0, y-int(radius)-2):min(frame_h, y+int(radius)+2),
                                   max(0, x-int(radius)-2):min(frame_w, x+int(radius)+2)]
                    if roi_check.size > 0:
                        avg_brightness = np.mean(roi_check[:, :, 2])
                        if avg_brightness < 120:  # Too dark to be actively lit
                            continue
                    
                    traffic_lights.append({
                        "x": x,
                        "y": y,
                        "radius": int(radius),
                        "color": color_name,
                    })
        
        # Remove duplicates (lights detected in overlapping regions)
        unique_lights = self._deduplicate_detections(traffic_lights, distance_threshold=35)
        
        print(f"   ✅ Found {len(unique_lights)} active traffic lights: {[(tl['color'], (tl['x'], tl['y'])) for tl in unique_lights]}")
        return unique_lights
    
    def _deduplicate_detections(self, detections, distance_threshold=35):
        """Remove duplicate detections that are too close"""
        if not detections:
            return []
        
        unique = []
        for det in detections:
            is_duplicate = False
            for existing in unique:
                dist = np.sqrt((det["x"] - existing["x"])**2 + (det["y"] - existing["y"])**2)
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(det)
        
        return unique
    
    def detect_pedestrian_crossings(self, frame):
        """
        Detect pedestrian crossing zones using ONLY zebra stripe pattern detection
        NO YOLO pedestrian detection
        Returns: list of crossing zones (bounding boxes)
        """
        print("🚶 Analyzing pedestrian crossing areas (zebra patterns ONLY)...")
        
        zebra_zones = self._detect_zebra_pattern(frame)
        
        print(f"   ✅ Found {len(zebra_zones)} zebra crossing zones")
        return zebra_zones
    
    def _detect_zebra_pattern(self, frame):
        """
        Detect white striped zebra crossing patterns
        Zebra crossings have characteristic white horizontal stripes
        """
        zones = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect bright areas (zebra stripes are white)
        # Use high threshold to only get very bright pixels
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Look for horizontal stripe patterns (characteristic of zebra crossings)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
        stripes = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_h)
        
        # Dilate slightly to connect nearby stripes
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
        stripes = cv2.dilate(stripes, kernel_dilate, iterations=2)
        
        # Find contours of stripe patterns
        contours, _ = cv2.findContours(stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Zebra crossings are large areas (multiple stripes)
            if area < 2000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Zebra crossings have specific proportions
            if w < 80 or h < 40:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            # Typical zebra crossing aspect ratio is 2:1 to 4:1
            if aspect_ratio < 1.5 or aspect_ratio > 6:
                continue
            
            # Expand bounding box to create zone with margin
            margin = 30
            zones.append({
                "x1": max(0, x - margin),
                "y1": max(0, y - margin),
                "x2": min(frame.shape[1], x + w + margin),
                "y2": min(frame.shape[0], y + h + margin)
            })
        
        # Merge overlapping zones
        zones = self._merge_overlapping_boxes(zones)
        return zones
    
    def _merge_overlapping_boxes(self, boxes):
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            merged_box = box1.copy()
            used.add(i)
            
            # Find all overlapping boxes
            for j, box2 in enumerate(boxes[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Calculate intersection
                x1_min = max(merged_box["x1"], box2["x1"])
                y1_min = max(merged_box["y1"], box2["y1"])
                x2_max = min(merged_box["x2"], box2["x2"])
                y2_max = min(merged_box["y2"], box2["y2"])
                
                if x2_max > x1_min and y2_max > y1_min:
                    # Boxes overlap, merge them
                    merged_box["x1"] = min(merged_box["x1"], box2["x1"])
                    merged_box["y1"] = min(merged_box["y1"], box2["y1"])
                    merged_box["x2"] = max(merged_box["x2"], box2["x2"])
                    merged_box["y2"] = max(merged_box["y2"], box2["y2"])
                    used.add(j)
            
            merged.append(merged_box)
        
        return merged
    
    def analyze_scene(self, frame, frame_count=0):
        """
        Main preprocessing method: Analyze frame for traffic lights and crossings
        
        Traffic lights: Re-analyzed every 10 seconds (every 300 frames at 30fps)
        Pedestrian zones: Analyzed once at start (static)
        
        Args:
            frame: Current video frame
            frame_count: Current frame number (for periodic re-analysis)
        """
        # Re-analyze traffic lights every ~10 seconds
        if frame_count % self.light_analysis_interval == 0:
            if not self.analyzed:
                print("\n" + "=" * 80)
                print("SCENE ANALYSIS - Analyzing frame for static elements")
                print("=" * 80)
            else:
                print(f"\n📋 RE-ANALYZING TRAFFIC LIGHTS (Frame {frame_count})...")
            
            self.traffic_lights = self.detect_traffic_lights(frame)
        
        # Pedestrian zones are static, only detect once
        if not self.analyzed:
            self.pedestrian_zones = self.detect_pedestrian_crossings(frame)
            self.analyzed = True
            
            print("=" * 80)
            print(f"✅ SCENE ANALYSIS COMPLETE")
            print(f"   Traffic Lights: {len(self.traffic_lights)} (re-analyzed every 10 seconds)")
            print(f"   Zebra Crossings: {len(self.pedestrian_zones)} (static)")
            print("=" * 80 + "\n")
        elif frame_count % self.light_analysis_interval == 0 and frame_count > 0:
            print(f"   Updated: {len(self.traffic_lights)} traffic lights found")
            print("=" * 80 + "\n")
        
        return self.traffic_lights, self.pedestrian_zones
    
    def get_traffic_light_regions(self):
        """Return list of traffic light detection regions"""
        return self.traffic_lights
    
    def get_pedestrian_zones(self):
        """Return list of pedestrian crossing zones (zebra crossings only)"""
        return self.pedestrian_zones
