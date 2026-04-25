import cv2
import numpy as np
from collections import defaultdict, deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class RedLightDetector:
    """
    Detects red-light jumping violations.
    Uses YOLO for traffic light detection with HSV color analysis fallback.
    """
    
    def __init__(self, frame_width=1150, frame_height=840, model_path="models/yolov8n.pt"):
        """
        Initialize red-light detector.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            model_path: Path to YOLO model for traffic light detection
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # YOLO model for traffic light detection
        self.yolo_model = None
        self.use_yolo = False
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(model_path)
                self.use_yolo = True
                print("   ✅ YOLO loaded for traffic light detection")
            except Exception as e:
                print(f"   ⚠️  YOLO loading failed ({e}), using HSV fallback")
        
        # Define traffic light region (typically top area of frame)
        # Can be customized based on actual camera position
        self.light_region = {
            'x1': int(frame_width * 0.35),
            'y1': int(frame_height * 0.05),
            'x2': int(frame_width * 0.65),
            'y2': int(frame_height * 0.15)
        }
        
        # Detected traffic light regions (set by preprocessor)
        self.detected_traffic_lights = []
        
        # Track light state and history
        self.light_state = 'green'  # green, red, yellow
        self.light_state_history = []
        self.state_change_frames = 0
        # Temporal smoothing buffer for stability (require 2 consecutive detections)
        self._recent_states = deque(maxlen=5)
        self._state_confirm_required = 2
        # Brightness and area thresholds for robust detection
        self._brightness_threshold = 120
        self._area_ratio_threshold = 0.008
        self._min_pixel_count = 40
        
        # YOLO confidence threshold
        self._yolo_conf_threshold = 0.5
        
        # Track vehicles crossing the stop line
        self.stop_line_y = int(frame_height * 0.5)  # Usually middle of frame
        self.vehicles_crossing = defaultdict(dict)
        self.red_light_violations = []
    
    def set_traffic_light_regions(self, traffic_lights):
        """Set detected traffic light regions for analysis"""
        self.detected_traffic_lights = traffic_lights
        if traffic_lights:
            print(f"   ✅ Red-Light Detector: Configured for {len(traffic_lights)} traffic lights")
        
    def detect_light_color(self, frame):
        """
        Detect traffic light color using YOLO detection with HSV color analysis.
        Prioritizes YOLO detection, falls back to HSV, defaults to green if uncertain.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            light_state: 'red' or 'green' (never 'unknown')
        """
        detection = 'green'  # Default fallback
        
        # Try YOLO detection first if available
        if self.use_yolo and self.yolo_model is not None:
            detection = self._detect_with_yolo(frame)
        
        # If YOLO didn't work or unavailable, use HSV detection
        if detection == 'green':
            detection = self._detect_with_hsv(frame)
        
        # Temporal smoothing: require multiple matching detections before switching
        self._recent_states.append(detection)
        if self._recent_states.count(detection) >= self._state_confirm_required:
            self.light_state = detection
        
        return self.light_state
    
    def _detect_with_yolo(self, frame):
        """
        Detect traffic light using YOLO model.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            light_state: 'red', 'green', or 'green' (default)
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=self._yolo_conf_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return 'green'
            
            result = results[0]
            
            # Filter detections for traffic light class
            # YOLO traffic light class ID is typically 9 or we can search by name
            traffic_lights = []
            for box in result.boxes:
                conf = float(box.conf)
                class_id = int(box.cls)
                
                # Class 9 in COCO is traffic light, but verify
                # Alternative: check class name
                if hasattr(result, 'names'):
                    class_name = result.names.get(class_id, '')
                    if 'traffic' in class_name.lower() or 'light' in class_name.lower() or class_id == 9:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        traffic_lights.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'class_id': class_id
                        })
                elif class_id == 9:  # Fallback: check class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    traffic_lights.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class_id': class_id
                    })
            
            if not traffic_lights:
                return 'green'
            
            # Analyze detected traffic lights for color
            color_votes = {'red': 0, 'green': 0}
            
            for tl in traffic_lights:
                x1, y1, x2, y2 = tl['bbox']
                roi = frame[max(0, y1):min(frame.shape[0], y2), 
                           max(0, x1):min(frame.shape[1], x2)]
                
                if roi.size == 0:
                    continue
                
                # Analyze HSV in detected region
                color = self._analyze_hsv_colors(roi)
                if color in color_votes:
                    color_votes[color] += 1
            
            # Return majority color
            if color_votes['red'] > color_votes['green']:
                return 'red'
            else:
                return 'green'
        
        except Exception as e:
            print(f"   ⚠️  YOLO detection error: {e}")
            return 'green'
    
    def _detect_with_hsv(self, frame):
        """
        Detect traffic light color using HSV analysis.
        Uses detected traffic light regions if available, otherwise uses default region.
        
        Returns:
            light_state: 'red' or 'green' (never 'unknown')
        """
        # Use detected traffic lights if available
        if self.detected_traffic_lights:
            detection = self._detect_from_regions(frame, self.detected_traffic_lights)
        else:
            # Fallback to default region
            detection = self._detect_from_region(frame, self.light_region)
        
        # Never return unknown - default to green
        if detection == 'unknown':
            return 'green'
        
        return detection
    
    def _detect_from_region(self, frame, region):
        """Detect light color from a single region"""
        # Extract region of interest for traffic light
        roi = frame[
            region['y1']:region['y2'],
            region['x1']:region['x2']
        ]
        
        if roi.size == 0:
            return 'green'
        
        # Create circular mask around center to focus on light bulb
        h, w = roi.shape[:2]
        cx_rel = w // 2
        cy_rel = h // 2
        radius = max(4, min(w, h) // 6)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx_rel, cy_rel), int(radius), 255, -1)
        
        return self._analyze_hsv_colors(roi, mask=mask)
    
    def _detect_from_regions(self, frame, traffic_lights):
        """Detect light color from multiple detected traffic light positions"""
        detected_colors = {'red': 0, 'green': 0, 'yellow': 0}
        
        for tl in traffic_lights:
            x, y, radius = tl["x"], tl["y"], tl["radius"]
            
            # Extract region around traffic light
            y1 = max(0, y - radius - 10)
            y2 = min(frame.shape[0], y + radius + 10)
            x1 = max(0, x - radius - 10)
            x2 = min(frame.shape[1], x + radius + 10)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Circular mask to focus on the lit area and reduce background influence
            h, w = roi.shape[:2]
            cx_rel = int((x - x1))
            cy_rel = int((y - y1))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx_rel, cy_rel), int(radius) + 3, 255, -1)
            
            color = self._analyze_hsv_colors(roi, mask=mask)
            if color in detected_colors:
                detected_colors[color] += 1
        
        # Return most frequently detected color among detected traffic lights
        # Default to green if no clear detection
        if detected_colors['red'] > 0 or detected_colors['green'] > 0:
            return 'red' if detected_colors['red'] > detected_colors['green'] else 'green'
        
        # If we detected yellow, treat as caution but lean green
        return 'green'
    
    def _analyze_hsv_colors(self, roi, mask=None):
        """
        Analyze HSV colors in a region of interest with an optional circular mask.
        Uses brightness filtering and area ratio checks to avoid background bias.
        Prevents truck brake lights and yellow backgrounds from causing false detections.
        Never returns 'unknown' - defaults to 'green'.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Color ranges (tuned for saturated/bright light bulbs) - more tolerant
        # Red: wraps around in HSV, so two ranges
        lower_red1 = np.array([0, 80, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 80, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Green: broader range to catch various green tones
        lower_green = np.array([30, 60, 100])
        upper_green = np.array([90, 255, 255])
        
        # Yellow: for reference but we'll lean red/green
        lower_yellow = np.array([15, 80, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Ensure mask is valid
        h, w = hsv.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Brightness filter to focus on lit areas (reduce background influence)
        v = hsv[:, :, 2]
        bright_mask = (v >= self._brightness_threshold).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_and(mask, bright_mask)
        
        red_final = cv2.bitwise_and(mask_red, mask_red, mask=combined_mask)
        green_final = cv2.bitwise_and(mask_green, mask_green, mask=combined_mask)
        yellow_final = cv2.bitwise_and(mask_yellow, mask_yellow, mask=combined_mask)
        
        red_pixels = cv2.countNonZero(red_final)
        green_pixels = cv2.countNonZero(green_final)
        yellow_pixels = cv2.countNonZero(yellow_final)
        
        masked_area = max(1, cv2.countNonZero(mask))
        red_ratio = red_pixels / masked_area
        green_ratio = green_pixels / masked_area
        yellow_ratio = yellow_pixels / masked_area
        
        ratios = {'red': red_ratio, 'green': green_ratio, 'yellow': yellow_ratio}
        counts = {'red': red_pixels, 'green': green_pixels, 'yellow': yellow_pixels}
        
        # Primary decision: require minimum ratio and minimum pixel count
        best = max(ratios, key=ratios.get)
        if ratios[best] >= self._area_ratio_threshold and counts[best] >= self._min_pixel_count:
            return best
        
        # Fallback: check raw counts within the circular mask (no brightness filter)
        red_raw = cv2.bitwise_and(mask_red, mask_red, mask=mask)
        green_raw = cv2.bitwise_and(mask_green, mask_green, mask=mask)
        yellow_raw = cv2.bitwise_and(mask_yellow, mask_yellow, mask=mask)
        red_raw_cnt = cv2.countNonZero(red_raw)
        green_raw_cnt = cv2.countNonZero(green_raw)
        yellow_raw_cnt = cv2.countNonZero(yellow_raw)
        raw_total = red_raw_cnt + green_raw_cnt + yellow_raw_cnt
        
        if raw_total > 0:
            raw_norm = {'red': red_raw_cnt / raw_total, 'green': green_raw_cnt / raw_total, 'yellow': yellow_raw_cnt / raw_total}
            raw_best = max(raw_norm, key=raw_norm.get)
            if raw_norm[raw_best] > 0.5 and max(red_raw_cnt, green_raw_cnt, yellow_raw_cnt) > 30:
                return raw_best
        
        # Default to green if uncertain
        return 'green'
    
    def update(self, frame, vehicle_id, cx, cy, vehicle_type='car'):
        """
        Update red-light detection for a vehicle.
        
        Args:
            frame: Input frame
            vehicle_id: Unique vehicle identifier
            cx, cy: Vehicle center coordinates
            vehicle_type: Type of vehicle
            
        Returns:
            is_violation: Boolean indicating red-light violation
            violation_record: Dictionary with violation details or None
        """
        # Detect current light state
        current_state = self.detect_light_color(frame)
        self.light_state = current_state
        
        # Track vehicle approach to stop line
        if vehicle_id not in self.vehicles_crossing:
            self.vehicles_crossing[vehicle_id] = {
                'first_approach': None,
                'crossed_at_time': None,
                'status': 'approaching'
            }
        
        vehicle_data = self.vehicles_crossing[vehicle_id]
        
        # Check if vehicle is approaching stop line
        approaching = cy >= (self.stop_line_y - 50)
        
        if approaching:
            if vehicle_data['status'] == 'approaching':
                vehicle_data['first_approach'] = self.light_state
                vehicle_data['status'] = 'at_line'
        
        # Check if vehicle crosses stop line
        crossing = cy >= self.stop_line_y
        
        violation = False
        violation_record = None
        
        if crossing and vehicle_data['status'] == 'at_line':
            # Vehicle is crossing stop line
            if self.light_state == 'red' or vehicle_data['first_approach'] == 'red':
                violation = True
                violation_record = {
                    'id': vehicle_id,
                    'violation': 'red_light_jumping',
                    'vehicle_type': vehicle_type,
                    'time': None,  # Will be set by caller
                    'light_state': self.light_state,
                    'position': (cx, cy)
                }
                self.red_light_violations.append(violation_record)
            
            vehicle_data['status'] = 'crossed'
        
        return violation, violation_record
    
    def cleanup_vehicle(self, vehicle_id):
        """Remove vehicle from tracking when it leaves the scene."""
        if vehicle_id in self.vehicles_crossing:
            del self.vehicles_crossing[vehicle_id]
    
    def get_light_state(self):
        """Get current traffic light state."""
        return self.light_state
    
    def draw_light_indicator(self, frame):
        """Draw traffic light detection region on frame."""
        roi = self.light_region
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        
        color = color_map.get(self.light_state, (0, 255, 0))  # Default green
        
        # Draw rectangle around light detection region
        cv2.rectangle(
            frame,
            (roi['x1'], roi['y1']),
            (roi['x2'], roi['y2']),
            color,
            2
        )
        
        # Add label
        cv2.putText(
            frame,
            f"Light: {self.light_state.upper()}",
            (roi['x1'], roi['y1'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    
    def draw_stop_line(self, frame):
        """Draw stop line reference on frame."""
        cv2.line(
            frame,
            (0, self.stop_line_y),
            (self.frame_width, self.stop_line_y),
            (0, 0, 255),  # Red line
            2
        )
        cv2.putText(
            frame,
            "STOP LINE",
            (10, self.stop_line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
