import cv2
import numpy as np
from collections import defaultdict

class RedLightDetector:
    """
    Detects red-light jumping violations.
    Uses color-based traffic light detection and tracks vehicle crossing behavior.
    """
    
    def __init__(self, frame_width=1150, frame_height=840):
        """
        Initialize red-light detector.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Define traffic light region (typically top area of frame)
        # Can be customized based on actual camera position
        self.light_region = {
            'x1': int(frame_width * 0.35),
            'y1': int(frame_height * 0.05),
            'x2': int(frame_width * 0.65),
            'y2': int(frame_height * 0.15)
        }
        
        # Track light state and history
        self.light_state = 'green'  # green, red, yellow, unknown
        self.light_state_history = []
        self.state_change_frames = 0
        
        # Track vehicles crossing the stop line
        self.stop_line_y = int(frame_height * 0.5)  # Usually middle of frame
        self.vehicles_crossing = defaultdict(dict)
        self.red_light_violations = []
        
    def detect_light_color(self, frame):
        """
        Detect traffic light color using HSV color space.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            light_state: 'red', 'green', 'yellow', or 'unknown'
        """
        # Extract region of interest for traffic light
        roi = frame[
            self.light_region['y1']:self.light_region['y2'],
            self.light_region['x1']:self.light_region['x2']
        ]
        
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Red light (hue wraps around, so we check two ranges)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Green light
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        # Yellow light
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Count pixels for each color
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        # Determine dominant color
        if red_pixels > green_pixels and red_pixels > yellow_pixels and red_pixels > 100:
            return 'red'
        elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > 100:
            return 'green'
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > 100:
            return 'yellow'
        else:
            return 'unknown'
    
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
            'yellow': (0, 255, 255),
            'unknown': (128, 128, 128)
        }
        
        color = color_map.get(self.light_state, (128, 128, 128))
        
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
