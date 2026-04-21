import cv2
import numpy as np
from collections import defaultdict

class LaneViolationDetector:
    """
    Detects lane violations including:
    - Improper lane changes
    - Wrong-way driving
    - Lane markings crossing
    """
    
    def __init__(self, frame_width=1150, frame_height=840, num_lanes=3):
        """
        Initialize lane violation detector.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            num_lanes: Number of lanes to detect (default 3)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_lanes = num_lanes
        
        # Define lane boundaries (vertical x-coordinates)
        self.lane_boundaries = self._calculate_lane_boundaries()
        
        # Track vehicle lane positions
        self.vehicle_lanes = defaultdict(dict)
        self.lane_violations = []
        
        # Calibration parameters
        # These can be tuned based on actual lane positions
        self.lane_width = frame_width // num_lanes
        self.lane_crossing_threshold = 20  # pixels allowed before violation
        
    def _calculate_lane_boundaries(self):
        """Calculate lane boundary x-coordinates."""
        boundaries = []
        for i in range(self.num_lanes + 1):
            x = int((i / self.num_lanes) * self.frame_width)
            boundaries.append(x)
        return boundaries
    
    def detect_lanes_hough(self, frame):
        """
        Detect lane markings using Hough line transform.
        
        Args:
            frame: Input frame
            
        Returns:
            lanes: List of detected lane lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        return lines
    
    def get_vehicle_lane(self, cx, cy):
        """
        Determine which lane a vehicle is in based on x-coordinate.
        
        Args:
            cx: Vehicle center x-coordinate
            cy: Vehicle center y-coordinate
            
        Returns:
            lane_id: Lane number (0 to num_lanes-1)
        """
        for i in range(self.num_lanes):
            if self.lane_boundaries[i] <= cx < self.lane_boundaries[i + 1]:
                return i
        return -1  # Out of bounds
    
    def update(self, vehicle_id, cx, cy, vehicle_type='car', direction_vector=None):
        """
        Update lane violation detection for a vehicle.
        
        Args:
            vehicle_id: Unique vehicle identifier
            cx, cy: Vehicle center coordinates
            vehicle_type: Type of vehicle
            direction_vector: (dx, dy) movement vector for direction checking
            
        Returns:
            is_violation: Boolean indicating lane violation
            violation_record: Dictionary with violation details or None
        """
        current_lane = self.get_vehicle_lane(cx, cy)
        
        if vehicle_id not in self.vehicle_lanes:
            self.vehicle_lanes[vehicle_id] = {
                'previous_lane': current_lane,
                'lane_changes': 0,
                'lane_change_frames': 0,
                'wrong_way_detected': False
            }
        
        vehicle_data = self.vehicle_lanes[vehicle_id]
        previous_lane = vehicle_data['previous_lane']
        
        violation = False
        violation_record = None
        
        # Check for lane change violation (abrupt lane switching)
        if previous_lane != -1 and current_lane != -1:
            if abs(current_lane - previous_lane) > 1:
                # Vehicle jumped more than one lane (dangerous maneuver)
                vehicle_data['lane_changes'] += 1
                vehicle_data['lane_change_frames'] += 1
                
                if vehicle_data['lane_change_frames'] > 5:  # Confirm violation over frames
                    violation = True
                    violation_record = {
                        'id': vehicle_id,
                        'violation': 'improper_lane_change',
                        'vehicle_type': vehicle_type,
                        'time': None,
                        'from_lane': previous_lane,
                        'to_lane': current_lane,
                        'position': (cx, cy)
                    }
                    self.lane_violations.append(violation_record)
                    vehicle_data['lane_change_frames'] = 0
            else:
                # Normal lane change or same lane
                vehicle_data['lane_change_frames'] = 0
        
        # Check for wrong-way driving (direction vector check)
        if direction_vector is not None:
            dx, dy = direction_vector
            # If moving left to right at bottom of frame, typically wrong way
            if dy < 0 and cy > (self.frame_height * 0.7):
                if not vehicle_data['wrong_way_detected']:
                    violation = True
                    violation_record = {
                        'id': vehicle_id,
                        'violation': 'wrong_way_driving',
                        'vehicle_type': vehicle_type,
                        'time': None,
                        'position': (cx, cy),
                        'direction': (dx, dy)
                    }
                    self.lane_violations.append(violation_record)
                    vehicle_data['wrong_way_detected'] = True
        
        # Update previous lane
        vehicle_data['previous_lane'] = current_lane
        
        return violation, violation_record
    
    def cleanup_vehicle(self, vehicle_id):
        """Remove vehicle from tracking when it leaves the scene."""
        if vehicle_id in self.vehicle_lanes:
            del self.vehicle_lanes[vehicle_id]
    
    def draw_lanes(self, frame, alpha=0.3):
        """
        Draw detected lane boundaries on frame.
        
        Args:
            frame: Input frame to draw on
            alpha: Transparency level
        """
        overlay = frame.copy()
        
        colors = [
            (0, 255, 0),   # Green
            (255, 255, 0), # Cyan
            (0, 165, 255), # Orange
            (255, 0, 0),   # Blue
        ]
        
        for i in range(len(self.lane_boundaries) - 1):
            x_start = self.lane_boundaries[i]
            x_end = self.lane_boundaries[i + 1]
            
            color = colors[i % len(colors)]
            
            # Draw lane boundary lines
            cv2.line(overlay, (x_start, 0), (x_start, self.frame_height), color, 2)
            
            # Fill lane area slightly
            cv2.rectangle(
                overlay,
                (x_start, 0),
                (x_end, self.frame_height),
                color,
                -1
            )
        
        # Draw last boundary
        cv2.line(
            overlay,
            (self.lane_boundaries[-1], 0),
            (self.lane_boundaries[-1], self.frame_height),
            colors[-1],
            2
        )
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def draw_vehicle_lane_indicator(self, frame, cx, cy, vehicle_id):
        """
        Draw lane indicator for a specific vehicle.
        
        Args:
            frame: Input frame
            cx, cy: Vehicle center coordinates
            vehicle_id: Vehicle identifier
        """
        lane = self.get_vehicle_lane(cx, cy)
        
        if lane >= 0:
            color = (0, 255, 0) if lane >= 0 else (0, 0, 255)
            cv2.putText(
                frame,
                f"Lane {lane}",
                (cx - 30, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
