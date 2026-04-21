import cv2
import numpy as np
from collections import deque
from datetime import datetime

class CongestionAnalyzer:
    """
    Analyzes traffic density and predicts congestion levels.
    """
    
    def __init__(self, frame_width=1150, frame_height=840, window_size=30):
        """
        Initialize congestion analyzer.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            window_size: Number of frames to use for moving average
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.window_size = window_size
        
        # Historical data
        self.vehicle_counts = deque(maxlen=window_size)
        self.vehicle_densities = deque(maxlen=window_size)
        self.avg_speeds = deque(maxlen=window_size)
        self.congestion_levels = deque(maxlen=window_size)
        
        # Frame grid for density calculation
        self.grid_cols = 4
        self.grid_rows = 3
        self.grid_cells = self.grid_rows * self.grid_cols
        self.cell_width = frame_width // self.grid_cols
        self.cell_height = frame_height // self.grid_rows
        self.cell_vehicle_count = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        
        # Thresholds
        self.low_density_threshold = 0.1
        self.moderate_density_threshold = 0.3
        self.high_density_threshold = 0.6
        self.very_high_density_threshold = 0.8
        
    def get_cell_index(self, cx, cy):
        """Get grid cell indices for a position."""
        col = min(int(cx / self.cell_width), self.grid_cols - 1)
        row = min(int(cy / self.cell_height), self.grid_rows - 1)
        return row, col
    
    def update(self, vehicle_positions, vehicle_speeds):
        """
        Update congestion analysis.
        
        Args:
            vehicle_positions: List of (cx, cy) tuples for all vehicles
            vehicle_speeds: List of speeds for all vehicles
            
        Returns:
            congestion_data: Dictionary with current congestion metrics
        """
        # Reset grid
        self.cell_vehicle_count = [[0] * self.grid_cols for _ in range(self.grid_rows)]
        
        # Count vehicles in each grid cell
        for cx, cy in vehicle_positions:
            row, col = self.get_cell_index(cx, cy)
            self.cell_vehicle_count[row][col] += 1
        
        # Calculate metrics
        total_vehicles = len(vehicle_positions)
        max_vehicles_per_cell = 20  # Calibration parameter
        
        # Calculate vehicle density (vehicles per pixel area)
        total_area = self.frame_width * self.frame_height
        vehicle_density = min(total_vehicles / (total_area / 10000), 1.0)
        
        # Calculate average speed
        avg_speed = np.mean(vehicle_speeds) if vehicle_speeds else 0
        
        # Determine congestion level
        congestion_level = self._calculate_congestion_level(
            total_vehicles,
            vehicle_density,
            avg_speed
        )
        
        # Store historical data
        self.vehicle_counts.append(total_vehicles)
        self.vehicle_densities.append(vehicle_density)
        self.avg_speeds.append(avg_speed)
        self.congestion_levels.append(congestion_level)
        
        congestion_data = {
            'total_vehicles': total_vehicles,
            'vehicle_density': round(vehicle_density, 3),
            'avg_speed_kmh': round(avg_speed, 1),
            'congestion_level': congestion_level,
            'congestion_percentage': self._get_congestion_percentage(congestion_level),
            'trend': self._get_trend(),
            'hotspots': self._identify_hotspots(),
            'recommendations': self._get_recommendations(congestion_level)
        }
        
        return congestion_data
    
    def _calculate_congestion_level(self, total_vehicles, density, avg_speed):
        """
        Calculate congestion level based on multiple factors.
        
        Returns:
            level: 'free_flow', 'low', 'moderate', 'high', 'severe'
        """
        if total_vehicles < 5 and density < self.low_density_threshold:
            return 'free_flow'
        elif density < self.moderate_density_threshold and avg_speed > 40:
            return 'low'
        elif density < self.high_density_threshold and avg_speed > 20:
            return 'moderate'
        elif density < self.very_high_density_threshold and avg_speed > 10:
            return 'high'
        else:
            return 'severe'
    
    def _get_congestion_percentage(self, level):
        """Convert congestion level to percentage."""
        levels = {
            'free_flow': 0,
            'low': 25,
            'moderate': 50,
            'high': 75,
            'severe': 100
        }
        return levels.get(level, 0)
    
    def _get_trend(self):
        """Determine if congestion is improving or worsening."""
        if len(self.congestion_levels) < 2:
            return 'stable'
        
        # Convert levels to numeric values
        level_values = {
            'free_flow': 0,
            'low': 1,
            'moderate': 2,
            'high': 3,
            'severe': 4
        }
        
        recent_levels = list(self.congestion_levels)[-10:]
        recent_values = [level_values.get(l, 0) for l in recent_levels]
        
        if len(recent_values) < 2:
            return 'stable'
        
        avg_recent = np.mean(recent_values[-5:])
        avg_previous = np.mean(recent_values[:5]) if len(recent_values) >= 10 else avg_recent
        
        if avg_recent > avg_previous + 0.5:
            return 'worsening'
        elif avg_recent < avg_previous - 0.5:
            return 'improving'
        else:
            return 'stable'
    
    def _identify_hotspots(self):
        """Identify areas with highest vehicle density."""
        hotspots = []
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                count = self.cell_vehicle_count[row][col]
                if count > 3:  # Threshold for hotspot
                    cell_center_x = col * self.cell_width + self.cell_width // 2
                    cell_center_y = row * self.cell_height + self.cell_height // 2
                    
                    hotspots.append({
                        'position': (cell_center_x, cell_center_y),
                        'vehicle_count': count,
                        'grid_cell': (row, col)
                    })
        
        # Sort by vehicle count
        hotspots.sort(key=lambda x: x['vehicle_count'], reverse=True)
        return hotspots[:5]  # Top 5 hotspots
    
    def _get_recommendations(self, level):
        """Get traffic management recommendations."""
        recommendations = {
            'free_flow': [
                'Traffic is flowing smoothly',
                'No immediate action required'
            ],
            'low': [
                'Light traffic detected',
                'Monitor for incidents'
            ],
            'moderate': [
                'Traffic is moderately congested',
                'Consider alternate routes',
                'Increase traffic signal timing'
            ],
            'high': [
                'Heavy congestion detected',
                'Recommend route diversions',
                'Activate emergency lane if available',
                'Increase police presence'
            ],
            'severe': [
                'Severe congestion - ALERT',
                'All diversionary routes recommended',
                'Consider temporary road closures',
                'Deploy traffic management teams',
                'Public notification advised'
            ]
        }
        
        return recommendations.get(level, [])
    
    def get_historical_stats(self):
        """Get historical congestion statistics."""
        if not self.congestion_levels:
            return {}
        
        return {
            'avg_vehicles': round(np.mean(list(self.vehicle_counts)), 1),
            'max_vehicles': max(self.vehicle_counts),
            'avg_density': round(np.mean(list(self.vehicle_densities)), 3),
            'avg_speed': round(np.mean(list(self.avg_speeds)), 1),
            'most_common_level': self._get_mode(list(self.congestion_levels)),
            'peak_congestion': max([c for c in self.congestion_levels] 
                                   if self.congestion_levels else [0])
        }
    
    def _get_mode(self, items):
        """Get most common item in a list."""
        if not items:
            return None
        return max(set(items), key=items.count)
    
    def draw_grid_heatmap(self, frame):
        """
        Draw grid heatmap showing vehicle density per cell.
        
        Args:
            frame: Input frame to draw on
        """
        overlay = frame.copy()
        
        max_count = max([max(row) for row in self.cell_vehicle_count])
        if max_count == 0:
            return
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                count = self.cell_vehicle_count[row][col]
                density = count / max_count if max_count > 0 else 0
                
                # Color based on density (blue to red gradient)
                blue = int(255 * (1 - density))
                red = int(255 * density)
                green = 0
                
                x1 = col * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (blue, green, red), -1)
                
                # Add vehicle count text
                if count > 0:
                    text_x = x1 + self.cell_width // 2 - 10
                    text_y = y1 + self.cell_height // 2
                    cv2.putText(
                        overlay,
                        str(count),
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    def draw_congestion_indicator(self, frame, congestion_data):
        """Draw congestion status on frame."""
        level = congestion_data['congestion_level']
        percentage = congestion_data['congestion_percentage']
        
        color_map = {
            'free_flow': (0, 255, 0),      # Green
            'low': (0, 255, 255),           # Cyan
            'moderate': (0, 165, 255),      # Orange
            'high': (0, 100, 255),          # Red-Orange
            'severe': (0, 0, 255)           # Red
        }
        
        color = color_map.get(level, (255, 255, 255))
        
        # Draw bar at bottom
        bar_width = int((percentage / 100) * self.frame_width)
        cv2.rectangle(frame, (0, self.frame_height - 20), (bar_width, self.frame_height), color, -1)
        
        # Add text
        text = f"Congestion: {level.upper()} ({percentage}%)"
        cv2.putText(
            frame,
            text,
            (10, self.frame_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
