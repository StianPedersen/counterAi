import numpy as np
from typing import List, Tuple, Dict, Optional
from utils import Config, CounterLogger
from line_definition import CountingLine

class LineCrossingDetector:
    def __init__(self, config: Config, counting_line: CountingLine):
        self.config = config
        self.counting_line = counting_line
        self.logger = CounterLogger("crossing_detection")
        self.frame_width = None
        self.frame_height = None
        self.crossing_history = {}
        self.min_trajectory_points = 3
        
    def set_frame_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height
    
    def detect_crossing(self, object_id: int, trajectory: List[Tuple[int, int]]) -> Optional[str]:
        if len(trajectory) < self.min_trajectory_points:
            return None
            
        if self.frame_width is None or self.frame_height is None:
            self.logger.warning("Frame dimensions not set")
            return None
        
        line_coords = self.counting_line.get_line_coordinates(self.frame_width, self.frame_height)
        x1, y1, x2, y2 = line_coords
        
        recent_trajectory = trajectory[-self.min_trajectory_points:]
        
        crossing_direction = self._analyze_trajectory_crossing(recent_trajectory, line_coords)
        
        if crossing_direction:
            if object_id not in self.crossing_history:
                self.crossing_history[object_id] = []
            
            if crossing_direction not in self.crossing_history[object_id]:
                self.crossing_history[object_id].append(crossing_direction)
                
                if self._is_valid_crossing_direction(crossing_direction):
                    self.logger.info(f"Object {object_id} crossed line: {crossing_direction}")
                    return crossing_direction
        
        return None
    
    def _analyze_trajectory_crossing(self, trajectory: List[Tuple[int, int]], 
                                   line_coords: Tuple[int, int, int, int]) -> Optional[str]:
        x1, y1, x2, y2 = line_coords
        
        if len(trajectory) < 2:
            return None
        
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        start_side = self._get_point_side(start_point, line_coords)
        end_side = self._get_point_side(end_point, line_coords)
        
        if start_side == 0 or end_side == 0:
            return None
        
        if start_side != end_side:
            if self.counting_line.line_type == "vertical":
                if start_side == -1 and end_side == 1:
                    return "left_to_right"
                elif start_side == 1 and end_side == -1:
                    return "right_to_left"
            else:
                if start_side == -1 and end_side == 1:
                    return "top_to_bottom"
                elif start_side == 1 and end_side == -1:
                    return "bottom_to_top"
        
        return None
    
    def _get_point_side(self, point: Tuple[int, int], line_coords: Tuple[int, int, int, int]) -> int:
        px, py = point
        x1, y1, x2, y2 = line_coords
        
        # For vertical lines (x1 == x2), use simpler logic
        if abs(x2 - x1) < 1e-6:  # Vertical line
            if abs(px - x1) < 1e-6:
                return 0  # Point is on the line
            elif px < x1:
                return -1  # Point is to the left of the line
            else:
                return 1   # Point is to the right of the line
        
        # For non-vertical lines, use cross product
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        
        if abs(cross_product) < 1e-6:
            return 0
        elif cross_product > 0:
            return 1
        else:
            return -1
    
    def _is_valid_crossing_direction(self, detected_direction: str) -> bool:
        expected_direction = self.counting_line.direction
        return detected_direction == expected_direction
    
    def has_crossed_line(self, object_id: int, trajectory: List[Tuple[int, int]]) -> bool:
        if len(trajectory) < 2:
            return False
            
        line_coords = self.counting_line.get_line_coordinates(self.frame_width, self.frame_height)
        
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            
            if self._segments_intersect(p1, p2, line_coords):
                return True
        
        return False
    
    def _segments_intersect(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                           line_coords: Tuple[int, int, int, int]) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3, x4, y4 = line_coords
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def get_crossing_point(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self.frame_width is None or self.frame_height is None:
            return None
            
        line_coords = self.counting_line.get_line_coordinates(self.frame_width, self.frame_height)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3, x4, y4 = line_coords
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        if 0 <= t <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None
    
    def reset_crossing_history(self, object_id: Optional[int] = None):
        if object_id is None:
            self.crossing_history.clear()
            self.logger.info("All crossing history cleared")
        elif object_id in self.crossing_history:
            del self.crossing_history[object_id]
            self.logger.info(f"Crossing history cleared for object {object_id}")
    
    def get_crossing_statistics(self) -> Dict[str, int]:
        stats = {
            "total_objects_tracked": len(self.crossing_history),
            "left_to_right": 0,
            "right_to_left": 0,
            "top_to_bottom": 0,
            "bottom_to_top": 0
        }
        
        for object_id, crossings in self.crossing_history.items():
            for crossing in crossings:
                if crossing in stats:
                    stats[crossing] += 1
        
        return stats

class TrajectoryAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("trajectory_analyzer")
        self.smoothing_window = 5
        
    def smooth_trajectory(self, trajectory: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(trajectory) <= self.smoothing_window:
            return trajectory
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)
            
            window_points = trajectory[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            
            smoothed.append((int(avg_x), int(avg_y)))
        
        return smoothed
    
    def calculate_velocity(self, trajectory: List[Tuple[int, int]]) -> List[float]:
        if len(trajectory) < 2:
            return []
        
        velocities = []
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            velocities.append(distance)
        
        return velocities
    
    def detect_direction_change(self, trajectory: List[Tuple[int, int]], threshold: float = 45.0) -> List[int]:
        if len(trajectory) < 3:
            return []
        
        direction_changes = []
        
        for i in range(2, len(trajectory)):
            p1 = trajectory[i-2]
            p2 = trajectory[i-1]
            p3 = trajectory[i]
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            angle = self._angle_between_vectors(v1, v2)
            
            if angle > threshold:
                direction_changes.append(i-1)
        
        return direction_changes
    
    def _angle_between_vectors(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = np.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        return np.degrees(np.arccos(cos_angle)) 