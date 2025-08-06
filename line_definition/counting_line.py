import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from utils import Config, CounterLogger

class CountingLine:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("line_definition")
        self.line_config = config.get("counting_line", {})
        self.coordinates = self.line_config.get("coordinates", [320, 0, 320, 480])
        self.direction = self.line_config.get("direction", "left_to_right")
        self.line_type = self.line_config.get("type", "vertical")
        self.thickness = 3
        self.color = (0, 255, 0)
        
    def get_line_coordinates(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        if self.line_config.get("position") == "middle":
            if self.line_type == "vertical":
                x = frame_width // 2
                return (x, 0, x, frame_height)
            else:
                y = frame_height // 2
                return (0, y, frame_width, y)
        else:
            return tuple(self.coordinates)
    
    def set_line_coordinates(self, x1: int, y1: int, x2: int, y2: int):
        self.coordinates = [x1, y1, x2, y2]
        self.config.set("counting_line.coordinates", self.coordinates)
        self.logger.info(f"Line coordinates updated: {self.coordinates}")
    
    def set_direction(self, direction: str):
        valid_directions = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]
        if direction in valid_directions:
            self.direction = direction
            self.config.set("counting_line.direction", direction)
            self.logger.info(f"Direction set to: {direction}")
        else:
            self.logger.error(f"Invalid direction: {direction}")
    
    def draw_line(self, image: np.ndarray, frame_width: int = None, frame_height: int = None) -> np.ndarray:
        if frame_width is None:
            frame_height, frame_width = image.shape[:2]
        
        x1, y1, x2, y2 = self.get_line_coordinates(frame_width, frame_height)
        
        vis_image = image.copy()
        cv2.line(vis_image, (x1, y1), (x2, y2), self.color, self.thickness)
        
        direction_arrow = self._get_direction_arrow(x1, y1, x2, y2)
        if direction_arrow:
            arrow_start, arrow_end = direction_arrow
            cv2.arrowedLine(vis_image, arrow_start, arrow_end, self.color, 2)
        
        label_pos = self._get_label_position(x1, y1, x2, y2)
        cv2.putText(vis_image, f"Counting Line ({self.direction})", 
                   label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        
        return vis_image
    
    def _get_direction_arrow(self, x1: int, y1: int, x2: int, y2: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        arrow_length = 30  # Smaller arrow
        
        if self.direction == "left_to_right":
            # Horizontal arrow pointing right
            arrow_start = (mid_x - arrow_length, mid_y)
            arrow_end = (mid_x + arrow_length, mid_y)
            return (arrow_start, arrow_end)
        elif self.direction == "right_to_left":
            # Horizontal arrow pointing left
            arrow_start = (mid_x + arrow_length, mid_y)
            arrow_end = (mid_x - arrow_length, mid_y)
            return (arrow_start, arrow_end)
        elif self.direction == "top_to_bottom":
            # Vertical arrow pointing down
            arrow_start = (mid_x, mid_y - arrow_length)
            arrow_end = (mid_x, mid_y + arrow_length)
            return (arrow_start, arrow_end)
        elif self.direction == "bottom_to_top":
            # Vertical arrow pointing up
            arrow_start = (mid_x, mid_y + arrow_length)
            arrow_end = (mid_x, mid_y - arrow_length)
            return (arrow_start, arrow_end)
        
        return None
    
    def _get_label_position(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
        if self.line_type == "vertical":
            return (x1 + 10, min(y1, y2) + 30)
        else:
            return (min(x1, x2) + 10, y1 - 10)
    
    def is_valid_line(self, frame_width: int, frame_height: int) -> bool:
        x1, y1, x2, y2 = self.get_line_coordinates(frame_width, frame_height)
        
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return False
        if x1 >= frame_width or x2 >= frame_width:
            return False
        if y1 >= frame_height or y2 >= frame_height:
            return False
        
        if x1 == x2 and y1 == y2:
            return False
        
        return True
    
    def get_line_equation(self, frame_width: int, frame_height: int) -> Dict[str, float]:
        x1, y1, x2, y2 = self.get_line_coordinates(frame_width, frame_height)
        
        if x2 - x1 == 0:
            return {"type": "vertical", "x": x1}
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        return {"type": "linear", "slope": slope, "intercept": intercept}
    
    def distance_to_line(self, point: Tuple[int, int], frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = self.get_line_coordinates(frame_width, frame_height)
        px, py = point
        
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_length_sq == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        return np.sqrt((px - projection_x) ** 2 + (py - projection_y) ** 2)

class InteractiveLineEditor:
    def __init__(self, counting_line: CountingLine):
        self.counting_line = counting_line
        self.logger = CounterLogger("line_editor")
        self.editing = False
        self.dragging_point = None
        self.points = []
        
    def start_editing(self, image: np.ndarray) -> np.ndarray:
        self.editing = True
        frame_height, frame_width = image.shape[:2]
        x1, y1, x2, y2 = self.counting_line.get_line_coordinates(frame_width, frame_height)
        self.points = [(x1, y1), (x2, y2)]
        
        cv2.namedWindow("Line Editor")
        cv2.setMouseCallback("Line Editor", self._mouse_callback)
        
        self.logger.info("Interactive line editing started. Click and drag endpoints to adjust.")
        return self._draw_editing_interface(image)
    
    def _mouse_callback(self, event, x, y, flags, param):
        if not self.editing:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging_point = self._find_nearest_point(x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            self.points[self.dragging_point] = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point is not None:
                self.points[self.dragging_point] = (x, y)
                self.dragging_point = None
                self._update_line_coordinates()
    
    def _find_nearest_point(self, x: int, y: int) -> Optional[int]:
        min_distance = float('inf')
        nearest_index = None
        
        for i, (px, py) in enumerate(self.points):
            distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if distance < min_distance and distance < 20:
                min_distance = distance
                nearest_index = i
                
        return nearest_index
    
    def _update_line_coordinates(self):
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            self.counting_line.set_line_coordinates(x1, y1, x2, y2)
    
    def _draw_editing_interface(self, image: np.ndarray) -> np.ndarray:
        vis_image = image.copy()
        
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(vis_image, (x2, y2), 8, (255, 0, 0), -1)
            
            cv2.putText(vis_image, "Drag endpoints to adjust line", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, "Press 'q' to quit, 's' to save", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def stop_editing(self):
        self.editing = False
        cv2.destroyWindow("Line Editor")
        self.logger.info("Line editing completed") 