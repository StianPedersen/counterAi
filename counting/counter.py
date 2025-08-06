import csv
import json
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from utils import Config, CounterLogger

class ObjectCounter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("counting")
        self.counted_objects = set()
        self.count_by_direction = {
            "left_to_right": 0,
            "right_to_left": 0,
            "top_to_bottom": 0,
            "bottom_to_top": 0
        }
        self.total_count = 0
        self.session_start_time = datetime.now()
        self.counting_log = []
        self.output_dir = config.get("output.output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def process_crossing(self, object_id: int, direction: str, 
                        centroid: Tuple[int, int], confidence: float, 
                        class_name: str, frame_number: int = 0) -> bool:
        if object_id in self.counted_objects:
            return False
        
        min_confidence = self.config.get("counting.min_confidence", 0.9)
        if confidence < min_confidence:
            self.logger.debug(f"Object {object_id} below confidence threshold: {confidence}")
            return False
        
        target_classes = self.config.get("data.target_classes", ["red_box"])
        if class_name not in target_classes:
            self.logger.debug(f"Object {object_id} not in target classes: {class_name}")
            return False
        
        self.counted_objects.add(object_id)
        
        if direction in self.count_by_direction:
            self.count_by_direction[direction] += 1
        
        self.total_count += 1
        
        count_entry = {
            "timestamp": datetime.now().isoformat(),
            "object_id": object_id,
            "direction": direction,
            "centroid": centroid,
            "confidence": float(confidence),  # Convert numpy float32 to Python float
            "class_name": class_name,
            "frame_number": frame_number,
            "total_count": self.total_count
        }
        
        self.counting_log.append(count_entry)
        
        self.logger.info(
            f"Object {object_id} counted! Direction: {direction}, "
            f"Total: {self.total_count}, Confidence: {confidence:.2f}"
        )
        
        return True
    
    def get_current_count(self) -> Dict[str, int]:
        return {
            "total": self.total_count,
            **self.count_by_direction
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            "session_start": self.session_start_time.isoformat(),
            "session_duration_seconds": session_duration,
            "total_count": self.total_count,
            "count_by_direction": self.count_by_direction.copy(),
            "unique_objects_counted": len(self.counted_objects),
            "counting_rate_per_minute": (self.total_count / session_duration * 60) if session_duration > 0 else 0,
            "target_classes": self.config.get("data.target_classes", ["red_box"]),
            "confidence_threshold": self.config.get("counting.min_confidence", 0.9)
        }
    
    def save_results_csv(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"counting_results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        fieldnames = [
            "timestamp", "object_id", "direction", "centroid_x", "centroid_y", 
            "confidence", "class_name", "frame_number", "total_count"
        ]
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.counting_log:
                row = entry.copy()
                if "centroid" in row:
                    row["centroid_x"] = row["centroid"][0]
                    row["centroid_y"] = row["centroid"][1]
                    del row["centroid"]
                writer.writerow(row)
        
        self.logger.info(f"Results saved to CSV: {filepath}")
        return filepath
    
    def save_results_json(self, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"counting_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        results = {
            "summary": self.get_detailed_statistics(),
            "counting_log": self.counting_log
        }
        
        with open(filepath, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2)
        
        self.logger.info(f"Results saved to JSON: {filepath}")
        return filepath
    
    def reset_counter(self):
        self.counted_objects.clear()
        self.count_by_direction = {
            "left_to_right": 0,
            "right_to_left": 0,
            "top_to_bottom": 0,
            "bottom_to_top": 0
        }
        self.total_count = 0
        self.session_start_time = datetime.now()
        self.counting_log.clear()
        self.logger.info("Counter reset")
    
    def is_object_counted(self, object_id: int) -> bool:
        return object_id in self.counted_objects
    
    def remove_object_from_count(self, object_id: int) -> bool:
        if object_id in self.counted_objects:
            self.counted_objects.remove(object_id)
            self.logger.info(f"Object {object_id} removed from counted objects")
            return True
        return False

class CountingVisualizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("counting_visualizer")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)
        
    def draw_count_display(self, image: np.ndarray, counter: ObjectCounter) -> np.ndarray:
        vis_image = image.copy()
        
        counts = counter.get_current_count()
        
        y_offset = 30
        line_height = 30
        
        count_text = f"Total Count: {counts['total']}"
        self._draw_text_with_background(vis_image, count_text, (10, y_offset))
        y_offset += line_height
        
        direction_text = f"L->R: {counts['left_to_right']} | R->L: {counts['right_to_left']}"
        self._draw_text_with_background(vis_image, direction_text, (10, y_offset))
        y_offset += line_height
        
        if self.config.get("counting_line.type") == "horizontal":
            direction_text = f"T->B: {counts['top_to_bottom']} | B->T: {counts['bottom_to_top']}"
            self._draw_text_with_background(vis_image, direction_text, (10, y_offset))
        
        return vis_image
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, position: Tuple[int, int]):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )
        
        x, y = position
        
        cv2.rectangle(image, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + baseline + 5), self.bg_color, -1)
        
        cv2.putText(image, text, (x, y), self.font, self.font_scale, self.text_color, self.thickness)
    
    def create_count_overlay(self, image: np.ndarray, counter: ObjectCounter, 
                           frame_number: int = 0) -> np.ndarray:
        overlay = image.copy()
        
        height, width = image.shape[:2]
        
        overlay_height = 120
        overlay = cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        
        counts = counter.get_current_count()
        stats = counter.get_detailed_statistics()
        
        texts = [
            f"Frame: {frame_number}",
            f"Total Objects: {counts['total']}",
            f"Session Time: {stats['session_duration_seconds']:.1f}s",
            f"Rate: {stats['counting_rate_per_minute']:.1f}/min"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 25 + i * 25
            cv2.putText(overlay, text, (10, y_pos), self.font, 
                       self.font_scale, self.text_color, self.thickness)
        
        return overlay 