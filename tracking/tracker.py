import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import math
from utils import Config, CounterLogger

class TrackableObject:
    def __init__(self, object_id: int, centroid: Tuple[int, int], bbox: List[float]):
        self.object_id = object_id
        self.centroids = [centroid]
        self.bboxes = [bbox]
        self.disappeared = 0
        self.confidence = 0.0
        self.class_name = ""
        self.direction = None
        self.counted = False
        
    def update(self, centroid: Tuple[int, int], bbox: List[float], confidence: float):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.confidence = confidence
        self.disappeared = 0
        
        if len(self.centroids) > 50:
            self.centroids = self.centroids[-50:]
            self.bboxes = self.bboxes[-50:]

class MultiObjectTracker:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("tracking")
        self.objects = OrderedDict()
        self.next_object_id = 0
        self.max_disappeared = config.get("tracking.max_disappeared", 30)
        self.max_distance = config.get("tracking.max_distance", 50)
        
    def register(self, centroid: Tuple[int, int], bbox: List[float], 
                confidence: float, class_name: str) -> int:
        obj = TrackableObject(self.next_object_id, centroid, bbox)
        obj.confidence = confidence
        obj.class_name = class_name
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj.object_id
    
    def deregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, TrackableObject]:
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        input_centroids = []
        input_bboxes = []
        input_confidences = []
        input_classes = []
        
        for detection in detections:
            bbox = detection["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int((bbox[1] + bbox[3]) / 2.0)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            input_confidences.append(detection["confidence"])
            input_classes.append(detection["class_name"])
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], 
                             input_confidences[i], input_classes[i])
        else:
            object_centroids = [obj.centroids[-1] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            distances = self._compute_distances(object_centroids, input_centroids)
            
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id].update(
                    input_centroids[col], input_bboxes[col], input_confidences[col]
                )
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)
            
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.objects[object_id].disappeared += 1
                    
                    if self.objects[object_id].disappeared > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col],
                                 input_confidences[col], input_classes[col])
        
        return self.objects
    
    def _compute_distances(self, object_centroids: List[Tuple[int, int]], 
                          input_centroids: List[Tuple[int, int]]) -> np.ndarray:
        distances = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), 
            axis=2
        )
        return distances
    
    def get_tracks(self) -> Dict[int, Dict]:
        tracks = {}
        for object_id, obj in self.objects.items():
            if len(obj.centroids) > 0:
                tracks[object_id] = {
                    "centroid": obj.centroids[-1],
                    "bbox": obj.bboxes[-1],
                    "confidence": obj.confidence,
                    "class_name": obj.class_name,
                    "trajectory": obj.centroids,
                    "disappeared": obj.disappeared,
                    "counted": obj.counted
                }
        return tracks
    
    def visualize_tracks(self, image: np.ndarray) -> np.ndarray:
        vis_image = image.copy()
        
        for object_id, obj in self.objects.items():
            if len(obj.centroids) == 0:
                continue
                
            bbox = obj.bboxes[-1]
            centroid = obj.centroids[-1]
            
            bbox = [int(x) for x in bbox]
            
            color = (0, 255, 0) if not obj.counted else (255, 0, 0)
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            cv2.circle(vis_image, centroid, 4, color, -1)
            
            if len(obj.centroids) > 1:
                for i in range(1, len(obj.centroids)):
                    cv2.line(vis_image, obj.centroids[i-1], obj.centroids[i], color, 2)
            
            label = f"ID: {object_id} ({obj.class_name})"
            if obj.counted:
                label += " [COUNTED]"
                
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def reset(self):
        self.objects.clear()
        self.next_object_id = 0
        self.logger.info("Tracker reset")

class KalmanFilterTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        
    def predict(self):
        return self.kalman.predict()
    
    def update(self, measurement):
        self.kalman.correct(measurement)
    
    def set_state(self, x, y, vx=0, vy=0):
        self.kalman.statePre = np.array([x, y, vx, vy], dtype=np.float32) 