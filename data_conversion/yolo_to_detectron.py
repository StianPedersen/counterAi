import os
import json
import cv2
from typing import List, Dict, Tuple
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from utils import Config, CounterLogger

class YOLOToDetectron2Converter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("data_conversion")
        self.class_mapping = config.get("data.class_mapping", {"0": "red_box"})
        self.images_path = config.get("data.images_path")
        self.labels_path = config.get("data.labels_path")
        
    def convert_yolo_to_detectron(self, dataset_name: str = "red_box_dataset") -> str:
        self.logger.info("Starting YOLOv8 to Detectron2 conversion")
        
        image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.png')])
        label_files = sorted([f for f in os.listdir(self.labels_path) if f.endswith('.txt')])
        
        self.logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")
        
        dataset_dicts = []
        for img_file, label_file in zip(image_files, label_files):
            img_path = os.path.join(self.images_path, img_file)
            label_path = os.path.join(self.labels_path, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            record = self._convert_single_annotation(img_path, label_path)
            if record:
                dataset_dicts.append(record)
        
        self.logger.info(f"Converted {len(dataset_dicts)} samples")
        
        self._register_dataset(dataset_name, dataset_dicts)
        return dataset_name
    
    def _convert_single_annotation(self, img_path: str, label_path: str) -> Dict:
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        height, width = img.shape[:2]
        
        record = {
            "file_name": img_path,
            "image_id": os.path.basename(img_path).split('.')[0],
            "height": height,
            "width": width,
            "annotations": []
        }
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, w, h = map(float, parts)
                
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                
                annotation = {
                    "bbox": [x1, y1, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(class_id),
                    "iscrowd": 0
                }
                record["annotations"].append(annotation)
        
        return record
    
    def _register_dataset(self, dataset_name: str, dataset_dicts: List[Dict]):
        if dataset_name in DatasetCatalog:
            DatasetCatalog.remove(dataset_name)
            
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)
        MetadataCatalog.get(dataset_name).set(thing_classes=list(self.class_mapping.values()))
        
        self.logger.info(f"Registered dataset: {dataset_name}")
    
    def split_dataset(self, dataset_name: str) -> Tuple[str, str]:
        train_split = self.config.get("data.train_split", 0.8)
        
        if dataset_name not in DatasetCatalog:
            raise ValueError(f"Dataset {dataset_name} not found")
            
        dataset_dicts = DatasetCatalog.get(dataset_name)
        if callable(dataset_dicts):
            dataset_dicts = dataset_dicts()
        random.shuffle(dataset_dicts)
        
        split_idx = int(len(dataset_dicts) * train_split)
        train_data = dataset_dicts[:split_idx]
        val_data = dataset_dicts[split_idx:]
        
        train_name = f"{dataset_name}_train"
        val_name = f"{dataset_name}_val"
        
        self._register_dataset(train_name, train_data)
        self._register_dataset(val_name, val_data)
        
        self.logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} val")
        
        return train_name, val_name 