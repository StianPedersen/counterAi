import os
import json
import cv2
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from utils import Config, CounterLogger

class XMLToDetectron2Converter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("data_conversion")
        self.class_mapping = config.get("data.class_mapping", {"0": "red_box"})
        self.images_path = config.get("data.images_path")
        self.labels_path = config.get("data.labels_path")
        
    def convert_xml_to_detectron(self, dataset_name: str = "red_box_dataset") -> str:
        """Convert XML annotations to Detectron2 format"""
        self.logger.info("Starting XML to Detectron2 conversion")
        
        image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.png')])
        label_files = sorted([f for f in os.listdir(self.labels_path) if f.endswith('.xml')])
        
        self.logger.info(f"Found {len(image_files)} images and {len(label_files)} XML labels")
        
        dataset_dicts = []
        for img_file in image_files:
            # Find corresponding XML file
            base_name = os.path.splitext(img_file)[0]
            xml_file = base_name + '.xml'
            
            img_path = os.path.join(self.images_path, img_file)
            xml_path = os.path.join(self.labels_path, xml_file)
            
            if not os.path.exists(xml_path):
                self.logger.warning(f"No XML annotation found for {img_file}")
                continue
                
            record = self._convert_single_annotation(img_path, xml_path)
            if record:
                dataset_dicts.append(record)
        
        self.logger.info(f"Converted {len(dataset_dicts)} samples from XML")
        
        self._register_dataset(dataset_name, dataset_dicts)
        return dataset_name
    
    def _convert_single_annotation(self, img_path: str, xml_path: str) -> Dict:
        """Convert a single XML annotation to Detectron2 format"""
        img = cv2.imread(img_path)
        if img is None:
            self.logger.warning(f"Could not load image: {img_path}")
            return None
            
        height, width = img.shape[:2]
        
        record = {
            "file_name": img_path,
            "image_id": os.path.basename(img_path).split('.')[0],
            "height": height,
            "width": width,
            "annotations": []
        }
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Parse XML annotations
            for obj in root.findall('object'):
                # Get bounding box coordinates
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                    
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Get object name (class)
                name_elem = obj.find('name')
                if name_elem is not None:
                    class_name = name_elem.text
                else:
                    # Fallback to 'n' element if 'name' doesn't exist (as seen in the XML files)
                    n_elem = obj.find('n')
                    class_name = n_elem.text if n_elem is not None else "red_box"
                
                # Map class name to class ID using config class_mapping
                class_id = self._get_class_id_from_name(class_name)
                if class_id is None:
                    self.logger.warning(f"Unknown class '{class_name}' in {xml_path}, skipping object")
                    continue
                
                annotation = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": class_id,
                    "iscrowd": 0
                }
                record["annotations"].append(annotation)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML file {xml_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing XML file {xml_path}: {e}")
            return None
        
        return record
    
    def _get_class_id_from_name(self, class_name: str) -> int:
        """Map class name to class ID using config class_mapping"""
        # Create reverse mapping from class name to class ID
        name_to_id = {v: int(k) for k, v in self.class_mapping.items()}
        
        # Return class ID if found, otherwise None
        return name_to_id.get(class_name, None)
    
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