import json
import os
from typing import Dict, List, Tuple, Any

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        default_config = {
            "data": {
                "images_path": "data/images",
                "labels_path": "data/labels",
                "class_mapping": {
                    "0": "red_box"
                },
                "target_classes": ["red_box"],
                "train_split": 0.8
            },
            "detection": {
                "confidence_threshold": 0.9,
                "nms_threshold": 0.7,
                "model_config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                "num_classes": 1,
                "max_iter": 3000,
                "batch_size": 2,
                "learning_rate": 0.00025
            },
            "tracking": {
                "max_disappeared": 30,
                "max_distance": 50,
                "tracker_type": "deepsort"
            },
            "counting_line": {
                "type": "vertical",
                "position": "middle",
                "coordinates": [320, 0, 320, 480],
                "direction": "left_to_right"
            },
            "counting": {
                "enable_size_filter": False,
                "min_object_size": 0,
                "enable_confidence_filter": True,
                "min_confidence": 0.9
            },
            "output": {
                "save_video": True,
                "save_csv": True,
                "show_realtime": True,
                "output_dir": "output",
                "log_level": "INFO"
            },
            "performance": {
                "target_fps": 5,
                "device": "cuda"
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                self._merge_configs(default_config, loaded_config)
        
        self.config = default_config
        self.save_config()
    
    def _merge_configs(self, default: Dict, loaded: Dict):
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_configs(default[key], value)
                else:
                    default[key] = value
    
    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save_config() 