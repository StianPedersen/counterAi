import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import pickle
from utils import Config, CounterLogger

class RedBoxDetector:
    def __init__(self, config: Config):
        self.config = config
        self.logger = CounterLogger("detection")
        self.cfg = None
        self.predictor = None
        self.model_path = None
        
    def setup_config(self, dataset_name: str, num_classes: int = 1):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            self.config.get("detection.model_config")
        ))
        
        cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        cfg.DATASETS.TEST = (f"{dataset_name}_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            self.config.get("detection.model_config")
        )
        
        cfg.SOLVER.IMS_PER_BATCH = self.config.get("detection.batch_size", 2)
        cfg.SOLVER.BASE_LR = self.config.get("detection.learning_rate", 0.00025)
        cfg.SOLVER.MAX_ITER = self.config.get("detection.max_iter", 3000)
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.GAMMA = 0.1
        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        
        cfg.TEST.EVAL_PERIOD = 500
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        self.cfg = cfg
        self.logger.info("Detectron2 config setup completed")
        
    def train(self, dataset_name: str):
        if self.cfg is None:
            raise ValueError("Config not setup. Call setup_config first.")
            
        self.logger.info("Starting training...")
        
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        self.model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.logger.info(f"Training completed. Model saved to: {self.model_path}")
        
        return self.model_path
    
    def load_model(self, model_path: Optional[str] = None):
        if self.cfg is None:
            raise ValueError("Config not setup. Call setup_config first.")
            
        if model_path:
            self.model_path = model_path
            self.cfg.MODEL.WEIGHTS = model_path
        elif self.model_path:
            self.cfg.MODEL.WEIGHTS = self.model_path
        else:
            self.logger.warning("Using pre-trained weights")
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.get(
            "detection.confidence_threshold", 0.9
        )
        
        self.predictor = DefaultPredictor(self.cfg)
        self.logger.info("Model loaded successfully")
    
    def detect(self, image: np.ndarray) -> Dict:
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model first.")
            
        outputs = self.predictor(image)
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        detections = []
        for i in range(len(boxes)):
            detection = {
                "bbox": boxes[i],
                "confidence": scores[i],
                "class_id": classes[i],
                "class_name": self.get_class_name(classes[i])
            }
            detections.append(detection)
        
        return {
            "detections": detections,
            "raw_output": outputs
        }
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
    
    def get_class_name(self, class_id: int) -> str:
        class_mapping = self.config.get("data.class_mapping", {"0": "red_box"})
        return class_mapping.get(str(class_id), f"class_{class_id}")
    
    def apply_nms(self, detections: List[Dict], iou_threshold: float = None) -> List[Dict]:
        if not detections:
            return detections
            
        if iou_threshold is None:
            iou_threshold = self.config.get("detection.nms_threshold", 0.7)
        
        boxes = np.array([det["bbox"] for det in detections])
        scores = np.array([det["confidence"] for det in detections])
        
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            self.config.get("detection.confidence_threshold", 0.9),
            iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        
        return []
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection["bbox"].astype(int)
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image
    
    def evaluate(self, dataset_name: str):
        if self.cfg is None or self.predictor is None:
            raise ValueError("Model not loaded")
            
        evaluator = COCOEvaluator(f"{dataset_name}_val", self.cfg, False, 
                                 output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, f"{dataset_name}_val")
        results = inference_on_dataset(self.predictor.model, val_loader, evaluator)
        
        self.logger.info(f"Evaluation results: {results}")
        return results 