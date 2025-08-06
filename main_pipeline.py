import cv2
import numpy as np
import os
import time
from typing import Optional, Dict, Any
from utils import Config, CounterLogger
from data_conversion import DataConverter
from detection import RedBoxDetector
from tracking import MultiObjectTracker
from line_definition import CountingLine
from crossing_detection import LineCrossingDetector
from counting import ObjectCounter, CountingVisualizer

class CounterAIPipeline:
    def __init__(self, config_path: str = "config.json"):
        self.config = Config(config_path)
        self.logger = CounterLogger("main_pipeline")
        
        self.data_converter = None
        self.detector = None
        self.tracker = None
        self.counting_line = None
        self.crossing_detector = None
        self.counter = None
        self.visualizer = None
        
        self.is_trained = False
        self.is_initialized = False
        
        self.logger.info("CounterAI Pipeline initialized")
    
    def setup_pipeline(self):
        try:
            self.data_converter = DataConverter(self.config)
            self.detector = RedBoxDetector(self.config)
            self.tracker = MultiObjectTracker(self.config)
            self.counting_line = CountingLine(self.config)
            self.crossing_detector = LineCrossingDetector(self.config, self.counting_line)
            self.counter = ObjectCounter(self.config)
            self.visualizer = CountingVisualizer(self.config)
            
            self.is_initialized = True
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup pipeline: {e}")
            raise
    
    def train_model(self) -> str:
        if not self.is_initialized:
            self.setup_pipeline()
        
        self.logger.info("Starting model training pipeline...")
        
        try:
            dataset_name = self.data_converter.convert_to_detectron()
            train_name, val_name = self.data_converter.split_dataset(dataset_name)
            
            self.detector.setup_config(dataset_name, num_classes=self.config.get("detection.num_classes", 1))
            
            model_path = self.detector.train(dataset_name)
            
            self.detector.load_model(model_path)
            
            evaluation_results = self.detector.evaluate(dataset_name)
            self.logger.info(f"Model evaluation completed: {evaluation_results}")
            
            self.is_trained = True
            self.logger.info("Model training completed successfully")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def load_pretrained_model(self, model_path: Optional[str] = None):
        if not self.is_initialized:
            self.setup_pipeline()
        
        try:
            dataset_name = self.data_converter.convert_to_detectron()
            self.detector.setup_config(dataset_name, num_classes=self.config.get("detection.num_classes", 1))
            self.detector.load_model(model_path)
            
            self.is_trained = True
            self.logger.info("Pre-trained model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     show_realtime: bool = True) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained or loaded. Call train_model() or load_pretrained_model() first.")
        
        self.logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.crossing_detector.set_frame_dimensions(width, height)
        
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                processed_frame = self._process_frame(frame, frame_count)
                
                if show_realtime:
                    cv2.imshow("CounterAI - Object Counting", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if video_writer:
                    video_writer.write(processed_frame)
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    self.logger.info(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        results = {
            "processing_time": processing_time,
            "frames_processed": frame_count,
            "average_fps": frame_count / processing_time,
            "counting_results": self.counter.get_detailed_statistics()
        }
        
        csv_path = self.counter.save_results_csv()
        json_path = self.counter.save_results_json()
        
        results["output_files"] = {
            "csv": csv_path,
            "json": json_path,
            "video": output_path
        }
        
        self.logger.info(f"Video processing completed. Results saved.")
        return results
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        detection_results = self.detector.detect(frame)
        detections = detection_results["detections"]
        
        detections = self.detector.apply_nms(detections)
        
        self.tracker.update(detections)
        tracked_objects = self.tracker.get_tracks()
        
        for object_id, track_data in tracked_objects.items():
            trajectory = track_data["trajectory"]
            centroid = track_data["centroid"]
            confidence = track_data["confidence"]
            class_name = track_data["class_name"]
            
            crossing_direction = self.crossing_detector.detect_crossing(object_id, trajectory)
            
            if crossing_direction:
                self.counter.process_crossing(
                    object_id, crossing_direction, centroid, 
                    confidence, class_name, frame_number
                )
                
                # Mark object as counted in the tracker
                if object_id in self.tracker.objects:
                    self.tracker.objects[object_id].counted = True
        
        vis_frame = self.tracker.visualize_tracks(frame)
        
        vis_frame = self.counting_line.draw_line(vis_frame)
        
        vis_frame = self.visualizer.draw_count_display(vis_frame, self.counter)
        
        return vis_frame
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = image.shape[:2]
        self.crossing_detector.set_frame_dimensions(width, height)
        
        processed_image = self._process_frame(image, 0)
        
        if output_path:
            cv2.imwrite(output_path, processed_image)
        
        return {
            "input_image": image_path,
            "output_image": output_path,
            "counting_results": self.counter.get_detailed_statistics()
        }
    
    def live_camera_processing(self, camera_index: int = 0):
        if not self.is_trained:
            raise ValueError("Model not trained or loaded.")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_index}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.crossing_detector.set_frame_dimensions(width, height)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_frame = self._process_frame(frame, frame_count)
                
                cv2.imshow("CounterAI - Live Camera", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.counter.reset_counter()
                    self.tracker.reset()
                    self.logger.info("Counter and tracker reset")
                elif key == ord('s'):
                    self.counter.save_results_csv()
                    self.counter.save_results_json()
                    self.logger.info("Results saved")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.is_initialized,
            "trained": self.is_trained,
            "current_count": self.counter.get_current_count() if self.counter else {},
            "config": self.config.config
        }

def main():
    pipeline = CounterAIPipeline()
    
    pipeline.setup_pipeline()
    
    if not os.path.exists("output/model_final.pth"):
        print("Training new model...")
        pipeline.train_model()
    else:
        print("Loading existing model...")
        pipeline.load_pretrained_model("output/model_final.pth")
    
    video_path = "inference_video/output.mp4"
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        
        output_video = "output/processed_video.mp4"
        results = pipeline.process_video(video_path, output_video, show_realtime=True)
        
        print("Processing completed!")
        print(f"Results: {results}")
    else:
        print(f"Video file not found: {video_path}")
        print("Starting live camera mode...")
        pipeline.live_camera_processing()

if __name__ == "__main__":
    main() 