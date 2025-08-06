from .yolo_to_detectron import YOLOToDetectron2Converter
from .xml_to_detectron import XMLToDetectron2Converter
from utils import Config

class DataConverter:
    """Unified data converter that switches between XML and YOLO formats based on config"""
    
    def __init__(self, config: Config):
        self.config = config
        self.source_format = config.get("data.source_label_format", "yolo")
        
        if self.source_format.lower() == "xml":
            self.converter = XMLToDetectron2Converter(config)
        elif self.source_format.lower() == "yolo":
            self.converter = YOLOToDetectron2Converter(config)
        else:
            raise ValueError(f"Unsupported source_label_format: {self.source_format}. Use 'xml' or 'yolo'")
    
    def convert_to_detectron(self, dataset_name: str = "red_box_dataset") -> str:
        """Convert data to Detectron2 format based on source_label_format config"""
        if self.source_format.lower() == "xml":
            return self.converter.convert_xml_to_detectron(dataset_name)
        else:
            return self.converter.convert_yolo_to_detectron(dataset_name)
    
    def split_dataset(self, dataset_name: str):
        """Split dataset into train/validation sets"""
        return self.converter.split_dataset(dataset_name)
    
    # Legacy methods for backward compatibility
    def convert_yolo_to_detectron(self, dataset_name: str = "red_box_dataset") -> str:
        """Legacy method for backward compatibility"""
        if hasattr(self.converter, 'convert_yolo_to_detectron'):
            return self.converter.convert_yolo_to_detectron(dataset_name)
        else:
            raise ValueError("YOLO conversion not available when source_label_format is 'xml'")

__all__ = ['YOLOToDetectron2Converter', 'XMLToDetectron2Converter', 'DataConverter'] 