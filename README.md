# CounterAI - Object Counting System

A modular and maintainable AI system for object counting using computer vision with Detectron2 and DeepSORT tracking.

## Features

- **Object Detection**: Detectron2-based detection with training capabilities
- **Multi-Object Tracking**: DeepSORT tracking for consistent object IDs
- **Line Crossing Detection**: Configurable counting line with direction detection
- **Real-time Counting**: Live camera processing and video analysis
- **Modular Architecture**: Clean separation of concerns for easy debugging
- **Export Results**: CSV and JSON output formats
- **Visualization**: Real-time display with bounding boxes and trajectories

## System Architecture

```
counter_ai/
├── detection/          # Detectron2 object detection
├── tracking/           # Multi-object tracking with DeepSORT
├── line_definition/    # Configurable counting line
├── crossing_detection/ # Line crossing detection logic  
├── counting/          # Object counting and statistics
├── utils/             # Configuration and logging
├── data_conversion/   # YOLO/XML to Detectron2 conversion
└── main_pipeline.py   # Main orchestrator
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd counterAi
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Detectron2** (if not automatically installed)
```bash
# For CUDA 11.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# For CPU only
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
```

## Configuration

The system uses a `config.json` file for configuration. Key settings include:

- **Detection**: Confidence threshold (90%), NMS threshold (70%)
- **Counting Line**: Vertical line in middle, left-to-right direction
- **Target Classes**: Red box detection
- **Output**: Real-time display + CSV/JSON export

## Usage

### Basic Usage

```python
from main_pipeline import CounterAIPipeline

# Initialize pipeline
pipeline = CounterAIPipeline()
pipeline.setup_pipeline()

# Train model (first time)
pipeline.train_model()

# Or load existing model
pipeline.load_pretrained_model("path/to/model.pth")

# Process video
results = pipeline.process_video("input_video.mp4", "output_video.mp4")
```

### Command Line Usage

```bash
# Run with default video
python main_pipeline.py

# The system will:
# 1. Check for existing trained model
# 2. Train new model if needed using data/ folder
# 3. Process inference_video/output.mp4
# 4. Show real-time visualization
# 5. Save results to output/ folder
```

### Live Camera Processing

```python
# Live camera with controls
pipeline.live_camera_processing(camera_index=0)

# Controls:
# 'q' - quit
# 'r' - reset counter
# 's' - save results
```

## Data Format

The system supports both YOLO and XML annotation formats. Configure the format in `config.json`:

```json
{
  "data": {
    "source_label_format": "yolo",  // or "xml"
    "images_path": "data/images",
    "labels_path": "data/labels",
    "class_mapping": {
      "0": "red_box",
      "1": "blue_box", 
      "2": "battery"
    },
    "target_classes": ["red_box", "blue_box", "battery"]
  },
  "detection": {
    "num_classes": 3  // Must match number of classes
  }
}
```

### YOLO Format (Default)
```
data/
├── images/           # PNG images (frame_0001.png, etc.)
└── labels/           # YOLOv8 format labels (frame_0001.txt, etc.)
```

**Label Format:**
```
class_id center_x center_y width height
0 0.431262 0.186851 0.037649 0.069180
```

### XML Format
```
data_xml/
├── images/           # PNG images (2025-07-02--11-28-19-950612.png, etc.)
└── labels/           # XML annotations (2025-07-02--11-28-19-950612.xml, etc.)
```

**Label Format:**
```xml
<annotation>
  <size>
    <width>2592</width>
    <height>1944</height>
  </size>
  <object>
    <name>red_box</name>
    <bndbox>
      <xmin>176</xmin>
      <ymin>1593</ymin>
      <xmax>702</xmax>
      <ymax>1944</ymax>
    </bndbox>
  </object>
</annotation>
```

### Output Results
- **CSV**: Detailed counting log with timestamps
- **JSON**: Complete session statistics
- **Video**: Processed video with visualizations

## Configuration Options

### Counting Line Configuration
```json
{
  "counting_line": {
    "type": "vertical",
    "position": "middle", 
    "coordinates": [320, 0, 320, 480],
    "direction": "left_to_right"
  }
}
```

### Detection Settings
```json
{
  "detection": {
    "confidence_threshold": 0.9,
    "nms_threshold": 0.7,
    "model_config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
  }
}
```

## Performance

- **Target FPS**: 5 fps (configurable)
- **GPU**: Recommended for training and inference
- **Memory**: ~4GB GPU memory for training
- **Dataset**: Optimized for 5000-10000 images

## Modules Overview

### Detection Module
- Detectron2 integration
- Training and inference
- NMS post-processing
- Confidence filtering

### Tracking Module  
- Multi-object tracking
- Kalman filtering
- Trajectory management
- ID consistency

### Line Definition
- Configurable counting lines
- Interactive line editor
- Direction specification
- Validation logic

### Crossing Detection
- Trajectory analysis
- Line intersection detection
- Direction validation
- Anti-double-counting

### Counting Module
- Object counting logic
- Statistics tracking
- Result export
- Visualization

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in config
   - Use smaller input resolution

2. **Model not training**
   - Check data format
   - Verify file paths
   - Check GPU availability

3. **Poor detection accuracy**
   - Increase training iterations
   - Adjust confidence threshold
   - Add more training data

## Future Enhancements

- Support for multiple object classes
- Web interface for configuration
- Real-time analytics dashboard
- Cloud deployment options
- Mobile app integration

## License

MIT License - see LICENSE file for details. 