# CounterAI Usage Guide

## 🚀 Quick Start Commands

### 1. Model Training Script
Train a new CounterAI model on your custom data.

```bash
# Basic training (uses data/images and data/labels)
source .venv/bin/activate && python3 main_train.py

# Custom data paths
source .venv/bin/activate && python3 main_train.py --images custom/images --labels custom/labels

# Custom training parameters
source .venv/bin/activate && python3 main_train.py --max-iter 5000 --batch-size 4

# Validate data only (no training)
source .venv/bin/activate && python3 main_train.py --validate-only

# Force retrain existing model
source .venv/bin/activate && python3 main_train.py --force
```

### 2. Video Processing Script
Process a video file to count objects crossing the counting line.

```bash
# Basic usage
source .venv/bin/activate && python3 main_process_video.py your_video.mp4

# With custom output
source .venv/bin/activate && python3 main_process_video.py input.mp4 --output results.mp4

# Without real-time display (faster)
source .venv/bin/activate && python3 main_process_video.py input.mp4 --no-display

# With custom config
source .venv/bin/activate && python3 main_process_video.py input.mp4 --config custom_config.json
```

### 3. Live Camera Script
Process live camera feed with automatic camera detection.

```bash
# Auto-detect cameras and let user select
source .venv/bin/activate && python3 main_live_camera.py

# Use specific camera
source .venv/bin/activate && python3 main_live_camera.py --camera 0

# Save processed video
source .venv/bin/activate && python3 main_live_camera.py --save-video

# Skip camera test
source .venv/bin/activate && python3 main_live_camera.py --no-test
```

## 📹 Live Camera Controls

When running the live camera script:
- **'q'** - Quit the application
- **'s'** - Save screenshot
- **'r'** - Reset counter and tracker
- **SPACE** - Pause/resume processing

## 🎯 Features

### Model Training (`main_train.py`)
- ✅ Automatic data validation and format conversion
- ✅ YOLOv8 to Detectron2 data conversion
- ✅ Customizable training parameters
- ✅ Training time estimation
- ✅ Progress monitoring and logging
- ✅ Checkpoint management
- ✅ Model evaluation and metrics
- ✅ Comprehensive error handling

### Video Processing (`main_process_video.py`)
- ✅ Automatic model loading
- ✅ Real-time progress display  
- ✅ Automatic output file naming
- ✅ Complete counting statistics
- ✅ CSV/JSON results export
- ✅ Command line argument support
- ✅ Auto-fitted display windows (preserves aspect ratio)

### Live Camera (`main_live_camera.py`)
- ✅ Automatic camera detection
- ✅ Camera selection interface
- ✅ Camera testing before processing
- ✅ Real-time FPS display
- ✅ Live controls (pause, reset, screenshot)
- ✅ Optional video recording
- ✅ Session statistics
- ✅ Auto-fitted display windows (preserves aspect ratio)

## 📊 Output Files

### Training Script (`main_train.py`)
- **model_final.pth** - Trained Detectron2 model (315MB+)
- **metrics.json** - Training metrics and loss curves
- **last_checkpoint** - Checkpoint information for resuming
- **events.out.tfevents.*** - TensorBoard training logs

### Processing Scripts (`main_process_video.py`, `main_live_camera.py`)
- **Processed video** - Video with counting visualization (improved arrow design)
- **CSV results** - Detailed counting log with timestamps
- **JSON results** - Session statistics and summary
- **Screenshots** - (Live camera only, on demand)

## ⚙️ Configuration

Edit `config.json` to customize:
- Counting line position and direction
- Detection confidence thresholds  
- Target object classes
- Output preferences

## 🔧 System Requirements

- Python 3.12+ with virtual environment
- Trained CounterAI model (`output/model_final.pth`)
- OpenCV-compatible camera (for live mode)
- CUDA-capable GPU (recommended)

## 🚀 Performance Tips

- Use `--no-display` for faster video processing
- Lower camera resolution for better live performance
- Ensure good lighting for better detection accuracy
- Position camera for clear view of objects crossing the line 