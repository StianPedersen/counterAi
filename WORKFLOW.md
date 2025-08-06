# CounterAI Complete Workflow Guide

## 🚀 Step-by-Step Usage Workflow

### Step 1: Prepare Your Data
Ensure your data is in YOLOv8 format:
```
data/
├── images/           # Your training images (.png, .jpg, .jpeg)
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── labels/           # Corresponding YOLO labels (.txt)
    ├── frame_0001.txt
    ├── frame_0002.txt
    └── ...
```

### Step 2: Validate Your Data
Check that your data is properly formatted:
```bash
source .venv/bin/activate && python3 main_train.py --validate-only
```

### Step 3: Train Your Model
Train a custom CounterAI model:
```bash
# Basic training with default parameters
source .venv/bin/activate && python3 main_train.py

# Or with custom parameters
source .venv/bin/activate && python3 main_train.py --max-iter 5000 --batch-size 4
```

**Expected Output:**
- `output/model_final.pth` - Your trained model
- Training logs and metrics
- Evaluation results (should show high AP scores)

### Step 4: Test on Video
Process a test video to verify your model works:
```bash
source .venv/bin/activate && python3 main_process_video.py test_video.mp4
```

**Expected Output:**
- Processed video with counting visualization
- CSV/JSON results with object counts
- Real-time progress display

### Step 5: Use Live Camera
Deploy your model for live counting:
```bash
source .venv/bin/activate && python3 main_live_camera.py
```

**Features:**
- Auto-detects available cameras
- Real-time object counting
- Live controls (pause, reset, screenshot)

## 🔄 Typical Workflows

### 🎯 New Object Type Training
```bash
# 1. Prepare new data in data/images and data/labels
# 2. Update config.json with new class names
# 3. Train new model
python3 main_train.py --force

# 4. Test on video
python3 main_process_video.py test_video.mp4

# 5. Deploy live
python3 main_live_camera.py
```

### 📹 Video Processing Batch
```bash
# Process multiple videos
for video in *.mp4; do
    python3 main_process_video.py "$video" --no-display
done
```

### 🎛️ Custom Configuration
```bash
# 1. Copy and modify config.json
cp config.json custom_config.json
# Edit custom_config.json with your settings

# 2. Train with custom config
python3 main_train.py --config custom_config.json

# 3. Use custom config for processing
python3 main_process_video.py video.mp4 --config custom_config.json
```

## ⚙️ Common Parameters

### Training Parameters
- `--max-iter`: Number of training iterations (default: 3000)
- `--batch-size`: Training batch size (default: 2)
- `--learning-rate`: Learning rate (default: 0.00025)

### Processing Parameters
- `--no-display`: Skip real-time video display (faster processing)
- `--output`: Custom output file path
- `--model`: Use specific model file

### Camera Parameters
- `--camera`: Use specific camera ID
- `--save-video`: Record processed video
- `--no-test`: Skip camera test before processing

## 🔍 Troubleshooting

### Training Issues
```bash
# Check data format
python3 main_train.py --validate-only

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce batch size if out of memory
python3 main_train.py --batch-size 1
```

### Processing Issues
```bash
# Test with a small video first
python3 main_process_video.py small_video.mp4 --no-display

# Check model file exists
ls -la output/model_final.pth

# Use verbose logging
# (Check logs/ directory for detailed error messages)
```

### Camera Issues
```bash
# List available cameras
python3 main_live_camera.py --camera 0 --no-test

# Test specific camera
python3 main_live_camera.py --camera 1
```

## 📈 Performance Tips

### Training Performance
- Use GPU for faster training (automatic if available)
- Increase batch size if you have more GPU memory
- Use more iterations for better accuracy

### Processing Performance
- Use `--no-display` for batch processing
- Lower camera resolution for live processing
- Ensure good lighting for better detection

### Storage Management
- Training creates large model files (300MB+)
- Processed videos can be large
- Use CSV output for analysis, video for visualization

## 🎯 Next Steps

1. **Experiment** with different training parameters
2. **Collect more data** for better accuracy
3. **Fine-tune** counting line position and direction
4. **Deploy** for production use with live cameras
5. **Monitor** counting accuracy and adjust as needed 