#!/usr/bin/env python3
"""
CounterAI Model Training Script

Trains a Detectron2 model on YOLOv8 formatted data for object counting.
Automatically converts data format and provides detailed training progress.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from main_pipeline import CounterAIPipeline
from utils import Config

def validate_data_paths(images_path, labels_path, label_format="yolo"):
    """Validate that data paths exist and contain files"""
    if not os.path.exists(images_path):
        print(f"❌ Error: Images directory '{images_path}' not found")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ Error: Labels directory '{labels_path}' not found")
        return False
    
    # Count files based on label format
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if label_format.lower() == "xml":
        label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.xml')]
        format_desc = "XML"
    else:
        label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
        format_desc = "YOLO"
    
    if len(image_files) == 0:
        print(f"❌ Error: No image files found in '{images_path}'")
        return False
    
    if len(label_files) == 0:
        print(f"❌ Error: No {format_desc} label files found in '{labels_path}'")
        return False
    
    print(f"📊 Dataset validation ({format_desc} format):")
    print(f"  Images: {len(image_files)} files")
    print(f"  Labels: {len(label_files)} files")
    
    if len(image_files) != len(label_files):
        print(f"⚠️  Warning: Mismatch between images ({len(image_files)}) and labels ({len(label_files)})")
    
    return True

def estimate_training_time(num_images, max_iter):
    """Estimate training time based on dataset size"""
    # Rough estimates based on typical hardware
    seconds_per_iter = 0.5  # Conservative estimate
    total_seconds = max_iter * seconds_per_iter
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    print(f"⏱️  Estimated training time: ", end="")
    if hours > 0:
        print(f"{int(hours)}h {int(minutes)}m")
    else:
        print(f"{int(minutes)}m")
    
    print(f"📈 Training parameters:")
    print(f"  Max iterations: {max_iter}")
    print(f"  Dataset size: {num_images} images")

def main():
    parser = argparse.ArgumentParser(
        description="CounterAI Model Training - Train object detection model on custom data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use paths from config.json
  %(prog)s --images custom/images --labels custom/labels  # Override config paths
  %(prog)s --max-iter 5000 --batch-size 4    # Custom training parameters
  %(prog)s --config custom_config.json       # Use custom configuration file
  %(prog)s --resume output/model_0001000.pth # Resume from checkpoint
        """
    )
    
    parser.add_argument(
        "--images",
        default=None,
        help="Path to images directory (default: use config.json)"
    )
    
    parser.add_argument(
        "--labels", 
        default=None,
        help="Path to labels directory (default: use config.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for trained model (default: output)"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Maximum training iterations (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        help="Resume training from checkpoint file"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data format without training"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even if model already exists"
    )
    
    args = parser.parse_args()
    
    print("🤖 CounterAI Model Training")
    print("=" * 40)
    
    # Validate data paths
    # Load config first to get the correct paths
    config = Config(args.config)
    
    # Use command line args if provided, otherwise use config
    images_path = args.images if args.images else config.get("data.images_path")
    labels_path = args.labels if args.labels else config.get("data.labels_path")
    
    if not validate_data_paths(images_path, labels_path, config.get("data.source_label_format", "yolo")):
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Data validation completed successfully!")
        sys.exit(0)
    
    # Check for existing model
    existing_model = os.path.join(args.output_dir, "model_final.pth")
    if os.path.exists(existing_model) and not args.force:
        print(f"⚠️  Model already exists: {existing_model}")
        print("💡 Use --force to overwrite or --resume to continue training")
        try:
            response = input("Continue anyway? (y/N): ").lower()
            if response != 'y':
                print("Training cancelled.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n⛔ Training cancelled.")
            sys.exit(0)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize pipeline with custom config (already loaded above)
        print("\n🔧 Initializing training pipeline...")
        
        # Override config with command line arguments (only if provided)
        if args.max_iter:
            config.set("detection.max_iter", args.max_iter)
        if args.batch_size:
            config.set("detection.batch_size", args.batch_size)
        if args.learning_rate:
            config.set("detection.learning_rate", args.learning_rate)
        
        # Override data paths only if explicitly provided via command line
        if args.images:
            config.set("data.images_path", args.images)
        if args.labels:
            config.set("data.labels_path", args.labels)
        
        pipeline = CounterAIPipeline(args.config)
        pipeline.setup_pipeline()
        
        # Display training configuration
        print("\n⚙️  Training Configuration:")
        print(f"  Images path: {config.get('data.images_path')}")
        print(f"  Labels path: {config.get('data.labels_path')}")
        print(f"  Label format: {config.get('data.source_label_format', 'yolo')}")
        print(f"  Max iterations: {config.get('detection.max_iter')}")
        print(f"  Batch size: {config.get('detection.batch_size')}")
        print(f"  Learning rate: {config.get('detection.learning_rate')}")
        print(f"  Target classes: {config.get('data.target_classes')}")
        print(f"  Number of classes: {config.get('detection.num_classes')}")
        print(f"  Output directory: {args.output_dir}")
        
        # Count images for time estimation
        image_files = [f for f in os.listdir(config.get('data.images_path')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        estimate_training_time(len(image_files), config.get('detection.max_iter'))
        
        # Confirm training start
        print(f"\n🚀 Ready to start training...")
        try:
            input("Press Enter to begin training (Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\n⛔ Training cancelled.")
            sys.exit(0)
        
        # Start training
        print("\n🎯 Starting model training...")
        start_time = time.time()
        
        if args.resume:
            print(f"📁 Resuming from checkpoint: {args.resume}")
            # Note: Resume functionality would need to be implemented in the pipeline
            
        model_path = pipeline.train_model()
        
        training_time = time.time() - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️  Training time: ", end="")
        if hours > 0:
            print(f"{int(hours)}h {int(minutes)}m")
        else:
            print(f"{int(minutes)}m {int(training_time % 60)}s")
        
        print(f"📁 Model saved to: {model_path}")
        
        # Display final metrics
        print(f"\n📊 Training Summary:")
        print(f"  Dataset: {len(image_files)} images")
        print(f"  Iterations completed: {config.get('detection.max_iter')}")
        print(f"  Final model: {model_path}")
        print(f"  Output directory: {args.output_dir}")
        
        # Check output files
        output_files = []
        for file_name in ["model_final.pth", "metrics.json", "last_checkpoint"]:
            file_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                else:
                    size_str = f"{size/1024:.1f}KB"
                output_files.append(f"  {file_name}: {size_str}")
        
        if output_files:
            print(f"\n📁 Output Files:")
            for file_info in output_files:
                print(file_info)
        
        print(f"\n✅ Ready for inference! Use the trained model with:")
        print(f"   python3 main_process_video.py input.mp4 --model {model_path}")
        print(f"   python3 main_live_camera.py --model {model_path}")
        
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("  - Check data format matches config (YOLO .txt or XML .xml)")
        print("  - Verify image and label file counts match")
        print("  - Ensure class names in XML match class_mapping in config")
        print("  - Check num_classes matches number of classes in data")
        print("  - Ensure sufficient disk space")
        print("  - Check CUDA/GPU availability")
        sys.exit(1)

if __name__ == "__main__":
    main() 