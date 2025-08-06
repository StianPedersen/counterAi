#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["output", "logs", "models"]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip3", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Training will be slower on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed. Cannot check GPU availability.")
        return False

def setup_config():
    """Setup initial configuration"""
    from utils.config import Config
    
    config = Config()
    print("✓ Configuration file created: config.json")
    
    # Adjust config based on system capabilities
    if not check_gpu_availability():
        config.set("performance.device", "cpu")
        config.set("detection.batch_size", 1)  # Reduce batch size for CPU
        print("✓ Configuration adjusted for CPU usage")

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_pipeline.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🎉 CounterAI Setup Complete!")
    print("="*60)
    print()
    print("Quick Start:")
    print("1. Train the model:")
    print("   python3 main_pipeline.py")
    print()
    print("2. Test with existing video:")
    print("   python3 -c \"from main_pipeline import CounterAIPipeline; p=CounterAIPipeline(); p.setup_pipeline(); p.load_pretrained_model(); p.process_video('inference_video/output.mp4', 'output/result.mp4')\"")
    print()
    print("3. Live camera processing:")
    print("   python3 -c \"from main_pipeline import CounterAIPipeline; p=CounterAIPipeline(); p.setup_pipeline(); p.load_pretrained_model(); p.live_camera_processing()\"")
    print()
    print("Configuration:")
    print("- Edit config.json to customize settings")
    print("- Line position, direction, thresholds, etc.")
    print()
    print("Data:")
    print("- Images: data/images/")
    print("- Labels: data/labels/ (YOLOv8 format)")
    print()
    print("Output:")
    print("- Processed videos: output/")
    print("- Results: CSV and JSON files")
    print("- Logs: logs/")
    print()
    print("Controls (Live mode):")
    print("- 'q': Quit")
    print("- 'r': Reset counter")
    print("- 's': Save results")

def main():
    print("CounterAI Setup Script")
    print("="*30)
    
    # Check system requirements
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency issues")
        return 1
    
    # Setup configuration
    setup_config()
    
    # Run tests
    if not run_tests():
        print("⚠️  Setup completed but some tests failed")
        print("You may still be able to use the system")
    
    # Print usage instructions
    print_usage_instructions()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 