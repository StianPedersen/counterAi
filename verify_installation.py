#!/usr/bin/env python3

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import detectron2
        print(f"✓ Detectron2 {detectron2.__version__}")
    except ImportError as e:
        print(f"✗ Detectron2: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    return True

def test_basic_components():
    """Test basic component imports"""
    print("\nTesting component imports...")
    
    try:
        from utils import Config, CounterLogger
        print("✓ Utils components")
    except ImportError as e:
        print(f"✗ Utils: {e}")
        return False
    
    try:
        from data_conversion import YOLOToDetectron2Converter, XMLToDetectron2Converter, DataConverter
        print("✓ Data conversion")
    except ImportError as e:
        print(f"✗ Data conversion: {e}")
        return False
    
    try:
        from detection import RedBoxDetector
        print("✓ Detection module")
    except ImportError as e:
        print(f"✗ Detection: {e}")
        return False
    
    try:
        from tracking import MultiObjectTracker
        print("✓ Tracking module")
    except ImportError as e:
        print(f"✗ Tracking: {e}")
        return False
    
    try:
        from line_definition import CountingLine
        print("✓ Line definition")
    except ImportError as e:
        print(f"✗ Line definition: {e}")
        return False
    
    try:
        from crossing_detection import LineCrossingDetector
        print("✓ Crossing detection")
    except ImportError as e:
        print(f"✗ Crossing detection: {e}")
        return False
    
    try:
        from counting import ObjectCounter
        print("✓ Counting module")
    except ImportError as e:
        print(f"✗ Counting: {e}")
        return False
    
    return True

def test_detectron2_functionality():
    """Test basic Detectron2 functionality"""
    print("\nTesting Detectron2 functionality...")
    
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        print("✓ Detectron2 config loading")
        
        return True
    except Exception as e:
        print(f"✗ Detectron2 functionality: {e}")
        return False

def main():
    print("=" * 50)
    print("CounterAI Installation Verification")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_basic_components():
        tests_passed += 1
    
    if test_detectron2_functionality():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 Installation verified successfully!")
        print("\nNext steps:")
        print("1. Run: python3 main_pipeline.py")
        print("2. The system will train on your data and process the test video")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 