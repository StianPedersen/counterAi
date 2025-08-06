#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from main_pipeline import CounterAIPipeline

def test_pipeline_setup():
    """Test basic pipeline initialization"""
    print("Testing pipeline setup...")
    
    try:
        pipeline = CounterAIPipeline()
        pipeline.setup_pipeline()
        
        status = pipeline.get_pipeline_status()
        print(f"Pipeline initialized: {status['initialized']}")
        print("✓ Pipeline setup successful")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline setup failed: {e}")
        return False

def test_data_conversion():
    """Test YOLOv8 to Detectron2 data conversion"""
    print("\nTesting data conversion...")
    
    try:
        pipeline = CounterAIPipeline()
        pipeline.setup_pipeline()
        
        # Test data conversion
        dataset_name = pipeline.data_converter.convert_yolo_to_detectron()
        train_name, val_name = pipeline.data_converter.split_dataset(dataset_name)
        
        print(f"✓ Data conversion successful: {dataset_name}")
        print(f"✓ Dataset split: {train_name}, {val_name}")
        return True
        
    except Exception as e:
        print(f"✗ Data conversion failed: {e}")
        return False

def test_detection_setup():
    """Test detection model setup"""
    print("\nTesting detection setup...")
    
    try:
        pipeline = CounterAIPipeline()
        pipeline.setup_pipeline()
        
        # Setup detector config
        dataset_name = pipeline.data_converter.convert_yolo_to_detectron()
        pipeline.detector.setup_config(dataset_name, num_classes=1)
        
        print("✓ Detection model setup successful")
        return True
        
    except Exception as e:
        print(f"✗ Detection setup failed: {e}")
        return False

def test_single_image():
    """Test processing a single image"""
    print("\nTesting single image processing...")
    
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_image.png", test_image)
        
        pipeline = CounterAIPipeline()
        pipeline.setup_pipeline()
        
        # Setup for inference (without training)
        dataset_name = pipeline.data_converter.convert_yolo_to_detectron()
        pipeline.detector.setup_config(dataset_name, num_classes=1)
        pipeline.detector.load_model()  # Load pre-trained weights
        pipeline.is_trained = True
        
        # Process test image
        results = pipeline.process_image("test_image.png", "test_output.png")
        
        print("✓ Single image processing successful")
        
        # Cleanup
        if os.path.exists("test_image.png"):
            os.remove("test_image.png")
        if os.path.exists("test_output.png"):
            os.remove("test_output.png")
            
        return True
        
    except Exception as e:
        print(f"✗ Single image processing failed: {e}")
        return False

def test_counting_line():
    """Test counting line functionality"""
    print("\nTesting counting line...")
    
    try:
        pipeline = CounterAIPipeline()
        pipeline.setup_pipeline()
        
        # Test line coordinates
        frame_width, frame_height = 640, 480
        line_coords = pipeline.counting_line.get_line_coordinates(frame_width, frame_height)
        
        print(f"✓ Line coordinates: {line_coords}")
        
        # Test line validation
        is_valid = pipeline.counting_line.is_valid_line(frame_width, frame_height)
        print(f"✓ Line validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"✗ Counting line test failed: {e}")
        return False

def test_config_management():
    """Test configuration management"""
    print("\nTesting configuration...")
    
    try:
        pipeline = CounterAIPipeline()
        
        # Test config access
        confidence = pipeline.config.get("detection.confidence_threshold")
        direction = pipeline.config.get("counting_line.direction")
        target_classes = pipeline.config.get("data.target_classes")
        
        print(f"✓ Confidence threshold: {confidence}")
        print(f"✓ Counting direction: {direction}")
        print(f"✓ Target classes: {target_classes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 
        'detectron2', 'PIL', 'matplotlib', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True

def main():
    print("=" * 50)
    print("CounterAI Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Pipeline Setup", test_pipeline_setup),
        ("Configuration", test_config_management),
        ("Data Conversion", test_data_conversion),
        ("Detection Setup", test_detection_setup),
        ("Counting Line", test_counting_line),
        ("Single Image", test_single_image),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 