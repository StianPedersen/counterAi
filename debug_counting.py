#!/usr/bin/env python3

import cv2
import numpy as np
from main_pipeline import CounterAIPipeline
from utils import Config
from line_definition import CountingLine
from crossing_detection import LineCrossingDetector
from counting import ObjectCounter

def debug_counting_pipeline():
    print("🔍 EXPERT DEBUGGING: CounterAI Line Crossing Issues")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    pipeline = CounterAIPipeline()
    pipeline.setup_pipeline()
    pipeline.load_pretrained_model('output/model_final.pth')
    
    # Test 1: Configuration verification
    print("\n📋 STEP 1: Configuration Analysis")
    print("-" * 30)
    print(f"Counting line type: {config.get('counting_line.type')}")
    print(f"Counting line position: {config.get('counting_line.position')}")
    print(f"Counting line coordinates: {config.get('counting_line.coordinates')}")
    print(f"Counting direction: {config.get('counting_line.direction')}")
    print(f"Target classes: {config.get('data.target_classes')}")
    print(f"Min confidence: {config.get('counting.min_confidence')}")
    
    # Test 2: Line coordinates for video resolution
    print("\n📏 STEP 2: Line Coordinates Analysis")
    print("-" * 30)
    video_path = 'inference_video/output.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    counting_line = CountingLine(config)
    line_coords = counting_line.get_line_coordinates(frame_width, frame_height)
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Line coordinates: {line_coords}")
    print(f"Line direction: {counting_line.direction}")
    print(f"Line type: {counting_line.line_type}")
    
    # Test 3: Process a few frames and examine detection/tracking
    print("\n🎯 STEP 3: Detection & Tracking Analysis")
    print("-" * 30)
    
    cap = cv2.VideoCapture(video_path)
    pipeline.crossing_detector.set_frame_dimensions(frame_width, frame_height)
    
    frame_count = 0
    detection_count = 0
    tracking_count = 0
    
    for i in range(50):  # Test first 50 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Get detections
        detection_results = pipeline.detector.detect(frame)
        detections = detection_results["detections"]
        detections = pipeline.detector.apply_nms(detections)
        
        if detections:
            detection_count += 1
            print(f"Frame {frame_count}: {len(detections)} detections")
            for i, det in enumerate(detections):
                print(f"  Det {i}: confidence={det['confidence']:.3f}, class={det['class_name']}, bbox={det['bbox']}")
        
        # Update tracking
        pipeline.tracker.update(detections)
        tracked_objects = pipeline.tracker.get_tracks()
        
        if tracked_objects:
            tracking_count += 1
            print(f"Frame {frame_count}: {len(tracked_objects)} tracked objects")
            
            for obj_id, track_data in tracked_objects.items():
                trajectory = track_data["trajectory"]
                centroid = track_data["centroid"]
                confidence = track_data["confidence"]
                class_name = track_data["class_name"]
                
                print(f"  Track {obj_id}: centroid={centroid}, confidence={confidence:.3f}, traj_len={len(trajectory)}")
                
                # Test crossing detection
                crossing_direction = pipeline.crossing_detector.detect_crossing(obj_id, trajectory)
                if crossing_direction:
                    print(f"    🚨 CROSSING DETECTED: {crossing_direction}")
                    
                    # Test counting
                    counted = pipeline.counter.process_crossing(
                        obj_id, crossing_direction, centroid, confidence, class_name, frame_count
                    )
                    print(f"    📊 COUNTED: {counted}")
                else:
                    # Debug why no crossing detected
                    if len(trajectory) >= 3:
                        print(f"    🔍 Trajectory analysis:")
                        print(f"      Start: {trajectory[0]}, End: {trajectory[-1]}")
                        
                        # Test line side calculation
                        start_side = pipeline.crossing_detector._get_point_side(trajectory[0], line_coords)
                        end_side = pipeline.crossing_detector._get_point_side(trajectory[-1], line_coords)
                        print(f"      Start side: {start_side}, End side: {end_side}")
                        
                        if start_side != end_side and start_side != 0 and end_side != 0:
                            # Should detect crossing, but doesn't - check direction validation
                            recent_traj = trajectory[-3:]
                            detected_dir = pipeline.crossing_detector._analyze_trajectory_crossing(recent_traj, line_coords)
                            print(f"      Detected direction: {detected_dir}")
                            
                            if detected_dir:
                                is_valid = pipeline.crossing_detector._is_valid_crossing_direction(detected_dir)
                                print(f"      Direction valid: {is_valid} (expected: {counting_line.direction})")
    
    cap.release()
    
    print(f"\n📊 SUMMARY:")
    print(f"Frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Frames with tracking: {tracking_count}")
    print(f"Total objects counted: {pipeline.counter.total_count}")
    
    # Test 4: Manual trajectory crossing test
    print("\n🧪 STEP 4: Manual Trajectory Crossing Test")
    print("-" * 30)
    
    # Test with manual trajectories that should cross the line
    test_trajectories = [
        # Left to right crossing (should count)
        [(800, 500), (900, 500), (1000, 500)],  # Crosses middle line left to right
        # Right to left crossing (should NOT count - wrong direction)
        [(1100, 500), (1000, 500), (900, 500)],  # Crosses middle line right to left
        # No crossing
        [(800, 500), (850, 500), (900, 500)],   # Stays on left side
    ]
    
    for i, traj in enumerate(test_trajectories):
        print(f"\nTest trajectory {i+1}: {traj}")
        crossing_dir = pipeline.crossing_detector._analyze_trajectory_crossing(traj, line_coords)
        print(f"  Detected direction: {crossing_dir}")
        
        if crossing_dir:
            is_valid = pipeline.crossing_detector._is_valid_crossing_direction(crossing_dir)
            print(f"  Valid direction: {is_valid}")
            
            if is_valid:
                # Test if it would be counted
                counted = pipeline.counter.process_crossing(
                    999 + i, crossing_dir, traj[-1], 0.95, "red_box", 999
                )
                print(f"  Would be counted: {counted}")

def test_line_side_calculation():
    """Test the line side calculation with known points"""
    print("\n🧮 BONUS: Line Side Calculation Test")
    print("-" * 30)
    
    config = Config()
    counting_line = CountingLine(config)
    
    # For 1920x1080 video, middle vertical line should be at x=960
    frame_width, frame_height = 1920, 1080
    line_coords = counting_line.get_line_coordinates(frame_width, frame_height)
    print(f"Line coordinates: {line_coords}")
    
    # Test points
    test_points = [
        (800, 500, "Left side"),   # Should be -1
        (960, 500, "On line"),     # Should be 0  
        (1100, 500, "Right side"), # Should be 1
    ]
    
    crossing_detector = LineCrossingDetector(config, counting_line)
    
    for x, y, desc in test_points:
        side = crossing_detector._get_point_side((x, y), line_coords)
        print(f"Point ({x}, {y}) - {desc}: side = {side}")

if __name__ == "__main__":
    debug_counting_pipeline()
    test_line_side_calculation() 