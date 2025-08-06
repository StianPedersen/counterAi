#!/usr/bin/env python3
"""
CounterAI Live Camera Processing Script

Processes live camera feed to count objects crossing a defined line.
Automatically detects available cameras and allows user selection.
"""

import argparse
import cv2
import os
import sys
import time
import numpy as np
from typing import List, Dict, Any
from main_pipeline import CounterAIPipeline
from utils import Config

def detect_cameras(max_cameras: int = 10) -> List[int]:
    """Detect available cameras by trying to open them"""
    available_cameras = []
    
    print("🔍 Detecting available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm camera is working
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"  📷 Camera {i}: {width}x{height} @ {fps:.1f}fps")
                available_cameras.append(i)
            cap.release()
        else:
            # Stop searching after first unavailable camera index
            if i > 0 and len(available_cameras) == 0:
                break
    
    return available_cameras

def select_camera(available_cameras: List[int]) -> int:
    """Let user select camera from available options"""
    if not available_cameras:
        print("❌ No cameras detected!")
        sys.exit(1)
    
    if len(available_cameras) == 1:
        camera_id = available_cameras[0]
        print(f"✅ Using only available camera: {camera_id}")
        return camera_id
    
    print(f"\n📷 Available cameras: {available_cameras}")
    
    while True:
        try:
            selection = input(f"Select camera ID {available_cameras}: ")
            camera_id = int(selection)
            if camera_id in available_cameras:
                return camera_id
            else:
                print(f"❌ Invalid selection. Choose from: {available_cameras}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n⛔ Cancelled by user")
            sys.exit(1)

def test_camera(camera_id: int, duration: int = 3) -> bool:
    """Test camera and show preview"""
    print(f"\n🧪 Testing camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return False
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Camera resolution: {width}x{height}")
    print(f"⚡ Camera FPS: {fps:.1f}")
    print(f"⏱️  Testing for {duration} seconds...")
    print("💡 Press 'q' to quit test early")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        # Add test overlay
        cv2.putText(frame, f"Camera {camera_id} Test", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to continue", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Setup window on first frame
        if frame_count == 1:
            cv2.namedWindow(f"Camera {camera_id} Test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(f"Camera {camera_id} Test", width, height)
        
        cv2.imshow(f"Camera {camera_id} Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or elapsed >= duration:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    actual_fps = frame_count / elapsed
    print(f"✅ Camera test completed. Actual FPS: {actual_fps:.1f}")
    return True

def live_camera_processing(camera_id: int, pipeline: CounterAIPipeline, save_video: bool = False):
    """Main live camera processing loop"""
    print(f"\n🎬 Starting live camera processing...")
    print("💡 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset counter")
    print("  - Press SPACE to pause/resume")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize pipeline for this resolution
    pipeline.crossing_detector.set_frame_dimensions(width, height)
    
    # Setup window with automatic fitting
    cv2.namedWindow("CounterAI Live Camera", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("CounterAI Live Camera", width, height)
    print(f"📐 Camera resolution: {width}x{height}")
    print(f"🖼️  Window: Auto-fitted to screen with aspect ratio preserved")
    
    # Video writer for saving
    video_writer = None
    if save_video:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"output/live_camera_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        print(f"💾 Saving video to: {output_path}")
    
    frame_count = 0
    start_time = time.time()
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Process frame through CounterAI pipeline
                processed_frame = pipeline._process_frame(frame, frame_count)
                
                # Add live info overlay
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(processed_frame, f"Live Camera {camera_id}", (10, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (10, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show processed frame
                cv2.imshow("CounterAI Live Camera", processed_frame)
                
                # Save frame if recording
                if video_writer:
                    video_writer.write(processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"output/screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, processed_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('r'):
                # Reset counter
                pipeline.counter.reset_counter()
                pipeline.tracker.reset()
                print("🔄 Counter and tracker reset")
            elif key == ord(' '):
                # Pause/resume
                paused = not paused
                print(f"⏸️ {'Paused' if paused else '▶️ Resumed'}")
    
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final results
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print(f"🎞️  Frames processed: {frame_count}")
        print(f"⚡ Average FPS: {avg_fps:.1f}")
        print(f"🎯 Total objects counted: {pipeline.counter.total_count}")
        
        # Save final results
        csv_path = pipeline.counter.save_results_csv("live_camera_session")
        json_path = pipeline.counter.save_results_json("live_camera_session")
        print(f"💾 Results saved: {csv_path}, {json_path}")

def main():
    parser = argparse.ArgumentParser(
        description="CounterAI Live Camera Processing - Count objects crossing a line from camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Auto-detect and select camera
  %(prog)s --camera 0               # Use specific camera
  %(prog)s --save-video             # Save processed video
  %(prog)s --no-test                # Skip camera test
  %(prog)s --config custom.json    # Use custom configuration
        """
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera ID to use (if not specified, will auto-detect and prompt)"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save processed video to output folder"
    )
    
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip camera test before processing"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--model",
        default="output/model_final.pth",
        help="Path to trained model file (default: output/model_final.pth)"
    )
    
    args = parser.parse_args()
    
    print("📹 CounterAI Live Camera Processing")
    print("=" * 40)
    
    # Validate model
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file '{args.model}' not found")
        print(f"💡 Run training first: python3 main_pipeline.py")
        sys.exit(1)
    
    # Camera selection
    if args.camera is not None:
        camera_id = args.camera
        print(f"📷 Using specified camera: {camera_id}")
    else:
        available_cameras = detect_cameras()
        camera_id = select_camera(available_cameras)
    
    # Test camera
    if not args.no_test:
        if not test_camera(camera_id):
            print("❌ Camera test failed")
            sys.exit(1)
        
        # Confirm to proceed
        try:
            input("\n✅ Camera test successful. Press Enter to start processing (Ctrl+C to cancel)...")
        except KeyboardInterrupt:
            print("\n⛔ Cancelled by user")
            sys.exit(1)
    
    try:
        # Initialize pipeline
        print("\n🔧 Initializing CounterAI pipeline...")
        pipeline = CounterAIPipeline(args.config)
        pipeline.setup_pipeline()
        
        # Load model
        print("📦 Loading trained model...")
        pipeline.load_pretrained_model(args.model)
        
        # Start live processing
        live_camera_processing(camera_id, pipeline, args.save_video)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 