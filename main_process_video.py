#!/usr/bin/env python3
"""
CounterAI Video Processing Script

Processes a video file to count objects crossing a defined line.
Automatically loads trained model and generates counting results.
"""

import argparse
import os
import sys
import cv2
from pathlib import Path
from main_pipeline import CounterAIPipeline
from utils import Config

def process_video_with_display(pipeline, input_video, output_video, show_display):
    """Custom video processing with fitted display window"""
    import time
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    pipeline.crossing_detector.set_frame_dimensions(frame_width, frame_height)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Setup window with automatic fitting
    if show_display:
        cv2.namedWindow("CounterAI Processing", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("CounterAI Processing", frame_width, frame_height)
        print(f"📐 Video resolution: {frame_width}x{frame_height}")
        print(f"🖼️  Window: Auto-fitted to screen with aspect ratio preserved")
    
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame = pipeline._process_frame(frame, frame_count)
            
            out.write(processed_frame)
            
            if show_display:
                cv2.imshow("CounterAI Processing", processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⛔ Processing stopped by user")
                    break
                elif key == ord(' '):
                    print("⏸️ Paused - Press any key to continue...")
                    cv2.waitKey(0)
                    print("▶️ Resumed")
            
            # Progress logging
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                progress = (frame_count / total_frames) * 100
                elapsed = current_time - start_time
                fps_actual = frame_count / elapsed
                print(f"⏳ Progress: {progress:.1f}% | FPS: {fps_actual:.1f} | Frame {frame_count}/{total_frames}")
                last_log_time = current_time
    
    finally:
        cap.release()
        out.release()
        if show_display:
            cv2.destroyAllWindows()
    
    processing_time = time.time() - start_time
    average_fps = frame_count / processing_time if processing_time > 0 else 0
    
    # Get counting results
    counting_results = pipeline.counter.get_detailed_statistics()
    
    # Save results
    csv_path = pipeline.counter.save_results_csv()
    json_path = pipeline.counter.save_results_json()
    
    return {
        'processing_time': processing_time,
        'frames_processed': frame_count,
        'average_fps': average_fps,
        'counting_results': counting_results,
        'output_files': {
            'video': output_video,
            'csv': csv_path,
            'json': json_path
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="CounterAI Video Processing - Count objects crossing a line in video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_video.mp4
  %(prog)s input_video.mp4 --output processed_video.mp4
  %(prog)s input_video.mp4 --no-display --output results.mp4
  %(prog)s input_video.mp4 --config custom_config.json
        """
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: auto-generated in output/ folder)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show real-time video display during processing"
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
    
    # Validate input video
    if not os.path.exists(args.input_video):
        print(f"❌ Error: Input video '{args.input_video}' not found")
        sys.exit(1)
    
    # Validate model
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file '{args.model}' not found")
        print(f"💡 Run training first: python3 main_pipeline.py")
        sys.exit(1)
    
    # Generate output path if not provided
    if not args.output:
        input_name = Path(args.input_video).stem
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/processed_{input_name}_{timestamp}.mp4"
    
    # Ensure output directory exists
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    print("🎯 CounterAI Video Processing")
    print("=" * 40)
    print(f"📹 Input video: {args.input_video}")
    print(f"💾 Output video: {args.output}")
    print(f"🤖 Model: {args.model}")
    print(f"⚙️  Config: {args.config}")
    print(f"🖥️  Real-time display: {'No' if args.no_display else 'Yes'}")
    print()
    
    try:
        # Initialize pipeline
        print("🔧 Initializing CounterAI pipeline...")
        pipeline = CounterAIPipeline(args.config)
        pipeline.setup_pipeline()
        
        # Load model
        print("📦 Loading trained model...")
        pipeline.load_pretrained_model(args.model)
        
        # Process video with fitted display
        print("🎬 Processing video...")
        results = process_video_with_display(
            pipeline,
            args.input_video,
            args.output,
            show_display=not args.no_display
        )
        
        # Display results
        print("\n✅ Video processing completed!")
        print("📊 COUNTING RESULTS:")
        print("-" * 20)
        counting_results = results['counting_results']
        print(f"🎯 Total objects counted: {counting_results['total_count']}")
        print(f"➡️  Left-to-right: {counting_results['count_by_direction']['left_to_right']}")
        print(f"⬅️  Right-to-left: {counting_results['count_by_direction']['right_to_left']}")
        print(f"🔍 Target classes: {', '.join(counting_results['target_classes'])}")
        print(f"⏱️  Processing time: {results['processing_time']:.1f} seconds")
        print(f"🎞️  Frames processed: {results['frames_processed']}")
        print(f"⚡ Average FPS: {results['average_fps']:.1f}")
        
        print(f"\n📁 OUTPUT FILES:")
        for file_type, file_path in results['output_files'].items():
            print(f"  {file_type.upper()}: {file_path}")
        
    except KeyboardInterrupt:
        print("\n⛔ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 