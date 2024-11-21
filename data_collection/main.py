"""
Entry point for the vehicle detection system.
"""
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

from src.detector.vehicle_detector import MovingVehicleDetector
from src.config.settings import DEFAULT_RTSP_URL, DEFAULT_OUTPUT_DIR

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Vehicle detection script for parking lot monitoring'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (local storage only, no API uploads)'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        default=DEFAULT_RTSP_URL,
        help='RTSP stream URL'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for detected vehicles'
    )
    return parser.parse_args()

def main():
    """Main function to start detection"""
    # Load environment variables from .env file
    load_dotenv()

    # Parse command line arguments
    args = parse_args()

    # Initialize detector with optimized parameters for parking lot scenario
    detector = MovingVehicleDetector(
        rtsp_url=args.rtsp_url,
        output_dir=args.output_dir,
        debug_mode=args.debug,
        confidence_threshold=0.5,   # YOLO detection confidence
        min_detection_interval=1.0, # Minimum time between saved detections
        image_size=(640, 640),  # Standard YOLO input size
        # Adjust based on testing (3% of pixels showing motion)
        motion_threshold=0.03,
        motion_history=5    # Number of frames to consider for motion
    )

    try:
        # Start collection
        detector.start_capture()

        # Run until interrupted
        print(f"Started moving vehicle detection in {
              'debug' if args.debug else 'production'} mode. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        detector.stop_capture()
        print("Detection stopped successfully.")

if __name__ == "__main__":
    main()
