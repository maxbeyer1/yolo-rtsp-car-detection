"""
This script demonstrates how to detect moving vehicles in a parking lot using YOLOv11 and OpenCV.
"""
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import time
import queue
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict
import cv2
from ultralytics import YOLO
import numpy as np


class MovingVehicleDetector:
    """
    Class to detect moving vehicles in a parking lot using YOLOv11 and background subtraction.
    
    Args:
        rtsp_url (str): RTSP stream URL for video capture
        output_dir (str): Directory to save detected vehicle images
        confidence_threshold (float): Minimum confidence threshold for YOLO detections
        min_detection_interval (float): Minimum time between saved detections
        image_size (Tuple[int, int]): Standard size for YOLO input images
        motion_threshold (float): Minimum fraction of pixels that must show motion
        motion_history (int): Number of frames to keep for motion analysis
    """
    def __init__(
        self,
        rtsp_url: str,
        output_dir: str,
        debug_mode: bool = False,
        confidence_threshold: float = 0.5,
        min_detection_interval: float = 1.0,
        image_size: Tuple[int, int] = (640, 640),
        motion_threshold: float = 0.03,  # Minimum fraction of pixels that must show motion
        motion_history: int = 5  # Number of frames to keep for motion analysis
    ):
        self.debug_mode = debug_mode
        
        # Initialize logging with different levels based on debug mode
        logging_level = logging.DEBUG if debug_mode else logging.INFO
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vehicle_collector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log debug mode status
        self.logger.info(f"Running in {'debug' if debug_mode else 'production'} mode")
        
        # Configuration
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.min_detection_interval = min_detection_interval
        self.image_size = image_size
        self.motion_threshold = motion_threshold
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model
        try:
            self.model = YOLO("yolo11s.pt")
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Motion detection variables
        self.motion_history = deque(maxlen=motion_history)
        
        self.last_save_time = 0
        self.frame_queue = queue.Queue(maxsize=30)
        self.is_running = False
        
        # Output format variables
        self.current_car_folder = None
        self.last_detection_time = 0
        self.car_event_timeout = 3.0  # Consider it a new car if more than 3 seconds have passed
        
    def start_capture(self):
        """Start the capture process in a separate thread"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.process_thread = threading.Thread(target=self._process_frames)
        
        self.capture_thread.start()
        self.process_thread.start()
        
    def stop_capture(self):
        """Stop the capture process"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            
    def _capture_frames(self):
        """Capture frames from RTSP stream"""
        retry_count = 0
        max_retries = 5
        
        while self.is_running:
            try:
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    raise ConnectionError("Failed to open RTSP stream")
                
                self.logger.info("Successfully connected to RTSP stream")
                retry_count = 0
                
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        raise ConnectionError("Failed to read frame")
                    
                    # Resize frame to standard size
                    #frame = cv2.resize(frame, self.image_size)
                    
                    # Get original dimensions
                    height, width = frame.shape[:2]
                    
                    # Calculate crop amount from left (keeping roughly 1680px worth of the original)
                    # Original ratio: 880:2560 ≈ 0.34375 (wall portion)
                    crop_ratio = 800 / 2560  # Crop slightly less than the wall portion
                    crop_pixels = int(width * crop_ratio)
                    
                    # Crop from left side
                    frame = frame[:, crop_pixels:]
                    
                    # Now resize to target size (640x640)
                    frame = cv2.resize(frame, self.image_size)
                    
                    # Add frame to queue, skip if queue is full
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        continue
                    
            except Exception as e:
                self.logger.error(f"Stream capture error: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    self.logger.error("Max retries reached. Stopping capture.")
                    self.is_running = False
                    break
                    
                time.sleep(5)  # Wait before retrying
                
            finally:
                if 'cap' in locals():
                    cap.release()
                    
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Detect motion in frame using background subtraction

        Args:
            frame: Input frame from video stream
            
        Returns:
            motion_detected: True if motion is detected in the frame
            fg_mask: Foreground mask showing motion areas
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (they are marked as 127 in the mask)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate the fraction of pixels showing motion
        motion_fraction = np.sum(fg_mask > 0) / fg_mask.size
        
        # Store motion history
        self.motion_history.append(motion_fraction)
        
        # Consider motion detected if the average motion over recent frames exceeds threshold
        motion_detected = np.mean(list(self.motion_history)) > self.motion_threshold
        
        return motion_detected, fg_mask
    
    def is_vehicle_moving(self, bbox: List[int], motion_mask: np.ndarray) -> bool:
        """
        Determine if a detected vehicle is moving based on the motion mask
        
        Args:
        bbox: [x1, y1, x2, y2]
        motion_mask: Foreground mask showing motion areas
        
        Returns: True if the vehicle is moving
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = motion_mask[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
            
        # Calculate the fraction of pixels showing motion in the vehicle's bounding box
        motion_fraction = np.sum(roi > 0) / roi.size
        return motion_fraction > self.motion_threshold
    
    def _process_frames(self):
        """Process frames for moving vehicle detection"""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Detect motion in frame
                motion_detected, motion_mask = self.detect_motion(frame)
                
                # Only process with YOLO if motion is detected
                if motion_detected:
                    # Run YOLO detection
                    results = self.model(frame, verbose=False)
                    
                    for result in results:
                        # Filter for car class (typically class 2 in COCO dataset)
                        car_detections = [
                            box for box in result.boxes
                            if box.cls == 2 and box.conf >= self.confidence_threshold
                        ]
                        
                        # Check each car detection for motion
                        moving_cars = []
                        for box in car_detections:
                            bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                            if self.is_vehicle_moving(bbox, motion_mask):
                                moving_cars.append(box)
                        
                        # Save detection if moving cars found and enough time has passed
                        current_time = time.time()
                        if moving_cars and (current_time - self.last_save_time) >= self.min_detection_interval:
                            self._save_detection(frame, result, motion_mask)
                            self.last_save_time = current_time
                            
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
                
    def _create_new_car_folder(self, timestamp: str) -> Path:
        """
        Create a new folder structure for a new car detection event
        
        Args:
            timestamp: Timestamp string for folder naming
            
        Returns:
            Path object for the new car folder
        """
        # Create main car folder
        car_folder = self.output_dir / timestamp
        car_folder.mkdir(parents=True, exist_ok=True)
        
        # Create annotated subfolder
        annotated_folder = car_folder / "annotated"
        annotated_folder.mkdir(parents=True, exist_ok=True)
        
        return car_folder
    
    def _save_detection(self, frame: np.ndarray, result, motion_mask: np.ndarray) -> None:
        """Save detected frame with timestamp and motion visualization
        
        Args:
            frame: Original frame from video stream
            result: YOLO detection result
            motion_mask: Foreground mask showing motion areas
        """
        try:
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Check if this is a new car event
            if (self.current_car_folder is None or 
                (current_time - self.last_detection_time) > self.car_event_timeout):
                # Create new car folder using this timestamp
                self.current_car_folder = self._create_new_car_folder(timestamp)
                self.logger.info(f"Started tracking new car event: {timestamp}")
            
            # Update last detection time
            self.last_detection_time = current_time
            
            # Save original frame in car folder root
            filename = self.current_car_folder / f"vehicle_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            
            # Create visualization with both YOLO detections and motion
            annotated_frame = result.plot()
            
            # Overlay motion mask (in semi-transparent red)
            motion_overlay = np.zeros_like(annotated_frame)
            motion_overlay[motion_mask > 0] = [0, 0, 255]  # Red color
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, motion_overlay, 0.3, 0)
            
            # Save annotated frame in annotated subfolder
            anno_filename = self.current_car_folder / "annotated" / f"vehicle_{timestamp}_annotated.jpg"
            cv2.imwrite(str(anno_filename), annotated_frame)
            
            self.logger.info(f"Saved detection in {self.current_car_folder.name}: {filename.name}")
            
            # If not in debug mode, upload to Roboflow API
            if not self.debug_mode:
                self._upload_to_roboflow(frame, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error saving detection: {e}")
            
    def _upload_to_roboflow(self, frame: np.ndarray, timestamp: str) -> None:
        """
        Upload frame to Roboflow API (placeholder for future implementation)
        """
        # TODO: Implement Roboflow API upload
        self.logger.debug(f"Would upload frame {timestamp} to Roboflow API in production mode")
        pass

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
        default="rtsp://wb:vQ7E4HiVkwr17bQqX2ild7XlAFvFhfUoqulBwSYm@camerapi:8554/parking-lot-cam",
        help='RTSP stream URL'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="vehicle_detections",
        help='Output directory for detected vehicles'
    )
    return parser.parse_args()

def main():
    """
    Main function to start detection
    """
    # Configuration
    # rtsp_url = "rtsp://wb:vQ7E4HiVkwr17bQqX2ild7XlAFvFhfUoqulBwSYm@camerapi:8554/parking-lot-cam"
    # output_dir = "vehicle_detections"
    
    # Parse command line arguments
    args = parse_args()
    
    # Initialize detector with optimized parameters for parking lot scenario
    detector = MovingVehicleDetector(
        rtsp_url=args.rtsp_url,
        output_dir=args.output_dir,
        debug_mode=args.debug,
        confidence_threshold=0.5,          # YOLO detection confidence
        min_detection_interval=1.0,        # Minimum time between saved detections
        image_size=(640, 640),            # Standard YOLO input size
        motion_threshold=0.03,             # Adjust based on testing (3% of pixels showing motion)
        motion_history=5                   # Number of frames to consider for motion
    )
    
    try:
        # Start collection
        detector.start_capture()
        
        # Run until interrupted
        print(f"Started moving vehicle detection in {'debug' if args.debug else 'production'} mode. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping detection...")
    finally:
        detector.stop_capture()
        print("Detection stopped successfully.")

if __name__ == "__main__":
    main()
