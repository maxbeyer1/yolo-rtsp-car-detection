"""
Core vehicle detector implementation.
"""
import os
import logging
import time
import queue
import threading
from pathlib import Path
from typing import Tuple
import shutil
from datetime import datetime
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import numpy as np

from src.detector.motion_detector import MotionDetector
from src.detector.utils import create_car_folder, get_timestamp, get_timestamps_in_folder
from src.config.settings import (
    FRAME_QUEUE_SIZE,
    CAR_EVENT_TIMEOUT,
    CROP_RATIO,
    MAX_CAPTURE_RETRIES,
    CAPTURE_RETRY_DELAY
)
from src.monitoring.system_metrics import SystemMetricsCollector


class MovingVehicleDetector:
    """Class to detect moving vehicles in a parking lot using YOLOv11 and background subtraction."""

    def __init__(self, rtsp_url: str, output_dir: str, debug_mode: bool = False,
                 confidence_threshold: float = 0.5, min_detection_interval: float = 1.0,

                 image_size: Tuple[int, int] = (640, 640),
                 motion_threshold: float = 0.03,
                 motion_history: int = 5):
        self.debug_mode = debug_mode

        # Initialize logging
        logging_level = logging.DEBUG if debug_mode else logging.INFO
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/vehicle_collector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Running in %s mode",
                         ('debug' if debug_mode else 'production'))

        # Initialize threads
        self.capture_thread = None
        self.process_thread = None
        self.cleanup_thread = None

        # Configuration
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.min_detection_interval = min_detection_interval
        self.image_size = image_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize YOLO model
        try:
            self.model = YOLO("yolo11s.pt")
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error("Failed to load YOLO model: %s", e)
            raise

        # Initialize motion detector
        self.motion_detector = MotionDetector(motion_threshold, motion_history)

        # Initialize Roboflow if config is provided
        self.roboflow_project = None
        if not debug_mode:
            try:
                rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
                self.roboflow_project = rf.workspace(os.getenv('ROBOFLOW_WORKSPACE_ID'))\
                    .project(os.getenv('ROBOFLOW_PROJECT_ID'))
                self.logger.info(
                    "Roboflow connection initialized successfully")
            except Exception as e:
                self.logger.error("Failed to initialize Roboflow: %s", e)

        self.last_save_time = 0
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.is_running = False

        # Output format variables
        self.current_car_folder = None
        self.last_detection_time = 0

        # Initialize system metrics collector for debugging
        self.metrics_collector = SystemMetricsCollector(self.output_dir)

    def _run_cleanup(self):
        """Periodic cleanup task"""
        while self.is_running:
            try:
                self._cleanup_old_detections()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
            time.sleep(3600)  # Run every hour

    def start_capture(self):
        """Start the capture process in a separate thread"""
        self.is_running = True
        self.metrics_collector.start()  # Start metrics collection
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.process_thread = threading.Thread(target=self._process_frames)
        self.cleanup_thread = threading.Thread(target=self._run_cleanup)

        self.capture_thread.start()
        self.process_thread.start()
        self.cleanup_thread.start()

    def stop_capture(self):
        """Stop the capture process"""
        self.is_running = False
        self.metrics_collector.stop()  # Stop and save metrics
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        if hasattr(self, 'cleanup_thread'):
            self.cleanup_thread.join()

    def _capture_frames(self):
        """Capture frames from RTSP stream"""
        retry_count = 0

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

                    height, width = frame.shape[:2]
                    crop_pixels = int(width * CROP_RATIO)
                    frame = frame[:, crop_pixels:]
                    frame = cv2.resize(frame, self.image_size)

                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        continue

            except Exception as e:
                self.logger.error("Stream capture error: %s", e)
                retry_count += 1

                if retry_count >= MAX_CAPTURE_RETRIES:
                    self.logger.error("Max retries reached. Stopping capture.")
                    self.is_running = False
                    break

                time.sleep(CAPTURE_RETRY_DELAY)

            finally:
                if 'cap' in locals():
                    cap.release()

    def _process_frames(self):
        """Process frames for moving vehicle detection"""
        consecutive_detections = 0
        frame_count = 0

        while self.is_running:
            try:
                start_time = time.time()
                frame_count += 1

                frame = self.frame_queue.get(timeout=1.0)
                motion_detected, motion_mask, motion_debug = self.motion_detector.detect_motion(
                    frame)

                if frame_count % 30 == 0:  # Log every 30 frames
                    self.logger.debug(
                        "Frame %d - Motion debug: %s", frame_count, motion_debug)

                if motion_detected:
                    results = self.model(frame, verbose=False)

                    for result in results:
                        car_detections = [
                            box for box in result.boxes
                            if box.cls == 2 and box.conf >= self.confidence_threshold
                        ]

                        # moving_cars = []
                        # for box in car_detections:
                        #     bbox = box.xyxy[0].cpu().numpy()
                        #     if self.motion_detector.is_vehicle_moving(bbox, motion_mask):
                        #         moving_cars.append(box)

                        # current_time = time.time()
                        # if moving_cars and (current_time - self.last_save_time) >= self.min_detection_interval:
                        #     self._save_detection(frame, result, motion_mask)
                        #     self.last_save_time = current_time

                        if car_detections:
                            self.logger.debug("Frame %d - Detected %d vehicles",
                                              frame_count, len(car_detections))

                            for i, box in enumerate(car_detections):
                                conf = float(box.conf)
                                bbox = box.xyxy[0].cpu().numpy()

                                is_moving, motion_info = self.motion_detector.is_vehicle_moving(
                                    bbox, motion_mask)

                                # self.logger.debug(
                                #     "Frame %d - Vehicle %d: confidence=%.3f, motion_info=%s",
                                #     frame_count, i, conf, motion_info
                                # )

                                if is_moving:
                                    consecutive_detections += 1
                                    self.logger.info(
                                        "Frame %d - Moving vehicle detected! consecutive_detections=%d",
                                        frame_count, consecutive_detections
                                    )
                                else:
                                    consecutive_detections = max(
                                        0, consecutive_detections - 1)
                        else:
                            consecutive_detections = max(
                                0, consecutive_detections - 1)

                        current_time = time.time()
                        if consecutive_detections >= 2 and (current_time - self.last_save_time) >= self.min_detection_interval:
                            self.logger.info(
                                "Frame %d - Saving detection (consecutive_detections=%d)",
                                frame_count, consecutive_detections
                            )
                            self._save_detection(frame, result, motion_mask)
                            self.last_save_time = current_time

                processing_time = time.time() - start_time
                self.metrics_collector.record_processing_time(processing_time)

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Frame processing error: %s", e)

    def _save_detection(self, frame, result, motion_mask):
        """Save detected frame with timestamp and motion visualization"""
        try:
            current_time = time.time()
            timestamp = get_timestamp()

            if (self.current_car_folder is None or
                    (current_time - self.last_detection_time) > CAR_EVENT_TIMEOUT):
                self.current_car_folder = create_car_folder(
                    self.output_dir, timestamp)
                self.logger.info(
                    "Started tracking new car event: %s", timestamp)

            self.last_detection_time = current_time

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            filename = self.current_car_folder / f"vehicle_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame, encode_params)

            annotated_frame = result.plot()
            motion_overlay = np.zeros_like(annotated_frame)
            motion_overlay[motion_mask > 0] = [0, 0, 255]
            annotated_frame = cv2.addWeighted(
                annotated_frame, 1.0, motion_overlay, 0.3, 0)

            anno_filename = self.current_car_folder / \
                "annotated" / f"vehicle_{timestamp}_annotated.jpg"
            cv2.imwrite(str(anno_filename), annotated_frame)

            self.logger.info("Saved detection in %s: %s",
                             self.current_car_folder.name, filename.name)

            if not self.debug_mode:
                self._upload_to_roboflow(frame, timestamp)

        except Exception as e:
            self.logger.error("Error saving detection: %s", e)

    def _cleanup_old_detections(self, max_age_days=7):
        """Remove processed images older than max_age_days that are confirmed uploaded"""
        current_time = time.time()

        for car_folder in self.output_dir.iterdir():
            if car_folder.is_dir() and car_folder.name != "metrics":
                folder_time = datetime.strptime(
                    car_folder.name, "%Y%m%d_%H%M%S_%f").timestamp()

                if (current_time - folder_time) > (max_age_days * 24 * 3600):
                    # Only delete if confirmed uploaded to Roboflow
                    if all((car_folder / f"{ts}.uploaded").exists()
                           for ts in get_timestamps_in_folder(car_folder)):
                        try:
                            shutil.rmtree(car_folder)
                            self.logger.info(
                                "Cleaned up %s", car_folder.name)
                        except Exception as e:
                            self.logger.error(
                                "Failed to clean up %s: %s", car_folder.name, e)

    def _upload_to_roboflow(self, frame: np.ndarray, timestamp: str) -> None:
        """Upload frame to Roboflow API with retry mechanism"""
        if not self.roboflow_project:
            return

        try:
            image_path = str(self.current_car_folder /
                             f"vehicle_{timestamp}.jpg")
            self.roboflow_project.upload(
                image_path=image_path,
                num_retry_uploads=3,
            )
            self.logger.info(
                "Successfully uploaded image %s to Roboflow", timestamp)

            # Mark as uploaded (could use a simple .uploaded file or database)
            (self.current_car_folder / f"{timestamp}.uploaded").touch()

        except Exception as e:
            self.logger.error(
                "Failed to upload image %s to Roboflow: %s", timestamp, e)
