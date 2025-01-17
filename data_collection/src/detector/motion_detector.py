"""
Motion detection functionality for vehicle tracking.
"""
from collections import deque
from typing import Tuple, List
import logging
import cv2
import numpy as np

from src.config.settings import (
    BG_SUBTRACTOR_HISTORY,
    BG_SUBTRACTOR_VAR_THRESHOLD,
    BG_SUBTRACTOR_DETECT_SHADOWS,
    MOTION_KERNEL_SIZE
)


class MotionDetector:
    def __init__(self, motion_threshold: float, motion_history: int):
        self.logger = logging.getLogger(__name__)
        self.motion_threshold = motion_threshold
        self.motion_history = deque(maxlen=motion_history)
        self.decay_factor = 0.7
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_SUBTRACTOR_HISTORY,
            varThreshold=BG_SUBTRACTOR_VAR_THRESHOLD,
            detectShadows=BG_SUBTRACTOR_DETECT_SHADOWS
        )
        self.last_motion_value = 0

    def _calculate_regional_motion(self, fg_mask: np.ndarray) -> float:
        """Calculate motion using regions to account for distance"""
        height, width = fg_mask.shape

        # Define regions (near, middle, far)
        regions = [
            (0, height//3),      # Near region (bottom third)
            (height//3, 2*height//3),  # Middle region
            (2*height//3, height)      # Far region (top third)
        ]

        # Different thresholds for different regions
        # Increase sensitivity for distant regions
        region_names = ['near', 'middle', 'far']
        region_weights = [1.0, 1.5, 2.0]

        region_metrics = {}
        total_motion = 0
        total_weight = 0

        for (start_y, end_y), weight, name in zip(regions, region_weights, region_names):
            region = fg_mask[start_y:end_y, :]
            region_motion = np.sum(region > 0) / region.size
            weighted_motion = region_motion * weight
            total_motion += weighted_motion
            total_weight += weight

            region_metrics = {
                'raw_motion': float(region_motion),
                'weighted_motion': float(weighted_motion),
                'pixels_in_motion': int(np.sum(region > 0)),
                'total_pixels': region.size
            }

        avg_motion = total_motion / total_weight
        return avg_motion, region_metrics

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect motion in frame using background subtraction with decay"""
        # Get initial foreground mask
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MOTION_KERNEL_SIZE)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Calculate current motion with decay
        current_motion, region_metrics = self._calculate_regional_motion(
            fg_mask)
        decayed_motion = (current_motion * (1 - self.decay_factor) +
                          self.last_motion_value * self.decay_factor)

        self.last_motion_value = decayed_motion
        self.motion_history.append(decayed_motion)

        # motion_fraction = np.sum(fg_mask > 0) / fg_mask.size
        # self.motion_history.append(motion_fraction)
        # motion_detected = np.mean(
        #     list(self.motion_history)) > self.motion_threshold

        # Use weighted average for motion detection
        weights = np.linspace(1, 0.5, len(self.motion_history))
        weights = weights / np.sum(weights)

        motion_average = np.average(list(self.motion_history), weights=weights)
        motion_detected = motion_average > self.motion_threshold

        debug_info = {
            'region_metrics': region_metrics,
            'current_motion': float(current_motion),
            'decayed_motion': float(decayed_motion),
            'motion_average': float(motion_average),
            'threshold': float(self.motion_threshold),
            'motion_detected': motion_detected
        }

        # self.logger.debug("Motion metrics: %s", debug_info)

        return motion_detected, fg_mask, debug_info

    def is_vehicle_moving(self, bbox: List[int], motion_mask: np.ndarray) -> bool:
        """Determine if a detected vehicle is moving based on the motion mask"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = motion_mask[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # Calculate the vertical position of the bbox center
        center_y = (y1 + y2) / 2
        image_height = motion_mask.shape[0]

        # Adjust threshold based on vertical position (distance from camera)
        # 1.0 at bottom, 2.0 at top
        position_factor = 1.0 + (center_y / image_height)
        adjusted_threshold = self.motion_threshold / \
            position_factor  # Lower threshold for distant objects

        motion_fraction = np.sum(roi > 0) / roi.size
        return motion_fraction > adjusted_threshold
