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
        self.decay_factor = 0.5
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_SUBTRACTOR_HISTORY,
            # Reduce threshold
            varThreshold=BG_SUBTRACTOR_VAR_THRESHOLD * 0.8,
            detectShadows=BG_SUBTRACTOR_DETECT_SHADOWS
        )
        self.last_motion_value = 0

    def _calculate_regional_motion(self, fg_mask: np.ndarray) -> float:
        """Calculate motion using smaller, overlapping regions"""
        height, width = fg_mask.shape

        # Create a grid of smaller regions (5x5 grid)
        region_height = height // 5
        region_width = width // 5

        region_metrics = {'regions': [], 'hotspots': []}
        max_regional_motion = 0

        # Analyze overlapping regions with 50% overlap
        for i in range(9):  # More vertical regions with overlap
            for j in range(9):  # More horizontal regions with overlap
                start_y = (i * region_height) // 2
                start_x = (j * region_width) // 2
                end_y = min(start_y + region_height, height)
                end_x = min(start_x + region_width, width)

                region = fg_mask[start_y:end_y, start_x:end_x]
                region_motion = np.sum(region > 0) / region.size

                if region_motion > max_regional_motion:
                    max_regional_motion = region_motion

                if region_motion > self.motion_threshold * 0.3:  # Lower threshold for regions
                    region_metrics['hotspots'].append({
                        'position': [start_x, start_y, end_x, end_y],
                        'motion': float(region_motion)
                    })

                region_metrics['regions'].append({
                    'position': [start_x, start_y, end_x, end_y],
                    'motion': float(region_motion)
                })

        region_metrics['max_regional_motion'] = float(max_regional_motion)
        region_metrics['num_hotspots'] = len(region_metrics['hotspots'])

        # Consider motion detected if we have any significant hotspots
        motion_score = max_regional_motion if region_metrics['num_hotspots'] > 0 else 0

        return motion_score, region_metrics

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
        motion_score, region_metrics = self._calculate_regional_motion(
            fg_mask)

        # Use decayed motion but with more emphasis on current frame
        decayed_motion = (motion_score * 0.7 + self.last_motion_value * 0.3)
        self.last_motion_value = decayed_motion
        self.motion_history.append(decayed_motion)

        # Weighted average with more emphasis on recent frames
        weights = np.linspace(1, 0.3, len(self.motion_history))
        weights = weights / np.sum(weights)
        motion_average = np.average(list(self.motion_history), weights=weights)

        # Motion is detected if either:
        # 1. We have strong localized motion (hotspots)
        # 2. We have sustained moderate motion
        motion_detected = (motion_score > self.motion_threshold * 0.5 or
                           (motion_average > self.motion_threshold * 0.3 and
                            region_metrics['num_hotspots'] > 0))

        debug_info = {
            'region_metrics': region_metrics,
            'motion_score': float(motion_score),
            'decayed_motion': float(decayed_motion),
            'motion_average': float(motion_average),
            'threshold': float(self.motion_threshold),
            'motion_detected': motion_detected,
            'num_hotspots': region_metrics['num_hotspots']
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
        position_factor = 1.0 + (center_y / image_height) * 0.5
        adjusted_threshold = (self.motion_threshold * 0.6) / \
            position_factor  # Lower threshold for distant objects

        motion_fraction = np.sum(roi > 0) / roi.size
        is_moving = motion_fraction > adjusted_threshold

        debug_info = {
            'bbox': [x1, y1, x2, y2],
            'center_y': float(center_y),
            'position_factor': float(position_factor),
            'adjusted_threshold': float(adjusted_threshold),
            'motion_fraction': float(motion_fraction),
            'is_moving': is_moving,
            'roi_size': roi.size,
            'moving_pixels': int(np.sum(roi > 0))
        }

        return is_moving, debug_info
