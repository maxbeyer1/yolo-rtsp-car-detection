"""
Motion detection functionality for vehicle tracking.
"""
import cv2
import numpy as np
from collections import deque
from typing import Tuple, List

from src.config.settings import (
    BG_SUBTRACTOR_HISTORY,
    BG_SUBTRACTOR_VAR_THRESHOLD,
    BG_SUBTRACTOR_DETECT_SHADOWS,
    MOTION_KERNEL_SIZE
)

class MotionDetector:
    def __init__(self, motion_threshold: float, motion_history: int):
        self.motion_threshold = motion_threshold
        self.motion_history = deque(maxlen=motion_history)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_SUBTRACTOR_HISTORY,
            varThreshold=BG_SUBTRACTOR_VAR_THRESHOLD,
            detectShadows=BG_SUBTRACTOR_DETECT_SHADOWS
        )

    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect motion in frame using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MOTION_KERNEL_SIZE)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        motion_fraction = np.sum(fg_mask > 0) / fg_mask.size
        self.motion_history.append(motion_fraction)
        motion_detected = np.mean(list(self.motion_history)) > self.motion_threshold

        return motion_detected, fg_mask

    def is_vehicle_moving(self, bbox: List[int], motion_mask: np.ndarray) -> bool:
        """Determine if a detected vehicle is moving based on the motion mask"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = motion_mask[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        motion_fraction = np.sum(roi > 0) / roi.size
        return motion_fraction > self.motion_threshold
