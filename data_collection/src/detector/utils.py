"""
Utility functions for vehicle detection system.
"""
import logging
from typing import List
from datetime import datetime
from pathlib import Path


def create_car_folder(base_dir: Path, timestamp: str) -> Path:
    """Create a new folder structure for a car detection event"""
    car_folder = base_dir / timestamp
    car_folder.mkdir(parents=True, exist_ok=True)

    annotated_folder = car_folder / "annotated"
    annotated_folder.mkdir(parents=True, exist_ok=True)

    return car_folder


def get_timestamp() -> str:
    """Generate timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def get_timestamps_in_folder(folder_path: Path) -> List[str]:
    """
    Get all timestamps from image files in the folder.

    Args:
        folder_path (Path): Path to the folder containing vehicle detection images

    Returns:
        List[str]: List of timestamps extracted from image filenames

    Example:
        For files like:
        - vehicle_20241121_153022_123456.jpg
        - vehicle_20241121_153023_234567.jpg
        Returns: ['20241121_153022_123456', '20241121_153023_234567']
    """
    timestamps = []

    # Only look at jpg files that start with 'vehicle_'
    for image_file in folder_path.glob('vehicle_*.jpg'):
        # Extract timestamp from filename
        # filename format: vehicle_YYYYMMDD_HHMMSS_microseconds.jpg
        try:
            # Remove 'vehicle_' prefix and '.jpg' suffix
            timestamp = image_file.stem.replace('vehicle_', '')
            timestamps.append(timestamp)
        except Exception as e:
            logging.getLogger(__name__).error(
                "Error extracting timestamp from %s: %s", image_file, e)
            continue

    # Don't include timestamps from the 'annotated' subdirectory
    return [t for t in timestamps if (folder_path / f"vehicle_{t}.jpg").exists()]
