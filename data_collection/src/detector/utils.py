"""
Utility functions for vehicle detection system.
"""
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
