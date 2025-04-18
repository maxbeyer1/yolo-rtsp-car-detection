# YOLO-RTSP Car Detection System

Advanced real-time vehicle detection system optimized for parking lot surveillance, leveraging computer vision and deep learning techniques.

## Overview

This system is designed to detect and track moving vehicles in parking lot environments using RTSP camera feeds. It uses a combination of sophisticated technologies for accurate detection:

- **YOLOv11** object detection for real-time vehicle identification
- **Motion detection** with adaptive background subtraction and regional analysis
- **Temporal filtering** with decay factors to reduce false positives
- **Multi-threading** approach for efficient frame capture and processing
- **Roboflow integration** for optional data collection and model improvement

The system is specifically optimized for parking lot scenarios where accurate vehicle movement detection is critical, even in challenging lighting conditions.

## Key Features

- **Moving Vehicle Focus**: Distinguishes between stationary and moving vehicles using sophisticated motion analysis
- **Automated Data Collection**: Captures and organizes vehicle detections with timestamp-based directory structure
- **Edge Processing**: Designed to run efficiently on edge devices (including ARM64 architecture)
- **Advanced Motion Detection**: Uses regional motion metrics and temporal filtering to minimize false positives
- **System Monitoring**: Built-in system metrics collection for performance monitoring
- **Containerized Deployment**: Docker support for easy deployment across environments
- **Configurable Parameters**: Extensive configuration options for different scenarios

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU with sufficient processing power
- Docker and Docker Compose (for containerized deployment)
- RTSP camera stream source

## Installation

### Local Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yolo-rtsp-car-detection.git
   cd yolo-rtsp-car-detection
   ```

2. Install dependencies:

   ```bash
   cd data_collection
   pip install -r requirements.txt
   ```

3. Create a `.env` file with Roboflow credentials (for data collection mode):
   ```
   ROBOFLOW_API_KEY=your_api_key
   ROBOFLOW_WORKSPACE_ID=your_workspace_id
   ROBOFLOW_PROJECT_ID=your_project_id
   ```

### Docker Installation (Recommended)

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/yolo-rtsp-car-detection.git
   cd yolo-rtsp-car-detection
   ```

2. Create a `.env` file in the `data_collection` directory with your credentials

3. (Optional) If you want to use Tailscale, create a `.env` file in the `docker/data-collection` directory with the following content:

   ```
   TS_AUTH_KEY=your_tailscale_auth_key
   ```

4. Build and run the container:
   ```bash
   docker-compose --profile data-collection up -d
   ```

## Usage

### Running the Detection System

```bash
# From the data_collection directory:
python main.py --rtsp-url rtsp://your-camera-url --output-dir vehicle_detections

# Debug mode (local storage only, no API uploads):
python main.py --debug --rtsp-url rtsp://your-camera-url

# With Docker:
docker-compose --profile data-collection up -d
```

### Configuration

Edit `data_collection/src/config/settings.py` to adjust detection parameters:

- Motion sensitivity thresholds
- Detection confidence levels
- Background subtractor parameters
- Image capture and processing settings

## Architecture

The system operates using a multi-threaded architecture:

- **Frame Capture Thread**: Continuously pulls frames from the RTSP stream
- **Processing Thread**: Applies motion detection and YOLO object detection
- **Cleanup Thread**: Manages storage by removing old detections

### Technical Implementation

- Background subtraction using OpenCV's MOG2 algorithm with adaptive parameters
- Regional motion analysis with overlapping grid approach
- Temporal filtering with weighted decay factors
- YOLOv11 model with optimized confidence thresholds
- Multi-level validation pipeline to minimize false positives

## Acknowledgments

- YOLOv11 model from [Ultralytics](https://github.com/ultralytics/yolov5)
