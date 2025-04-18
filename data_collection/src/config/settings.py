"""
Configuration settings for the vehicle detection system.
"""
# Default configurations
# DEFAULT_RTSP_URL = "rtsp://wb:vQ7E4HiVkwr17bQqX2ild7XlAFvFhfUoqulBwSYm@camerapi:8554/parking-lot-cam"
# DEFAULT_RTSP_URL = "rtsp://wb:vQ7E4HiVkwr17bQqX2ild7XlAFvFhfUoqulBwSYm@100.103.159.23:8554/parking-lot-cam"
DEFAULT_RTSP_URL = "rtsp://24.30.252.59/axis-media/media.amp?camera=1&videoframeskipmode=empty&videozprofile=classic&resolution=1280x720&timestamp=1&videocodec=h264"
DEFAULT_OUTPUT_DIR = "vehicle_detections"

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MIN_DETECTION_INTERVAL = 1.0
DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_MOTION_THRESHOLD = 0.03
DEFAULT_MOTION_HISTORY = 5

# Background subtractor settings
BG_SUBTRACTOR_HISTORY = 500
BG_SUBTRACTOR_VAR_THRESHOLD = 16
BG_SUBTRACTOR_DETECT_SHADOWS = True

# Motion detection settings
MOTION_KERNEL_SIZE = (3, 3)

# Car event timeout (seconds)
CAR_EVENT_TIMEOUT = 3.0

# Queue settings
FRAME_QUEUE_SIZE = 30

# Stream capture settings
MAX_CAPTURE_RETRIES = 5
CAPTURE_RETRY_DELAY = 5  # seconds

# Image crop settings
CROP_RATIO = 800 / 2560
