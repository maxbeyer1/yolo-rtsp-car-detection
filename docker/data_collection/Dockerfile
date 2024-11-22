# Start with the official Ultralytics CPU image as base
FROM ultralytics/ultralytics:latest-arm64

# Set working directory
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY data_collection/requirements.txt .

# Install additional Python packages not included in Ultralytics image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file first
COPY data_collection/.env .

# Copy the data collection module
COPY data_collection/ .

# Create directories for logs and detections
RUN mkdir -p logs vehicle_detections

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the detection script with IPC host setting
CMD ["python", "main.py"]