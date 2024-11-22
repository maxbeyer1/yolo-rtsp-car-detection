# Start with the official Ultralytics CPU image as base
FROM ultralytics/ultralytics:latest-cpu

# Set working directory
WORKDIR /app

# Install additional system dependencies
# Install EPEL repository first
RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
    yum update -y && \
    yum install -y \
        mesa-libGL \
        glib2 \
        python3-devel \
        gcc \
    && yum clean all \
    && rm -rf /var/cache/yum

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