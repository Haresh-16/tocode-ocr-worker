# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies required by OpenCV + MMOCR
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables to be overridden by ECS
ENV SQS_QUEUE_URL=""
ENV S3_BUCKET=""
ENV DYNAMODB_TABLE=""

# Run the OCR worker
CMD ["python", "worker.py"]
