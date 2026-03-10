FROM python:3.9-slim

# Install system dependencies (ffmpeg is required for librosa/audioread)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: In a production environment with GPU, usage of pytorch-cuda or a specific base image (e.g., nvcr.io/nvidia/pytorch) is recommended.
# Here we stick to CPU default for broad compatibility unless GPU is specified.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
