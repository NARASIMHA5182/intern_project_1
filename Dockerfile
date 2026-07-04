# Dockerfile - Credit Card Approval Prediction System
FROM python:3.11-slim

# Set environment paths and configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

WORKDIR /app

# Install system dependencies (build-essential and libgomp1 are required for XGBoost/LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY . .

# Run pre-training step so the system has a pre-trained ML model immediately on container boot
RUN python preprocessing/data_loader.py && python training/trainer.py

EXPOSE 5000

# Run with Gunicorn production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
