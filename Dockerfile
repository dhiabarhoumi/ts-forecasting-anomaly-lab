FROM python:3.11.8-slim-bookworm@sha256:3a891a748b60e3e9c62e6bb99c43f0bda2ebfcf9e428c6f3bb8f81e50e7d984a

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts/mlruns artifacts/reports data/retail data/energy

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=/app/artifacts/mlruns

# Default command
CMD ["python", "-m", "src.cli.backtest", "--help"]
