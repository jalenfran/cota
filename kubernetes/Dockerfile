FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY kubernetes/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY kubernetes/collector.py ./collector.py

# Copy GTFS static data (needed for delay computation)
COPY data/cota.gtfs/ ./data/cota.gtfs/

# Create data directory
RUN mkdir -p /data/realtime_history

# Run the collector with unbuffered output for logs
CMD ["python", "-u", "collector.py"]

