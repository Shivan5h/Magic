FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with timeout and retries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 300 -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Set working directory to backend
WORKDIR /app/backend

# Expose port (Railway will provide $PORT)
EXPOSE 8000

# Run the application (use Railway's $PORT)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
