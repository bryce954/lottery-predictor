# Dockerfile for Lottery Predictor
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run the API server (create_app is the ASGI factory in cloud.py)
CMD ["uvicorn", "lottery_predictor.cloud:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
