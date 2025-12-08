FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential needed for some numpy/pandas builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ /app/src/
COPY my_portfolio.txt /app/
COPY .env /app/
COPY models/ /app/models/

# Set Python path so imports work
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command (overridden by docker-compose)
CMD ["python", "src/bot_standalone.py"]
