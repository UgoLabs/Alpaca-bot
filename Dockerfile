FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY config/ /app/config/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY my_portfolio.txt /app/
COPY day_trade_list.txt /app/
COPY .env /app/

# Set Python path
ENV PYTHONPATH=/app

# Default command (overridden by docker-compose)
CMD ["python", "-m", "src.bots.swing_trader"]
