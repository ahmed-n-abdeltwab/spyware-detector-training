# ========== BUILD STAGE ==========
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========== RUNTIME STAGE ==========
FROM python:3.9-slim

# Set Python path
ENV PYTHONPATH=/app

# Create directory structure with proper permissions
RUN mkdir -p /app && \
    mkdir -p /app/data && \
    mkdir -p /app/models && \
    mkdir -p /app/release && \
    chmod -R 777 /app

# Create non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# Copy virtual environment
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD python -c "import sys; sys.exit(0)"

# Entry point
CMD ["python", "src/main.py"]
