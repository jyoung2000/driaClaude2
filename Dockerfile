FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=4144 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID/GID that works with unRAID
ARG PUID=99
ARG PGID=100
RUN groupadd -g ${PGID} appuser && \
    useradd -m -u ${PUID} -g appuser appuser

# Create application directories with proper permissions
RUN mkdir -p /app /app/uploads /app/outputs /app/voices /app/cache && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies as root (for system packages)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Create directories for model cache
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 4144

# Set volume mount points
VOLUME ["/app/uploads", "/app/outputs", "/app/voices", "/home/appuser/.cache/huggingface"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:4144/health || exit 1

# Run the enhanced application
CMD ["python", "app_enhanced.py"]