# ExamForge — Production Dockerfile
# Builds a containerized OpenEnv environment for HuggingFace Spaces
# Port: 7860 (required by HuggingFace)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir openenv-core fastapi uvicorn pydantic

# Create non-root user for security (HuggingFace best practice)
RUN useradd -m -u 1000 examforge && chown -R examforge:examforge /app
USER examforge

# HuggingFace Spaces REQUIRES port 7860
EXPOSE 7860

# Health check so HF knows when container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--timeout-keep-alive", "30"]
