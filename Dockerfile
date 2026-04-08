FROM python:3.11-slim

WORKDIR /app

# Copy server requirements first for Docker layer caching
COPY server/requirements.txt ./server_requirements.txt
RUN pip install --no-cache-dir -r server_requirements.txt

# Copy all project files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
