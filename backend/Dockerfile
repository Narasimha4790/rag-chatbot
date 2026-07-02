FROM python:3.12-slim

# Install light-weight compiler tooling if needed for package extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install python dependencies
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

# Copy API backend source code
COPY backend /workspace/backend

# Expose backend port
EXPOSE 8000

ENV PYTHONPATH=/workspace

# Start FastAPI app
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
