FROM python:3.12-slim

WORKDIR /workspace

# Copy and install python dependencies
COPY frontend/requirements.txt /workspace/frontend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/frontend/requirements.txt

# Copy frontend source code
COPY frontend /workspace/frontend

# Expose Streamlit web port
EXPOSE 8501

# Start Streamlit application
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
