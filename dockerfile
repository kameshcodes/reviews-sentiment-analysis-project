# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    python -m venv venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*  # Clean up after installation

# Copy application code
COPY . .

# Run additional setup script if needed
RUN . /app/venv/bin/activate && python /app/src/setup-nltk.py

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH" 

# Expose the Streamlit port
EXPOSE 8502

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
