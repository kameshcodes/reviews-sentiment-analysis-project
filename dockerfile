# Use a slim Python base image
FROM python:3.9-slim

# Set environment variables to avoid bytecode and buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Create and activate a virtual environment, then install dependencies
RUN python -m venv venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt

COPY . /app

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8502/health || exit 1

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
