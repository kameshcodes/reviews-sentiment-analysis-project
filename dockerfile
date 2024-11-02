FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install -r requirements.txt

ENV PATH="/app/venv/bin:$PATH"

COPY . .

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8502/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
