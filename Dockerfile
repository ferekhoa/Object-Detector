FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200  

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --retries 5 --timeout 1200 -r requirements.txt

COPY . .

RUN useradd -m appuser && \
    mkdir -p static/results && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
CMD ["/bin/sh","-lc","gunicorn -k gthread --threads 4 --timeout 120 -b 0.0.0.0:$PORT app:app"]
