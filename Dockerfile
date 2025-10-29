FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies only when needed. We keep the image lean by
# ensuring temporary apt caches are removed.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libportaudio2 \
        libportaudiocpp0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so Docker layer caching works when source
# files change less frequently than dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy only the source folders required by the FastAPI application.
COPY app ./app
COPY src ./src

EXPOSE 8000

# Start the FastAPI application with Uvicorn, honoring the PORT env var if provided.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
