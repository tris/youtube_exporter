FROM python:3.13-slim

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache Python deps on requirements hash
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 9473

# Prefer FFmpeg backend in OpenCV to avoid GStreamer
ENV OPENCV_VIDEOIO_PRIORITY_FFMPEG=1

CMD ["./app.py"]
