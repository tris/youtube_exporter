"""Flask routes module for YouTube exporter."""

import logging
import threading
import time

from flask import Flask, Response, request

from config import (
    CHANNEL_ID_PATTERN,
    DEFAULT_INTERVAL,
    MIN_INTERVAL,
    VIDEO_ID_PATTERN,
)
from metrics import (
    get_prometheus_metrics,
    update_channel_metrics,
    update_metrics,
)

logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Disable Flask/Werkzeug logs
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def periodic_update(
    video_id,
    interval=DEFAULT_INTERVAL,
    fetch_images=True,
    max_height=None,
    match=None,
):
    """Periodically update video metrics."""
    while True:
        update_metrics(video_id, fetch_images, max_height, match)
        time.sleep(interval)


def periodic_update_channel(
    channel_id,
    interval=DEFAULT_INTERVAL,
    fetch_images=True,
    disable_live=False,
    match=None,
):
    """Periodically update channel metrics."""
    while True:
        update_channel_metrics(channel_id, fetch_images, disable_live, match)
        time.sleep(interval)


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    video_id = request.args.get("video_id")
    channel_id = request.args.get("channel")

    # Check if both or neither parameters are provided
    if not video_id and not channel_id:
        return Response(
            "Missing required query parameter: video_id (YouTube video ID) or channel (YouTube channel ID)",
            status=400,
        )
    if video_id and channel_id:
        return Response(
            "Provide either video_id (YouTube video ID) or channel (YouTube channel ID), not both",
            status=400,
        )

    # Get optional parameters
    fetch_images = request.args.get("fetch_images", "true").lower() == "true"
    disable_live = request.args.get("disable_live", "false").lower() == "true"
    max_height_str = request.args.get("max_height")
    max_height = None
    if max_height_str:
        try:
            max_height = int(max_height_str)
            if max_height <= 0:
                max_height = None
        except (ValueError, TypeError):
            logger.warning(f"Invalid max_height parameter: {max_height_str}")
            max_height = None
    match = request.args.get("match")  # Object to count in the image
    logger.debug(
        f"Request params: video_id={video_id}, channel_id={channel_id}, fetch_images={fetch_images}, match={match}"
    )
    interval_str = request.args.get("interval", str(DEFAULT_INTERVAL))
    try:
        interval = int(interval_str)
        if interval < MIN_INTERVAL:  # Minimum 30 seconds
            interval = MIN_INTERVAL
    except ValueError:
        interval = DEFAULT_INTERVAL

    if video_id:
        # Handle video metrics
        if not VIDEO_ID_PATTERN.match(video_id):
            return Response("Invalid video ID format", status=400)

        # Fetch metrics immediately for the first request
        # Always do image processing if match is provided, otherwise skip for speed
        first_request_fetch_images = fetch_images or (match is not None)
        logger.debug(
            f"Calling update_metrics with fetch_images={first_request_fetch_images}, match={match}"
        )
        update_metrics(
            video_id,
            fetch_images=first_request_fetch_images,
            max_height=max_height,
            match=match,
        )

        # Start periodic update if not already running or parameters changed
        thread_key = f"video_{video_id}_{fetch_images}_{interval}"
        if not hasattr(metrics, "threads"):
            metrics.threads = {}

        if (
            thread_key not in metrics.threads
            or not metrics.threads[thread_key].is_alive()
        ):
            metrics.threads[thread_key] = threading.Thread(
                target=periodic_update,
                args=(video_id, interval, fetch_images, max_height, match),
                daemon=True,
            )
            metrics.threads[thread_key].start()

    elif channel_id:
        # Handle channel metrics
        if not CHANNEL_ID_PATTERN.match(channel_id):
            return Response("Invalid channel ID format", status=400)

        # Fetch metrics immediately for the first request (skip images for speed)
        # Don't pass match parameter to immediate call to avoid duplicate object detection
        update_channel_metrics(
            channel_id,
            fetch_images=False,
            disable_live=disable_live,
            match=None,
        )

        # Start periodic update if not already running or parameters changed
        thread_key = f"channel_{channel_id}_{fetch_images}_{disable_live}_{interval}_{match}"
        if not hasattr(metrics, "threads"):
            metrics.threads = {}

        if (
            thread_key not in metrics.threads
            or not metrics.threads[thread_key].is_alive()
        ):
            metrics.threads[thread_key] = threading.Thread(
                target=periodic_update_channel,
                args=(channel_id, interval, fetch_images, disable_live, match),
                daemon=True,
            )
            metrics.threads[thread_key].start()

    # Generate and return Prometheus metrics
    formatted_output = get_prometheus_metrics()
    return Response(formatted_output, mimetype="text/plain")


@app.route("/health")
def health():
    """Health check endpoint."""
    return Response("OK", status=200)


@app.route("/")
def index():
    """Root endpoint with usage information."""
    usage_info = """
YouTube Exporter

Usage:
- /metrics?video_id=VIDEO_ID - Get metrics for a specific video
- /metrics?channel=CHANNEL_ID - Get metrics for a specific channel
- /health - Health check endpoint

Optional parameters:
- fetch_images=true/false - Whether to fetch and analyze video frames (default: true)
- disable_live=true/false - Whether to disable live stream detection for channels (default: false)
- interval=SECONDS - Update interval in seconds (default: 300, minimum: 30)
- match=OBJECT - Count identifiable objects in the image (e.g., match=paraglider)

Examples:
- /metrics?video_id=dQw4w9WgXcQ
- /metrics?channel=UCuAXFkgsw1L7xaCfnd5JJOw
- /metrics?video_id=dQw4w9WgXcQ&fetch_images=false&interval=60
"""
    return Response(usage_info, mimetype="text/plain")


def get_flask_app():
    """Get the Flask application instance."""
    return app
