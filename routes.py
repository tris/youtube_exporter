import logging
import threading
import time

from flask import Flask, Response, request
from prometheus_client import generate_latest

from config import (
    CACHE_THRESHOLD,
    CHANNEL_ID_PATTERN,
    DEFAULT_INTERVAL,
    MIN_INTERVAL,
    VIDEO_ID_PATTERN,
)
from metrics import (
    registry,
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
    match_objects=None,
    debug=False,
    cache_threshold=CACHE_THRESHOLD,
):
    """Periodically update video metrics."""
    while True:
        update_metrics(
            video_id, fetch_images, match_objects, debug, cache_threshold
        )
        time.sleep(interval)


def periodic_update_channel(
    channel_id,
    interval=DEFAULT_INTERVAL,
    fetch_images=True,
    disable_live=False,
    match_objects=None,
    debug=False,
    cache_threshold=CACHE_THRESHOLD,
):
    """Periodically update channel metrics."""
    while True:
        update_channel_metrics(
            channel_id,
            fetch_images,
            disable_live,
            match_objects,
            debug,
            cache_threshold,
        )
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
    debug = request.args.get("debug", "false").lower() in ("true", "1")
    match = request.args.get(
        "match"
    )  # Objects to count in the image (format: "object1:threshold1,object2:threshold2")
    match_objects = {}
    if match:
        # Parse match parameter for multiple objects with individual thresholds
        for item in match.split(","):
            item = item.strip()
            if ":" in item:
                obj, thresh = item.split(":", 1)
                obj = obj.strip()
                try:
                    thresh = float(thresh.strip())
                    if thresh <= 0 or thresh > 1:
                        logger.warning(
                            f"Invalid threshold for {obj}: {thresh}, must be between 0 and 1, using default 0.1"
                        )
                        thresh = 0.1
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid threshold format for {obj}: {thresh}, using default 0.1"
                    )
                    thresh = 0.1
            else:
                obj = item
                thresh = 0.1  # default threshold
            match_objects[obj] = thresh

    logger.debug(
        f"Request params: video_id={video_id}, channel_id={channel_id}, fetch_images={fetch_images}, debug={debug}, match_objects={match_objects}"
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
            match_objects=match_objects,
            debug=debug,
            cache_threshold=interval,
        )

        # Start periodic update if not already running or parameters changed
        thread_key = f"video_{video_id}_{fetch_images}_{debug}_{interval}"
        if not hasattr(metrics, "threads"):
            metrics.threads = {}

        if (
            thread_key not in metrics.threads
            or not metrics.threads[thread_key].is_alive()
        ):
            metrics.threads[thread_key] = threading.Thread(
                target=periodic_update,
                args=(
                    video_id,
                    interval,
                    fetch_images,
                    match_objects,
                    debug,
                    interval,
                ),
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
            match_objects={},
            debug=debug,
            cache_threshold=interval,
        )

        # Start periodic update if not already running or parameters changed
        thread_key = f"channel_{channel_id}_{fetch_images}_{disable_live}_{debug}_{interval}_{match}"
        if not hasattr(metrics, "threads"):
            metrics.threads = {}

        if (
            thread_key not in metrics.threads
            or not metrics.threads[thread_key].is_alive()
        ):
            metrics.threads[thread_key] = threading.Thread(
                target=periodic_update_channel,
                args=(
                    channel_id,
                    interval,
                    fetch_images,
                    disable_live,
                    match_objects,
                    debug,
                    interval,
                ),
                daemon=True,
            )
            metrics.threads[thread_key].start()

    # Generate and return Prometheus metrics
    try:
        output = generate_latest(registry).decode("utf-8")
        return Response(output, mimetype="text/plain")
    except Exception as e:
        logger.error(f"Error in /metrics endpoint: {e}")
        # Return error metrics instead of crashing
        error_response = f"""# HELP youtube_exporter_endpoint_error Indicates an error in the metrics endpoint
# TYPE youtube_exporter_endpoint_error gauge
youtube_exporter_endpoint_error{{endpoint="/metrics"}} 1
# HELP youtube_exporter_endpoint_error_info Error information for metrics endpoint
# TYPE youtube_exporter_endpoint_error_info gauge
youtube_exporter_endpoint_error_info{{endpoint="/metrics",error_message="{str(e).replace('"', '\\"').replace('{', '{{').replace('}', '}}')}"}} 1
"""
        return Response(error_response, mimetype="text/plain", status=500)


@app.route("/health")
def health():
    """Health check endpoint."""
    return Response("OK", status=200)


def get_flask_app():
    """Get the Flask application instance."""
    return app
