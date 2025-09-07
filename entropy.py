"""Entropy calculation module for video frame analysis."""

import logging
import time
import urllib.request

import cv2
import numpy as np
import yt_dlp
from PIL import Image

from config import DEFAULT_FRAME_SKIP, MAX_VIDEO_HEIGHT

logger = logging.getLogger(__name__)


def calculate_spatial_entropy(image):
    """Calculate color-aware Shannon entropy across HSV channels."""
    img_array = np.array(image)
    if img_array.shape[2] != 3:
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    channels = cv2.split(hsv)
    channel_names = ["hue", "saturation", "value"]
    entropies = {}

    for name, channel in zip(channel_names, channels):
        # For hue, use 180 bins (0-179 in OpenCV)
        bins = 180 if name == "hue" else 256
        hist_range = [0, 180] if name == "hue" else [0, 256]
        hist = cv2.calcHist([channel], [0], None, [bins], hist_range)
        hist = hist / hist.sum()  # Normalize
        # Calculate entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        entropies[name] = entropy

    return entropies


def calculate_temporal_entropy(current_image, previous_image):
    """Calculate color-aware temporal entropy across HSV channel differences."""
    if previous_image is None:
        return {"hue": 0.0, "saturation": 0.0, "value": 0.0}

    current_array = np.array(current_image)
    previous_array = np.array(previous_image)
    if current_array.shape[2] != 3 or previous_array.shape[2] != 3:
        return

    # Convert to HSV
    current_hsv = cv2.cvtColor(current_array, cv2.COLOR_RGB2HSV)
    previous_hsv = cv2.cvtColor(previous_array, cv2.COLOR_RGB2HSV)
    current_channels = cv2.split(current_hsv)
    previous_channels = cv2.split(previous_hsv)

    channel_names = ["hue", "saturation", "value"]
    entropies = {}

    for name, curr_ch, prev_ch in zip(
        channel_names, current_channels, previous_channels
    ):
        # Calculate difference
        diff = cv2.absdiff(curr_ch, prev_ch)
        # For hue, use 180 bins
        bins = 180 if name == "hue" else 256
        hist_range = [0, 180] if name == "hue" else [0, 256]
        hist = cv2.calcHist([diff], [0], None, [bins], hist_range)
        hist = hist / hist.sum()
        # Calculate entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        entropies[name] = entropy

    return entropies


def fetch_two_spaced_frames(
    url, frame_skip=DEFAULT_FRAME_SKIP, max_height=None
):
    """Fetch two frames with some spacing between them from the YouTube live stream using yt-dlp."""
    if max_height is None:
        max_height = MAX_VIDEO_HEIGHT

    # Ensure max_height is a valid integer
    try:
        max_height = int(max_height)
        if max_height <= 0:
            max_height = MAX_VIDEO_HEIGHT
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid max_height value: {max_height}, using default"
        )
        max_height = MAX_VIDEO_HEIGHT

    logger.debug(f"Using max_height: {max_height} (type: {type(max_height)})")
    try:
        format_string = f"best[height<={max_height}]"
        logger.debug(f"Format string: {format_string}")
        ydl_opts = {
            "format": format_string,
            "quiet": True,
            "no_warnings": True,
        }
    except Exception as e:
        logger.warning(
            f"Failed to create format string with max_height {max_height}: {e}"
        )
        ydl_opts = {
            "format": "best",
            "quiet": True,
            "no_warnings": True,
        }
    bitrate = None
    resolution = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # For live streams, we need to get the stream URL
            if "formats" in info and info["formats"]:
                # Select the highest resolution format available (no height limit for max quality)
                formats = [
                    f for f in info["formats"] if f.get("height", 0) > 0
                ]
                if not formats:
                    formats = info["formats"]
                format = max(formats, key=lambda f: f.get("height", 0) or 0)
                resolution = (
                    f"{format.get('width', 0)}x{format.get('height', 0)}"
                )
                stream_url = format["url"]
                logger.debug(
                    f"Stream URL: {stream_url[:100]}..."
                )  # Log first 100 chars of URL

                # Calculate effective bitrate by measuring data for 1 second worth of playback frames

                # First, check if the stream URL is a manifest file
                is_manifest = False
                try:
                    with urllib.request.urlopen(
                        stream_url, timeout=5
                    ) as check_response:
                        first_bytes = check_response.read(100)
                        if b"#EXTM3U" in first_bytes or b"<MPD" in first_bytes:
                            is_manifest = True
                            logger.info(
                                "Stream URL points to HLS/DASH manifest file"
                            )
                        elif (
                            first_bytes.decode("utf-8", errors="ignore")
                            .strip()
                            .startswith("#")
                        ):
                            is_manifest = True
                            logger.info(
                                "Stream URL appears to be a playlist/manifest"
                            )
                except Exception as e:
                    logger.debug(f"Failed to check stream URL: {e}")

                if is_manifest:
                    # Handle manifest-based streams - measure actual bitrate from segments
                    logger.info(
                        "Processing HLS/DASH manifest for measured bitrate"
                    )
                    try:
                        import m3u8

                        playlist = m3u8.load(stream_url)
                        if playlist.segments:
                            logger.debug(
                                f"Found {len(playlist.segments)} segments in HLS playlist"
                            )
                            # Download minimum segments to get ~1 second of content
                            segment_bytes = 0
                            segments_downloaded = 0
                            total_segment_duration = 0.0
                            target_duration = 1.0  # Only need 1 second

                            for segment in playlist.segments:
                                if total_segment_duration >= target_duration:
                                    break
                                try:
                                    with urllib.request.urlopen(
                                        segment.uri, timeout=5
                                    ) as seg_response:
                                        seg_data = seg_response.read()
                                        segment_bytes += len(seg_data)
                                        segments_downloaded += 1
                                        # Add this segment's duration (default to 6 seconds if not specified)
                                        segment_duration = getattr(
                                            segment, "duration", 6.0
                                        )
                                        total_segment_duration += (
                                            segment_duration
                                        )
                                        logger.debug(
                                            f"Downloaded video segment {segments_downloaded}: {len(seg_data)} bytes ({segment_duration}s duration)"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to download segment: {e}"
                                    )
                                    continue

                            if (
                                segment_bytes > 0
                                and total_segment_duration > 0
                            ):
                                # Calculate measured bitrate from actual video segments
                                bitrate = (
                                    segment_bytes * 8
                                ) / total_segment_duration
                                logger.info(
                                    f"Measured bitrate: {bitrate:.0f} bps ({segment_bytes} bytes for {total_segment_duration:.2f}s of content)"
                                )
                            else:
                                logger.error(
                                    "Could not download video segments for bitrate measurement"
                                )
                                bitrate = None
                        else:
                            logger.error("No segments found in HLS playlist")
                            bitrate = None
                    except Exception as e:
                        logger.error(f"Failed to process manifest: {e}")
                        bitrate = None

                    # Get frames for entropy calculation from the manifest stream
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    frame1 = None
                    frame2 = None
                    if cap.isOpened():
                        ret1, frame1 = cap.read()
                        if ret1:
                            # Skip frames to get temporal separation
                            for i in range(frame_skip):
                                cap.read()
                            ret2, frame2 = cap.read()
                            if not ret2:
                                frame2 = frame1
                        cap.release()
                    else:
                        logger.error(
                            "Could not open manifest stream for frame capture"
                        )
                        frame1 = frame2 = None
                else:
                    # Handle direct video streams
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        logger.error(
                            f"Failed to open stream URL: {stream_url}"
                        )
                        return None, None, None, None

                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30  # Default fallback FPS
                        logger.warning(
                            f"Could not determine FPS, using default: {fps}"
                        )
                    else:
                        logger.info(f"Video FPS: {fps}")

                    # Calculate how many frames we need for 1 second of playback
                    frames_for_one_second = int(fps)
                    logger.info(
                        f"Will capture {frames_for_one_second} frames for 1 second of playback"
                    )

                    try:
                        # Capture frames for entropy calculation and timing
                        frames_captured = 0
                        frame1 = None
                        frame2 = None

                        # Capture frames for 1 second of playback plus frame_skip
                        target_frames = frames_for_one_second + frame_skip + 1
                        start_time = time.time()

                        while frames_captured < target_frames:
                            ret, frame = cap.read()
                            if not ret:
                                logger.warning(
                                    f"Stream ended after {frames_captured} frames"
                                )
                                break

                            # Store first and spaced frames for entropy calculation
                            if frames_captured == 0:
                                frame1 = frame.copy()
                            elif frames_captured == frame_skip:
                                frame2 = frame.copy()

                            frames_captured += 1

                            # Add a small delay to prevent overwhelming the stream
                            if frames_captured % 10 == 0:
                                time.sleep(0.01)  # 10ms pause every 10 frames

                        frame_capture_time = time.time() - start_time
                        logger.info(
                            f"Captured {frames_captured} frames in {frame_capture_time:.2f}s"
                        )

                        # Measure bitrate by downloading data for 1 second
                        playback_duration = 1.0  # Just measure 1 second
                        logger.info(
                            f"Measuring data download for {playback_duration:.2f}s of playback content"
                        )

                        bytes_downloaded = 0
                        download_start = time.time()
                        chunks_read = 0

                        logger.debug(
                            f"Starting download measurement for {playback_duration:.2f}s using urllib"
                        )
                        try:
                            with urllib.request.urlopen(
                                stream_url, timeout=10
                            ) as response:
                                # Download data for 1 second
                                while (
                                    time.time() - download_start
                                    < playback_duration
                                ):
                                    chunk = response.read(8192)
                                    if not chunk:
                                        logger.debug(
                                            f"Download ended after {chunks_read} chunks (empty chunk)"
                                        )
                                        break
                                    bytes_downloaded += len(chunk)
                                    chunks_read += 1
                        except Exception as e:
                            logger.warning(f"Urllib download failed: {e}")
                            bytes_downloaded = 0

                        if bytes_downloaded > 0:
                            bitrate = (
                                bytes_downloaded * 8
                            ) / playback_duration
                            logger.info(
                                f"Measured bitrate: {bitrate:.0f} bps ({bytes_downloaded} bytes for {playback_duration:.2f}s of content)"
                            )
                        else:
                            logger.error(
                                "Could not download data for bitrate measurement"
                            )
                            bitrate = None

                    except Exception as e:
                        logger.warning(f"Failed to measure bitrate: {e}")
                        bitrate = None

                if frame1 is None:
                    logger.error("No frames captured from stream")
                    return None, None, bitrate, resolution

                if frame2 is None:
                    logger.warning(
                        "Only one frame captured, using it for both entropy calculations"
                    )
                    frame2 = frame1

                return (
                    Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)),
                    Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)),
                    bitrate,
                    resolution,
                )
            else:
                logger.error("No formats available for the stream")
    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download error for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching frames from {url}: {e}")
    return None, None, bitrate, None
