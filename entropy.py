"""Entropy calculation module for video frame analysis."""

import logging
import cv2
import numpy as np
import yt_dlp
from PIL import Image
from config import MAX_VIDEO_HEIGHT, DEFAULT_FRAME_SKIP

logger = logging.getLogger(__name__)


def calculate_intra_entropy(image):
    """Calculate Shannon entropy of pixel intensities."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small value to avoid log(0)
    return entropy


def calculate_inter_entropy(current_image, previous_image):
    """Calculate entropy of the difference between frames."""
    if previous_image is None:
        return 0.0
    # Convert to grayscale
    current_gray = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(np.array(previous_image), cv2.COLOR_RGB2GRAY)
    # Calculate difference
    diff = cv2.absdiff(current_gray, previous_gray)
    # Calculate histogram of difference
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy


def fetch_two_spaced_frames(url, frame_skip=DEFAULT_FRAME_SKIP, max_height=None):
    """Fetch two frames with some spacing between them from the YouTube live stream using yt-dlp."""
    if max_height is None:
        max_height = MAX_VIDEO_HEIGHT
    ydl_opts = {
        'format': f'best[height<={max_height}]',  # Limit resolution for efficiency
        'quiet': True,
        'no_warnings': True,
    }
    bitrate = None
    resolution = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # For live streams, we need to get the stream URL
            if 'formats' in info and info['formats']:
                # Select the best format
                formats = [f for f in info['formats'] if f.get('height', 0) <= max_height]
                if not formats:
                    formats = info['formats']
                format = max(formats, key=lambda f: f.get('height', 0) or 0)
                resolution = f"{format.get('width', 0)}x{format.get('height', 0)}"
                stream_url = format['url']

                # Calculate effective bitrate by measuring data for 1 second worth of playback frames
                import time
                import urllib.request

                # Use OpenCV to capture frames and measure bitrate
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open stream URL: {stream_url}")
                    return None, None, None, None

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30  # Default fallback FPS
                    logger.warning(f"Could not determine FPS, using default: {fps}")
                else:
                    logger.info(f"Video FPS: {fps}")

                # Calculate how many frames we need for 1 second of playback
                frames_for_one_second = int(fps)
                logger.info(f"Will capture {frames_for_one_second} frames for 1 second of playback")

                try:
                    # First, capture frames for entropy calculation and timing
                    frames_captured = 0
                    frame1 = None
                    frame2 = None

                    # Capture frames for 1 second of playback plus frame_skip
                    target_frames = frames_for_one_second + frame_skip + 1
                    start_time = time.time()

                    while frames_captured < target_frames:
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning(f"Stream ended after {frames_captured} frames")
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
                    logger.info(f"Captured {frames_captured} frames in {frame_capture_time:.2f}s")

                    # Now measure data download for the equivalent playback duration
                    playback_duration = frames_captured / fps
                    if playback_duration > 0:
                        logger.info(f"Measuring data download for {playback_duration:.2f}s of playback content")

                        # Use yt-dlp's opener for data measurement
                        opener = ydl._opener
                        response = opener.open(stream_url, timeout=10)

                        bytes_downloaded = 0
                        download_start = time.time()

                        try:
                            # Download data for the calculated playback duration
                            while time.time() - download_start < playback_duration:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                bytes_downloaded += len(chunk)
                        finally:
                            response.close()

                        actual_download_time = time.time() - download_start
                        bitrate = (bytes_downloaded * 8) / playback_duration  # bits per second
                        logger.info(f"Effective bitrate: {bitrate:.0f} bps ({bytes_downloaded} bytes for {playback_duration:.2f}s of playback content, downloaded in {actual_download_time:.2f}s)")
                    else:
                        bitrate = None
                        logger.warning("Could not calculate effective bitrate - no frames captured")

                except Exception as e:
                    logger.warning(f"Failed to measure effective bitrate: {e}")
                    # Fallback to format info
                    if 'tbr' in format and format['tbr']:
                        bitrate = format['tbr'] * 1000
                        logger.info(f"Fallback bitrate from format tbr: {bitrate:.0f} bps")
                    else:
                        bitrate = None

                    # Still need to capture frames for entropy calculation
                    ret1, frame1 = cap.read()
                    if not ret1:
                        logger.error("Failed to read first frame from stream")
                        cap.release()
                        return None, None, bitrate, resolution

                    # Skip frames to get more temporal separation
                    frames_skipped = 0
                    for i in range(frame_skip):
                        ret_skip, _ = cap.read()
                        if ret_skip:
                            frames_skipped += 1
                        else:
                            logger.warning(f"Only skipped {frames_skipped} frames out of {frame_skip} requested")
                            break

                    # Read second frame after skipping
                    ret2, frame2 = cap.read()
                    if not ret2:
                        logger.error("Failed to read second frame from stream")
                        frame2 = frame1  # Use first frame twice if we can't get a second frame

                cap.release()

                if frame1 is None:
                    logger.error("No frames captured from stream")
                    return None, None, bitrate, resolution

                if frame2 is None:
                    logger.warning("Only one frame captured, using it for both entropy calculations")
                    frame2 = frame1

                return (Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)),
                        Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), bitrate, resolution)
            else:
                logger.error("No formats available for the stream")
    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download error for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching frames from {url}: {e}")
    return None, None, bitrate, None
