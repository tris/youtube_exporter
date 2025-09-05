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
                stream_url = format['url']

                # Calculate actual bitrate by using yt-dlp's opener to download data for 1 second
                import time
                import urllib.request
                
                try:
                    logger.info(f"Starting 1-second download test for bitrate calculation")
                    
                    # Use yt-dlp's opener which has all the proper headers and handling
                    opener = ydl._opener
                    
                    start_time = time.time()
                    bytes_downloaded = 0
                    
                    # Open the stream URL using yt-dlp's opener
                    response = opener.open(stream_url, timeout=5)
                    
                    try:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                                
                            bytes_downloaded += len(chunk)
                            elapsed = time.time() - start_time
                            
                            # Stop after approximately 1 second
                            if elapsed >= 1.0:
                                break
                    finally:
                        response.close()
                    
                    elapsed_time = time.time() - start_time
                    bitrate = (bytes_downloaded * 8) / elapsed_time  # bits per second
                    logger.info(f"Measured bitrate: {bitrate:.0f} bps ({bytes_downloaded} bytes in {elapsed_time:.2f}s)")
                    
                except Exception as e:
                    logger.warning(f"Failed to measure bitrate via yt-dlp opener: {e}")
                    # Fallback to format info
                    if 'tbr' in format and format['tbr']:
                        bitrate = format['tbr'] * 1000
                        logger.info(f"Fallback bitrate from format tbr: {bitrate:.0f} bps")
                    else:
                        bitrate = None

                # Use OpenCV to capture frames with spacing
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open stream URL: {stream_url}")
                    return None, None, bitrate

                # Read first frame
                ret1, frame1 = cap.read()
                if not ret1:
                    logger.error("Failed to read first frame from stream")
                    cap.release()
                    return None, None, bitrate

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
                cap.release()

                if not ret2:
                    logger.error("Failed to read second frame from stream")
                    # Return first frame twice if we can't get a second frame
                    return Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)), bitrate

                return (Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)),
                        Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)), bitrate)
            else:
                logger.error("No formats available for the stream")
    except yt_dlp.DownloadError as e:
        logger.error(f"yt-dlp download error for {url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching frames from {url}: {e}")
    return None, None, bitrate