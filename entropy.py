"""Entropy calculation module for video frame analysis."""

import logging
import traceback

import cv2
import numpy as np
import yt_dlp
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from config import DEFAULT_FRAME_SKIP, MAX_VIDEO_HEIGHT

logger = logging.getLogger(__name__)


def calculate_spatial_entropy(image):
    """Calculate Shannon entropy of pixel intensities."""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize
    # Calculate entropy
    entropy = -np.sum(
        hist * np.log2(hist + 1e-10)
    )  # Add small value to avoid log(0)
    return entropy


def calculate_temporal_entropy(current_image, previous_image):
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
                import time
                import urllib.request

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
                    cap = cv2.VideoCapture(stream_url)
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
                    cap = cv2.VideoCapture(stream_url)
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


def count_objects_in_video(
    video_id, object_type, max_height=None, reuse_frame=None
):
    """Count objects of specified type in a high-resolution snapshot using OWLv2.

    Args:
        video_id: YouTube video ID
        object_type: Type of object to detect
        max_height: Maximum height (ignored for object detection - always uses highest resolution)
        reuse_frame: Optional PIL Image to reuse instead of capturing new frame
    """
    logger.debug(
        f"Executing object detection for {video_id}, object_type: {object_type}"
    )
    # If we have a reusable frame from entropy calculation, use it
    if reuse_frame is not None:
        logger.info(
            "Reusing existing high-resolution frame for object detection"
        )
        image = reuse_frame
    else:
        # Capture a new frame using the same logic as fetch_two_spaced_frames
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Use highest resolution format
        ydl_opts = {
            "format": "best",
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if "formats" in info and info["formats"]:
                    # Select the highest resolution format available
                    formats = [
                        f for f in info["formats"] if f.get("height", 0) > 0
                    ]
                    if not formats:
                        formats = info["formats"]
                    format = max(
                        formats, key=lambda f: f.get("height", 0) or 0
                    )
                    stream_url = format["url"]
                    resolution = (
                        f"{format.get('width', 0)}x{format.get('height', 0)}"
                    )
                    selected_codec = format.get("vcodec", "unknown")

                    logger.info(
                        f"Selected format for object detection: {resolution} (codec: {selected_codec})"
                    )

                    # Capture a single high-resolution frame
                    ret = False
                    frame = None

                    cap = cv2.VideoCapture(stream_url)
                    if not cap.isOpened():
                        logger.error(
                            f"Failed to open stream URL: {stream_url} (codec: {selected_codec})"
                        )
                        logger.warning(
                            "This might be due to an incompatible codec. Trying alternative formats..."
                        )

                        # Try alternative formats if the selected one fails
                        for alt_fmt in sorted(
                            info["formats"],
                            key=lambda f: f.get("height", 0) or 0,
                            reverse=True,
                        )[
                            1:4
                        ]:  # Next 3 best
                            alt_height = alt_fmt.get("height", 0)
                            alt_codec = alt_fmt.get("vcodec", "unknown")
                            if alt_height > 0 and alt_height != format.get(
                                "height", 0
                            ):
                                logger.info(
                                    f"Trying alternative format: {alt_fmt.get('width', 0)}x{alt_height} (codec: {alt_codec})"
                                )
                                try:
                                    alt_cap = cv2.VideoCapture(alt_fmt["url"])
                                    if alt_cap.isOpened():
                                        alt_ret, alt_frame = alt_cap.read()
                                        alt_cap.release()
                                        if alt_ret:
                                            logger.info(
                                                f"Successfully captured frame with alternative format: {alt_fmt.get('width', 0)}x{alt_height}"
                                            )
                                            frame = alt_frame
                                            resolution = f"{alt_fmt.get('width', 0)}x{alt_height}"
                                            ret = True
                                            break
                                except Exception as e:
                                    logger.debug(
                                        f"Alternative format failed: {e}"
                                    )
                                    continue

                        if not ret:
                            logger.error("All format attempts failed")
                            return None
                    else:
                        # Primary format opened successfully
                        ret, frame = cap.read()
                        cap.release()

                    if not ret:
                        logger.error("Failed to read frame from stream")
                        return None

                    # Convert to PIL Image
                    image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )

        except yt_dlp.DownloadError as e:
            logger.error(f"yt-dlp download error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error fetching frame for object detection: {e}"
            )
            return None

    try:
        # Load OWLv2 model and processor
        logger.debug("Loading OwlViT model and processor...")

        # Check PyTorch availability and configuration
        import torch

        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        logger.debug(
            f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )

        processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        logger.debug(f"OwlViT processor loaded successfully")

        model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )
        logger.debug(f"OwlViT model loaded successfully")

        # Prepare inputs
        texts = [[f"a photo of a {object_type}"]]
        logger.debug(f"Prepared text prompt: {texts}")
        logger.debug(f"Image size: {image.size}, mode: {image.mode}")

        inputs = processor(text=texts, images=image, return_tensors="pt")
        logger.debug(f"Processor inputs prepared successfully")

        # Perform object detection
        logger.debug("Running model inference...")
        outputs = model(**inputs)
        logger.debug(f"Model inference completed successfully")

        # Get predictions
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        logger.debug(f"Target sizes: {target_sizes}")

        results = processor.post_process_grounded_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.1
        )
        logger.debug("Post-processing completed")

        # Count detected objects
        predictions = results[0]
        object_count = len(predictions["scores"])

        if object_count > 0:
            scores = predictions["scores"].tolist()
            logger.debug(
                f"Detected {object_count} '{object_type}' objects with scores: {[f'{s:.3f}' for s in scores]}"
            )

        logger.info(
            f"Detected {object_count} '{object_type}' objects in {video_id}"
        )
        return object_count

    except ImportError as e:
        logger.error(
            f"Object detection import error (missing dependencies): {e}"
        )
        return None
    except RuntimeError as e:
        logger.error(
            f"Object detection runtime error (likely hardware/CUDA issue): {e}"
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error in object detection: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None
