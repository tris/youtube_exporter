"""Object detection module using OWLv2 for video frame analysis."""

import logging
import os
import threading
import time
import traceback

import cv2
import yt_dlp
from PIL import Image, ImageDraw

from config import DEBUG_DIR, MODEL_CACHE_DIR

# Set up custom cache directory BEFORE importing transformers
if MODEL_CACHE_DIR:
    os.environ["HF_HOME"] = MODEL_CACHE_DIR
    # Ensure directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

from transformers import Owlv2ForObjectDetection, Owlv2Processor

logger = logging.getLogger(__name__)

# Global model variables - loaded once on import
_model = None
_processor = None
_inference_lock = threading.Lock()  # Ensure thread-safe model inference


def check_opencv_ffmpeg_support():
    """Check if OpenCV was built with FFMPEG support."""
    build_info = cv2.getBuildInformation()
    logger.debug("OpenCV Build Information:")

    # Check for FFMPEG in build info
    ffmpeg_enabled = "FFMPEG" in build_info and "YES" in build_info

    # Also check available backends
    backends = []
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append("CAP_FFMPEG")
    if hasattr(cv2, "CAP_GSTREAMER"):
        backends.append("CAP_GSTREAMER")
    if hasattr(cv2, "CAP_V4L2"):
        backends.append("CAP_V4L2")

    logger.info(f"OpenCV available video backends: {backends}")

    # Look for FFMPEG specifically in build info
    ffmpeg_lines = [
        line for line in build_info.split("\n") if "FFMPEG" in line.upper()
    ]
    if ffmpeg_lines:
        logger.info("FFMPEG build configuration:")
        for line in ffmpeg_lines:
            logger.info(f"  {line.strip()}")
    else:
        logger.warning("No FFMPEG configuration found in OpenCV build info")

    return ffmpeg_enabled, backends


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
    import threading

    thread_id = threading.current_thread().ident
    logger.debug(
        f"[THREAD-{thread_id}] Executing object detection for {video_id}, object_type: {object_type}"
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

                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
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
                                    alt_cap = cv2.VideoCapture(
                                        alt_fmt["url"], cv2.CAP_FFMPEG
                                    )
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
        # Use global model and processor loaded at import
        logger.debug(
            f"[THREAD-{thread_id}] Using pre-loaded OWLv2 model and processor..."
        )

        import torch

        # Use global model and processor loaded at import
        model, processor = _model, _processor

        if model is None or processor is None:
            logger.error(
                "Model or processor not loaded - this should not happen!"
            )
            return None

        final_device = next(model.parameters()).device
        logger.debug(
            f"[THREAD-{thread_id}] Using model on device: {final_device}"
        )

        # Prepare inputs
        texts = [[f"a photo of a {object_type}"]]
        logger.debug(f"Prepared text prompt: {texts}")
        logger.debug(f"Image size: {image.size}, mode: {image.mode}")

        inputs = processor(text=texts, images=image, return_tensors="pt")
        logger.debug(f"Processor inputs prepared successfully")
        logger.debug(
            f"Input tensor devices: {[inputs[k].device for k in inputs if hasattr(inputs[k], 'device')]}"
        )

        # Perform object detection with thread safety
        with _inference_lock:
            logger.debug("Running model inference...")
            # Use no_grad() to prevent memory accumulation during inference
            with torch.no_grad():
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

            # Debug: Save image with bounding boxes if DEBUG_DIR is set
            if DEBUG_DIR:
                try:
                    # Create a copy of the image for drawing
                    debug_image = image.copy()
                    draw = ImageDraw.Draw(debug_image)

                    # Draw bounding boxes and scores if objects detected
                    if object_count > 0:
                        boxes = predictions["boxes"].tolist()
                        scores = predictions["scores"].tolist()

                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = [int(coord) for coord in box]

                            # Draw rectangle
                            draw.rectangle(
                                [x1, y1, x2, y2], outline="red", width=3
                            )

                            # Draw score text
                            score_text = f"{score:.2f}"
                            draw.text((x1, y1 - 10), score_text, fill="red")
                    else:
                        # No objects detected, add text indicator
                        draw.text(
                            (10, 10),
                            f"No '{object_type}' detected",
                            fill="red",
                        )

                    # Create filename
                    timestamp = int(time.time())
                    filename = f"{video_id}_{object_type}_{object_count}_{timestamp}.png"
                    filepath = os.path.join(DEBUG_DIR, filename)

                    # Ensure directory exists
                    os.makedirs(DEBUG_DIR, exist_ok=True)

                    # Save image
                    debug_image.save(filepath)
                    logger.info(
                        f"Saved debug image: {filepath} (objects detected: {object_count})"
                    )

                except Exception as e:
                    logger.error(f"Failed to save debug image: {e}")

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


# Load models immediately upon module import
def _load_models():
    """Load models and assign to global variables."""
    global _model, _processor

    thread_id = threading.current_thread().ident
    logger.info(
        f"[STARTUP-{thread_id}] Loading OWLv2 models on module import..."
    )

    try:
        # Check OpenCV FFMPEG support
        ffmpeg_enabled, backends = check_opencv_ffmpeg_support()
        if not ffmpeg_enabled:
            logger.warning(
                "OpenCV may not have FFMPEG support - video capture might fail"
            )

        # Check PyTorch availability and configuration
        import torch

        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")

        # Load processor
        logger.info(f"[STARTUP-{thread_id}] Loading Owlv2Processor...")
        start_time = time.time()
        _processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble", use_fast=True
        )
        processor_time = time.time() - start_time
        logger.info(
            f"[STARTUP-{thread_id}] Owlv2Processor loaded in {processor_time:.1f}s"
        )

        # Load model
        logger.info(
            f"[STARTUP-{thread_id}] Loading Owlv2ForObjectDetection..."
        )
        start_time = time.time()
        _model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble", dtype=torch.float32
        )
        model_time = time.time() - start_time
        logger.info(
            f"[STARTUP-{thread_id}] Owlv2ForObjectDetection loaded in {model_time:.1f}s"
        )

        # Move to CPU and set to eval mode to save memory
        _model = _model.to("cpu")
        _model.eval()  # Set to evaluation mode to disable dropout/batch norm
        final_device = next(_model.parameters()).device
        logger.info(
            f"[STARTUP-{thread_id}] Models loaded successfully on device: {final_device}"
        )

    except Exception as e:
        logger.error(
            f"[STARTUP-{thread_id}] Failed to load models on import: {e}"
        )
        logger.error(
            f"[STARTUP-{thread_id}] Application will exit due to model loading failure"
        )
        import sys

        sys.exit(1)


# Execute model loading
_load_models()
