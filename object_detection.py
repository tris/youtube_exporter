"""Object detection module using OWLv2 for video frame analysis."""

import logging
import os
import threading
import time
import traceback

import cv2
import yt_dlp
from PIL import Image

from config import MODEL_CACHE_DIR

# Set up custom cache directory BEFORE importing transformers
if MODEL_CACHE_DIR:
    os.environ["HF_HOME"] = MODEL_CACHE_DIR
    # Ensure directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

from transformers import OwlViTForObjectDetection, OwlViTProcessor

logger = logging.getLogger(__name__)

# Global model cache with enhanced thread synchronization
_model_cache = {}
_model_lock = threading.RLock()  # Use RLock for reentrancy
_model_loading_event = (
    threading.Event()
)  # Event to signal when loading is complete
_model_loading_thread = None  # Track which thread is doing the loading
_model_loading_error = None  # Store any error that occurred during loading


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


def _load_models_thread_worker():
    """Worker function that actually loads the models. Should only be called by one thread."""
    global _model_cache, _model_loading_error

    thread_id = threading.current_thread().ident

    try:
        logger.info(
            f"[DOWNLOADER-{thread_id}] Starting model download and loading process..."
        )

        # Log which cache directory is being used
        if MODEL_CACHE_DIR:
            logger.info(
                f"[DOWNLOADER-{thread_id}] Using custom cache directory: {MODEL_CACHE_DIR}"
            )
        else:
            logger.debug(
                f"[DOWNLOADER-{thread_id}] Using default HuggingFace cache directory"
            )

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
        logger.debug(
            f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )

        # Load processor - this is the first component that downloads model files
        logger.info(
            f"[DOWNLOADER-{thread_id}] Starting OwlViTProcessor.from_pretrained() - downloading processor files..."
        )
        start_time = time.time()
        processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        processor_time = time.time() - start_time
        logger.info(
            f"[DOWNLOADER-{thread_id}] OwlViTProcessor loaded successfully in {processor_time:.1f}s"
        )

        # Load model - this downloads the main model files
        logger.info(
            f"[DOWNLOADER-{thread_id}] Starting OwlViTForObjectDetection.from_pretrained() - downloading model files..."
        )
        start_time = time.time()
        model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32", dtype=torch.float32
        )
        model_time = time.time() - start_time
        logger.info(
            f"[DOWNLOADER-{thread_id}] OwlViTForObjectDetection loaded successfully in {model_time:.1f}s"
        )

        # Explicitly move model to CPU if it's not already there
        model = model.to("cpu")
        final_device = next(model.parameters()).device
        logger.info(
            f"[DOWNLOADER-{thread_id}] Model loaded and moved to device: {final_device}"
        )

        # Verify model is on CPU
        if final_device.type != "cpu":
            error_msg = (
                f"Failed to load model on CPU, loaded on {final_device}"
            )
            logger.error(f"[DOWNLOADER-{thread_id}] {error_msg}")
            raise RuntimeError(error_msg)

        # Cache the loaded model and processor
        _model_cache["model"] = model
        _model_cache["processor"] = processor
        logger.info(
            f"[DOWNLOADER-{thread_id}] Model and processor cached successfully"
        )

        # Clear any previous error
        _model_loading_error = None

    except Exception as e:
        logger.error(f"[DOWNLOADER-{thread_id}] Failed to load models: {e}")
        logger.debug(
            f"[DOWNLOADER-{thread_id}] Full traceback: {traceback.format_exc()}"
        )
        _model_loading_error = e
        raise


def get_cached_model_and_processor():
    """Get cached OwlViT model and processor, loading them if necessary.
    Thread-safe singleton pattern with single-threaded downloading to prevent lock contention.
    """
    global _model_cache, _model_lock, _model_loading_event, _model_loading_thread, _model_loading_error

    thread_id = threading.current_thread().ident
    logger.debug(
        f"[THREAD-{thread_id}] get_cached_model_and_processor() called"
    )

    # Fast path: Check if model is already cached (no lock needed)
    if "model" in _model_cache and "processor" in _model_cache:
        logger.debug(
            f"[THREAD-{thread_id}] Using cached OwlViT model and processor (fast path)"
        )
        return _model_cache["model"], _model_cache["processor"]

    # Slow path: Need to potentially load the model
    with _model_lock:
        # Double-check: model might have been loaded while waiting for lock
        if "model" in _model_cache and "processor" in _model_cache:
            return _model_cache["model"], _model_cache["processor"]

        # Check if another thread is already loading
        if _model_loading_thread is not None:
            logger.info(
                f"[THREAD-{thread_id}] Another thread is downloading the model, waiting..."
            )
            # Release lock and wait for loading to complete
            _model_lock.release()
            try:
                # Wait for loading to complete (with timeout to prevent hanging)
                if _model_loading_event.wait(timeout=600):  # 10 minute timeout
                    logger.debug(
                        f"[THREAD-{thread_id}] Model loading completed by other thread"
                    )
                    # Re-acquire lock to check results
                    _model_lock.acquire()
                    if _model_loading_error:
                        logger.error(
                            f"[THREAD-{thread_id}] Model loading failed in other thread: {_model_loading_error}"
                        )
                        raise _model_loading_error
                    return _model_cache["model"], _model_cache["processor"]
                else:
                    logger.error(
                        f"[THREAD-{thread_id}] Timeout waiting for model loading"
                    )
                    _model_lock.acquire()
                    raise RuntimeError(
                        "Timeout waiting for model loading to complete"
                    )
            except Exception:
                _model_lock.acquire()
                raise

        # This thread will be responsible for loading
        logger.info(
            f"[THREAD-{thread_id}] This thread will download the model"
        )
        _model_loading_thread = thread_id
        _model_loading_event.clear()  # Reset the event

        try:
            # Perform the actual loading
            _load_models_thread_worker()

            # Signal that loading is complete
            _model_loading_event.set()
            logger.info(
                f"[THREAD-{thread_id}] Model loading complete, signaling other threads"
            )

            return _model_cache["model"], _model_cache["processor"]

        except Exception:
            # Signal that loading failed
            _model_loading_event.set()
            logger.error(
                f"[THREAD-{thread_id}] Model loading failed, signaling other threads"
            )
            raise
        finally:
            # Reset loading state
            _model_loading_thread = None


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
        # Use cached model and processor to prevent concurrent loading
        logger.debug(
            f"[THREAD-{thread_id}] Getting cached OwlViT model and processor..."
        )

        import torch

        model, processor = get_cached_model_and_processor()

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
