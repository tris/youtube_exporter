import logging
import os
import threading
import time
import traceback

import cv2
import yt_dlp
from PIL import Image, ImageDraw, ImageFont

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
# Note: Model inference is thread-safe on CPU when model is in eval mode


def draw_text_with_outline(
    draw,
    position,
    text,
    fill_color,
    outline_color="black",
    outline_width=2,
    font_size=16,
):
    """
    Draw text with a black outline to ensure visibility on any background.

    Args:
        draw: PIL ImageDraw object
        position: (x, y) tuple for text position
        text: Text to draw
        fill_color: Color for the main text
        outline_color: Color for the outline (default: black)
        outline_width: Width of the outline (default: 2)
        font_size: Size of the font (default: 16)
    """
    # Load a bold font if available, otherwise default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=font_size,
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    x, y = position
    # Draw outline by drawing text in all 8 directions around the main position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue  # Skip the center position for now
            draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
    # Draw the main text on top
    draw.text((x, y), text, fill=fill_color, font=font)


def draw_dashed_rectangle(draw, xy, outline, width=3, dash=(10, 5)):
    """
    Draw a dashed rectangle using line segments.

    Args:
        draw: PIL ImageDraw object
        xy: Bounding box as (x1, y1, x2, y2)
        outline: Color for the outline
        width: Line width
        dash: Tuple of (dash_length, gap_length)
    """
    x1, y1, x2, y2 = xy
    dash_len, gap_len = dash

    # Draw top side
    x = x1
    while x < x2:
        end_x = min(x + dash_len, x2)
        draw.line([(x, y1), (end_x, y1)], fill=outline, width=width)
        x += dash_len + gap_len

    # Draw bottom side
    x = x1
    while x < x2:
        end_x = min(x + dash_len, x2)
        draw.line([(x, y2), (end_x, y2)], fill=outline, width=width)
        x += dash_len + gap_len

    # Draw left side
    y = y1
    while y < y2:
        end_y = min(y + dash_len, y2)
        draw.line([(x1, y), (x1, end_y)], fill=outline, width=width)
        y += dash_len + gap_len

    # Draw right side
    y = y1
    while y < y2:
        end_y = min(y + dash_len, y2)
        draw.line([(x2, y), (x2, end_y)], fill=outline, width=width)
        y += dash_len + gap_len


def get_color_for_object(obj_type: str) -> str:
    """
    Deterministically choose a color for a given object type name.
    Uses a fixed palette and a stable hash based on character codes.
    """
    palette = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "white",
    ]
    idx = sum(ord(c) for c in str(obj_type)) % len(palette)
    return palette[idx]


def select_best_video_format(info: dict) -> dict:
    """
    Select the highest resolution video format from yt-dlp info.

    Args:
        info: yt-dlp extracted info dictionary

    Returns:
        dict: Selected format dictionary, or None if no valid format found

    Raises:
        ValueError: If info structure is invalid or no formats available
    """
    if not isinstance(info, dict) or "formats" not in info:
        raise ValueError("Invalid info structure: missing 'formats' key")

    formats = info["formats"]
    if not formats:
        raise ValueError("No video formats available")

    # Filter to formats with valid height, preferring video formats
    valid_formats = [
        fmt
        for fmt in formats
        if isinstance(fmt, dict) and fmt.get("height", 0) > 0
    ]

    # If no formats have height, fall back to all formats (audio-only might be included)
    if not valid_formats:
        logger.warning(
            "No formats with height found, using all available formats"
        )
        valid_formats = formats

    # Select format with maximum height
    try:
        best_format = max(valid_formats, key=lambda fmt: fmt.get("height", 0))
    except (TypeError, ValueError) as e:
        logger.error(f"Error selecting best format: {e}")
        raise ValueError("Unable to determine best video format") from e

    # Validate that the selected format has required fields
    required_fields = ["url"]
    missing_fields = [
        field for field in required_fields if field not in best_format
    ]
    if missing_fields:
        raise ValueError(
            f"Selected format missing required fields: {missing_fields}"
        )

    return best_format


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
    video_id, objects_to_thresholds, reuse_frame=None, debug=False
):
    """Count objects of specified types in a high-resolution snapshot using OWLv2.

    Args:
        video_id: YouTube video ID
        objects_to_thresholds: Dict of {object_type: threshold} pairs or single object_type string
        reuse_frame: Optional PIL Image to reuse instead of capturing new frame
        debug: Whether to save debug images (only if DEBUG_DIR is also set)
    """
    import threading

    thread_id = threading.current_thread().ident
    logger.debug(
        f"[THREAD-{thread_id}] Executing object detection for {video_id}, objects: {list(objects_to_thresholds.keys())}"
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
                    try:
                        selected_format = select_best_video_format(info)
                        stream_url = selected_format["url"]
                        resolution = f"{selected_format.get('width', 0)}x{selected_format.get('height', 0)}"
                        selected_codec = selected_format.get(
                            "vcodec", "unknown"
                        )

                        logger.info(
                            f"Selected format for object detection: {resolution} (codec: {selected_codec})"
                        )
                    except ValueError as e:
                        logger.error(f"Failed to select video format: {e}")
                        return None

                    # Capture a single high-resolution frame
                    ret = False
                    frame = None

                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        logger.error(
                            f"Failed to open stream URL: {stream_url} (codec: {selected_codec})"
                        )
                        return None

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

        # Prepare inputs for all objects
        texts = [[object_type for object_type in objects_to_thresholds.keys()]]
        logger.debug(f"Prepared text prompts: {texts}")
        logger.debug(f"Image size: {image.size}, mode: {image.mode}")

        inputs = processor(text=texts, images=image, return_tensors="pt")
        logger.debug(f"Processor inputs prepared successfully")
        logger.debug(
            f"Input tensor devices: {[inputs[k].device for k in inputs if hasattr(inputs[k], 'device')]}"
        )

        # Perform object detection (model is in eval mode, should be thread-safe for CPU inference)
        logger.debug("Running model inference for objects...")
        inference_start = time.time()
        # Use no_grad() to prevent memory accumulation during inference
        with torch.no_grad():
            outputs = model(**inputs)
        inference_duration = time.time() - inference_start
        logger.info(
            f"[THREAD-{thread_id}] Model inference completed in {inference_duration:.3f}s for {video_id}"
        )
        logger.debug(f"Model inference completed successfully for objects")

        # Get predictions
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        logger.debug(f"Target sizes: {target_sizes}")

        results = processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0,  # Use lowest threshold, filter later
        )
        logger.debug("Post-processing completed for objects")

        # Count detected objects for each type with individual thresholds
        predictions = results[0]
        object_counts = {}

        # Extract predictions as lists and group by label (prompt index)
        labels_raw = predictions.get("labels", [])
        scores_raw = predictions.get("scores", [])
        boxes_raw = predictions.get("boxes", [])
        labels_list = (
            labels_raw.tolist()
            if hasattr(labels_raw, "tolist")
            else list(labels_raw)
        )
        scores_list = (
            scores_raw.tolist()
            if hasattr(scores_raw, "tolist")
            else list(scores_raw)
        )
        boxes_list = (
            boxes_raw.tolist()
            if hasattr(boxes_raw, "tolist")
            else list(boxes_raw)
        )

        # Prepare combined debug image and overlay structures
        timestamp = int(time.time())
        debug_image = None
        draw = None
        overlay_lines_top_left = []  # list of tuples (obj_type, count, color)
        overlay_lines_bottom_left = (
            []
        )  # list of tuples (obj_type, [scores], color)
        if DEBUG_DIR and debug:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            debug_image = image.copy()
            draw = ImageDraw.Draw(debug_image)

        # Process each object type with its specific threshold
        for i, (obj_type, threshold) in enumerate(
            objects_to_thresholds.items()
        ):
            # Indices predicted for this prompt index i
            label_indices = [
                j for j, lab in enumerate(labels_list) if lab == i
            ]

            # Apply threshold filtering per object type
            valid_indices = [
                j for j in label_indices if scores_list[j] >= threshold
            ]
            object_count = len(valid_indices)

            object_counts[obj_type] = object_count

            # Debug draw (accumulate on a single image)
            if DEBUG_DIR and debug and draw is not None:
                try:
                    # Choose color per object type
                    color = get_color_for_object(obj_type)

                    # Draw bounding boxes and inline scores for all valid detections and invalid detections with score >= 0.1
                    for j in label_indices:
                        score = scores_list[j]
                        box = boxes_list[j]
                        x1, y1, x2, y2 = [int(coord) for coord in box]

                        # Draw solid rectangle for valid detections, dashed for invalid (if score >= 0.1)
                        if j in valid_indices:
                            draw.rectangle(
                                [x1, y1, x2, y2], outline=color, width=3
                            )
                        elif score >= 0.1:
                            draw_dashed_rectangle(
                                draw,
                                [x1, y1, x2, y2],
                                outline=color,
                                width=3,
                            )
                        else:
                            continue  # Skip invalid detections with score < 0.1

                        # Small inline score near the box (may overlap)
                        draw_text_with_outline(
                            draw,
                            (x1, max(0, y1 - 12)),
                            f"{score:.2f}",
                            color,
                        )

                    # Add overlay entries (only for valid detections)
                    overlay_lines_top_left.append(
                        (obj_type, object_count, color)
                    )
                    if valid_indices:
                        valid_scores = [scores_list[j] for j in valid_indices]
                        overlay_lines_bottom_left.append(
                            (
                                obj_type,
                                [f"{s:.2f}" for s in valid_scores],
                                color,
                            )
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to prepare debug overlay for {obj_type}: {e}"
                    )

            if object_count > 0:
                valid_scores = [scores_list[j] for j in valid_indices]
                logger.debug(
                    f"Detected {object_count} '{obj_type}' objects with scores: {[f'{s:.3f}' for s in valid_scores]} (threshold: {threshold})"
                )

            logger.info(
                f"Detected {object_count} '{obj_type}' objects in {video_id} (threshold: {threshold})"
            )

        # Compute total count of detections
        total_count = sum(object_counts.values())
        timestamp = int(time.time())
        filename_base = f"{video_id}_{total_count}_{timestamp}"

        # Save combined debug images (original and annotated)
        if DEBUG_DIR and debug:
            try:
                original_filepath = os.path.join(
                    DEBUG_DIR, f"{filename_base}.png"
                )
                image.save(original_filepath)

                if debug_image is None:
                    debug_image = image.copy()
                    draw = ImageDraw.Draw(debug_image)

                # Top-left: object types and counts
                tl_x, tl_y = 10, 10
                line_h = 16
                for obj_type, count, color in overlay_lines_top_left:
                    draw_text_with_outline(
                        draw, (tl_x, tl_y), f"{obj_type} (n={count})", color
                    )
                    tl_y += line_h

                # Bottom-left: scores per object type
                bl_x = 10
                bl_y = max(
                    10,
                    debug_image.size[1]
                    - (len(overlay_lines_bottom_left) * line_h)
                    - 10,
                )
                for obj_type, score_list, color in overlay_lines_bottom_left:
                    draw_text_with_outline(
                        draw,
                        (bl_x, bl_y),
                        f"{obj_type} scores: {', '.join(score_list)}",
                        color,
                    )
                    bl_y += line_h

                bbox_filepath = os.path.join(
                    DEBUG_DIR, f"{filename_base}_bbox.png"
                )
                debug_image.save(bbox_filepath)
                logger.info(
                    f"Saved combined debug images: {original_filepath}, {bbox_filepath}"
                )
            except Exception as e:
                logger.error(f"Failed to save combined debug images: {e}")

        return object_counts

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
