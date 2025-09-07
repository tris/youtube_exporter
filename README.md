# youtube_exporter

A Prometheus exporter that monitors YouTube videos and live streams for image entropy metrics and metadata to detect when streams go "dark" or get stuck. Typical channel and video statistics are included (views, videos, likes, subscribers), as well as stream bitrate estimation, and object detection via [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2).

## Features

### Entropy Calculation
- **Spatial entropy**: Shannon entropy calculated across HSV color channels (hue, saturation, value) within frames (0-8 bits range)
- **Temporal entropy**: Entropy of differences between frames separated by ~1 second, calculated per HSV channel (0-8 bits range)

### Bitrate Measurement
- **Measured Bitrates**: Real-time bitrate estimation from actual video segment downloads

### Performance & Reliability
- **Asynchronous Processing**: Non-blocking API responses with background video processing
- **Smart Caching**: 5-minute frame cache with automatic expiration and reuse logic

### Integration & Monitoring
- **YouTube API Integration**: Fetches comprehensive metadata (views, likes, concurrent viewers, live status, channel info)
- **Channel Monitoring**: Support for monitoring entire channels and their live streams
- **Object Detection**: Optional AI-powered object counting
- **Prometheus Export**: Exposes all metrics via HTTP endpoint for Prometheus scraping, with timestamps
- **Resource Management**: Built-in quota tracking, error monitoring, and process metrics

## Metrics

### Entropy Metrics
- `youtube_video_spatial_entropy_hue{video_id="...", title="..."}`: Hue component entropy of pixel intensities within a single frame (0-8 bits range)
- `youtube_video_spatial_entropy_saturation{video_id="...", title="..."}`: Saturation component entropy of pixel intensities within a single frame (0-8 bits range)
- `youtube_video_spatial_entropy_value{video_id="...", title="..."}`: Value component entropy of pixel intensities within a single frame (0-8 bits range)
- `youtube_video_temporal_entropy_hue{video_id="...", title="..."}`: Hue component entropy of pixel differences between frames (0-8 bits range)
- `youtube_video_temporal_entropy_saturation{video_id="...", title="..."}`: Saturation component entropy of pixel differences between frames (0-8 bits range)
- `youtube_video_temporal_entropy_value{video_id="...", title="..."}`: Value component entropy of pixel differences between frames (0-8 bits range)
- `youtube_video_bitrate{video_id="...", title="...", resolution="..."}`: **Measured** bitrate of the video stream in bits per second from actual video segment downloads (not format metadata)

### Object Detection Metrics
- `youtube_video_object_count{video_id="...", title="...", object_type="..."}`: Number of detected objects of specified type in high-resolution video frames using AI object detection

### YouTube API Metrics
- `youtube_video_view_count{video_id="...", title="..."}`: Total view count reported by YouTube
- `youtube_video_like_count{video_id="...", title="..."}`: Total like count reported by YouTube
- `youtube_video_concurrent_viewers{video_id="...", title="..."}`: Concurrent viewers (only non-zero while live)
- `youtube_video_live{video_id="...", title="..."}`: 1 if YouTube reports the video as currently live, else 0
- `youtube_video_live_status{video_id="...", title="...", state="..."}`: Infometric with state label; 1 for the current state ("live", "upcoming", or "none")

### Channel Metrics
- `youtube_channel_subscriber_count{channel_id="...", channel_title="..."}`: Total subscriber count reported by YouTube for this channel
- `youtube_channel_view_count{channel_id="...", channel_title="..."}`: Total view count reported by YouTube for this channel
- `youtube_channel_video_count{channel_id="...", channel_title="..."}`: Total video count reported by YouTube for this channel
- `youtube_channel_scrape_success{channel_id="...", channel_title="..."}`: 1 if the scrape of YouTube API succeeded, else 0

### System Metrics
- `youtube_api_quota_units_total{endpoint="...", key="..."}`: Total YouTube Data API quota units consumed, labeled by endpoint and key index
- `youtube_api_errors_total{code="...", endpoint="...", key="..."}`: Total YouTube API errors, labeled by error code, endpoint, and key index

### Process Metrics
- `process_cpu_seconds_total`: Total user and system CPU time spent in seconds
- `process_resident_memory_bytes`: Resident memory size in bytes
- `process_virtual_memory_bytes`: Virtual memory size in bytes
- `process_start_time_seconds`: Start time of the process since unix epoch in seconds

## Usage

1. Set one or more YouTube API keys (use multiple for load balancing):
    ```bash
    export YOUTUBE_API_KEY="key1,key2,key3"
    ```

2. (Optional) Set custom port:
    ```bash
    export PORT=8080
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the exporter:
    ```bash
    python main.py
    ```

4. Query metrics:
    ```
    http://localhost:9473/metrics?video_id=YOUR_VIDEO_ID
    ```

## Query Parameters

### Video Metrics
- `video_id`: YouTube video ID (11 characters)
- `fetch_images` (optional): Enable/disable image fetching for entropy calculation (default: true)
- `max_height` (optional): Maximum video height in pixels (default: 4320 for up to 8K resolution)
- `match` (optional): Object type to detect and count (e.g., "person", "car", "dog") - enables AI object detection
- `interval` (optional): Fetch interval in seconds (default: 300, min: 30)

### Channel Metrics
- `channel`: YouTube channel ID (UC followed by 22 characters)
- `fetch_images` (optional): Enable/disable image fetching for entropy calculation (default: true)
- `disable_live` (optional): Disable live stream detection for channels (default: false)
- `interval` (optional): Fetch interval in seconds (default: 300, min: 30)

### Performance Notes
- **Asynchronous Processing**: First request returns immediately with YouTube API data; entropy metrics appear in subsequent requests after background processing completes
- **Smart Caching**: Entropy data is cached for 5 minutes to avoid redundant processing

## Examples

### Video Metrics
Basic usage (maximum resolution entropy + measured bitrate):
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA
```

With AI object detection:
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&match=person
```

Disable image fetching (API data only):
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&fetch_images=false
```

Custom resolution limit (1080p max):
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&max_height=1080
```

Multiple features combined:
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&match=car&interval=30
```

### Channel Metrics
Basic channel usage (monitors all live streams):
```
http://localhost:9473/metrics?channel=UC029bcZGqxOsRkvWrZLhgKQ
```

Channel without live stream detection:
```
http://localhost:9473/metrics?channel=UC029bcZGqxOsRkvWrZLhgKQ&disable_live=true
```

### Performance Examples
First request (immediate response with API data):
```bash
curl "http://localhost:9473/metrics?video_id=yv2RtoIMNzA"
# Returns immediately with YouTube API metrics
# Background processing starts for entropy calculation
```

Second request (includes entropy metrics):
```bash
curl "http://localhost:9473/metrics?video_id=yv2RtoIMNzA"
# Returns API metrics + entropy metrics from background processing
# Uses cached high-resolution frames if object detection is also requested
```

## Configuration

### Environment Variables
- `YOUTUBE_API_KEY` (required): YouTube Data API v3 key(s)
- `PORT` (optional): Server port (default: 9473)
- `LOG_LEVEL` (optional): Logging level - DEBUG, INFO, WARNING, ERROR (default: INFO)
- `MODEL_CACHE_DIR` (optional): Custom directory for HuggingFace model cache to persist downloaded models between restarts

## Deployment

### Basic Deployment
```bash
export YOUTUBE_API_KEY="key1,key2,key3"
export LOG_LEVEL="INFO"
python main.py
```

### Deployment with Persistent Model Cache
```bash
export YOUTUBE_API_KEY="key1,key2,key3"
export MODEL_CACHE_DIR="/opt/youtube-exporter/model-cache"
export LOG_LEVEL="INFO"
python main.py
```

### Docker Deployment
Run as a service or container. Ensure yt-dlp dependencies (ffmpeg, etc.) are installed.

### Production Considerations
- **Memory usage**: High-resolution frame caching requires adequate RAM
- **Network bandwidth**: Measured bitrate calculation downloads video segments
- **CPU usage**: AI object detection and entropy calculation are CPU-intensive
- **API quotas**: Monitor YouTube API quota usage via built-in metrics

## Requirements

### Core Dependencies
- Python 3.11+
- YouTube Data API v3 key
- yt-dlp (with ffmpeg support)
- prometheus-client
- Pillow (PIL)
- numpy
- opencv-python-headless
- flask
- google-api-python-client
- google-auth

### AI/ML Dependencies (for object detection)
- transformers
- torch
- m3u8 (for HLS manifest parsing)

### System Dependencies
- ffmpeg (required by yt-dlp for video processing)
- libgl1-mesa-glx (for OpenCV on some Linux distributions)

## Prometheus Configuration

### Basic Video Monitoring
Monitor specific YouTube videos with entropy and bitrate metrics:

```yaml
- job_name: 'youtube'
  scrape_interval: 5m
  static_configs:
  - targets:
    - oefos36Y_Mo
    - vZEINdmawdc
    - JFkxBYPwkfY
    - eLt-QEwVnxc
  metrics_path: /metrics
  relabel_configs:
  - source_labels: [__address__]
    target_label: __param_video_id
  - target_label: __address__
    replacement: your-server:9473
```

### Channel Monitoring
Monitor entire YouTube channels and their live streams:

```yaml
- job_name: 'youtube_channel'
  scrape_interval: 5m
  static_configs:
  - targets:
    - UCTIuhsbotXQGmhLgmcRYVCQ
    - UC51Z-ATEJ7Va3BP-JH7bLNA
    - UCNLH3taW6_MpeSymek4acZg
    - UCZ_a4if3xoN4_BYtWdrUoRg
    - UC029bcZGqxOsRkvWrZLhgKQ
  metrics_path: /metrics
  relabel_configs:
  - source_labels: [__address__]
    target_label: __param_channel
  - target_label: __address__
    replacement: your-server:9473
```

### AI Object Detection
Monitor videos with object detection (e.g., count people in streams):

```yaml
- job_name: 'youtube_objects'
  scrape_interval: 5m
  static_configs:
  - targets:
    - oefos36Y_Mo
    - vZEINdmawdc
  metrics_path: /metrics
  relabel_configs:
  - source_labels: [__address__]
    target_label: __param_video_id
  - target_label: __param_match
    replacement: person
  - target_label: __address__
    replacement: your-server:9473
```

### Advanced Configuration
Combine multiple parameters for comprehensive monitoring:

```yaml
- job_name: 'youtube_advanced'
  scrape_interval: 3m
  static_configs:
  - targets:
    - JFkxBYPwkfY
  metrics_path: /metrics
  relabel_configs:
  - source_labels: [__address__]
    target_label: __param_video_id
  - target_label: __param_match
    replacement: car
  - target_label: __param_max_height
    replacement: 1080
  - target_label: __param_interval
    replacement: 180
  - target_label: __address__
    replacement: your-server:9473
```

### Configuration Notes
- **Scrape intervals**: Recommended 5 minutes to balance data freshness with API quota usage
- **Target format**: Use video IDs (11 characters) or channel IDs (UC + 22 characters) as targets
- **Server replacement**: Replace `your-server:9473` with your actual server hostname/IP and port
- **Object detection**: Use common object types like "person", "car", "dog", "cat", "bicycle", etc.
- **Performance**: Object detection and high-resolution processing may require longer scrape timeouts
