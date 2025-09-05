# YouTube Exporter

A comprehensive Prometheus exporter that monitors YouTube videos and live streams for entropy metrics and metadata to detect when streams go "dark" or get stuck.

## Features

- **Video Analysis**: Fetches frames from YouTube live streams and videos using yt-dlp
- **Entropy Calculation**:
  - Intra-image entropy (Shannon entropy of pixel intensities within frames)
  - Inter-image entropy (entropy of differences between frames separated by ~1 second)
- **Bitrate Measurement**: Real-time bitrate calculation from actual 1-second downloads using yt-dlp's HTTP client
- **YouTube API Integration**: Fetches comprehensive metadata (views, likes, concurrent viewers, live status, channel info)
- **Channel Monitoring**: Support for monitoring entire channels and their live streams
- **Prometheus Export**: Exposes all metrics via HTTP endpoint for Prometheus scraping with proper timestamps
- **Background Processing**: Asynchronous entropy computation for channel live streams to avoid blocking
- **Resource Management**: Optional image fetching, configurable intervals, quota tracking, error monitoring
- **Process Monitoring**: Built-in CPU, memory, and system metrics
- **Validation**: Video ID and channel ID validation with proper error handling

## Metrics

### Entropy Metrics
- `youtube_video_intra_entropy{video_id="...", title="..."}`: Shannon entropy of pixel intensities within a single frame (0-8 bits range)
- `youtube_video_inter_entropy{video_id="...", title="..."}`: Shannon entropy of pixel differences between frames separated by ~1 second (0-8 bits range)
- `youtube_video_bitrate{video_id="...", title="..."}`: Bitrate of the video stream in bits per second, calculated from 1-second download

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
- `youtube_api_quota_units_today{endpoint="..."}`: Estimated YouTube Data API quota units consumed today, labeled by endpoint
- `youtube_api_quota_units_total{endpoint="..."}`: Total YouTube Data API quota units consumed, labeled by endpoint
- `youtube_api_errors_total{code="...", endpoint="..."}`: Total YouTube API errors, labeled by error code and endpoint

### Process Metrics
- `process_cpu_seconds_total`: Total user and system CPU time spent in seconds
- `process_resident_memory_bytes`: Resident memory size in bytes
- `process_virtual_memory_bytes`: Virtual memory size in bytes
- `process_start_time_seconds`: Start time of the process since unix epoch in seconds

## Usage

1. Set up YouTube API key:
    ```bash
    export YOUTUBE_API_KEY="your_api_key_here"
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
- `video_id` (required for video metrics): YouTube video ID (11 characters)
- `fetch_images` (optional): Enable/disable image fetching for entropy calculation (default: true)
- `interval` (optional): Fetch interval in seconds (default: 300, min: 30)

### Channel Metrics
- `channel` (required for channel metrics): YouTube channel ID (UC followed by 22 characters)
- `fetch_images` (optional): Enable/disable image fetching for entropy calculation (default: true)
- `disable_live` (optional): Disable live stream detection for channels (default: false)
- `interval` (optional): Fetch interval in seconds (default: 300, min: 30)

## Examples

### Video Metrics
Basic usage:
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA
```

Disable image fetching (API data only):
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&fetch_images=false
```

Custom interval (30 seconds):
```
http://localhost:9473/metrics?video_id=yv2RtoIMNzA&interval=30
```

### Channel Metrics
Basic channel usage:
```
http://localhost:9473/metrics?channel=UC029bcZGqxOsRkvWrZLhgKQ
```

Channel without live stream detection:
```
http://localhost:9473/metrics?channel=UC029bcZGqxOsRkvWrZLhgKQ&disable_live=true
```

## Configuration

- Port: 9473 (configurable via PORT environment variable)
- YouTube API Key: Required via YOUTUBE_API_KEY environment variable
- Default fetch interval: 5 minutes
- Minimum fetch interval: 30 seconds

## Deployment

Run as a service or container. Ensure yt-dlp dependencies (ffmpeg, etc.) are installed.

## Requirements

- Python 3.8+
- YouTube Data API v3 key
- yt-dlp
- prometheus-client
- Pillow
- numpy
- opencv-python
- flask
- google-api-python-client
- google-auth
