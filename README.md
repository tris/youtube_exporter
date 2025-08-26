# youtube_exporter

youtube_exporter is a [Prometheus](https://prometheus.io/) exporter for
[YouTube](https://www.youtube.com/) videos and channels.

The exporter queries the
[YouTube Data API](https://developers.google.com/youtube/v3) to expose metrics
for individual videos or entire channels. It supports monitoring video
statistics (views, likes, concurrent viewers for live streams) and channel
statistics (subscribers, total views, video count, plus all live streams from
the channel).

## Install

Download from [releases](https://github.com/tris/youtube_exporter/releases)
or run from Docker:

```bash
docker run -d -p 9473:9473 -e YOUTUBE_API_KEY=your_api_key_here ghcr.io/tris/youtube_exporter
```

An alternate port may be defined using the `PORT` environment variable.

## Configuration

You must provide a YouTube Data API v3 key via the `YOUTUBE_API_KEY`
environment variable.

To get an API key:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API key)
5. Optionally restrict the key to YouTube Data API v3

## Usage

### Video metrics

To scrape metrics for a specific video, make a request to `/scrape` with the
video ID:

```bash
curl "http://localhost:9473/scrape?v=..."
```

The video ID is the 11-character identifier from YouTube URLs (e.g.,
`yv2RtoIMNzA` from `https://www.youtube.com/watch?v=yv2RtoIMNzA`).

### Channel metrics

To scrape metrics for an entire channel, make a request to `/scrape` with the
channel ID:

```bash
curl "http://localhost:9473/scrape?channel=..."
```

Channel IDs start with "UC" followed by 22 characters (e.g.,
`UC029bcZGqxOsRkvWrZLhgKQ`). Use a [YouTube Channel ID Finder](
https://www.streamweasels.com/tools/youtube-channel-id-and-user-id-convertor/)
if you don't know your ID.

## Metrics

### Video metrics

When querying individual videos (`?v=VIDEO_ID`):

- `youtube_video_view_count` - Total view count for the video
- `youtube_video_like_count` - Total like count for the video
- `youtube_video_concurrent_viewers` - Current concurrent viewers (only
  non-zero for live streams)
- `youtube_video_live` - Binary indicator (1 if currently live, 0 otherwise)
- `youtube_video_live_status` - Info metric with state label ("live",
  "upcoming", or "none")
- `youtube_video_scrape_success` - Scrape success indicator

All video metrics include `video_id` and `title` labels.

### Channel metrics

When querying channels (`?channel=CHANNEL_ID`):

- `youtube_channel_subscriber_count` - Total subscriber count for the
  channel
- `youtube_channel_view_count` - Total view count across all channel
  videos
- `youtube_channel_video_count` - Total number of videos on the channel
- `youtube_channel_scrape_success` - Scrape success indicator

Channel metrics include `channel_id` and `channel_title` labels.

Additionally, when querying a channel, the exporter will automatically
discover and include metrics for all currently live streams from that channel
using the same video metrics listed above.

## Example Prometheus config

```yaml
scrape_configs:
  - job_name: 'youtube_video'
    scrape_interval: 5m
    static_configs:
      - targets:
          - yv2RtoIMNzA
    metrics_path: /scrape
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_v
      - target_label: __address__
        replacement: localhost:9473

  - job_name: 'youtube_channel'
    scrape_interval: 5m
    static_configs:
      - targets:
          - UC029bcZGqxOsRkvWrZLhgKQ
    metrics_path: /scrape
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_channel
      - target_label: __address__
        replacement: localhost:9473

  - job_name: 'youtube_exporter'
    scrape_interval: 5m
    static_configs:
    - targets: ['localhost:9473']
```

## API rate limits

The YouTube Data API is metered. Each request consumes:
- Video details: 1 unit per batch (up to 50 videos)
- Channel details: 1 unit per channel
- Playlist items: 1 unit per page (50 items)
- Live stream search: 100 units per request (used sparingly; see below)

### Caching algorithm

We use a hybrid approach to minimize quota usage while ensuring full discovery:

**First request (per channel):**
- **Channels with ≤4,950 videos (99 pages)**: Uses full pagination through all videos (cheaper than Search API)
- **Channels with >4,950 videos**: Uses Search API (100 units) for comprehensive discovery
- Caches discovered live stream IDs in memory
- Results are cached to avoid repeating expensive operations

**Subsequent requests:**
- Skips expensive Search API entirely
- Uses efficient playlist method to check 50 most recent uploads (1-2 units)
- Refreshes cached stream details to verify they're still live (1 unit per 50 streams)
- Removes ended streams from cache automatically

**Quota usage breakdown:**

First request (>4,950 videos):
- Channel.List: 1 unit (for channel details)
- Search.List: 100 units (find all live streams)
- Videos.List: 1 unit (batched details for search results)
- PlaylistItems.List: 1 unit (check 50 recent videos)
- Videos.List: 1 unit (batched details for recent videos)
- **Total: ~104 units**

Subsequent requests (typical case with 1 live stream):
- Channel.List: 1 unit (for channel details)
- PlaylistItems.List: 1 unit (check 50 recent videos)
- Videos.List: 1 unit (batched details for recent videos)
- Videos.List: 0-1 units (refresh cached streams if any old ones exist)
- **Total: 3-4 units** (3 if all streams are recent, 4 if old cached streams exist)

**Summary:**
- **First scrape (≤4,950 videos)**: Up to 99 units (full pagination)
- **First scrape (>4,950 videos)**: ~104 units
- **Subsequent scrapes**: 3-4 units typically
- **With disable_live=true**: 1 unit (channel stats only)
- **Daily usage at 1-minute intervals**: ~4,400 units typical (well within 10,000 quota)

This approach ensures:
- Long-running streams (even years old) are never missed
- New streams are detected within one polling interval
- Minimal ongoing quota consumption
- Automatic cleanup of ended streams

The default quota is 10,000 units per day. Monitor your usage in the Google
Cloud Console and adjust scrape intervals accordingly.

### Query parameters

- `v`: YouTube video ID (for single video metrics)
- `channel`: YouTube channel ID (for channel metrics)
- `disable_live`: Set to `true` to skip live stream detection (channel requests only)

## Building

```bash
go build -o youtube_exporter .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
