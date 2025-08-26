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

### Video Metrics

To scrape metrics for a specific YouTube video, make a request to `/scrape`
with the video ID:

```bash
curl "http://localhost:9473/scrape?v=..."
```

The video ID is the 11-character identifier from YouTube URLs (e.g.,
`yv2RtoIMNzA` from `https://www.youtube.com/watch?v=yv2RtoIMNzA`).

### Channel Metrics

To scrape metrics for an entire YouTube channel, make a request to `/scrape`
with the channel ID:

```bash
curl "http://localhost:9473/scrape?channel=..."
```

Channel IDs start with "UC" followed by 22 characters (e.g.,
`UC029bcZGqxOsRkvWrZLhgKQ`). Use a [YouTube Channel ID Finder](
https://www.streamweasels.com/tools/youtube-channel-id-and-user-id-convertor/)
if you don't know your ID.

## Metrics

### Video Metrics

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

### Channel Metrics

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

## Example Prometheus Config

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

## API Rate Limits

The YouTube Data API v3 has quota limits. Each request consumes quota units:
- Video details: 1 unit per video
- Channel details: 1 unit per channel
- Live stream search: 100 units per request

The default quota is 10,000 units per day. Monitor your usage in the Google
Cloud Console and adjust scrape intervals accordingly.

## Building

```bash
go build -o youtube_exporter .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
