"""Metrics collection and Prometheus module for YouTube exporter."""

import re
import time
import logging
import psutil
import os
from prometheus_client import generate_latest, CollectorRegistry
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector
from quota import api_quota_used, api_quota_total
from youtube_client import fetch_video_details, fetch_channel_details, fetch_channel_live_streams
from entropy import fetch_two_spaced_frames, calculate_intra_entropy, calculate_inter_entropy
from api_errors import api_errors

logger = logging.getLogger(__name__)

# Global variables for metrics storage
metrics_data = {}  # Key: video_id, Value: metrics dict
channel_metrics_data = {}  # Key: channel_id, Value: channel metrics dict
# Separate storage for entropy values that can be updated asynchronously
entropy_data = {}  # Key: video_id, Value: {'intra_entropy': float, 'inter_entropy': float, 'timestamp': float}

# Prometheus registry
registry = CollectorRegistry()


class TimestampedMetricsCollector(Collector):
    """Custom collector that supports timestamps on metrics."""

    def collect(self):
        """Collect metrics with timestamps."""
        # Always create metric families, regardless of data availability
        
        # Entropy metrics
        intra_family = GaugeMetricFamily(
            'youtube_video_intra_entropy',
            'Shannon entropy of pixel intensities within a single frame (bits, 0-8 range)',
            labels=['video_id', 'title']
        )
        inter_family = GaugeMetricFamily(
            'youtube_video_inter_entropy',
            'Shannon entropy of pixel differences between frames separated by ~1 second (bits, 0-8 range)',
            labels=['video_id', 'title']
        )
        bitrate_family = GaugeMetricFamily(
            'youtube_video_bitrate',
            'Bitrate of the video stream in bits per second, calculated from 1-second download',
            labels=['video_id', 'title', 'resolution']
        )

        # YouTube API metrics
        view_family = GaugeMetricFamily(
            'youtube_video_view_count',
            'Total view count reported by YouTube for this video',
            labels=['video_id', 'title']
        )
        like_family = GaugeMetricFamily(
            'youtube_video_like_count',
            'Total like count reported by YouTube for this video',
            labels=['video_id', 'title']
        )
        concurrent_family = GaugeMetricFamily(
            'youtube_video_concurrent_viewers',
            'Concurrent viewers (only non-zero while live) reported by YouTube for this video',
            labels=['video_id', 'title']
        )
        live_family = GaugeMetricFamily(
            'youtube_video_live',
            '1 if YouTube reports the video as currently live, else 0',
            labels=['video_id', 'title']
        )
        live_status_family = GaugeMetricFamily(
            'youtube_video_live_status',
            'Infometric with state label; 1 for the current state ("live", "upcoming", or "none")',
            labels=['video_id', 'title', 'state']
        )

        # Channel metrics
        channel_subscriber_family = GaugeMetricFamily(
            'youtube_channel_subscriber_count',
            'Total subscriber count reported by YouTube for this channel',
            labels=['channel_id', 'channel_title']
        )
        channel_view_family = GaugeMetricFamily(
            'youtube_channel_view_count',
            'Total view count reported by YouTube for this channel',
            labels=['channel_id', 'channel_title']
        )
        channel_video_family = GaugeMetricFamily(
            'youtube_channel_video_count',
            'Total video count reported by YouTube for this channel',
            labels=['channel_id', 'channel_title']
        )
        channel_scrape_success_family = GaugeMetricFamily(
            'youtube_channel_scrape_success',
            '1 if the scrape of YouTube API succeeded, else 0',
            labels=['channel_id', 'channel_title']
        )

        # Quota metrics
        quota_family = GaugeMetricFamily(
            'youtube_api_quota_units_today',
            'Estimated YouTube Data API quota units consumed today, labeled by endpoint',
            labels=['endpoint']
        )
        quota_total_family = CounterMetricFamily(
            'youtube_api_quota_units_total',
            'Total YouTube Data API quota units consumed, labeled by endpoint',
            labels=['endpoint']
        )

        # API errors metrics
        errors_family = CounterMetricFamily(
            'youtube_api_errors_total',
            'Total YouTube API errors, labeled by error code and endpoint',
            labels=['code', 'endpoint']
        )

        # Process metrics
        process_cpu_family = CounterMetricFamily(
            'process_cpu_seconds_total',
            'Total user and system CPU time spent in seconds'
        )
        process_memory_family = GaugeMetricFamily(
            'process_resident_memory_bytes',
            'Resident memory size in bytes'
        )
        process_virtual_memory_family = GaugeMetricFamily(
            'process_virtual_memory_bytes',
            'Virtual memory size in bytes'
        )
        process_start_time_family = GaugeMetricFamily(
            'process_start_time_seconds',
            'Start time of the process since unix epoch in seconds'
        )

        # Process video metrics data
        if metrics_data:
            for video_id, data in metrics_data.items():
                # Get title for metrics
                title = data.get('api_data', {}).get('title', '') if 'api_data' in data else ''
                
                # Check for entropy data in separate storage (preferred) or in metrics_data (backward compat)
                if video_id in entropy_data:
                    entropy_info = entropy_data[video_id]
                    entropy_timestamp = entropy_info.get('timestamp', data['timestamp'])
                    if 'intra_entropy' in entropy_info:
                        intra_family.add_metric(
                            [video_id, title],
                            entropy_info['intra_entropy'],
                            timestamp=entropy_timestamp
                        )
                    if 'inter_entropy' in entropy_info:
                        inter_family.add_metric(
                            [video_id, title],
                            entropy_info['inter_entropy'],
                            timestamp=entropy_timestamp
                        )
                    if 'bitrate' in entropy_info and entropy_info['bitrate'] is not None:
                        bitrate_family.add_metric(
                            [video_id, title, entropy_info.get('resolution', 'unknown')],
                            entropy_info['bitrate'],
                            timestamp=entropy_timestamp
                        )
                elif 'intra_entropy' in data or 'inter_entropy' in data:
                    # Backward compatibility: check metrics_data
                    if 'intra_entropy' in data:
                        intra_family.add_metric(
                            [video_id, title],
                            data['intra_entropy'],
                            timestamp=data['timestamp']
                        )
                    if 'inter_entropy' in data:
                        inter_family.add_metric(
                            [video_id, title],
                            data['inter_entropy'],
                            timestamp=data['timestamp']
                        )

                # YouTube API metrics
                if 'api_data' in data:
                    api_data = data['api_data']
                    labels = [video_id, api_data.get('title', '')]

                    view_family.add_metric(labels, api_data.get('view_count', 0), timestamp=data['timestamp'])
                    like_family.add_metric(labels, api_data.get('like_count', 0), timestamp=data['timestamp'])
                    concurrent_family.add_metric(labels, api_data.get('concurrent_viewers', 0), timestamp=data['timestamp'])
                    live_family.add_metric(labels, api_data.get('live_binary', 0), timestamp=data['timestamp'])

                    # Live status infometric
                    live_status_labels = [video_id, api_data.get('title', ''), api_data.get('live_broadcast_state', 'none')]
                    live_status_family.add_metric(live_status_labels, 1, timestamp=data['timestamp'])

        # Process channel metrics data
        if channel_metrics_data:
            for channel_id, channel_data in channel_metrics_data.items():
                if 'channel_info' in channel_data:
                    channel_info = channel_data['channel_info']
                    channel_labels = [channel_id, channel_info.get('title', '')]

                    channel_subscriber_family.add_metric(channel_labels, channel_info.get('subscriber_count', 0), timestamp=channel_data['timestamp'])
                    channel_view_family.add_metric(channel_labels, channel_info.get('view_count', 0), timestamp=channel_data['timestamp'])
                    channel_video_family.add_metric(channel_labels, channel_info.get('video_count', 0), timestamp=channel_data['timestamp'])
                    channel_scrape_success_family.add_metric(channel_labels, 1, timestamp=channel_data['timestamp'])

                # Live stream metrics for channels
                if 'live_streams' in channel_data:
                    logger.info(f"Processing {len(channel_data['live_streams'])} live streams for channel {channel_id}")
                    for stream in channel_data['live_streams']:
                        stream_video_id = stream['video_id']
                        stream_labels = [stream_video_id, stream['title']]
                        
                        # Check for entropy data in the separate entropy_data storage
                        has_entropy = stream_video_id in entropy_data
                        if has_entropy:
                            entropy_info = entropy_data[stream_video_id]
                            logger.info(f"Found entropy data for {stream_video_id}: intra={entropy_info.get('intra_entropy', 0):.2f}, inter={entropy_info.get('inter_entropy', 0):.2f}")
                        else:
                            logger.debug(f"No entropy data yet for {stream_video_id}")

                        view_family.add_metric(stream_labels, stream.get('view_count', 0), timestamp=channel_data['timestamp'])
                        like_family.add_metric(stream_labels, stream.get('like_count', 0), timestamp=channel_data['timestamp'])
                        concurrent_family.add_metric(stream_labels, stream.get('concurrent_viewers', 0), timestamp=channel_data['timestamp'])
                        live_family.add_metric(stream_labels, stream.get('live_binary', 0), timestamp=channel_data['timestamp'])

                        # Add entropy metrics from separate storage if available
                        if stream_video_id in entropy_data:
                            entropy_info = entropy_data[stream_video_id]
                            entropy_timestamp = entropy_info.get('timestamp', channel_data['timestamp'])
                            if 'intra_entropy' in entropy_info:
                                logger.info(f"Adding intra_entropy metric for {stream_video_id}: {entropy_info['intra_entropy']}")
                                intra_family.add_metric(stream_labels, entropy_info['intra_entropy'], timestamp=entropy_timestamp)
                            if 'inter_entropy' in entropy_info:
                                logger.info(f"Adding inter_entropy metric for {stream_video_id}: {entropy_info['inter_entropy']}")
                                inter_family.add_metric(stream_labels, entropy_info['inter_entropy'], timestamp=entropy_timestamp)
                            if 'bitrate' in entropy_info and entropy_info['bitrate'] is not None:
                                logger.info(f"Adding bitrate metric for {stream_video_id}: {entropy_info['bitrate']}")
                                bitrate_family.add_metric([stream_video_id, stream['title'], entropy_info.get('resolution', 'unknown')], entropy_info['bitrate'], timestamp=entropy_timestamp)

                        # Live status infometric
                        live_status_labels = [stream_video_id, stream['title'], stream.get('live_broadcast_state', 'none')]
                        live_status_family.add_metric(live_status_labels, 1, timestamp=channel_data['timestamp'])

        # Process quota metrics
        for endpoint, units in api_quota_used.items():
            quota_family.add_metric([endpoint], units)

        for endpoint, units in api_quota_total.items():
            quota_total_family.add_metric([endpoint], units)

        # Process API errors metrics
        for (code, endpoint), count in api_errors.items():
            errors_family.add_metric([str(code), endpoint], count)

        # Process metrics
        try:
            process = psutil.Process(os.getpid())
            cpu_times = process.cpu_times()
            memory_info = process.memory_info()
            process_cpu_family.add_metric([], cpu_times.user + cpu_times.system)
            process_memory_family.add_metric([], memory_info.rss)
            process_virtual_memory_family.add_metric([], memory_info.vms)
            process_start_time_family.add_metric([], process.create_time())
        except Exception as e:
            logger.warning(f"Failed to collect process metrics: {e}")

        # Always yield all metric families
        yield intra_family
        yield inter_family
        yield bitrate_family
        yield view_family
        yield like_family
        yield concurrent_family
        yield live_family
        yield live_status_family
        yield channel_subscriber_family
        yield channel_view_family
        yield channel_video_family
        yield channel_scrape_success_family
        yield quota_family
        yield quota_total_family
        yield errors_family
        yield process_cpu_family
        yield process_memory_family
        yield process_virtual_memory_family
        yield process_start_time_family


def format_integer_metrics(output):
    """Format integer metrics to remove unnecessary decimal places."""
    # Pattern to match metrics that should be integers (counts, binary values)
    integer_metric_patterns = [
        r'(youtube_video_view_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_video_like_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_video_concurrent_viewers\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_video_live\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_video_live_status\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_channel_subscriber_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_channel_view_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_channel_video_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_api_quota_units_today\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_api_quota_units_total\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
        r'(youtube_api_errors_total\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
    ]
    
    for pattern in integer_metric_patterns:
        output = re.sub(pattern, r'\1\2\3', output)
    
    return output


def process_channel_data(channel, live_videos, fetch_images=True):
    """Process YouTube API channel response into channel snapshot."""
    stats = channel.get('statistics', {})
    snippet = channel.get('snippet', {})

    subscriber_count = int(stats.get('subscriberCount', 0))
    view_count = int(stats.get('viewCount', 0))
    video_count = int(stats.get('videoCount', 0))
    title = snippet.get('title', '')

    # Process live streams
    live_streams = []
    for video in live_videos:
        stream_data = process_video_data_for_channel(video, fetch_images)
        live_streams.append(stream_data)

    return {
        'channel_id': channel.get('id'),
        'title': title,
        'subscriber_count': subscriber_count,
        'view_count': view_count,
        'video_count': video_count,
        'live_streams': live_streams
    }


def compute_and_store_entropy(video_id, title=None, max_height=None):
    """Compute entropy for a video and store it in the global entropy_data."""
    global entropy_data
    
    logger.info(f"Computing entropy for video {video_id}")
    url = f"https://www.youtube.com/watch?v={video_id}"
    frame1, frame2, bitrate, resolution = fetch_two_spaced_frames(url, max_height=max_height)
    
    if frame1 is not None and frame2 is not None:
        # Calculate intra-entropy using the second frame
        intra_entropy = calculate_intra_entropy(frame2)
        # Calculate inter-entropy between the two spaced frames
        inter_entropy = calculate_inter_entropy(frame2, frame1)
        
        entropy_data[video_id] = {
            'intra_entropy': intra_entropy,
            'inter_entropy': inter_entropy,
            'bitrate': bitrate,
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        bitrate_str = f"{bitrate:.0f}" if bitrate is not None else "N/A"
        logger.info(f"Stored entropy for {video_id}: intra={intra_entropy:.2f}, inter={inter_entropy:.2f}, bitrate={bitrate_str} bps, resolution={resolution}")
        return intra_entropy, inter_entropy, bitrate, resolution
    else:
        logger.warning(f"Failed to fetch frames for {video_id}")
        return None, None, None


def process_video_data_for_channel(video, fetch_images=True):
    """Process video data for channel live streams."""
    stats = video.get('statistics', {})
    lsd = video.get('liveStreamingDetails', {})
    snippet = video.get('snippet', {})

    view_count = int(stats.get('viewCount', 0))
    like_count = int(stats.get('likeCount', 0))
    concurrent = int(lsd.get('concurrentViewers', 0))

    live_broadcast_content = snippet.get('liveBroadcastContent', 'none')
    live_binary = 1 if live_broadcast_content == 'live' else 0
    title = snippet.get('title', '')
    video_id = video.get('id')

    stream_data = {
        'video_id': video_id,
        'title': title,
        'view_count': view_count,
        'like_count': like_count,
        'concurrent_viewers': concurrent,
        'live_broadcast_state': live_broadcast_content,
        'live_binary': live_binary
    }

    # Don't compute entropy here - it will be done asynchronously
    # Just return the basic stream data
    logger.debug(f"Processed video data for {video_id}: live={live_binary}, fetch_images={fetch_images}")

    return stream_data


def update_channel_metrics(channel_id, fetch_images=True, disable_live=False):
    """Fetch channel data and live streams, update storage."""
    global channel_metrics_data, metrics_data, entropy_data
    timestamp = time.time()

    # Fetch channel details
    channel_data = fetch_channel_details(channel_id)
    if not channel_data:
        logger.warning(f"Failed to fetch channel data for {channel_id}")
        return

    # Fetch live streams
    live_videos = fetch_channel_live_streams(channel_data, disable_live)

    # Process channel data (without entropy computation)
    channel_snapshot = process_channel_data(channel_data, live_videos, fetch_images=False)

    # Store channel metrics
    channel_metrics_data[channel_id] = {
        'channel_info': {
            'title': channel_snapshot['title'],
            'subscriber_count': channel_snapshot['subscriber_count'],
            'view_count': channel_snapshot['view_count'],
            'video_count': channel_snapshot['video_count']
        },
        'live_streams': channel_snapshot['live_streams'],
        'timestamp': timestamp
    }

    # If fetch_images is True, kick off background entropy computation for live streams
    if fetch_images and not disable_live:
        import threading
        for stream in channel_snapshot['live_streams']:
            video_id = stream.get('video_id')
            if video_id and stream.get('live_binary') == 1:
                # Check if we already have recent entropy data
                if video_id in entropy_data:
                    age = time.time() - entropy_data[video_id].get('timestamp', 0)
                    if age < 300:  # Skip if entropy was computed in last 5 minutes
                        logger.debug(f"Skipping entropy computation for {video_id}, data is {age:.0f}s old")
                        continue
                
                # Start background thread to compute entropy
                logger.info(f"Starting background entropy computation for {video_id}")
                thread = threading.Thread(
                    target=compute_and_store_entropy,
                    args=(video_id, stream.get('title'), None),
                    daemon=True
                )
                thread.start()

    # Log status
    entropy_count = sum(1 for stream in channel_snapshot['live_streams']
                       if stream.get('video_id') in entropy_data)
    logger.info(f"Updated channel metrics for {channel_id}: {len(live_videos)} live streams, {entropy_count} with existing entropy data")


def update_metrics(video_id, fetch_images=True, max_height=None):
    """Fetch video data and frames, calculate metrics, update storage."""
    global metrics_data, channel_metrics_data, entropy_data
    timestamp = time.time()

    # Always try to fetch YouTube API data
    video_data = fetch_video_details(video_id)
    api_data = {}

    if video_data:
        # Extract API data similar to Go version
        stats = video_data.get('statistics', {})
        lsd = video_data.get('liveStreamingDetails', {})
        snippet = video_data.get('snippet', {})

        view_count = int(stats.get('viewCount', 0))
        like_count = int(stats.get('likeCount', 0))
        concurrent = int(lsd.get('concurrentViewers', 0))

        live_broadcast_content = snippet.get('liveBroadcastContent', 'none')
        live_binary = 1 if live_broadcast_content == 'live' else 0

        title = snippet.get('title', '')
        channel_id = snippet.get('channelId', '')

        api_data = {
            'view_count': view_count,
            'like_count': like_count,
            'concurrent_viewers': concurrent,
            'live_broadcast_state': live_broadcast_content,
            'live_binary': live_binary,
            'title': title,
            'channel_id': channel_id
        }

        # Also fetch channel information for this video
        if channel_id and channel_id not in channel_metrics_data:
            channel_data = fetch_channel_details(channel_id)
            if channel_data:
                channel_snapshot = process_channel_data(channel_data, [], False)  # No live streams for video requests, no image fetching
                channel_metrics_data[channel_id] = {
                    'channel_info': {
                        'title': channel_snapshot['title'],
                        'subscriber_count': channel_snapshot['subscriber_count'],
                        'view_count': channel_snapshot['view_count'],
                        'video_count': channel_snapshot['video_count']
                    },
                    'live_streams': [],
                    'timestamp': timestamp
                }
                logger.info(f"Fetched channel info for {channel_id} from video {video_id}")

        logger.info(f"YouTube API data for {video_id}: views={view_count}, likes={like_count}, live={live_binary}")
    else:
        logger.warning(f"Failed to fetch YouTube API data for {video_id}")

    # Store API data
    metrics_data[video_id] = {
        'api_data': api_data,
        'timestamp': timestamp
    }

    # Handle entropy computation
    if fetch_images:
        # Check if we already have recent entropy data
        if video_id in entropy_data:
            age = time.time() - entropy_data[video_id].get('timestamp', 0)
            if age < 300:  # Skip if entropy was computed in last 5 minutes
                logger.debug(f"Using existing entropy data for {video_id}, data is {age:.0f}s old")
                # Copy entropy data to metrics_data for backward compatibility
                metrics_data[video_id]['intra_entropy'] = entropy_data[video_id].get('intra_entropy', 0.0)
                metrics_data[video_id]['inter_entropy'] = entropy_data[video_id].get('inter_entropy', 0.0)
                return
        
        # Compute entropy synchronously for individual video requests
        intra_entropy, inter_entropy, bitrate, resolution = compute_and_store_entropy(video_id, api_data.get('title'), max_height)
        
        if intra_entropy is not None and inter_entropy is not None:
            # Also store in metrics_data for backward compatibility
            metrics_data[video_id]['intra_entropy'] = intra_entropy
            metrics_data[video_id]['inter_entropy'] = inter_entropy
            logger.info(f"Updated metrics for {video_id}: intra={intra_entropy:.2f}, inter={inter_entropy:.2f}")
        else:
            # Store zero values if entropy computation failed
            metrics_data[video_id]['intra_entropy'] = 0.0
            metrics_data[video_id]['inter_entropy'] = 0.0
    else:
        logger.info(f"Skipped image fetching for {video_id}, updated API data only")


def get_prometheus_metrics():
    """Generate Prometheus metrics output."""
    output = generate_latest(registry).decode('utf-8')
    formatted_output = format_integer_metrics(output)
    return formatted_output


# Register the custom collector
registry.register(TimestampedMetricsCollector())