"""Metrics collection and Prometheus module for YouTube exporter."""

import re
import time
import logging
import psutil
import os
import threading
from prometheus_client import generate_latest, CollectorRegistry
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from prometheus_client.registry import Collector
from quota import api_quota_used, api_quota_total
from youtube_client import fetch_video_details, fetch_channel_details, fetch_channel_live_streams
from entropy import fetch_two_spaced_frames, calculate_spatial_entropy, calculate_temporal_entropy, count_objects_in_video
from api_errors import api_errors

logger = logging.getLogger(__name__)

# Global variables for metrics storage
metrics_data = {}  # Key: video_id, Value: metrics dict
channel_metrics_data = {}  # Key: channel_id, Value: channel metrics dict
# Separate storage for entropy values that can be updated asynchronously
entropy_data = {}  # Key: video_id, Value: {'spatial_entropy': float, 'temporal_entropy': float, 'timestamp': float}
# Separate storage for object detection results
object_data = {}  # Key: f"{video_id}_{object_type}", Value: {'object_count': int, 'object_type': str, 'timestamp': float}
# Track active object detection tasks to prevent duplicates
active_object_detection = set()  # Set of f"{video_id}_{object_type}" currently being processed
# Track active entropy computation tasks to prevent duplicates
active_entropy_computation = set()  # Set of video_id currently being processed

# Prometheus registry
registry = CollectorRegistry()


class TimestampedMetricsCollector(Collector):
    """Custom collector that supports timestamps on metrics."""

    def collect(self):
        """Collect metrics with timestamps."""
        # Always create metric families, regardless of data availability

        # Entropy metrics
        spatial_family = GaugeMetricFamily(
            'youtube_video_spatial_entropy',
            'Shannon entropy of pixel intensities within a single frame (bits, 0-8 range)',
            labels=['video_id', 'title']
        )
        temporal_family = GaugeMetricFamily(
            'youtube_video_temporal_entropy',
            'Shannon entropy of pixel differences between frames separated by ~1 second (bits, 0-8 range)',
            labels=['video_id', 'title']
        )
        object_count_family = GaugeMetricFamily(
            'youtube_video_object_count',
            'Number of detected objects of specified type in the video frame',
            labels=['video_id', 'title', 'object_type']
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

        # Track processed video IDs to avoid duplicates
        processed_videos = set()

        # Process video metrics data
        if metrics_data:
            for video_id, data in metrics_data.items():
                processed_videos.add(video_id)
                # Get title for metrics
                title = data.get('api_data', {}).get('title', '') if 'api_data' in data else ''

                # Object count metrics from separate storage (emit all cached results)
                for key, obj_info in object_data.items():
                    if obj_info.get('video_id') == video_id:
                        object_count_family.add_metric(
                            [video_id, title, obj_info['object_type']],
                            obj_info['object_count'],
                            timestamp=obj_info['timestamp']
                        )

                # Check for entropy data in separate storage (preferred) or in metrics_data (backward compat)
                if video_id in entropy_data:
                    entropy_info = entropy_data[video_id]
                    entropy_timestamp = entropy_info.get('timestamp', data['timestamp'])
                    if 'spatial_entropy' in entropy_info:
                        spatial_family.add_metric(
                            [video_id, title],
                            entropy_info['spatial_entropy'],
                            timestamp=entropy_timestamp
                        )
                    if 'temporal_entropy' in entropy_info:
                        temporal_family.add_metric(
                            [video_id, title],
                            entropy_info['temporal_entropy'],
                            timestamp=entropy_timestamp
                        )
                    if 'bitrate' in entropy_info and entropy_info['bitrate'] is not None:
                        bitrate_family.add_metric(
                            [video_id, title, entropy_info.get('resolution', 'unknown')],
                            entropy_info['bitrate'],
                            timestamp=entropy_timestamp
                        )
                elif 'spatial_entropy' in data or 'temporal_entropy' in data:
                    # Backward compatibility: check metrics_data
                    if 'spatial_entropy' in data:
                        spatial_family.add_metric(
                            [video_id, title],
                            data['spatial_entropy'],
                            timestamp=data['timestamp']
                        )
                    if 'temporal_entropy' in data:
                        temporal_family.add_metric(
                            [video_id, title],
                            data['temporal_entropy'],
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

                        # Skip if this video was already processed in the individual video metrics section
                        if stream_video_id in processed_videos:
                            logger.debug(f"Skipping duplicate metrics for {stream_video_id} (already processed as individual video)")
                            continue

                        stream_labels = [stream_video_id, stream['title']]

                        # Check for entropy data in the separate entropy_data storage
                        has_entropy = stream_video_id in entropy_data
                        if has_entropy:
                            entropy_info = entropy_data[stream_video_id]
                            logger.info(f"Found entropy data for {stream_video_id}: spatial={entropy_info.get('spatial_entropy', 0):.2f}, temporal={entropy_info.get('temporal_entropy', 0):.2f}")
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
                            if 'spatial_entropy' in entropy_info:
                                logger.info(f"Adding spatial_entropy metric for {stream_video_id}: {entropy_info['spatial_entropy']}")
                                spatial_family.add_metric(stream_labels, entropy_info['spatial_entropy'], timestamp=entropy_timestamp)
                            if 'temporal_entropy' in entropy_info:
                                logger.info(f"Adding temporal_entropy metric for {stream_video_id}: {entropy_info['temporal_entropy']}")
                                temporal_family.add_metric(stream_labels, entropy_info['temporal_entropy'], timestamp=entropy_timestamp)
                            if 'bitrate' in entropy_info and entropy_info['bitrate'] is not None:
                                logger.info(f"Adding bitrate metric for {stream_video_id}: {entropy_info['bitrate']}")
                                bitrate_family.add_metric([stream_video_id, stream['title'], entropy_info.get('resolution', 'unknown')], entropy_info['bitrate'], timestamp=entropy_timestamp)

                        # Add object detection metrics from separate storage if available
                        for key, obj_info in object_data.items():
                            if obj_info.get('video_id') == stream_video_id:
                                object_count_family.add_metric(
                                    [stream_video_id, stream['title'], obj_info['object_type']],
                                    obj_info['object_count'],
                                    timestamp=obj_info['timestamp']
                                )
                                logger.debug(f"Adding object_count metric for {stream_video_id}: {obj_info['object_count']} '{obj_info['object_type']}' objects")

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
        yield spatial_family
        yield temporal_family
        yield bitrate_family
        yield object_count_family
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
        r'(youtube_video_object_count\{[^}]*\}\s+)(\d+)\.0+(\s+\d+)?',
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
    """Compute entropy for a video and store it in the global entropy_data.
    Returns the high-resolution frame for potential reuse in object detection."""
    global entropy_data, active_entropy_computation

    try:
        logger.info(f"Computing entropy for video {video_id}")
        url = f"https://www.youtube.com/watch?v={video_id}"
        frame1, frame2, bitrate, resolution = fetch_two_spaced_frames(url, max_height=max_height)

        if frame1 is not None and frame2 is not None:
            # Calculate spatial-entropy using the second frame
            spatial_entropy = calculate_spatial_entropy(frame2)
            # Calculate temporal-entropy between the two spaced frames
            temporal_entropy = calculate_temporal_entropy(frame2, frame1)

            entropy_data[video_id] = {
                'spatial_entropy': spatial_entropy,
                'temporal_entropy': temporal_entropy,
                'bitrate': bitrate,
                'resolution': resolution,
                'timestamp': time.time(),
                'reusable_frame': frame2  # Store high-res frame for object detection reuse
            }

            bitrate_str = f"{bitrate:.0f}" if bitrate is not None else "N/A"
            logger.info(f"Stored entropy for {video_id}: spatial={spatial_entropy:.2f}, temporal={temporal_entropy:.2f}, bitrate={bitrate_str} bps, resolution={resolution}")
            return spatial_entropy, temporal_entropy, bitrate, resolution, frame2
        else:
            logger.warning(f"Failed to fetch frames for {video_id}")
            return None, None, None, None, None
    finally:
        # Always remove from active set when done (success or failure)
        active_entropy_computation.discard(video_id)


def compute_and_store_objects(video_id, object_type, max_height=None):
    """Compute object detection for a video and store it in the global object_data.
    Tries to reuse high-resolution frame from entropy calculation if available."""
    global object_data, active_object_detection, entropy_data

    key = f"{video_id}_{object_type}"
    try:
        logger.debug(f"Computing object detection for video {video_id}, object_type: {object_type}")

        # Check if we have a reusable high-resolution frame from recent entropy calculation
        reuse_frame = None
        if video_id in entropy_data:
            entropy_info = entropy_data[video_id]
            frame_age = time.time() - entropy_info.get('timestamp', 0)
            if frame_age < 300 and 'reusable_frame' in entropy_info:  # Use frame if < 5 minutes old
                reuse_frame = entropy_info['reusable_frame']
                logger.debug(f"Reusing high-resolution frame from entropy calculation (age: {frame_age:.0f}s)")
            else:
                logger.debug(f"Cannot reuse frame - age: {frame_age:.0f}s, has_frame: {'reusable_frame' in entropy_info}")
        else:
            logger.debug(f"No entropy data available for {video_id}, will capture new frame")

        object_count = count_objects_in_video(video_id, object_type, max_height, reuse_frame=reuse_frame)

        if object_count is not None:
            object_data[key] = {
                'object_count': object_count,
                'object_type': object_type,
                'video_id': video_id,
                'timestamp': time.time()
            }
            logger.info(f"Stored object detection for {video_id}: {object_count} '{object_type}' objects")
            return object_count
        else:
            logger.warning(f"Failed to detect objects for {video_id}")
            return None
    except Exception as e:
        logger.error(f"Exception in compute_and_store_objects: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None
    finally:
        # Always remove from active set when done (success or failure)
        active_object_detection.discard(key)


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


def update_channel_metrics(channel_id, fetch_images=True, disable_live=False, match=None):
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

                # Check if entropy computation is already in progress
                if video_id in active_entropy_computation:
                    logger.debug(f"Entropy computation already in progress for {video_id}")
                    continue

                # Start background thread to compute entropy
                logger.info(f"Starting background entropy computation for {video_id}")
                active_entropy_computation.add(video_id)
                thread = threading.Thread(
                    target=compute_and_store_entropy,
                    args=(video_id, stream.get('title'), None),
                    daemon=True
                )
                thread.start()

    # If match is provided, also do object detection for live streams
    if match and not disable_live:
        import threading
        for stream in channel_snapshot['live_streams']:
            video_id = stream.get('video_id')
            if video_id and stream.get('live_binary') == 1:
                # Object counting mode - check if we need to start background detection
                key = f"{video_id}_{match}"
                if key in object_data:
                    stored_data = object_data[key]
                    # Check if existing data is recent enough
                    age = time.time() - stored_data.get('timestamp', 0)
                    if age < 300:  # Use existing data if recent
                        logger.debug(f"Using existing object data for {video_id}, object_type: {match}, data is {age:.0f}s old")
                        continue
                    else:
                        logger.debug(f"Existing object data is stale (age: {age:.0f}s), will recompute")

                # Check if detection is already in progress (with thread-safe add)
                if key in active_object_detection:
                    logger.debug(f"Object detection already in progress for {video_id}, object_type: {match}")
                    continue

                # Add to active set BEFORE starting thread to prevent race conditions
                active_object_detection.add(key)
                logger.info(f"Starting object detection for {video_id}, object_type: {match}")
                thread = threading.Thread(
                    target=compute_and_store_objects,
                    args=(video_id, match, None),
                    daemon=True,
                    name=f"ChannelObjectDetection-{video_id}-{match}"
                )
                thread.start()

    # Log status
    entropy_count = sum(1 for stream in channel_snapshot['live_streams']
                       if stream.get('video_id') in entropy_data)
    logger.info(f"Updated channel metrics for {channel_id}: {len(live_videos)} live streams, {entropy_count} with existing entropy data")


def update_metrics(video_id, fetch_images=True, max_height=None, match=None):
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

    # Handle entropy computation and object counting
    if fetch_images:
        # Check if we already have recent entropy data
        if video_id in entropy_data:
            age = time.time() - entropy_data[video_id].get('timestamp', 0)
            if age < 300:  # Skip if entropy was computed in last 5 minutes
                logger.debug(f"Using existing entropy data for {video_id}, data is {age:.0f}s old")
                # Copy entropy data to metrics_data for backward compatibility
                metrics_data[video_id]['spatial_entropy'] = entropy_data[video_id].get('spatial_entropy', 0.0)
                metrics_data[video_id]['temporal_entropy'] = entropy_data[video_id].get('temporal_entropy', 0.0)
            else:
                # Check if entropy computation is already in progress
                if video_id in active_entropy_computation:
                    logger.debug(f"Entropy computation already in progress for {video_id}")
                else:
                    # Start background entropy computation for individual video requests too
                    logger.info(f"Starting background entropy computation for {video_id}")
                    active_entropy_computation.add(video_id)
                    import threading
                    thread = threading.Thread(
                        target=compute_and_store_entropy,
                        args=(video_id, api_data.get('title'), max_height),
                        daemon=True
                    )
                    thread.start()
        else:
            # Check if entropy computation is already in progress
            if video_id in active_entropy_computation:
                logger.debug(f"Entropy computation already in progress for {video_id}")
            else:
                # Start background entropy computation for individual video requests too
                logger.info(f"Starting background entropy computation for {video_id}")
                active_entropy_computation.add(video_id)
                import threading
                thread = threading.Thread(
                    target=compute_and_store_entropy,
                    args=(video_id, api_data.get('title'), max_height),
                    daemon=True
                )
                thread.start()

        # If match is provided, also do object detection
        if match:
            logger.debug(f"Object detection requested for video_id={video_id}, match={match}")
            # Object counting mode - check if we need to start background detection
            key = f"{video_id}_{match}"
            if key in object_data:
                stored_data = object_data[key]
                # Check if existing data is recent enough
                age = time.time() - stored_data.get('timestamp', 0)
                if age < 300:  # Use existing data if recent
                    logger.debug(f"Using existing object data for {video_id}, object_type: {match}, data is {age:.0f}s old")
                    return
                else:
                    logger.debug(f"Existing object data is stale (age: {age:.0f}s), will recompute")

            # Check if detection is already in progress (with thread-safe add)
            if key in active_object_detection:
                logger.debug(f"Object detection already in progress for {video_id}, object_type: {match}")
                return

            # Add to active set BEFORE starting thread to prevent race conditions
            active_object_detection.add(key)
            logger.info(f"Starting object detection for {video_id}, object_type: {match}")
            import threading
            thread = threading.Thread(
                target=compute_and_store_objects,
                args=(video_id, match, max_height),
                daemon=True,
                name=f"ObjectDetection-{video_id}-{match}"
            )
            thread.start()
        else:
            logger.debug(f"No 'match' parameter provided, skipping object detection for {video_id}")
    else:
        logger.info(f"Skipped image fetching for {video_id}, updated API data only")


def get_prometheus_metrics():
    """Generate Prometheus metrics output."""
    output = generate_latest(registry).decode('utf-8')
    formatted_output = format_integer_metrics(output)
    return formatted_output


# Register the custom collector
registry.register(TimestampedMetricsCollector())
