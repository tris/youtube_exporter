"""YouTube API client module for fetching video and channel data."""

import logging

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from api_errors import api_errors
from cache import get_live_stream_cache
from config import (
    BATCH_SIZE,
    CHANNEL_VIDEO_THRESHOLD,
    MAX_RESULTS,
    YOUTUBE_API_KEY,
)
from quota import add_quota_units, check_quota_reset

logger = logging.getLogger(__name__)

# Global YouTube service instance
youtube_service = None


def get_youtube_service():
    """Get or create YouTube API service."""
    global youtube_service
    if youtube_service is None and YOUTUBE_API_KEY:
        try:
            youtube_service = build(
                "youtube", "v3", developerKey=YOUTUBE_API_KEY
            )
            logger.info("YouTube API service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API service: {e}")
    return youtube_service


def fetch_video_details(video_id, retries=2):
    """Fetch video details from YouTube API with retry logic."""
    logger.debug(f"Calling check_quota_reset for video {video_id}")
    check_quota_reset()  # Check for daily reset

    service = get_youtube_service()
    if not service:
        logger.error("YouTube API service not available")
        return None

    for attempt in range(retries + 1):
        try:
            request = service.videos().list(
                part="snippet,statistics,liveStreamingDetails", id=video_id
            )
            response = request.execute()

            # Track quota usage
            add_quota_units("videos.list")

            if not response.get("items"):
                return None

            video_data = response["items"][0]
            snippet = video_data.get("snippet", {})
            title = snippet.get("title", "")
            logger.debug(
                f"Fetched video {video_id}: title='{title}' (length: {len(title)})"
            )
            return video_data
        except HttpError as e:
            logger.error(
                f"YouTube API error for video {video_id} (attempt {attempt+1}/{retries+1}): {e}"
            )
            code = e.resp.status
            endpoint = "videos.list"
            key = (code, endpoint)
            api_errors[key] = api_errors.get(key, 0) + 1
            if attempt < retries and code in [
                500,
                502,
                503,
                504,
            ]:  # Retry on server errors
                import time

                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error fetching video {video_id} (attempt {attempt+1}/{retries+1}): {e}"
            )
            if attempt < retries:
                import time

                time.sleep(1 * (attempt + 1))
                continue
            return None


def fetch_channel_details(channel_id):
    """Fetch channel details from YouTube API."""
    service = get_youtube_service()
    if not service:
        return None

    try:
        request = service.channels().list(
            part="snippet,statistics,contentDetails", id=channel_id
        )
        response = request.execute()

        # Track quota usage
        add_quota_units("channels.list")

        if not response.get("items"):
            return None

        return response["items"][0]
    except HttpError as e:
        logger.error(f"YouTube API error for channel {channel_id}: {e}")
        code = e.resp.status
        endpoint = "channels.list"
        key = (code, endpoint)
        api_errors[key] = api_errors.get(key, 0) + 1
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching channel {channel_id}: {e}")
        return None


def refresh_cached_videos(cached_ids):
    """Check if cached video IDs are still live."""
    if not cached_ids:
        return [], set()

    service = get_youtube_service()
    if not service:
        return [], set()

    still_live_videos = []
    still_live_ids = set()

    # Fetch in batches of 50
    id_list = list(cached_ids)
    for i in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[i : i + BATCH_SIZE]
        try:
            request = service.videos().list(
                part="snippet,statistics,liveStreamingDetails",
                id=",".join(batch),
            )
            response = request.execute()

            # Track quota usage
            add_quota_units("videos.list")

            # Check which are still live
            for video in response.get("items", []):
                snippet = video.get("snippet", {})
                if snippet.get("liveBroadcastContent") == "live":
                    still_live_videos.append(video)
                    still_live_ids.add(video["id"])

        except HttpError as e:
            logger.error(
                f"YouTube API error refreshing cached videos batch {i//BATCH_SIZE}: {e}"
            )
            code = e.resp.status
            endpoint = "videos.list"
            key = (code, endpoint)
            api_errors[key] = api_errors.get(key, 0) + 1
        except Exception as e:
            logger.error(
                f"Unexpected error refreshing cached videos batch {i//BATCH_SIZE}: {e}"
            )

    return still_live_videos, still_live_ids


def fetch_channel_live_streams_comprehensive(channel_id):
    """Fetch all live streams for a channel using Search API (expensive)."""
    logger.info(
        f"Performing comprehensive live stream search for channel {channel_id} (100 units)"
    )

    service = get_youtube_service()
    if not service:
        return []

    try:
        search_request = service.search().list(
            part="id",
            channelId=channel_id,
            eventType="live",
            type="video",
            maxResults=MAX_RESULTS,
        )
        search_response = search_request.execute()

        # Track quota usage
        add_quota_units("search.list")

        if not search_response.get("items"):
            return []

        # Extract video IDs
        video_ids = []
        for item in search_response["items"]:
            if item.get("id", {}).get("videoId"):
                video_ids.append(item["id"]["videoId"])

        if not video_ids:
            return []

        # Fetch full video details
        videos_request = service.videos().list(
            part="snippet,statistics,liveStreamingDetails",
            id=",".join(video_ids),
        )
        videos_response = videos_request.execute()

        # Track quota usage
        add_quota_units("videos.list")

        return videos_response.get("items", [])

    except HttpError as e:
        logger.error(
            f"YouTube API error in comprehensive search for channel {channel_id}: {e}"
        )
        code = e.resp.status
        endpoint = "search.list"  # Assuming it's the search call that failed
        key = (code, endpoint)
        api_errors[key] = api_errors.get(key, 0) + 1
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error in comprehensive search for channel {channel_id}: {e}"
        )
        return []


def fetch_recent_channel_videos(channel):
    """Fetch recent videos from channel's uploads playlist."""
    content_details = channel.get("contentDetails", {})
    related_playlists = content_details.get("relatedPlaylists", {})
    uploads_playlist_id = related_playlists.get("uploads")

    if not uploads_playlist_id:
        return []

    service = get_youtube_service()
    if not service:
        return []

    try:
        # Fetch recent videos from uploads playlist
        playlist_request = service.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=MAX_RESULTS,  # Check up to 50 recent videos
        )
        playlist_response = playlist_request.execute()

        # Track quota usage
        add_quota_units("playlistItems.list")

        video_ids = []
        for item in playlist_response.get("items", []):
            content_details = item.get("contentDetails", {})
            if content_details.get("videoId"):
                video_ids.append(content_details["videoId"])

        if not video_ids:
            return []

        # Fetch video details in batches
        live_videos = []
        for i in range(0, len(video_ids), BATCH_SIZE):
            batch = video_ids[i : i + BATCH_SIZE]
            try:
                videos_request = service.videos().list(
                    part="snippet,statistics,liveStreamingDetails",
                    id=",".join(batch),
                )
                videos_response = videos_request.execute()

                # Track quota usage
                add_quota_units("videos.list")

                # Filter for live videos
                for video in videos_response.get("items", []):
                    snippet = video.get("snippet", {})
                    if snippet.get("liveBroadcastContent") == "live":
                        live_videos.append(video)
            except HttpError as e:
                logger.error(
                    f"YouTube API error fetching video batch for channel {channel.get('id')}: {e}"
                )
                code = e.resp.status
                endpoint = "videos.list"
                key = (code, endpoint)
                api_errors[key] = api_errors.get(key, 0) + 1

        return live_videos

    except HttpError as e:
        logger.error(
            f"YouTube API error fetching recent videos for channel {channel.get('id')}: {e}"
        )
        code = e.resp.status
        endpoint = "playlistItems.list"
        key = (code, endpoint)
        api_errors[key] = api_errors.get(key, 0) + 1
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error fetching recent videos for channel {channel.get('id')}: {e}"
        )
        return []


def fetch_channel_live_streams(channel, disable_live=False):
    """Fetch live streams for a channel with caching."""
    if disable_live:
        logger.info(
            f"Live stream fetching disabled for channel {channel.get('id')}"
        )
        return []

    channel_id = channel.get("id")
    if not channel_id:
        return []

    cache = get_live_stream_cache()
    channel_cache = cache.get_channel_cache(channel_id)

    # Determine strategy based on video count
    stats = channel.get("statistics", {})
    video_count = int(stats.get("videoCount", 0))
    use_full_pagination = video_count <= CHANNEL_VIDEO_THRESHOLD

    # If this is the first time, decide between comprehensive search or playlist method
    if not channel_cache.initialized:
        if use_full_pagination:
            logger.info(
                f"Channel {channel_id} has {video_count} videos (≤{CHANNEL_VIDEO_THRESHOLD}), using playlist method"
            )
            recent_videos = fetch_recent_channel_videos(channel)
            if recent_videos:
                new_cache = set(video["id"] for video in recent_videos)
                cache.update_cache(channel_id, new_cache)
                logger.info(
                    f"Cached {len(new_cache)} live streams for channel {channel_id}"
                )
                return recent_videos
        else:
            logger.info(
                f"Channel {channel_id} has {video_count} videos (>{CHANNEL_VIDEO_THRESHOLD}), using Search API"
            )
            comprehensive_videos = fetch_channel_live_streams_comprehensive(
                channel_id
            )
            if comprehensive_videos:
                new_cache = set(video["id"] for video in comprehensive_videos)
                cache.update_cache(channel_id, new_cache)
                logger.info(
                    f"Cached {len(new_cache)} live streams from comprehensive search for channel {channel_id}"
                )
                return comprehensive_videos

    # Always fetch recent videos using the efficient playlist method
    recent_live_videos = fetch_recent_channel_videos(channel)

    # Create a map to track all live videos (deduplication)
    all_live_videos = {video["id"]: video for video in recent_live_videos}

    # If we have cached IDs, refresh their status
    if channel_cache.initialized and channel_cache.cached_live_ids:
        cached_videos, still_live_ids = refresh_cached_videos(
            channel_cache.cached_live_ids
        )

        # Add still-live cached videos
        for video in cached_videos:
            all_live_videos[video["id"]] = video

        # Update cache with only still-live IDs
        cache.update_cache(channel_id, still_live_ids)

    result = list(all_live_videos.values())
    logger.info(
        f"Total live streams for channel {channel_id}: {len(result)} (recent: {len(recent_live_videos)}, cached: {len(result) - len(recent_live_videos)})"
    )

    return result
