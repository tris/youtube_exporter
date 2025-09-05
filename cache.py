"""Cache management module for live stream data."""

import time
import logging

logger = logging.getLogger(__name__)


class ChannelLiveCache:
    """Cache for a single channel's live stream data."""
    
    def __init__(self):
        self.initialized = False
        self.cached_live_ids = set()  # Set of video IDs
        self.last_updated = None


class LiveStreamCache:
    """Global cache manager for live stream data."""
    
    def __init__(self):
        self.cache = {}  # channel_id -> ChannelLiveCache

    def get_channel_cache(self, channel_id):
        """Get or create cache for a specific channel."""
        if channel_id not in self.cache:
            self.cache[channel_id] = ChannelLiveCache()
        return self.cache[channel_id]

    def update_cache(self, channel_id, live_ids):
        """Update cache with new live stream IDs for a channel."""
        channel_cache = self.get_channel_cache(channel_id)
        channel_cache.cached_live_ids = set(live_ids) if live_ids else set()
        channel_cache.last_updated = time.time()
        channel_cache.initialized = True
        logger.debug(f"Updated cache for channel {channel_id} with {len(channel_cache.cached_live_ids)} live streams")

    def get_cached_live_ids(self, channel_id):
        """Get cached live stream IDs for a channel."""
        channel_cache = self.get_channel_cache(channel_id)
        return channel_cache.cached_live_ids.copy() if channel_cache.initialized else set()

    def is_cache_initialized(self, channel_id):
        """Check if cache is initialized for a channel."""
        channel_cache = self.get_channel_cache(channel_id)
        return channel_cache.initialized

    def get_cache_age(self, channel_id):
        """Get age of cache in seconds for a channel."""
        channel_cache = self.get_channel_cache(channel_id)
        if channel_cache.last_updated is None:
            return None
        return time.time() - channel_cache.last_updated

    def clear_cache(self, channel_id=None):
        """Clear cache for a specific channel or all channels."""
        if channel_id:
            if channel_id in self.cache:
                del self.cache[channel_id]
                logger.info(f"Cleared cache for channel {channel_id}")
        else:
            self.cache.clear()
            logger.info("Cleared all channel caches")

    def get_cache_stats(self):
        """Get statistics about the cache."""
        stats = {
            'total_channels': len(self.cache),
            'initialized_channels': 0,
            'total_cached_streams': 0,
            'channels': {}
        }
        
        for channel_id, channel_cache in self.cache.items():
            if channel_cache.initialized:
                stats['initialized_channels'] += 1
                stats['total_cached_streams'] += len(channel_cache.cached_live_ids)
            
            stats['channels'][channel_id] = {
                'initialized': channel_cache.initialized,
                'live_streams': len(channel_cache.cached_live_ids),
                'last_updated': channel_cache.last_updated,
                'age_seconds': self.get_cache_age(channel_id)
            }
        
        return stats


# Global cache instance
live_stream_cache = LiveStreamCache()


def get_live_stream_cache():
    """Get the global live stream cache instance."""
    return live_stream_cache