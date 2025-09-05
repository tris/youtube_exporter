"""Configuration module for YouTube Entropy Exporter."""

import os
import re
import sys

# Environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Validate required environment variables
def validate_config():
    """Validate that all required configuration is present."""
    if not YOUTUBE_API_KEY:
        print("ERROR: YOUTUBE_API_KEY environment variable is not set.", file=sys.stderr)
        print("Please set your YouTube Data API v3 key and try again.", file=sys.stderr)
        sys.exit(1)

# Regex patterns for validation
VIDEO_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{11}$')
CHANNEL_ID_PATTERN = re.compile(r'^UC[A-Za-z0-9_-]{22}$')

# Default values
DEFAULT_INTERVAL = 300  # 5 minutes
MIN_INTERVAL = 30  # 30 seconds minimum
DEFAULT_FRAME_SKIP = 30  # Skip 30 frames (~1 second at 30fps)
MAX_RESULTS = 50  # Maximum results for API calls
BATCH_SIZE = 50  # Batch size for API calls

# Flask configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = int(os.getenv('PORT', 9473))

# Video quality settings
MAX_VIDEO_HEIGHT = 720  # Limit resolution for efficiency

# Cache settings
CHANNEL_VIDEO_THRESHOLD = 4950  # Up to 99 pages of 50 videos

# Quota costs (YouTube Data API v3)
QUOTA_COSTS = {
    'videos.list': 1,
    'channels.list': 1,
    'search.list': 100,
    'playlistItems.list': 1
}
