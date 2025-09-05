"""Quota tracking module for YouTube Data API usage."""

import logging
import time

from config import QUOTA_COSTS

logger = logging.getLogger(__name__)

# Global quota tracking variables
api_quota_used = {}  # Key: endpoint, Value: units used today
api_quota_total = {}  # Key: endpoint, Value: total units used
quota_last_reset = None


def add_quota_units(endpoint, units=None):
    """Add quota units used for an endpoint."""
    global api_quota_used, api_quota_total

    # Use predefined cost if units not specified
    if units is None:
        units = QUOTA_COSTS.get(endpoint, 1)

    if endpoint not in api_quota_used:
        api_quota_used[endpoint] = 0
    old_today = api_quota_used[endpoint]
    api_quota_used[endpoint] += units

    if endpoint not in api_quota_total:
        api_quota_total[endpoint] = 0
    old_total = api_quota_total[endpoint]
    api_quota_total[endpoint] += units

    logger.info(
        f"Added {units} quota units for {endpoint}, today: {old_today} -> {api_quota_used[endpoint]}, total: {old_total} -> {api_quota_total[endpoint]}"
    )


def reset_quota_usage():
    """Reset quota usage counters (called daily at midnight Pacific time)."""
    global api_quota_used, quota_last_reset
    logger.info(
        f"Resetting quota usage. Current api_quota_used: {api_quota_used}"
    )
    api_quota_used = {}
    quota_last_reset = time.time()
    logger.info("Reset YouTube API quota usage counters")


def check_quota_reset():
    """Check if quota should be reset (midnight Pacific time)."""
    global quota_last_reset
    if quota_last_reset is None:
        logger.info("quota_last_reset is None, resetting quota usage")
        reset_quota_usage()
        return

    # Pacific timezone
    pacific_tz = time.timezone - (7 * 3600)  # UTC-7 for Pacific
    now = time.time()
    pacific_time = time.gmtime(now - pacific_tz)

    logger.debug(
        f"Checking quota reset: now={now}, pacific_time={pacific_time.tm_hour}:{pacific_time.tm_min}, last_reset={quota_last_reset}"
    )

    # Reset at midnight Pacific time
    if pacific_time.tm_hour == 0 and pacific_time.tm_min == 0:
        # Check if we haven't reset today yet
        last_reset_pacific = time.gmtime(quota_last_reset - pacific_tz)
        logger.debug(
            f"Reset condition met: last_reset_day={last_reset_pacific.tm_mday}, current_day={pacific_time.tm_mday}"
        )
        if last_reset_pacific.tm_mday != pacific_time.tm_mday:
            logger.info(
                f"Resetting quota: last reset day {last_reset_pacific.tm_mday}, current day {pacific_time.tm_mday}"
            )
            reset_quota_usage()


def get_quota_usage():
    """Get current quota usage statistics."""
    return {
        "today": api_quota_used.copy(),
        "total": api_quota_total.copy(),
        "last_reset": quota_last_reset,
    }


def get_total_quota_today():
    """Get total quota units used today across all endpoints."""
    return sum(api_quota_used.values())


def get_total_quota_ever():
    """Get total quota units used ever across all endpoints."""
    return sum(api_quota_total.values())
