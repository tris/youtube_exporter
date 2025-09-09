import logging

from config import QUOTA_COSTS

logger = logging.getLogger(__name__)

# Global quota tracking variables
api_quota_total = {}  # Key: (key_index, endpoint), Value: total units used


def add_quota_units(endpoint, key_index, units=None):
    """Add quota units used for an endpoint."""
    global api_quota_total

    if units is None:
        units = QUOTA_COSTS.get(endpoint, 1)

    key = (key_index, endpoint)

    if key not in api_quota_total:
        api_quota_total[key] = 0
    old_total = api_quota_total[key]
    api_quota_total[key] += units

    logger.info(
        f"Added {units} quota units for key {key_index}, endpoint {endpoint}, total: {old_total} -> {api_quota_total[key]}"
    )


def get_total_quota_ever():
    """Get total quota units used ever across all endpoints."""
    return sum(api_quota_total.values())
