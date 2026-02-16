"""
Shared utilities module.

Contains:
- logger: Non-blocking logging configuration
- usage_tracker: Usage/cost tracking for calls
"""

from utils.logger import (
    configure_non_blocking_logging,
    get_log_listener,
    stop_logging,
    is_logging_configured,
    DEFAULT_LOG_FORMAT,
    DEFAULT_DATE_FORMAT,
)

from utils.usage_tracker import (
    UsageRecord,
    ComponentConfig,
    UsageCollector,
    PricingRate,
    is_component_tracking_enabled,
    attach_usage_collector,
    calculate_call_cost,
)

__all__ = [
    # Logger
    "configure_non_blocking_logging",
    "get_log_listener",
    "stop_logging",
    "is_logging_configured",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    # Usage Tracker
    "UsageRecord",
    "ComponentConfig",
    "UsageCollector",
    "PricingRate",
    "is_component_tracking_enabled",
    "attach_usage_collector",
    "calculate_call_cost",
]

