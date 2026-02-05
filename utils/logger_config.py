"""
Non-Blocking Logging Configuration for Async Applications

This module provides a QueueHandler-based logging setup that moves all I/O operations
(writing to console/file) to a separate thread. This prevents the asyncio event loop
from blocking on log writes, which is critical for real-time voice applications.

Benefits:
- Zero blocking: Log calls return in microseconds (just queue.put())
- Thread-safe: QueueHandler is designed for concurrent access
- Graceful shutdown: atexit handler ensures all logs are flushed
- Drop-in replacement: No changes needed to existing logger.info() calls

Usage:
    from utils.logger_config import configure_non_blocking_logging, get_log_listener
    
    # At application startup (before any logging)
    listener = configure_non_blocking_logging()
    
    # At shutdown (optional, atexit handles this automatically)
    listener.stop()
"""

import atexit
import logging
import logging.handlers
import os
import queue
import sys
from typing import Optional

# Module-level reference to the listener for shutdown handling
_log_listener: Optional[logging.handlers.QueueListener] = None

# Default format matching existing codebase style (exported for use in other modules)
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%H:%M:%S"


def _resolve_log_level(value: str | None) -> int:
    """Resolve log level from string or integer value."""
    if value is None:
        return logging.INFO
    
    stripped = value.strip().upper()
    
    # Standard level names
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    if stripped in level_map:
        return level_map[stripped]
    
    # Try numeric value
    try:
        return int(value.strip())
    except ValueError:
        return logging.INFO


def configure_non_blocking_logging(
    level: int | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    queue_size: int = -1,  # -1 = infinite
    silence_noisy_libs: bool = True,
) -> logging.handlers.QueueListener:
    """
    Configure non-blocking logging using QueueHandler pattern.
    
    This function replaces the default synchronous logging with an async-safe
    implementation that uses a queue and background thread for I/O.
    
    Args:
        level: Log level (default: from LOG_LEVEL env var or INFO)
        log_format: Format string for log messages
        date_format: Format string for timestamps
        queue_size: Max queue size (-1 for infinite, or e.g. 10000 to prevent memory leaks)
        silence_noisy_libs: If True, set noisy libraries to WARNING level
    
    Returns:
        QueueListener instance (store reference for graceful shutdown)
    
    Example:
        # At startup
        listener = configure_non_blocking_logging()
        
        # Normal logging works unchanged
        logger = logging.getLogger(__name__)
        logger.info("This won't block the event loop!")
    """
    global _log_listener
    
    # Resolve log level from env if not specified
    if level is None:
        level = _resolve_log_level(os.getenv("LOG_LEVEL"))
    
    # 1. Create a thread-safe queue for log records
    log_queue: queue.Queue = queue.Queue(queue_size)
    
    # 2. Create the actual console handler (runs in background thread)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # 3. Create the QueueListener (background thread that processes the queue)
    listener = logging.handlers.QueueListener(
        log_queue,
        console_handler,
        respect_handler_level=True,
    )
    listener.start()
    
    # 4. Create the QueueHandler (non-blocking, just puts records in queue)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(level)
    
    # 5. Configure the root logger
    root = logging.getLogger()
    root.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(queue_handler)
    
    # 6. Silence noisy libraries that flood logs
    if silence_noisy_libs:
        noisy_loggers = [
            "httpcore",
            "httpx",
            "asyncio",
            "urllib3",
            "websockets",
            "aiohttp",
            "google.auth",
            "google.auth.transport",
            "google_genai",
            "google_genai._api_client",
            "google_genai.models",
            # Analysis modules - keep at INFO to avoid polluting main.py logs
            "analysis.merged_analytics",
            "analysis.gemini_client",
            "analysis.lead_info_extractor",
            "analysis.lead_bookings_extractor",
            "analysis.sentiment_analyzer",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # 7. Register atexit handler for graceful shutdown
    def _shutdown_listener():
        try:
            listener.stop()
        except Exception:
            pass  # Ignore errors during shutdown
    
    atexit.register(_shutdown_listener)
    
    # Store reference for manual shutdown if needed
    _log_listener = listener
    
    return listener


def get_log_listener() -> Optional[logging.handlers.QueueListener]:
    """Get the current log listener for manual shutdown if needed."""
    return _log_listener


def stop_logging() -> None:
    """
    Stop the background logging thread and flush remaining logs.
    
    Call this during application shutdown to ensure all logs are written.
    Note: atexit handler calls this automatically, but explicit call is
    recommended for clean shutdown in async applications.
    """
    global _log_listener
    if _log_listener is not None:
        try:
            _log_listener.stop()
        except Exception:
            pass
        _log_listener = None


def is_logging_configured() -> bool:
    """Check if non-blocking logging has been configured."""
    return _log_listener is not None
