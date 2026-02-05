"""
Background Analysis Runner - Dedicated Thread for Post-Call Analysis

This module provides a thread-safe singleton that runs analysis tasks in a
dedicated daemon thread with its own event loop. This decouples analysis
from LiveKit's worker lifecycle, ensuring:

1. Worker is marked "available" immediately after cleanup (no waiting for analysis)
2. Analysis survives LiveKit executor shutdown
3. asyncio.to_thread() works reliably (uses our own executor, not LiveKit's)
4. Multiple concurrent analyses don't block each other

Architecture:
    ┌─────────────────────────┐            ┌────────────────────────────┐
    │  LiveKit Event Loop     │   submit   │  Analysis Thread (Daemon)  │
    │  (managed by LiveKit)   │───────────►│  - Own event loop          │
    │                         │            │  - Own thread executor     │
    │  cleanup_and_save()     │            │  - Survives job shutdown   │
    │  returns IMMEDIATELY    │            │  - Processes tasks async   │
    └─────────────────────────┘            └────────────────────────────┘

Usage:
    from analysis.background_runner import submit_analysis_task
    
    # Submit analysis coroutine to run in background thread
    submit_analysis_task(
        my_async_analysis_function(args),
        name="analysis_call123"
    )
"""

import asyncio
import logging
import threading
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


class AnalysisBackgroundRunner:
    """
    Singleton that manages a dedicated thread for running analysis tasks.
    
    This thread has its own event loop and executor, completely independent
    of LiveKit's event loop. Tasks submitted here will continue running
    even after the LiveKit job completes and the worker is freed.
    """
    _instance: "AnalysisBackgroundRunner | None" = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "AnalysisBackgroundRunner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_thread()
    
    def _start_thread(self) -> None:
        """Start the background thread with its own event loop."""
        def run_loop() -> None:
            # Create a new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            logger.info("Analysis background thread started with dedicated event loop")
            self._started.set()
            
            # Run forever - this thread stays alive for the entire process lifetime
            try:
                self._loop.run_forever()
            finally:
                # Cleanup when loop stops (process shutdown)
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
                logger.info("Analysis background thread stopped")
        
        self._thread = threading.Thread(
            target=run_loop,
            daemon=True,  # Daemon thread - doesn't prevent process exit
            name="AnalysisBackgroundRunner"
        )
        self._thread.start()
        
        # Wait for the thread to initialize its event loop
        if not self._started.wait(timeout=5.0):
            logger.error("Analysis background thread failed to start within 5 seconds")
    
    def submit(self, coro: Coroutine[Any, Any, Any], name: str = "analysis_task") -> None:
        """
        Submit an async coroutine to run in the background thread.
        
        This is thread-safe and can be called from any thread (including
        LiveKit's event loop thread).
        
        Args:
            coro: The coroutine to run
            name: Human-readable name for logging
        """
        if not self._loop or not self._loop.is_running():
            logger.error(
                "Analysis background loop not running, cannot submit task '%s'. "
                "Analysis will be skipped.",
                name
            )
            # Close the coroutine to prevent "coroutine was never awaited" warning
            coro.close()
            return
        
        # Thread-safe submission to the background event loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        # Add a callback to log completion/errors (optional, helps debugging)
        def on_done(fut: asyncio.Future) -> None:
            try:
                fut.result()  # This will raise if the task failed
                logger.debug("Analysis task '%s' completed successfully", name)
            except asyncio.CancelledError:
                logger.warning("Analysis task '%s' was cancelled", name)
            except Exception as exc:
                logger.error("Analysis task '%s' failed: %s", name, exc, exc_info=True)
        
        future.add_done_callback(on_done)
        logger.info("Submitted analysis task '%s' to background thread", name)
    
    @property
    def is_running(self) -> bool:
        """Check if the background thread and loop are running."""
        return (
            self._thread is not None 
            and self._thread.is_alive() 
            and self._loop is not None 
            and self._loop.is_running()
        )


# =============================================================================
# PUBLIC API
# =============================================================================

_runner: AnalysisBackgroundRunner | None = None
_runner_lock = threading.Lock()


def get_analysis_runner() -> AnalysisBackgroundRunner:
    """Get or create the global analysis runner singleton."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = AnalysisBackgroundRunner()
        return _runner


def submit_analysis_task(coro: Coroutine[Any, Any, Any], name: str = "analysis_task") -> None:
    """
    Submit an analysis task to run in the dedicated background thread.
    
    This function is safe to call from any context (sync or async, any thread).
    The task will run in a completely separate event loop from LiveKit's,
    ensuring it doesn't block worker capacity and survives job shutdown.
    
    Args:
        coro: The async coroutine to run (e.g., analysis function)
        name: Human-readable name for logging (e.g., "analysis_<call_id>")
    
    Example:
        submit_analysis_task(
            _run_analysis_background(ctx, data),
            name=f"analysis_{call_log_id}"
        )
    """
    runner = get_analysis_runner()
    runner.submit(coro, name)
