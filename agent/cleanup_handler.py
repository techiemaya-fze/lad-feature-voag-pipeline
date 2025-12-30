"""
Cleanup Handler Module.

Handles post-call cleanup operations after a voice agent call ends.
Extracted from entry.py for modular architecture.

Phase 16: Updated for lad_dev schema:
- Added tenant_id to CleanupContext
- Column names: duration_seconds (not call_duration), transcripts (not transcriptions)

Responsibilities:
1. Stop call recording and get recording URL
2. Save transcription to database
3. Calculate call cost (component-based or duration-based)
4. Update call status (ended, failed, etc.)
5. Trigger post-call analysis
6. Stop background audio players
7. Release call semaphore
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# =============================================================================
# CLEANUP CONTEXT
# =============================================================================

class CleanupContext:
    """
    Context object holding all resources needed for cleanup.
    
    Passed to cleanup functions to avoid global state.
    """
    
    def __init__(
        self,
        call_recorder: Any = None,
        call_log_id: str | None = None,
        tenant_id: str | None = None,  # Phase 16: Multi-tenancy
        call_storage: Any = None,
        silence_monitor: Any = None,
        latency_mask_player: Any = None,
        ambience_player: Any = None,
        background_audio_task: asyncio.Task | None = None,
        call_semaphore: asyncio.Semaphore | None = None,
        acquired_call_slot: bool = False,
        usage_collector: Any = None,
        job_id: str | None = None,
    ):
        self.call_recorder = call_recorder
        self.call_log_id = call_log_id
        self.tenant_id = tenant_id
        self.call_storage = call_storage
        self.silence_monitor = silence_monitor
        self.latency_mask_player = latency_mask_player
        self.ambience_player = ambience_player
        self.background_audio_task = background_audio_task
        self.call_semaphore = call_semaphore
        self.acquired_call_slot = acquired_call_slot
        self.usage_collector = usage_collector
        self.job_id = job_id


# =============================================================================
# RECORDING CLEANUP
# =============================================================================

async def stop_and_save_recording(ctx: CleanupContext) -> tuple[str | None, dict | None, float | None]:
    """
    Stop recording and retrieve artifacts.
    
    Args:
        ctx: Cleanup context with call_recorder
        
    Returns:
        Tuple of (recording_url, transcription_data, trimmed_duration)
    """
    call_recorder = ctx.call_recorder
    call_log_id = ctx.call_log_id
    
    recording_url: str | None = None
    transcription_data: dict | None = None
    trimmed_duration: float | None = None
    
    if not call_recorder or not call_log_id:
        return recording_url, transcription_data, trimmed_duration
    
    # Stop recording
    try:
        await call_recorder.stop_recording()
    except Exception as exc:
        logger.error("Failed to stop recording for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
    
    # Get trimmed duration
    try:
        trimmed_duration = call_recorder.get_trimmed_duration_seconds()
        if trimmed_duration is not None:
            logger.info("Call audio duration after trimming: %.2fs for call_log_id=%s", trimmed_duration, call_log_id)
    except Exception as exc:
        logger.error("Failed to get trimmed duration for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
    
    # Get recording URL
    try:
        recording_url = call_recorder.get_recording_url()
    except Exception as exc:
        logger.error("Failed to resolve recording URL for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
    
    # Get transcription
    try:
        transcription_data = call_recorder.get_transcription_dict()
    except Exception as exc:
        logger.error("Failed to serialize transcription for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
        transcription_data = None
    
    return recording_url, transcription_data, trimmed_duration


async def save_recording_to_db(
    ctx: CleanupContext,
    recording_url: str | None,
    transcription_data: dict | None,
    trimmed_duration: float | None,
) -> None:
    """
    Persist recording and transcription to database.
    
    Args:
        ctx: Cleanup context with call_storage
        recording_url: GCS URL of recording
        transcription_data: Transcription dict
        trimmed_duration: Duration in seconds after trimming
    """
    call_storage = ctx.call_storage
    call_log_id = ctx.call_log_id
    
    if not call_storage or not call_log_id:
        return
    
    if recording_url:
        try:
            await call_storage.update_call_recording(
                call_log_id=call_log_id,
                recording_url=recording_url,
                transcripts=transcription_data,
            )
            
            if trimmed_duration is not None:
                await call_storage.update_call_metadata(
                    call_log_id,
                    duration_seconds=trimmed_duration,  # Phase 16: new column name
                )
                logger.info("Saved duration_seconds=%.2fs for call_log_id=%s", trimmed_duration, call_log_id)
            
            logger.info("Saved recording and transcription for call_log_id=%s", call_log_id)
        except Exception as exc:
            logger.error("Failed to persist recording for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
    elif transcription_data:
        try:
            await call_storage.update_call_metadata(
                call_log_id,
                transcripts=transcription_data,  # Phase 16: new column name
            )
            logger.info("Saved transcripts without recording for call_log_id=%s", call_log_id)
        except Exception as exc:
            logger.error("Failed to persist transcripts for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
    else:
        logger.warning("No recording or transcription captured for call_log_id=%s", call_log_id)


# =============================================================================
# COST CALCULATION
# =============================================================================

async def calculate_and_save_cost(
    ctx: CleanupContext,
    duration_seconds: float | None,
) -> tuple[float | None, list | None]:
    """
    Calculate call cost using component-based or duration-based method.
    
    Args:
        ctx: Cleanup context with usage_collector
        duration_seconds: Call duration in seconds
        
    Returns:
        Tuple of (call_cost, cost_breakdown)
    """
    from utils.usage_tracker import calculate_call_cost
    
    call_cost: float | None = None
    cost_breakdown: list | None = None
    
    usage_collector = ctx.usage_collector
    call_storage = ctx.call_storage
    call_log_id = ctx.call_log_id
    
    duration_hours = duration_seconds / 3600.0 if duration_seconds else None
    
    # Mode 1: Component-based cost tracking
    if usage_collector and not usage_collector.is_empty():
        logger.info(f"UsageCollector state: is_empty={usage_collector.is_empty()}, summary={usage_collector.get_summary()}")
        
        if duration_seconds and duration_seconds > 0:
            usage_collector.add_telephony_seconds(duration_seconds, provider="vonage")
            usage_collector.add_vm_infrastructure_seconds(duration_seconds, provider="digitalocean")
        
        try:
            usage_summary = usage_collector.get_summary()
            if usage_summary and call_storage:
                pricing_rates = await call_storage.get_pricing_rates()
                await call_storage.save_call_usage(
                    call_log_id=str(call_log_id),
                    usage_records=usage_summary,
                    pricing_rates=pricing_rates,
                )
                
                total_cost, cost_breakdown = await calculate_call_cost(
                    usage_records=usage_summary,
                    pricing_rates=pricing_rates,
                )
                
                logger.info("=" * 60)
                logger.info("COMPONENT-BASED COST BREAKDOWN")
                logger.info("=" * 60)
                
                for item in cost_breakdown:
                    component = item["component"]
                    provider = item["provider"]
                    model = item["model"]
                    amount = item["amount"]
                    unit = item["unit"]
                    cost = item["cost"]
                    rate = item["rate"]
                    
                    if rate is not None:
                        logger.info(
                            f"  {component.upper()} ({provider}/{model}): "
                            f"{amount:.4f} {unit} × ${rate:.8f} = ${cost:.6f}"
                        )
                    else:
                        logger.warning(
                            f"  {component.upper()} ({provider}/{model}): "
                            f"{amount:.4f} {unit} - NO PRICING RATE"
                        )
                
                if total_cost > 0:
                    call_cost = float(total_cost)
                    logger.info(f"TOTAL COST: ${call_cost:.6f}")
        except Exception as exc:
            logger.warning(f"Failed to calculate component-based cost: {exc}", exc_info=True)
    
    # Mode 2: Duration-based cost (fallback)
    if call_cost is None and duration_hours is not None:
        cost_per_hour = float(os.getenv("CALL_COST_PER_HOUR", "0.10"))
        call_cost = duration_hours * cost_per_hour
        logger.info(f"Duration-based cost: {duration_seconds:.2f}s = ${call_cost:.4f}")
    
    return call_cost, cost_breakdown


def determine_final_status(existing_status: str | None) -> str:
    """
    Determine final call status based on existing status.
    
    Args:
        existing_status: Current call status
        
    Returns:
        Final status string
    """
    unchanged_terminal_statuses = {
        "failed", "declined", "rejected", "not_reachable",
        "no_answer", "busy", "error", "canceled", "cancelled",
    }
    
    if existing_status in unchanged_terminal_statuses:
        return existing_status
    if existing_status in {"ongoing", "running", "pending", "in_queue", "started"}:
        return "ended"
    return existing_status or "ended"


async def update_call_status(
    ctx: CleanupContext,
    ended_at: datetime,
    call_cost: float | None,
    cost_breakdown: list | None,
) -> None:
    """
    Update call status in database.
    
    Args:
        ctx: Cleanup context
        ended_at: When call ended
        call_cost: Calculated cost
        cost_breakdown: Cost details
    """
    call_storage = ctx.call_storage
    call_log_id = ctx.call_log_id
    
    if not call_storage or not call_log_id:
        return
    
    try:
        call_details = await call_storage.get_call_by_id(call_log_id)
    except Exception as exc:
        logger.error("Failed to load call details for call_log_id=%s: %s", call_log_id, exc, exc_info=True)
        call_details = None
    
    existing_status = None
    if call_details:
        raw_status = call_details.get('status')
        if isinstance(raw_status, str) and raw_status.strip():
            existing_status = raw_status.strip().lower()
    
    final_status = determine_final_status(existing_status)
    
    try:
        await call_storage.update_call_status(
            call_log_id=call_log_id,
            status=final_status,
            ended_at=ended_at,
            cost=call_cost,
            cost_breakdown=cost_breakdown,
        )
    except Exception as exc:
        logger.error("Failed to update call status for call_log_id=%s: %s", call_log_id, exc, exc_info=True)


# =============================================================================
# POST-CALL ANALYSIS
# =============================================================================

async def trigger_post_call_analysis(
    ctx: CleanupContext,
    transcription_data: dict | None,
    duration_seconds: float | None,
    call_details: dict | None,
) -> None:
    """
    Run post-call analysis on transcription.
    
    Args:
        ctx: Cleanup context
        transcription_data: Transcription dict
        duration_seconds: Call duration
        call_details: Full call record
    """
    from analysis import run_post_call_analysis
    
    call_log_id = ctx.call_log_id
    call_storage = ctx.call_storage
    tenant_id = ctx.tenant_id  # Multi-tenancy support
    
    if not transcription_data:
        logger.warning("Skipping post-call analysis; no transcript for call_log_id=%s", call_log_id)
        return
    
    try:
        await run_post_call_analysis(
            call_log_id=str(call_log_id),
            transcription_json=transcription_data,
            duration_seconds=duration_seconds,
            call_details=call_details or {},
            db_config=getattr(call_storage, "db_config", None),
            tenant_id=str(tenant_id) if tenant_id else None,  # Pass tenant_id
        )
    except Exception as exc:
        logger.error("Post-call analysis failed for call_log_id=%s: %s", call_log_id, exc, exc_info=True)


async def trigger_lead_bookings_extraction(
    ctx: CleanupContext,
    transcription_data: dict | None,
    call_details: dict | None,
) -> None:
    """
    Extract and save lead bookings from call transcription.
    
    Args:
        ctx: Cleanup context
        transcription_data: Transcription dict
        call_details: Full call record
    """
    try:
        from analysis.lead_bookings_extractor import LeadBookingsExtractor
    except ImportError:
        logger.warning("Lead bookings extractor not available, skipping extraction")
        return
    
    call_log_id = ctx.call_log_id
    
    if not transcription_data:
        logger.warning("Skipping lead bookings extraction; no transcript for call_log_id=%s", call_log_id)
        return
    
    if not call_log_id:
        logger.warning("Skipping lead bookings extraction; no call_log_id")
        return
    
    try:
        extractor = LeadBookingsExtractor()
        try:
            booking_data = await extractor.process_call_log(str(call_log_id))
            if booking_data:
                save_results = await extractor.save_booking(booking_data)
                if save_results.get("db"):
                    logger.info("Lead booking extracted and saved for call_log_id=%s", call_log_id)
                elif save_results.get("errors"):
                    logger.warning("Lead booking extraction completed with errors for call_log_id=%s: %s",
                                 call_log_id, save_results["errors"])
            else:
                logger.debug("No booking data extracted for call_log_id=%s", call_log_id)
        finally:
            await extractor.close()
    except Exception as exc:
        logger.error("Lead bookings extraction failed for call_log_id=%s: %s", call_log_id, exc, exc_info=True)


# =============================================================================
# AUDIO CLEANUP
# =============================================================================

async def stop_background_audio(ctx: CleanupContext) -> None:
    """Stop all background audio players and tasks."""
    ambient_logger = logging.getLogger("ambient.audio")
    
    # Cancel background audio task
    if ctx.background_audio_task and not ctx.background_audio_task.done():
        ctx.background_audio_task.cancel()
        try:
            await ctx.background_audio_task
        except asyncio.CancelledError:
            pass
    
    # Stop audio players
    for player in (ctx.latency_mask_player, ctx.ambience_player):
        if player:
            stop_callable = getattr(player, "stop", None)
            if callable(stop_callable):
                try:
                    maybe_coro = stop_callable()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                except Exception:
                    ambient_logger.warning("Failed to stop background audio player", exc_info=True)


# =============================================================================
# MAIN CLEANUP FUNCTION
# =============================================================================

async def cleanup_and_save(ctx: CleanupContext) -> None:
    """
    Main cleanup handler called when call ends.
    
    Performs all cleanup operations in order:
    1. Disable silence monitoring
    2. Stop recording and get artifacts
    3. Save recording to database
    4. Calculate and save cost
    5. Update call status
    6. Run post-call analysis
    7. Extract and save lead bookings
    8. Stop background audio
    9. Release semaphore
    
    Args:
        ctx: Cleanup context with all resources
    """
    # 1. Disable silence monitoring
    if ctx.silence_monitor:
        ctx.silence_monitor.disable()
    
    # 2. Stop recording and get artifacts
    recording_url, transcription_data, trimmed_duration = await stop_and_save_recording(ctx)
    
    # 3. Save recording to database
    await save_recording_to_db(ctx, recording_url, transcription_data, trimmed_duration)
    
    # Get call details for cost calculation
    call_details = None
    duration_seconds = None
    ended_at = datetime.now(timezone.utc)
    
    if ctx.call_storage and ctx.call_log_id:
        try:
            call_details = await ctx.call_storage.get_call_by_id(ctx.call_log_id)
        except Exception as exc:
            logger.error("Failed to load call details: %s", exc, exc_info=True)
        
        if call_details and call_details.get('started_at'):
            started_at = call_details['started_at']
            duration_seconds = (ended_at - started_at).total_seconds()
    
    # 4. Calculate and save cost
    call_cost, cost_breakdown = await calculate_and_save_cost(ctx, duration_seconds)
    
    # 5. Update call status
    await update_call_status(ctx, ended_at, call_cost, cost_breakdown)
    
    # Log transcription
    if ctx.call_recorder:
        try:
            logger.info("Call transcription:\n%s", ctx.call_recorder.get_transcription_text())
        except Exception:
            pass
    
    # 6. Run post-call analysis
    await trigger_post_call_analysis(ctx, transcription_data, duration_seconds, call_details)
    
    # 7. Extract and save lead bookings
    await trigger_lead_bookings_extraction(ctx, transcription_data, call_details)
    
    # 8. Stop background audio
    await stop_background_audio(ctx)
    
    # 9. Release semaphore
    if ctx.acquired_call_slot and ctx.call_semaphore:
        ctx.call_semaphore.release()
        logger.info("Released semaphore for job %s", ctx.job_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CleanupContext",
    "cleanup_and_save",
    "stop_and_save_recording",
    "save_recording_to_db",
    "calculate_and_save_cost",
    "determine_final_status",
    "update_call_status",
    "trigger_post_call_analysis",
    "trigger_lead_bookings_extraction",
    "stop_background_audio",
]
