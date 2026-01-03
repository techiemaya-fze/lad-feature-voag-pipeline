"""
Batch Routes Module (V2 API).

Handles batch call operations with kebab-case naming:
- POST /trigger-batch-call: Start batch call job
- GET /batch-status/{batch_id}: Check batch job status (now uses /calls/status)
- POST /batch-cancel/{batch_id}: Cancel batch job (now uses /calls/cancel)

Note: Batch status and cancel are consolidated into calls.py for a unified API.
This file provides the trigger-batch-call endpoint and forwards others.
"""

import asyncio
import csv
import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from api.models import (
    CallJobResponse,
    BatchCallJsonRequest,
    BatchCallJsonEntry,
    JobStatus,
    CallMode,
    CallJob,
    CallAttemptResult,
    BatchStatusResponse,
    BatchEntryStatusModel,
)
from api.services.call_service import (
    get_call_service,
    BatchCallEntry,
    _normalize_llm_provider,
    _clean_optional_text,
    DEFAULT_GLINKS_KB_STORE_IDS,
)
from db.storage import BatchStorage, VoiceStorage
from db.connection_pool import get_db_connection
from db.db_config import get_db_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])

# Lazy initialization
_batch_storage: BatchStorage | None = None
_voice_storage: VoiceStorage | None = None


def _get_batch_storage() -> BatchStorage:
    global _batch_storage
    if _batch_storage is None:
        _batch_storage = BatchStorage()
    return _batch_storage


def _get_voice_storage() -> VoiceStorage:
    global _voice_storage
    if _voice_storage is None:
        _voice_storage = VoiceStorage()
    return _voice_storage


async def _resolve_tenant_id_from_user(user_id: str | None) -> str | None:
    """
    Resolve tenant_id from user's primary_tenant_id in lad_dev.users table.
    
    Args:
        user_id: User UUID (initiated_by)
        
    Returns:
        Tenant UUID or None if not found
    """
    if not user_id:
        return None
    
    try:
        # Use standard connection pattern (same as old code)
        with get_db_connection(get_db_config()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT primary_tenant_id FROM lad_dev.users WHERE id = %s",
                    (user_id,)
                )
                result = cur.fetchone()
                if result and result[0]:
                    return str(result[0])
                return None
    except Exception as e:
        logger.error(f"Failed to resolve tenant_id for user {user_id}: {e}")
        return None


# E.164 pattern for validation
import re
E164_PATTERN = re.compile(r"^(\+[1-9]\d{1,14}|0\d{9,14})$")


def _validate_optional_number(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if not E164_PATTERN.match(text):
        raise HTTPException(status_code=400, detail=f"{field_name} must be E.164 formatted")
    return text


def _validate_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{field_name} must be an integer")
    if not isinstance(value, int):
        return None
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{field_name} must be positive")
    return value


def _clean_optional_context(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    return text if text else None


def _parse_batch_csv(file_bytes: bytes) -> list[BatchCallEntry]:
    """Parse CSV file into batch call entries."""
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = file_bytes.decode("latin-1")
    
    reader = csv.DictReader(io.StringIO(content))
    entries = []
    
    for row in reader:
        to_number = row.get("to_number") or row.get("phone") or row.get("number")
        if not to_number:
            continue
        
        to_number = to_number.strip()
        if not E164_PATTERN.match(to_number):
            logger.warning("Skipping invalid number in CSV: %s", to_number[:5] + "...")
            continue
        
        entries.append(BatchCallEntry(
            to_number=to_number,
            context=row.get("context") or row.get("added_context"),
            lead_name=row.get("lead_name") or row.get("name"),
        ))
    
    return entries


# =============================================================================
# POST /trigger-batch-call - Start batch call job
# =============================================================================

@router.post("/trigger-batch-call", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def trigger_batch_call(request: Request) -> dict[str, Any]:
    """
    Start a batch call job.
    
    Supports two formats:
    1. JSON payload with entries array
    2. Form data with CSV file upload
    
    Returns batch job ID for status tracking.
    """
    call_service = get_call_service()
    batch_storage = _get_batch_storage()
    voice_storage = _get_voice_storage()
    
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    entries: list[BatchCallEntry]
    clean_from_number: str | None
    clean_context: str | None
    initiator_id: int | None
    assigned_agent_id: int | None
    llm_provider_override: str | None
    llm_model_override: str | None
    voice_id: str

    if content_type == "application/json":
        try:
            payload_dict = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc.msg}")

        try:
            json_payload = BatchCallJsonRequest(**payload_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail={"message": "Invalid request body", "errors": exc.errors()})

        voice_id = json_payload.voice_id
        clean_from_number = json_payload.from_number
        clean_context = _clean_optional_context(json_payload.added_context)
        initiator_id = json_payload.initiated_by
        assigned_agent_id = json_payload.agent_id
        llm_provider_override = _normalize_llm_provider(json_payload.llm_provider)
        llm_model_override = _clean_optional_text(json_payload.llm_model)
        entries = [
            BatchCallEntry(
                to_number=item.to_number,
                context=item.added_context,
                lead_name=item.lead_name,
                lead_id=item.lead_id,
                knowledge_base_store_ids=item.knowledge_base_store_ids,
            )
            for item in json_payload.entries
        ]
    else:
        form = await request.form()
        voice_id_value = form.get("voice_id")
        if not voice_id_value or not str(voice_id_value).strip():
            raise HTTPException(status_code=400, detail="voice_id cannot be empty")
        voice_id = str(voice_id_value).strip()

        csv_upload = form.get("csv_file")
        if not isinstance(csv_upload, UploadFile):
            raise HTTPException(status_code=400, detail="csv_file is required")

        clean_from_number = _validate_optional_number(str(form.get("from_number")) if form.get("from_number") else None, "from_number")
        clean_context = _clean_optional_context(str(form.get("added_context")) if form.get("added_context") else None)
        initiator_id = _validate_optional_positive_int(form.get("initiated_by"), "initiated_by")
        assigned_agent_id = _validate_optional_positive_int(form.get("agent_id"), "agent_id")
        llm_provider_override = _normalize_llm_provider(str(form.get("llm_provider")) if form.get("llm_provider") else None)
        llm_model_override = _clean_optional_text(str(form.get("llm_model")) if form.get("llm_model") else None)

        file_bytes = await csv_upload.read()
        entries = _parse_batch_csv(file_bytes)

    if not entries:
        raise HTTPException(status_code=400, detail="No valid entries found")

    if not voice_id or not voice_id.strip():
        raise HTTPException(status_code=400, detail="voice_id cannot be empty")

    # Resolve voice
    try:
        resolved_voice_id, voice_context = await call_service.resolve_voice(voice_id, assigned_agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Resolve KB store IDs
    frontend_id = request.headers.get("X-Frontend-ID")
    batch_kb_store_ids: list[str] | None = None
    
    if await call_service.should_use_glinks_kb(assigned_agent_id, frontend_id, batch_kb_store_ids):
        batch_kb_store_ids = DEFAULT_GLINKS_KB_STORE_IDS
        logger.info(
            "Auto-assigned default Glinks KB for batch: agent_id=%s, frontend_id=%s",
            assigned_agent_id, frontend_id,
        )
    
    # Apply resolved KB to entries without their own KB
    if batch_kb_store_ids:
        for entry in entries:
            if not entry.knowledge_base_store_ids:
                entry.knowledge_base_store_ids = batch_kb_store_ids

    # Generate job_id with batch- prefix
    job_id = f"batch-{uuid.uuid4().hex}"
    
    # Resolve tenant_id from initiator's primary_tenant_id
    tenant_id = await _resolve_tenant_id_from_user(initiator_id)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not resolve tenant_id for initiated_by user"
        )
    
    # Create batch record in database
    batch_id = await batch_storage.create_batch(
        tenant_id=tenant_id,
        total_calls=len(entries),
        job_id=job_id,
        initiated_by_user_id=initiator_id,
        agent_id=assigned_agent_id,
        voice_id=voice_context.db_voice_id,
        from_number_id=None,  # Would need _resolve_from_number_id helper
        base_context=clean_context,
        llm_provider=llm_provider_override,
        llm_model=llm_model_override,
    )
    
    if not batch_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch record in database"
        )
    
    logger.info("Created batch record: batch_id=%s, job_id=%s, total_calls=%d", batch_id, job_id, len(entries))
    
    # Create batch entry records and track their IDs
    entry_ids: list[str] = []
    for i, entry in enumerate(entries):
        # Store lead_name in metadata since table uses lead_id (UUID FK)
        entry_metadata = {"entry_index": i}
        if entry.lead_name:
            entry_metadata["lead_name"] = entry.lead_name
        if entry.context:
            entry_metadata["added_context"] = entry.context
        
        entry_id = await batch_storage.create_batch_entry(
            tenant_id=tenant_id,
            batch_id=batch_id,
            to_phone=entry.to_number,
            lead_id=None,  # Would need lead resolution
            call_log_id=None,
            metadata=entry_metadata,
        )
        if entry_id:
            entry_ids.append(entry_id)
        else:
            logger.warning("Failed to create batch entry %d", i)
            entry_ids.append("")  # Empty placeholder

    # Fire and forget the batch processing
    async def _process_batch():
        call_service = get_call_service()
        batch_storage = _get_batch_storage()
        
        for i, entry in enumerate(entries):
            entry_id = entry_ids[i] if i < len(entry_ids) else None
            if not entry_id:
                logger.warning("Skipping entry %d - no entry_id", i)
                continue
            
            try:
                # Update entry status to running
                await batch_storage.update_batch_entry_status(entry_id, "running")
                
                # Dispatch call
                result = await call_service.dispatch_call(
                    job_id=job_id,
                    voice_id=resolved_voice_id,
                    voice_context=voice_context,
                    from_number=clean_from_number,
                    to_number=entry.to_number,
                    context=entry.context or clean_context,
                    initiated_by=initiator_id,
                    agent_id=assigned_agent_id,
                    llm_provider=llm_provider_override,
                    llm_model=llm_model_override,
                    knowledge_base_store_ids=entry.knowledge_base_store_ids,
                    lead_id_override=entry.lead_id,
                )
                
                # Update entry with call log ID and status
                await batch_storage.update_batch_entry_call_log(batch_id, entry_id, result.call_log_id)
                await batch_storage.update_batch_entry_status(entry_id, "dispatched")
                
                await batch_storage.increment_batch_counters(batch_id, completed_delta=1)
                
            except Exception as exc:
                logger.exception("Batch entry %d failed: %s", i, exc)
                await batch_storage.update_batch_entry_status(entry_id, "failed", error_message=str(exc)[:500])
                await batch_storage.increment_batch_counters(batch_id, failed_delta=1)
        
        # Mark batch as completed
        await batch_storage.update_batch_status(batch_id, "completed")
        logger.info("Batch %s completed", job_id)
        
        # Trigger batch report generation and email (fire and forget)
        try:
            from analysis.batch_report import generate_batch_report
            asyncio.create_task(_generate_and_send_batch_report(batch_id))
        except ImportError as e:
            logger.warning("Batch report module not available: %s", e)
        except Exception as e:
            logger.error("Error triggering batch report: %s", e)
    
    async def _generate_and_send_batch_report(batch_id: str):
        """Generate and send batch report email."""
        try:
            from analysis.batch_report import generate_batch_report
            # generate_batch_report gets initiator from batch_info internally
            result = await generate_batch_report(batch_id, send_email=True)
            if result.get("status") == "success":
                logger.info("Batch report sent for batch_id=%s", batch_id)
            else:
                logger.warning("Batch report failed for batch_id=%s: %s", batch_id, result.get("message"))
        except Exception as e:
            logger.error("Error generating batch report: %s", e, exc_info=True)

    asyncio.create_task(_process_batch())
    
    return {
        "job_id": job_id,
        "batch_id": str(batch_id),
        "total_entries": len(entries),
        "status": "accepted",
        "message": f"Batch job started with {len(entries)} calls",
    }


# =============================================================================
# GET /batch-status/{batch_id} - Get batch status
# =============================================================================

@router.get("/batch-status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """
    Get status of a batch call job.
    
    Args:
        batch_id: Job ID (batch-xxx format) or batch UUID
    
    Returns:
        BatchStatusResponse with status and entries
    """
    batch_storage = _get_batch_storage()
    
    # Try as job_id first
    if batch_id.startswith("batch-"):
        batch_record = await batch_storage.get_batch_by_job_id(batch_id)
    else:
        batch_record = await batch_storage.get_batch_by_id(batch_id)
    
    if not batch_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch not found: {batch_id}"
        )
    
    entries = await batch_storage.get_batch_entries(batch_record["id"])
    
    pending = sum(1 for e in entries if e["status"] == "pending")
    running = sum(1 for e in entries if e["status"] == "running")
    
    entry_models = [
        BatchEntryStatusModel(
            entry_index=e["entry_index"],
            to_number=e["to_number"],
            lead_name=e.get("lead_name"),
            status=e["status"],
            call_log_id=e.get("call_log_id"),
            call_status=e.get("call_status"),
            call_duration=e.get("call_duration"),
            call_recording_url=e.get("call_recording_url"),
            error_message=e.get("error_message"),
        )
        for e in entries
    ]
    
    return BatchStatusResponse(
        batch_id=batch_record["id"],
        job_id=batch_record["job_id"],
        status=batch_record["status"],
        total_calls=batch_record["total_calls"],
        completed_calls=batch_record["completed_calls"],
        failed_calls=batch_record["failed_calls"],
        cancelled_calls=batch_record["cancelled_calls"],
        pending_calls=pending,
        running_calls=running,
        initiated_by=batch_record.get("initiated_by"),
        agent_id=batch_record.get("agent_id"),
        created_at=batch_record.get("created_at"),
        updated_at=batch_record.get("updated_at"),
        stopped_at=batch_record.get("stopped_at"),
        completed_at=batch_record.get("completed_at"),
        entries=entry_models,
    )


# =============================================================================
# POST /batch-cancel/{batch_id} - Cancel batch job
# =============================================================================

@router.post("/batch-cancel/{batch_id}", response_model=dict)
async def cancel_batch(batch_id: str) -> dict[str, Any]:
    """
    Cancel a running batch job.
    
    Stops processing and marks pending entries as cancelled.
    Running calls will complete naturally.
    """
    batch_storage = _get_batch_storage()
    
    if batch_id.startswith("batch-"):
        batch_record = await batch_storage.get_batch_by_job_id(batch_id)
    else:
        batch_record = await batch_storage.get_batch_by_id(batch_id)
    
    if not batch_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch not found: {batch_id}"
        )
    
    current_status = batch_record.get("status")
    if current_status in ("stopped", "completed", "cancelled"):
        return {
            "batch_id": batch_record["id"],
            "status": current_status,
            "cancelled_count": 0,
            "message": f"Batch already in terminal state: {current_status}",
        }
    
    # Mark batch as stopped
    await batch_storage.update_batch_status(batch_record["id"], "stopped")
    
    # Mark pending entries as cancelled
    cancelled_count = await batch_storage.mark_pending_entries_cancelled(batch_record["id"])
    if cancelled_count > 0:
        await batch_storage.increment_batch_counters(batch_record["id"], cancelled_delta=cancelled_count)
    
    logger.info("Batch %s cancelled: %d entries cancelled", batch_id, cancelled_count)
    
    return {
        "batch_id": batch_record["id"],
        "job_id": batch_record["job_id"],
        "status": "stopped",
        "cancelled_count": cancelled_count,
        "message": f"Batch stopped. {cancelled_count} pending entries cancelled.",
    }


__all__ = ["router"]
