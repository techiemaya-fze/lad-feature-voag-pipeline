"""
Database storage for batch call operations.

Updated for lad_dev schema (Phase 12):
- Table: lad_dev.voice_call_batches (was batch_logs_voiceagent)
- Table: lad_dev.voice_call_batch_entries (was batch_call_entries_voiceagent)
- Added: tenant_id (required)
- Changed: agent_id and voice_id are now UUIDs
- Changed: job_id, base_context, llm_provider/model moved to metadata JSONB
- Changed: entry uses lead_id (UUID FK) instead of lead_name
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from dotenv import load_dotenv

from db.connection_pool import get_db_connection, get_raw_connection, return_connection, USE_CONNECTION_POOLING
from db.db_config import get_db_config

load_dotenv()

logger = logging.getLogger(__name__)

# Schema and table constants
SCHEMA = "lad_dev"
BATCH_TABLE = "voice_call_batches"
ENTRY_TABLE = "voice_call_batch_entries"
CALL_TABLE = "voice_call_logs"
FULL_BATCH_TABLE = f"{SCHEMA}.{BATCH_TABLE}"
FULL_ENTRY_TABLE = f"{SCHEMA}.{ENTRY_TABLE}"
FULL_CALL_TABLE = f"{SCHEMA}.{CALL_TABLE}"


class DatabaseError(Exception):
    """Raised when a database operation fails due to connection or query issues"""
    pass


class BatchStorage:
    """
    Handles database operations for batch call jobs.
    
    Uses lad_dev schema with:
    - tenant_id (required) for multi-tenancy
    - metadata JSONB for job_id, base_context, llm settings
    - lead_id UUID FK on entries (replaces lead_name)
    """

    def __init__(self):
        self.db_config = get_db_config()

    def _get_connection(self):
        """Get raw database connection (must be returned manually with _return_connection)"""
        return get_raw_connection(self.db_config)
    
    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled"""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)

    def _prepare_jsonb(self, data: dict | None) -> Optional[Json]:
        """Prepare dict for JSONB storage."""
        return Json(data) if data is not None else None

    # =========================================================================
    # CREATE
    # =========================================================================

    async def create_batch(
        self,
        tenant_id: str,
        total_calls: int,
        *,
        job_id: str | None = None,
        initiated_by_user_id: str | None = None,
        agent_id: str | None = None,
        voice_id: str | None = None,
        from_number_id: str | None = None,
        scheduled_at: datetime | None = None,
        base_context: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> Optional[str]:
        """
        Create a new batch job record.
        
        Args:
            tenant_id: Required tenant UUID
            total_calls: Total number of calls in the batch
            job_id: Unique job identifier (stored in metadata)
            initiated_by_user_id: UUID of user who initiated
            agent_id: Agent UUID
            voice_id: Voice UUID
            from_number_id: Outbound number UUID
            scheduled_at: When to start the batch
            base_context: Base context for all calls (stored in metadata)
            llm_provider: LLM provider (stored in metadata)
            llm_model: LLM model (stored in metadata)
        
        Returns:
            batch_id (UUID string) if successful, None otherwise
        """
        if not tenant_id:
            logger.error("tenant_id is required for create_batch")
            return None
        
        try:
            # Build metadata object for fields that moved to JSONB
            metadata = {}
            if job_id:
                metadata["job_id"] = job_id
            if base_context:
                metadata["base_context"] = base_context
            if llm_provider:
                metadata["llm_provider"] = llm_provider
            if llm_model:
                metadata["llm_model"] = llm_model
            
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {FULL_BATCH_TABLE}
                        (tenant_id, status, total_calls, completed_calls, failed_calls,
                         initiated_by_user_id, agent_id, voice_id, from_number_id,
                         scheduled_at, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            tenant_id,
                            "queued",
                            total_calls,
                            0,
                            0,
                            initiated_by_user_id,
                            agent_id,
                            voice_id,
                            from_number_id,
                            scheduled_at,
                            self._prepare_jsonb(metadata or {}),
                        )
                    )
                    batch_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.info(
                        "Created batch: id=%s, tenant=%s, total_calls=%d",
                        batch_id,
                        tenant_id,
                        total_calls,
                    )
                    return str(batch_id)
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to create batch: %s",
                exc,
                exc_info=True
            )
            return None

    async def create_batch_entry(
        self,
        tenant_id: str,
        batch_id: str,
        to_phone: str,
        *,
        lead_id: str | None = None,
        call_log_id: str | None = None,
        metadata: dict | None = None,
    ) -> Optional[str]:
        """
        Create a batch entry for a single call in the batch.
        
        Args:
            tenant_id: Required tenant UUID
            batch_id: UUID of the parent batch
            to_phone: Phone number to call
            lead_id: UUID of the lead (optional FK)
            call_log_id: UUID of the associated call log
            metadata: Additional entry metadata
        
        Returns:
            entry_id (UUID string) if successful, None otherwise
        """
        if not tenant_id:
            logger.error("tenant_id is required for create_batch_entry")
            return None
        
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {FULL_ENTRY_TABLE}
                        (tenant_id, batch_id, to_phone, lead_id, call_log_id,
                         status, last_error, metadata, is_deleted, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            tenant_id,
                            batch_id,
                            to_phone,
                            lead_id,
                            call_log_id,
                            "pending",
                            None,  # last_error - null initially
                            self._prepare_jsonb(metadata or {}),
                            False,  # is_deleted
                            datetime.utcnow(),
                            datetime.utcnow(),
                        )
                    )
                    entry_id = cur.fetchone()[0]
                    conn.commit()
                    return str(entry_id)
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to create batch entry for batch_id=%s: %s",
                batch_id,
                exc,
                exc_info=True
            )
            return None

    # =========================================================================
    # UPDATE
    # =========================================================================

    async def update_batch_entry_call_log(
        self,
        batch_id: str,
        entry_id: str,
        call_log_id: str,
    ) -> bool:
        """
        Link a call_log_id to a batch entry after the call is created.
        
        Args:
            batch_id: UUID of the parent batch
            entry_id: UUID of the entry
            call_log_id: UUID of the call log to link
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {FULL_ENTRY_TABLE}
                        SET call_log_id = %s, updated_at = %s
                        WHERE id = %s AND batch_id = %s
                        """,
                        (call_log_id, datetime.now(timezone.utc), entry_id, batch_id)
                    )
                    conn.commit()
                    return True
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to update batch entry call_log: %s",
                exc,
                exc_info=True
            )
            return False

    async def update_batch_entry_status(
        self,
        entry_id: str,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """
        Update the status of a batch entry.
        
        Args:
            entry_id: UUID of the entry
            status: New status (pending, running, completed, failed, cancelled)
            error_message: Error message if failed
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {FULL_ENTRY_TABLE}
                        SET status = %s, last_error = %s, updated_at = %s
                        WHERE id = %s
                        """,
                        (status, error_message, datetime.now(timezone.utc), entry_id)
                    )
                    conn.commit()
                    return True
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to update batch entry status: %s",
                exc,
                exc_info=True
            )
            return False

    async def update_batch_status(
        self,
        batch_id: str,
        status: str,
        completed_calls: int | None = None,
        failed_calls: int | None = None,
    ) -> bool:
        """
        Update the status of a batch job.
        
        Args:
            batch_id: UUID of the batch
            status: New status (pending, running, completed, stopped, failed)
            completed_calls: Number of completed calls
            failed_calls: Number of failed calls
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    set_clauses = ["status = %s", "updated_at = %s"]
                    params: list[Any] = [status, datetime.now(timezone.utc)]
                    
                    if completed_calls is not None:
                        set_clauses.append("completed_calls = %s")
                        params.append(completed_calls)
                    if failed_calls is not None:
                        set_clauses.append("failed_calls = %s")
                        params.append(failed_calls)
                    
                    # Set timestamps based on status
                    if status == "running":
                        set_clauses.append("started_at = COALESCE(started_at, %s)")
                        params.append(datetime.now(timezone.utc))
                    elif status in ("completed", "stopped", "failed"):
                        set_clauses.append("finished_at = %s")
                        params.append(datetime.now(timezone.utc))
                    
                    params.append(batch_id)
                    
                    cur.execute(
                        f"""
                        UPDATE {FULL_BATCH_TABLE}
                        SET {', '.join(set_clauses)}
                        WHERE id = %s
                        """,
                        params
                    )
                    conn.commit()
                    return True
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to update batch status: %s",
                exc,
                exc_info=True
            )
            return False

    async def increment_batch_counters(
        self,
        batch_id: str,
        completed_delta: int = 0,
        failed_delta: int = 0,
    ) -> bool:
        """
        Atomically increment batch counters.
        
        Args:
            batch_id: UUID of the batch
            completed_delta: Amount to add to completed_calls
            failed_delta: Amount to add to failed_calls
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {FULL_BATCH_TABLE}
                        SET 
                            completed_calls = completed_calls + %s,
                            failed_calls = failed_calls + %s,
                            updated_at = %s
                        WHERE id = %s
                        """,
                        (completed_delta, failed_delta, datetime.now(timezone.utc), batch_id)
                    )
                    conn.commit()
                    return True
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to increment batch counters: %s",
                exc,
                exc_info=True
            )
            return False

    # =========================================================================
    # READ
    # =========================================================================

    async def get_batch_by_id(self, batch_id: str) -> Optional[dict]:
        """Get batch details by UUID."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT 
                            id, tenant_id, status, total_calls, completed_calls, failed_calls,
                            initiated_by_user_id, agent_id, voice_id, from_number_id,
                            scheduled_at, started_at, finished_at, metadata,
                            created_at, updated_at
                        FROM {FULL_BATCH_TABLE}
                        WHERE id = %s AND is_deleted = FALSE
                        """,
                        (batch_id,)
                    )
                    result = cur.fetchone()
                    
                    if not result:
                        return None
                    
                    batch = dict(result)
                    # Extract legacy fields from metadata for backwards compatibility
                    metadata = batch.get("metadata") or {}
                    batch["job_id"] = metadata.get("job_id")
                    batch["base_context"] = metadata.get("base_context")
                    batch["llm_provider"] = metadata.get("llm_provider")
                    batch["llm_model"] = metadata.get("llm_model")
                    
                    return batch
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to get batch by id=%s: %s",
                batch_id,
                exc,
                exc_info=True
            )
            return None

    async def get_batch_by_job_id(self, job_id: str) -> Optional[dict]:
        """Get batch details by job_id (stored in metadata)."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT 
                            id, tenant_id, status, total_calls, completed_calls, failed_calls,
                            initiated_by_user_id, agent_id, voice_id, from_number_id,
                            scheduled_at, started_at, finished_at, metadata,
                            created_at, updated_at
                        FROM {FULL_BATCH_TABLE}
                        WHERE metadata->>'job_id' = %s AND is_deleted = FALSE
                        """,
                        (job_id,)
                    )
                    result = cur.fetchone()
                    
                    if not result:
                        return None
                    
                    batch = dict(result)
                    metadata = batch.get("metadata") or {}
                    batch["job_id"] = metadata.get("job_id")
                    batch["base_context"] = metadata.get("base_context")
                    batch["llm_provider"] = metadata.get("llm_provider")
                    batch["llm_model"] = metadata.get("llm_model")
                    
                    return batch
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to get batch by job_id=%s: %s",
                job_id,
                exc,
                exc_info=True
            )
            raise DatabaseError(f"Database error while fetching batch: {exc}") from exc

    async def get_batch_entries(self, batch_id: str) -> list[dict]:
        """Get all entries for a batch with call log details."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT 
                            e.id, e.tenant_id, e.batch_id, e.lead_id,
                            e.to_phone as to_number,
                            e.status, e.call_log_id, 
                            e.last_error as error_message,
                            e.metadata,
                            e.created_at, e.updated_at,
                            c.status as call_status, 
                            c.recording_url as call_recording_url, 
                            c.duration_seconds as call_duration,
                            -- Extract lead_name and entry_index from metadata
                            (e.metadata->>'lead_name') as lead_name,
                            (e.metadata->>'entry_index')::int as entry_index
                        FROM {FULL_ENTRY_TABLE} e
                        LEFT JOIN {FULL_CALL_TABLE} c ON e.call_log_id = c.id
                        WHERE e.batch_id = %s AND e.is_deleted = FALSE
                        ORDER BY e.created_at
                        """,
                        (batch_id,)
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to get batch entries: %s",
                exc,
                exc_info=True
            )
            raise DatabaseError(f"Database error while fetching batch entries: {exc}") from exc

    async def get_pending_entries(self, batch_id: str) -> list[dict]:
        """Get pending entries for a batch (not yet started or running)."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT 
                            id, tenant_id, batch_id, lead_id, to_phone, status, call_log_id
                        FROM {FULL_ENTRY_TABLE}
                        WHERE batch_id = %s AND status IN ('pending', 'running') 
                        AND is_deleted = FALSE
                        ORDER BY created_at
                        """,
                        (batch_id,)
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to get pending entries: %s",
                exc,
                exc_info=True
            )
            return []

    async def mark_pending_entries_cancelled(self, batch_id: str) -> int:
        """Mark all pending entries as cancelled and return count."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {FULL_ENTRY_TABLE}
                        SET status = 'cancelled', updated_at = %s
                        WHERE batch_id = %s AND status = 'pending'
                        """,
                        (datetime.now(timezone.utc), batch_id)
                    )
                    cancelled_count = cur.rowcount
                    conn.commit()
                    return cancelled_count
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to mark pending entries cancelled: %s",
                exc,
                exc_info=True
            )
            return 0

    async def get_running_entries_with_call_logs(self, batch_id: str) -> list[dict]:
        """Get running entries with their call_log details for cleanup."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT 
                            e.id, e.call_log_id, e.to_phone,
                            c.metadata->>'room_name' as room_name,
                            c.status as call_status
                        FROM {FULL_ENTRY_TABLE} e
                        LEFT JOIN {FULL_CALL_TABLE} c ON e.call_log_id = c.id
                        WHERE e.batch_id = %s AND e.status = 'running'
                        """,
                        (batch_id,)
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to get running entries: %s",
                exc,
                exc_info=True
            )
            return []

    async def is_batch_stopped(self, batch_id: str) -> bool:
        """Check if a batch has been marked as stopped."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT status FROM {FULL_BATCH_TABLE}
                        WHERE id = %s
                        """,
                        (batch_id,)
                    )
                    result = cur.fetchone()
                    if not result:
                        return False
                    return result[0] in ("stopped", "cancelled", "failed")
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to check batch stopped status: %s",
                exc,
                exc_info=True
            )
            return False

    async def check_and_complete_batch(
        self,
        batch_id: str,
    ) -> dict:
        """
        Check if all entries in the batch are done. If so, mark batch completed.
        
        Args:
            batch_id: UUID of the batch
        
        Returns:
            Dict with 'completed' (bool), 'should_report' (bool), and stats
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    # Get batch status and entry counts
                    cur.execute(
                        f"""
                        SELECT 
                            b.status,
                            b.total_calls,
                            COALESCE(b.completed_calls, 0),
                            COALESCE(b.failed_calls, 0),
                            (SELECT COUNT(*) FROM {FULL_ENTRY_TABLE} 
                             WHERE batch_id = b.id AND status IN ('completed', 'failed')) as done_count
                        FROM {FULL_BATCH_TABLE} b
                        WHERE b.id = %s
                        """,
                        (batch_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        logger.warning(f"Batch {batch_id} not found")
                        return {"completed": False, "should_report": False}
                    
                    status, total_calls, completed_calls, failed_calls, done_count = row
                    
                    logger.debug(
                        f"Batch {batch_id}: status={status}, total={total_calls}, "
                        f"completed={completed_calls}, failed={failed_calls}, done_entries={done_count}"
                    )
                    
                    # Already completed?
                    if status == "completed":
                        return {"completed": True, "should_report": False}
                    
                    # Check if all entries are done
                    all_done = done_count >= total_calls if total_calls > 0 else False
                    
                    if all_done:
                        # Mark batch as completed
                        cur.execute(
                            f"""
                            UPDATE {FULL_BATCH_TABLE}
                            SET status = 'completed', finished_at = NOW(), updated_at = NOW()
                            WHERE id = %s AND status = 'running'
                            RETURNING id
                            """,
                            (batch_id,)
                        )
                        updated = cur.fetchone()
                        conn.commit()
                        
                        if updated:
                            logger.info(f"Batch {batch_id} marked completed (all {total_calls} entries done)")
                            return {
                                "completed": True,
                                "should_report": True,
                                "total_calls": total_calls,
                                "completed_calls": completed_calls,
                                "failed_calls": failed_calls,
                            }
                        else:
                            # Already updated by another worker
                            return {"completed": True, "should_report": False}
                    
                    return {"completed": False, "should_report": False}
            finally:
                self._return_connection(conn)

        except Exception as exc:
            logger.error(
                "Failed to check batch completion: %s",
                exc,
                exc_info=True
            )
            return {"completed": False, "should_report": False}

