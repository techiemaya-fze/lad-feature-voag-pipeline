"""
Lead Bookings Storage - v2 Refactored

Database storage for lead bookings extraction from voice_call_logs transcriptions.
Uses psycopg2 with connection pooling (v2 pattern).
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager (v2 pattern)
from db.connection_pool import get_db_connection, return_connection, USE_CONNECTION_POOLING
# Import centralized DB config (respects USE_LOCAL_DB toggle)
from db.db_config import get_db_config

load_dotenv()

logger = logging.getLogger(__name__)

# Schema and table constants
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")
CALL_LOGS_TABLE = "voice_call_logs"
VOICE_AGENTS_TABLE = "voice_agents"
LEAD_BOOKINGS_TABLE = "lead_bookings"
LEADS_TABLE = "leads"


class LeadBookingsStorageError(Exception):
    """Exception raised for lead bookings storage errors."""
    pass


class LeadBookingsStorage:
    """
    Database storage for lead bookings extraction.
    
    Uses psycopg2 with connection pooling (v2 pattern).
    """
    
    def __init__(self):
        self.db_config = get_db_config()
    
    def _get_connection(self):
        """Get database connection (pooled or direct based on feature flag)"""
        return get_db_connection(self.db_config)
    
    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled"""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)
    
    async def close(self):
        """Close any resources (no-op for pooled connections)"""
        pass  # Connection pooling handles cleanup
    
    async def get_call_log(self, call_log_id: str) -> Optional[Dict]:
        """Get call log data from voice_call_logs table"""
        if not call_log_id:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT 
                            id, tenant_id, lead_id, transcripts,
                            initiated_by_user_id, agent_id, started_at, status
                        FROM {SCHEMA}.{CALL_LOGS_TABLE}
                        WHERE id = %s::uuid
                    """, (str(call_log_id),))
                    row = cur.fetchone()
                    
                    if not row:
                        return None
                    
                    return {
                        "id": str(row['id']) if row['id'] else None,
                        "tenant_id": str(row['tenant_id']) if row['tenant_id'] else None,
                        "lead_id": str(row['lead_id']) if row['lead_id'] else None,
                        "transcripts": row['transcripts'],
                        "initiated_by_user_id": str(row['initiated_by_user_id']) if row['initiated_by_user_id'] else None,
                        "agent_id": row['agent_id'],
                        "started_at": row['started_at'],
                        "status": row.get('status')
                    }
        except Exception as exc:
            logger.error("Failed to get call log %s: %s", call_log_id, exc, exc_info=True)
            return None
    
    async def get_voice_id_from_agent_id(self, agent_id: Any) -> Optional[str]:
        """Get voice_id from voice_agents table by matching agent_id"""
        if not agent_id:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    agent_id_str = str(agent_id).strip()
                    
                    # Try as integer first (most common case)
                    try:
                        agent_id_int = int(agent_id)
                        cur.execute(f"""
                            SELECT voice_id 
                            FROM {SCHEMA}.{VOICE_AGENTS_TABLE} 
                            WHERE id = %s
                        """, (agent_id_int,))
                        row = cur.fetchone()
                        if row:
                            return str(row[0]) if row[0] else None
                    except (ValueError, TypeError):
                        pass
                    
                    # Try as text match
                    cur.execute(f"""
                        SELECT voice_id 
                        FROM {SCHEMA}.{VOICE_AGENTS_TABLE} 
                        WHERE id::text = %s
                    """, (agent_id_str,))
                    row = cur.fetchone()
                    if row:
                        return str(row[0]) if row[0] else None
                    
                    return None
        except Exception as exc:
            logger.error("Failed to get voice_id for agent %s: %s", agent_id, exc, exc_info=True)
            return None
    
    async def list_calls(self, limit: Optional[int] = 100) -> List[Dict]:
        """List all calls from voice_call_logs"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if limit is None:
                        cur.execute(f"""
                            SELECT id, tenant_id, lead_id, started_at, initiated_by_user_id, agent_id
                            FROM {SCHEMA}.{CALL_LOGS_TABLE}
                            ORDER BY started_at DESC
                        """)
                    else:
                        cur.execute(f"""
                            SELECT id, tenant_id, lead_id, started_at, initiated_by_user_id, agent_id
                            FROM {SCHEMA}.{CALL_LOGS_TABLE}
                            ORDER BY started_at DESC
                            LIMIT %s
                        """, (limit,))
                    
                    rows = cur.fetchall()
                    return [
                        {
                            "id": str(row['id']),
                            "tenant_id": str(row['tenant_id']) if row['tenant_id'] else None,
                            "lead_id": str(row['lead_id']) if row['lead_id'] else None,
                            "started_at": row['started_at'].isoformat() if row['started_at'] else None,
                            "initiated_by_user_id": str(row['initiated_by_user_id']) if row['initiated_by_user_id'] else None,
                            "agent_id": str(row['agent_id']) if row['agent_id'] else None
                        }
                        for row in rows
                    ]
        except Exception as exc:
            logger.error("Failed to list calls: %s", exc, exc_info=True)
            return []
    
    async def count_bookings_by_lead_id(self, lead_id: str) -> int:
        """Count existing bookings for a lead_id"""
        if not lead_id:
            return 0
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT COUNT(*)
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE lead_id = %s::uuid
                        AND is_deleted = false
                    """, (str(lead_id),))
                    row = cur.fetchone()
                    return row[0] if row else 0
        except Exception as exc:
            logger.error("Failed to count bookings for lead %s: %s", lead_id, exc, exc_info=True)
            return 0
    
    async def count_bookings_by_lead_id_and_booking_type(self, lead_id: str, booking_type: str) -> int:
        """Count existing bookings for a lead_id and booking_type combination"""
        if not lead_id or not booking_type:
            return 0
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT COUNT(*)
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE lead_id = %s::uuid
                        AND booking_type = %s
                        AND is_deleted = false
                    """, (str(lead_id), booking_type))
                    row = cur.fetchone()
                    return row[0] if row else 0
        except Exception as exc:
            logger.error("Failed to count bookings for lead %s type %s: %s", lead_id, booking_type, exc, exc_info=True)
            return 0
    
    async def get_max_retry_count_by_lead_id(self, lead_id: str) -> int:
        """Get the maximum retry_count for a lead_id"""
        if not lead_id:
            return 0
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT MAX(retry_count)
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE lead_id = %s::uuid
                        AND is_deleted = false
                    """, (str(lead_id),))
                    row = cur.fetchone()
                    return row[0] if row and row[0] is not None else 0
        except Exception as exc:
            logger.error("Failed to get max retry for lead %s: %s", lead_id, exc, exc_info=True)
            return 0
    
    async def get_max_retry_count_by_lead_id_and_booking_type(self, lead_id: str, booking_type: str) -> int:
        """Get the maximum retry_count for a lead_id and booking_type combination"""
        if not lead_id or not booking_type:
            return 0
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT MAX(retry_count)
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE lead_id = %s::uuid
                        AND booking_type = %s
                        AND is_deleted = false
                    """, (str(lead_id), booking_type))
                    row = cur.fetchone()
                    return row[0] if row and row[0] is not None else 0
        except Exception as exc:
            logger.error("Failed to get max retry for lead %s type %s: %s", lead_id, booking_type, exc, exc_info=True)
            return 0
    
    async def get_original_booking_by_lead_id_and_booking_type(self, lead_id: str, booking_type: str) -> Optional[Dict]:
        """Get the original booking for a lead_id and booking_type (where parent_booking_id IS NULL)"""
        if not lead_id or not booking_type:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT id, tenant_id, lead_id, scheduled_at, created_at, parent_booking_id, metadata
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE lead_id = %s::uuid
                        AND booking_type = %s
                        AND parent_booking_id IS NULL
                        AND is_deleted = false
                        ORDER BY created_at ASC
                        LIMIT 1
                    """, (str(lead_id), booking_type))
                    row = cur.fetchone()
                    
                    if not row:
                        return None
                    
                    # Parse metadata to get call_id
                    metadata = row.get('metadata')
                    call_id_from_metadata = None
                    if metadata:
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        call_id_from_metadata = metadata.get('call_id') if isinstance(metadata, dict) else None
                    
                    return {
                        "id": str(row['id']) if row['id'] else None,
                        "tenant_id": str(row['tenant_id']) if row['tenant_id'] else None,
                        "lead_id": str(row['lead_id']) if row['lead_id'] else None,
                        "scheduled_at": row['scheduled_at'],
                        "created_at": row['created_at'],
                        "parent_booking_id": str(row['parent_booking_id']) if row['parent_booking_id'] else None,
                        "call_id": call_id_from_metadata
                    }
        except Exception as exc:
            logger.error("Failed to get original booking for lead %s type %s: %s", lead_id, booking_type, exc, exc_info=True)
            return None
    
    async def get_booking_by_call_id_in_metadata(self, call_id: str) -> Optional[Dict]:
        """Get a booking by call_id stored in metadata column"""
        if not call_id:
            return None
        
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First try original booking (parent_booking_id IS NULL)
                    cur.execute(f"""
                        SELECT id, tenant_id, lead_id, booking_type, scheduled_at, created_at, parent_booking_id
                        FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                        WHERE metadata->>'call_id' = %s
                        AND parent_booking_id IS NULL
                        AND is_deleted = false
                        ORDER BY created_at ASC
                        LIMIT 1
                    """, (str(call_id),))
                    row = cur.fetchone()
                    
                    # If not found, try any booking
                    if not row:
                        cur.execute(f"""
                            SELECT id, tenant_id, lead_id, booking_type, scheduled_at, created_at, parent_booking_id
                            FROM {SCHEMA}.{LEAD_BOOKINGS_TABLE}
                            WHERE metadata->>'call_id' = %s
                            AND is_deleted = false
                            ORDER BY created_at ASC
                            LIMIT 1
                        """, (str(call_id),))
                        row = cur.fetchone()
                    
                    if not row:
                        return None
                    
                    return {
                        "id": str(row['id']) if row['id'] else None,
                        "tenant_id": str(row['tenant_id']) if row['tenant_id'] else None,
                        "lead_id": str(row['lead_id']) if row['lead_id'] else None,
                        "booking_type": row.get('booking_type'),
                        "scheduled_at": row['scheduled_at'],
                        "created_at": row['created_at'],
                        "parent_booking_id": str(row['parent_booking_id']) if row['parent_booking_id'] else None
                    }
        except Exception as exc:
            logger.error("Failed to get booking by call_id %s: %s", call_id, exc, exc_info=True)
            return None
    
    async def save_booking(self, booking_data: Dict) -> Optional[str]:
        """Save a booking to the database"""
        if not booking_data.get('lead_id'):
            raise LeadBookingsStorageError("lead_id is required")
        if not booking_data.get('tenant_id'):
            raise LeadBookingsStorageError("tenant_id is required")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Parse scheduled_at if it's a string
                    scheduled_at = booking_data.get('scheduled_at')
                    if isinstance(scheduled_at, str):
                        try:
                            scheduled_at = datetime.fromisoformat(scheduled_at.replace('Z', '+00:00'))
                        except ValueError:
                            scheduled_at = datetime.now()  # Default if parsing fails
                    
                    # Ensure required NOT NULL fields have defaults
                    booking_type = booking_data.get('booking_type') or 'auto_followup'
                    booking_source = booking_data.get('booking_source') or 'system'
                    status = booking_data.get('status') or 'scheduled'
                    if not scheduled_at:
                        scheduled_at = datetime.now()
                    
                    # Parse buffer_until if it's a string
                    buffer_until = booking_data.get('buffer_until')
                    if isinstance(buffer_until, str):
                        try:
                            buffer_until = datetime.fromisoformat(buffer_until.replace('Z', '+00:00'))
                        except ValueError:
                            buffer_until = None
                    
                    cur.execute(f"""
                        INSERT INTO {SCHEMA}.{LEAD_BOOKINGS_TABLE} (
                            id, tenant_id, lead_id, assigned_user_id, booking_type, booking_source,
                            scheduled_at, timezone, status, retry_count, parent_booking_id,
                            notes, metadata, created_by, is_deleted, buffer_until
                        ) VALUES (
                            %s::uuid, %s::uuid, %s::uuid, %s::uuid, %s, %s,
                            %s, %s, %s, %s, %s::uuid,
                            %s, %s, %s::uuid, %s, %s
                        )
                        RETURNING id
                    """, (
                        booking_data.get('id'),
                        booking_data.get('tenant_id'),
                        booking_data.get('lead_id'),
                        booking_data.get('assigned_user_id'),
                        booking_type,
                        booking_source,
                        scheduled_at,
                        booking_data.get('timezone', 'GST'),
                        status,
                        booking_data.get('retry_count', 0),
                        booking_data.get('parent_booking_id'),
                        booking_data.get('notes'),
                        json.dumps(booking_data.get('metadata', {})),
                        booking_data.get('created_by'),
                        booking_data.get('is_deleted', False),
                        buffer_until
                    ))
                    
                    row = cur.fetchone()
                    conn.commit()
                    
                    if row:
                        return str(row[0])
                    return None
        except Exception as exc:
            logger.error("Failed to save booking: %s", exc, exc_info=True)
            raise LeadBookingsStorageError(f"Failed to save booking: {exc}")
