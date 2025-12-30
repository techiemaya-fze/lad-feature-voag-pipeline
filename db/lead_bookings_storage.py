"""
Lead Bookings Storage
Database storage for lead bookings extraction from voice_call_logs transcriptions.
Manages database connections and operations for lead bookings.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Note: Using asyncpg directly for true async operations (new file pattern)

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import asyncpg
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("asyncpg not installed. Install with: pip install asyncpg")


class LeadBookingsStorageError(Exception):
    """Exception raised for lead bookings storage errors."""
    pass


class LeadBookingsStorage:
    """
    Database storage for lead bookings extraction from voice_call_logs transcriptions.
    
    Manages:
    - Database connections using asyncpg connection pool
    - Querying voice_call_logs table
    - Querying voice_agents table
    """
    
    SCHEMA = "lad_dev"
    CALL_LOGS_TABLE = "voice_call_logs"
    VOICE_AGENTS_TABLE = "voice_agents"
    LEAD_BOOKINGS_TABLE = "lead_bookings"
    LEADS_TABLE = "leads"  # Table that stores lead assignments
    
    def __init__(self):
        # Get DB config from environment (asyncpg uses direct config)
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "salesmaya_agent"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
        
        # Connection pool for asyncpg (will be created on first use)
        self._pool = None
    
    async def _get_connection(self):
        """Get database connection from pool (async)"""
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        # Create connection pool if it doesn't exist
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=1,
                max_size=10
            )
        
        # Get connection from pool
        return await self._pool.acquire()
    
    async def _return_connection(self, conn):
        """Return connection to pool"""
        if self._pool and conn:
            await self._pool.release(conn)
    
    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def get_call_log(self, call_log_id: str) -> Optional[Dict]:
        """
        Get call log data from voice_call_logs table
        
        Args:
            call_log_id: Call log ID (UUID string)
        
        Returns:
            Dictionary with call log data or None if not found
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            call_data = await conn.fetchrow(f"""
                SELECT 
                    id,
                    tenant_id,
                    lead_id,
                    transcripts,
                    initiated_by_user_id,
                    agent_id,
                    started_at,
                    status
                FROM {self.SCHEMA}.{self.CALL_LOGS_TABLE}
                WHERE id = $1::uuid
            """, str(call_log_id))
            
            if not call_data:
                return None
            
            # Convert UUID objects to strings (asyncpg returns UUID objects)
            return {
                "id": str(call_data['id']) if call_data['id'] else None,
                "tenant_id": str(call_data['tenant_id']) if call_data['tenant_id'] else None,
                "lead_id": str(call_data['lead_id']) if call_data['lead_id'] else None,
                "transcripts": call_data['transcripts'],
                "initiated_by_user_id": str(call_data['initiated_by_user_id']) if call_data['initiated_by_user_id'] else None,
                "agent_id": call_data['agent_id'],  # Keep as-is for database queries
                "started_at": call_data['started_at'],
                "status": call_data.get('status')  # Status from voice_call_logs (e.g., "declined", "no_answer", etc.)
            }
        finally:
            await self._return_connection(conn)
    
    async def get_voice_id_from_agent_id(self, agent_id) -> Optional[str]:
        """
        Get voice_id from voice_agents table by matching agent_id (async)
        
        Args:
            agent_id: Agent ID (can be integer or UUID)
        
        Returns:
            voice_id string or None if not found
        """
        if not agent_id:
            return None
        
        conn = await self._get_connection()
        
        try:
            agent_id_str = str(agent_id).strip()
            
            # Try as integer first (most common case)
            try:
                agent_id_int = int(agent_id)
                result = await conn.fetchval(f"""
                    SELECT voice_id 
                    FROM {self.SCHEMA}.{self.VOICE_AGENTS_TABLE} 
                    WHERE id = $1
                """, agent_id_int)
                if result:
                    return str(result) if result else None
            except (ValueError, TypeError):
                # Not an integer, continue to other methods
                pass
            except Exception as e:
                logger.debug(f"Integer query failed: {e}")
            
            # Check if it looks like a UUID (has dashes and is 36 chars)
            is_uuid = len(agent_id_str) == 36 and agent_id_str.count('-') == 4
            
            if is_uuid:
                # Try as UUID
                try:
                    result = await conn.fetchval(f"""
                        SELECT voice_id 
                        FROM {self.SCHEMA}.{self.VOICE_AGENTS_TABLE} 
                        WHERE id = $1::uuid
                    """, agent_id_str)
                    if result:
                        return str(result) if result else None
                except Exception as e:
                    logger.debug(f"UUID query failed: {e}")
            
            # Try as text match (cast id to text) - this should work for any type
            try:
                result = await conn.fetchval(f"""
                    SELECT voice_id 
                    FROM {self.SCHEMA}.{self.VOICE_AGENTS_TABLE} 
                    WHERE id::text = $1
                """, agent_id_str)
                if result:
                    return str(result) if result else None
            except Exception as e:
                logger.debug(f"Text query failed: {e}")
            
            return None
        finally:
            await self._return_connection(conn)
    
    async def list_calls(self, limit: Optional[int] = 100) -> List[Dict]:
        """
        List all calls from voice_call_logs (async)
        
        Args:
            limit: Maximum number of calls to return (None for all)
        
        Returns:
            List of call dictionaries
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            if limit is None:
                # No limit - get all calls
                rows = await conn.fetch(f"""
                    SELECT 
                        id,
                        tenant_id,
                        lead_id,
                        started_at,
                        initiated_by_user_id,
                        agent_id
                    FROM {self.SCHEMA}.{self.CALL_LOGS_TABLE}
                    ORDER BY started_at DESC
                """)
            else:
                rows = await conn.fetch(f"""
                    SELECT 
                        id,
                        tenant_id,
                        lead_id,
                        started_at,
                        initiated_by_user_id,
                        agent_id
                    FROM {self.SCHEMA}.{self.CALL_LOGS_TABLE}
                    ORDER BY started_at DESC
                    LIMIT $1
                """, limit)
            
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
        finally:
            await self._return_connection(conn)
    
    async def get_latest_booking_by_lead_id(self, lead_id: str) -> Optional[Dict]:
        """
        Get the latest booking for a lead_id from lead_bookings table
        
        Args:
            lead_id: Lead ID (UUID string)
        
        Returns:
            Dictionary with latest booking data or None if not found
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            booking_data = await conn.fetchrow(f"""
                SELECT 
                    id,
                    tenant_id,
                    lead_id,
                    scheduled_at,
                    created_at,
                    parent_booking_id
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE lead_id = $1::uuid
                AND is_deleted = false
                ORDER BY created_at DESC
                LIMIT 1
            """, str(lead_id))
            
            if not booking_data:
                return None
            
            # Convert UUID objects to strings
            return {
                "id": str(booking_data['id']) if booking_data['id'] else None,
                "tenant_id": str(booking_data['tenant_id']) if booking_data['tenant_id'] else None,
                "lead_id": str(booking_data['lead_id']) if booking_data['lead_id'] else None,
                "scheduled_at": booking_data['scheduled_at'],
                "created_at": booking_data['created_at'],
                "parent_booking_id": str(booking_data['parent_booking_id']) if booking_data['parent_booking_id'] else None
            }
        finally:
            await self._return_connection(conn)
    
    async def get_original_booking_by_lead_id(self, lead_id: str) -> Optional[Dict]:
        """
        Get the original booking for a lead_id (where parent_booking_id IS NULL)
        This is the first booking for this lead.
        
        Args:
            lead_id: Lead ID (UUID string)
        
        Returns:
            Dictionary with original booking data or None if not found
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            booking_data = await conn.fetchrow(f"""
                SELECT 
                    id,
                    tenant_id,
                    lead_id,
                    scheduled_at,
                    created_at,
                    parent_booking_id,
                    metadata
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE lead_id = $1::uuid
                AND parent_booking_id IS NULL
                AND is_deleted = false
                ORDER BY created_at ASC
                LIMIT 1
            """, str(lead_id))
            
            if not booking_data:
                return None
            
            # Parse metadata to get call_id if it exists
            metadata = booking_data.get('metadata')
            call_id_from_metadata = None
            if metadata:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                call_id_from_metadata = metadata.get('call_id') if isinstance(metadata, dict) else None
            
            # Convert UUID objects to strings
            return {
                "id": str(booking_data['id']) if booking_data['id'] else None,
                "tenant_id": str(booking_data['tenant_id']) if booking_data['tenant_id'] else None,
                "lead_id": str(booking_data['lead_id']) if booking_data['lead_id'] else None,
                "scheduled_at": booking_data['scheduled_at'],
                "created_at": booking_data['created_at'],
                "parent_booking_id": str(booking_data['parent_booking_id']) if booking_data['parent_booking_id'] else None,
                "call_id": call_id_from_metadata  # call_id from metadata
            }
        finally:
            await self._return_connection(conn)
    
    async def get_booking_by_call_id_in_metadata(self, call_id: str) -> Optional[Dict]:
        """
        Get a booking by call_id stored in metadata column.
        This finds the booking where metadata->>'call_id' matches the given call_id.
        Prioritizes the original booking (where parent_booking_id IS NULL).
        
        Args:
            call_id: Call ID (UUID string) stored in metadata
        
        Returns:
            Dictionary with booking data or None if not found
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            # First, try to find the original booking (where parent_booking_id IS NULL)
            booking_data = await conn.fetchrow(f"""
                SELECT 
                    id,
                    tenant_id,
                    lead_id,
                    scheduled_at,
                    created_at,
                    parent_booking_id,
                    metadata
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE metadata->>'call_id' = $1
                AND parent_booking_id IS NULL
                AND is_deleted = false
                ORDER BY created_at ASC
                LIMIT 1
            """, str(call_id))
            
            # If not found, try to find any booking with that call_id
            if not booking_data:
                booking_data = await conn.fetchrow(f"""
                    SELECT 
                        id,
                        tenant_id,
                        lead_id,
                        scheduled_at,
                        created_at,
                        parent_booking_id,
                        metadata
                    FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                    WHERE metadata->>'call_id' = $1
                    AND is_deleted = false
                    ORDER BY created_at ASC
                    LIMIT 1
                """, str(call_id))
            
            if not booking_data:
                return None
            
            # Convert UUID objects to strings
            return {
                "id": str(booking_data['id']) if booking_data['id'] else None,
                "tenant_id": str(booking_data['tenant_id']) if booking_data['tenant_id'] else None,
                "lead_id": str(booking_data['lead_id']) if booking_data['lead_id'] else None,
                "scheduled_at": booking_data['scheduled_at'],
                "created_at": booking_data['created_at'],
                "parent_booking_id": str(booking_data['parent_booking_id']) if booking_data['parent_booking_id'] else None
            }
        finally:
            await self._return_connection(conn)
    
    async def count_bookings_by_lead_id(self, lead_id: str) -> int:
        """
        Count existing bookings for a lead_id
        
        Args:
            lead_id: Lead ID (UUID string)
        
        Returns:
            Count of existing bookings (integer)
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            count = await conn.fetchval(f"""
                SELECT COUNT(*)
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE lead_id = $1::uuid
                AND is_deleted = false
            """, str(lead_id))
            
            return count if count is not None else 0
        finally:
            await self._return_connection(conn)
    
    async def get_max_retry_count_by_lead_id(self, lead_id: str) -> int:
        """
        Get the maximum retry_count for a lead_id
        
        Args:
            lead_id: Lead ID (UUID string)
        
        Returns:
            Maximum retry_count (integer), or 0 if no bookings exist
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            max_retry = await conn.fetchval(f"""
                SELECT MAX(retry_count)
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE lead_id = $1::uuid
                AND is_deleted = false
            """, str(lead_id))
            
            return max_retry if max_retry is not None else 0
        finally:
            await self._return_connection(conn)
    
    async def count_bookings_by_parent_booking_id(self, parent_booking_id: str) -> int:
        """
        Count existing bookings for a parent_booking_id
        
        Args:
            parent_booking_id: Parent booking ID (UUID string)
        
        Returns:
            Count of existing bookings (integer)
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            count = await conn.fetchval(f"""
                SELECT COUNT(*)
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE parent_booking_id = $1::uuid
                AND is_deleted = false
            """, str(parent_booking_id))
            
            return count if count is not None else 0
        finally:
            await self._return_connection(conn)
    
    async def count_bookings_by_call_id_in_metadata(self, call_id: str) -> int:
        """
        Count existing bookings with the same call_id in metadata
        
        Args:
            call_id: Call ID (UUID string) stored in metadata
        
        Returns:
            Count of existing bookings (integer)
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            count = await conn.fetchval(f"""
                SELECT COUNT(*)
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE metadata->>'call_id' = $1
                AND is_deleted = false
            """, str(call_id))
            
            return count if count is not None else 0
        finally:
            await self._return_connection(conn)
    
    async def get_max_retry_count_by_call_id_in_metadata(self, call_id: str) -> int:
        """
        Get the maximum retry_count for bookings with the same call_id in metadata
        
        Args:
            call_id: Call ID (UUID string) stored in metadata
        
        Returns:
            Maximum retry_count (integer), or 0 if no bookings exist
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            max_retry = await conn.fetchval(f"""
                SELECT MAX(retry_count)
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE metadata->>'call_id' = $1
                AND is_deleted = false
            """, str(call_id))
            
            return max_retry if max_retry is not None else 0
        finally:
            await self._return_connection(conn)
    
    async def get_latest_booking_by_parent_booking_id(self, parent_booking_id: str) -> Optional[Dict]:
        """
        Get the latest booking for a parent_booking_id from lead_bookings table
        
        Args:
            parent_booking_id: Parent booking ID (UUID string)
        
        Returns:
            Dictionary with latest booking data or None if not found
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        conn = await self._get_connection()
        
        try:
            booking_data = await conn.fetchrow(f"""
                SELECT 
                    id,
                    tenant_id,
                    lead_id,
                    scheduled_at,
                    created_at,
                    parent_booking_id,
                    metadata
                FROM {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE}
                WHERE parent_booking_id = $1::uuid
                AND is_deleted = false
                ORDER BY created_at DESC
                LIMIT 1
            """, str(parent_booking_id))
            
            if not booking_data:
                return None
            
            # Parse metadata to get call_id if it exists
            metadata = booking_data.get('metadata')
            call_id_from_metadata = None
            if metadata:
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                call_id_from_metadata = metadata.get('call_id') if isinstance(metadata, dict) else None
            
            # Convert UUID objects to strings
            return {
                "id": str(booking_data['id']) if booking_data['id'] else None,
                "tenant_id": str(booking_data['tenant_id']) if booking_data['tenant_id'] else None,
                "lead_id": str(booking_data['lead_id']) if booking_data['lead_id'] else None,
                "scheduled_at": booking_data['scheduled_at'],
                "created_at": booking_data['created_at'],
                "parent_booking_id": str(booking_data['parent_booking_id']) if booking_data['parent_booking_id'] else None,
                "call_id": call_id_from_metadata  # call_id from metadata
            }
        finally:
            await self._return_connection(conn)
    
    async def _ensure_lead_assigned(self, conn, lead_id: str, assigned_user_id: str, tenant_id: Optional[str] = None):
        """
        Ensure lead is assigned to a user before saving booking.
        If lead is not assigned, try to assign it using assigned_user_id.
        
        The trigger function `set_booking_assigned_user()` looks for `assigned_user_id` column
        in the leads table, so we must update that column specifically.
        
        Args:
            conn: Database connection
            lead_id: Lead ID
            assigned_user_id: User ID to assign the lead to
            tenant_id: Tenant ID (optional, for multi-tenant support)
        """
        try:
            # First, try to find the leads table by querying information_schema
            leads_table_info = await self._find_leads_table(conn)
            
            if not leads_table_info:
                logger.warning(f"Could not find leads table. Skipping lead assignment check.")
                return
            
            table_name = leads_table_info['table_name']
            schema_name = leads_table_info['schema_name']
            assignment_column = leads_table_info.get('assignment_column')
            
            # Check if lead exists and what columns are available
            lead_data = None
            has_assigned_user_id_col = False
            has_user_id_col = False
            
            # First, check what columns exist in the leads table
            try:
                columns_query = """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = $1
                    AND table_name = $2
                    AND column_name IN ('assigned_user_id', 'user_id')
                """
                columns = await conn.fetch(columns_query, schema_name, table_name)
                column_names = [col['column_name'] for col in columns]
                has_assigned_user_id_col = 'assigned_user_id' in column_names
                has_user_id_col = 'user_id' in column_names
            except Exception as e:
                logger.debug(f"Could not check columns: {e}")
            
            # Query lead data to check current assignment
            query_columns = ['id']
            if has_assigned_user_id_col:
                query_columns.append('assigned_user_id')
            if has_user_id_col:
                query_columns.append('user_id')
            if assignment_column and assignment_column not in query_columns:
                query_columns.append(assignment_column)
            
            query = f"""
                SELECT {', '.join(query_columns)}
                FROM {schema_name}.{table_name}
                WHERE id = $1::uuid
            """
            
            try:
                lead_data = await conn.fetchrow(query, lead_id)
            except Exception as e:
                logger.debug(f"Query failed: {e}")
                lead_data = None
            
            # Check if lead is already assigned (check assigned_user_id first, then user_id, then assignment_column)
            lead_assigned = None
            if lead_data:
                if has_assigned_user_id_col and lead_data.get('assigned_user_id'):
                    lead_assigned = lead_data.get('assigned_user_id')
                elif has_user_id_col and lead_data.get('user_id'):
                    lead_assigned = lead_data.get('user_id')
                elif assignment_column and lead_data.get(assignment_column):
                    lead_assigned = lead_data.get(assignment_column)
                
                if lead_assigned:
                    logger.debug(f"Lead {lead_id} is already assigned to user {lead_assigned}")
                    # Even if assigned via user_id, we should ensure assigned_user_id is also set
                    # because the trigger specifically looks for assigned_user_id
                    if has_assigned_user_id_col and not lead_data.get('assigned_user_id'):
                        logger.info(f"Lead {lead_id} has user_id but assigned_user_id is NULL. Updating assigned_user_id.")
                        # Update assigned_user_id to match user_id
                        update_query = f"""
                            UPDATE {schema_name}.{table_name}
                            SET assigned_user_id = $1::uuid
                            WHERE id = $2::uuid
                        """
                        try:
                            await conn.execute(update_query, lead_assigned, lead_id)
                            logger.info(f"Updated assigned_user_id for lead {lead_id}")
                        except Exception as e:
                            logger.warning(f"Could not update assigned_user_id: {e}")
                    return  # Lead is already assigned
            
            # If lead is not assigned and we have assigned_user_id, try to assign it
            if not lead_assigned and assigned_user_id:
                logger.info(f"Lead {lead_id} is not assigned. Attempting to assign to user {assigned_user_id}")
                
                # Build UPDATE query to set all relevant assignment columns
                update_parts = []
                update_values = []
                param_num = 1
                
                # The trigger specifically looks for assigned_user_id, so update that first
                if has_assigned_user_id_col:
                    update_parts.append(f"assigned_user_id = ${param_num}::uuid")
                    update_values.append(assigned_user_id)
                    param_num += 1
                
                # Also update user_id if it exists (for consistency)
                if has_user_id_col:
                    update_parts.append(f"user_id = ${param_num}::uuid")
                    update_values.append(assigned_user_id)
                    param_num += 1
                
                # Update the assignment_column if it's different from the above
                if assignment_column and assignment_column not in ['assigned_user_id', 'user_id']:
                    update_parts.append(f"{assignment_column} = ${param_num}::uuid")
                    update_values.append(assigned_user_id)
                    param_num += 1
                
                if update_parts:
                    # Add lead_id as the last parameter
                    update_values.append(lead_id)
                    
                    update_query = f"""
                        UPDATE {schema_name}.{table_name}
                        SET {', '.join(update_parts)}
                        WHERE id = ${param_num}::uuid
                    """
                    
                    try:
                        result = await conn.execute(update_query, *update_values)
                        if result and "UPDATE" in result:
                            # Verify the assignment was successful by checking assigned_user_id specifically
                            if has_assigned_user_id_col:
                                verify_query = f"""
                                    SELECT assigned_user_id
                                    FROM {schema_name}.{table_name}
                                    WHERE id = $1::uuid
                                """
                            elif has_user_id_col:
                                verify_query = f"""
                                    SELECT user_id
                                    FROM {schema_name}.{table_name}
                                    WHERE id = $1::uuid
                                """
                            else:
                                verify_query = f"""
                                    SELECT {assignment_column}
                                    FROM {schema_name}.{table_name}
                                    WHERE id = $1::uuid
                                """
                            
                            verify_row = await conn.fetchrow(verify_query, lead_id)
                            if verify_row:
                                verified_value = None
                                if has_assigned_user_id_col:
                                    verified_value = verify_row.get('assigned_user_id')
                                elif has_user_id_col:
                                    verified_value = verify_row.get('user_id')
                                elif assignment_column:
                                    verified_value = verify_row.get(assignment_column)
                                
                                if verified_value:
                                    logger.info(f"Successfully assigned lead {lead_id} to user {assigned_user_id} and verified")
                                    return
                                else:
                                    logger.warning(f"Assignment update executed but verification shows NULL. Lead may not exist in {schema_name}.{table_name}")
                            else:
                                logger.warning(f"Assignment update executed but verification query returned no rows. Lead may not exist in {schema_name}.{table_name}")
                        else:
                            logger.warning(f"Assignment update did not affect any rows. Lead {lead_id} may not exist in {schema_name}.{table_name}")
                    except Exception as e:
                        logger.warning(f"Could not assign lead {lead_id} to user {assigned_user_id}: {e}")
                else:
                    logger.warning(f"No assignment columns found to update for lead {lead_id}")
            elif not lead_assigned:
                logger.warning(f"Lead {lead_id} is not assigned and no assigned_user_id provided. Database may reject the booking.")
                
        except Exception as e:
            logger.warning(f"Error checking/assigning lead: {e}. Proceeding with booking save attempt.")
            # Don't raise - continue with booking save attempt
    
    async def _find_leads_table(self, conn) -> Optional[Dict]:
        """
        Find the leads table by querying information_schema.
        Looks for tables that might contain lead assignments.
        
        Returns:
            Dictionary with table_name, schema_name, and assignment_column, or None if not found
        """
        try:
            # First, try to find exact 'leads' table name (highest priority)
            exact_leads_tables = [
                (self.SCHEMA, 'leads'),
                ('public', 'leads'),
            ]
            
            assignment_columns = ['assigned_to', 'assigned_user_id', 'user_id', 'owner_id', 'assigned_by', 'created_by']
            
            for schema, table in exact_leads_tables:
                try:
                    # Check if table exists and has assignment column
                    columns_query = """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_schema = $1
                        AND table_name = $2
                        AND (column_name = 'id' OR column_name IN ('assigned_to', 'assigned_user_id', 'user_id', 'owner_id', 'assigned_by', 'created_by'))
                    """
                    columns = await conn.fetch(columns_query, schema, table)
                    
                    has_id = any(col['column_name'] == 'id' for col in columns)
                    assignment_col = None
                    
                    for col in columns:
                        if col['column_name'] in assignment_columns:
                            assignment_col = col['column_name']
                            break
                    
                    if has_id and assignment_col:
                        logger.info(f"Found exact leads table: {schema}.{table} with assignment column: {assignment_col}")
                        return {
                            'schema_name': schema,
                            'table_name': table,
                            'assignment_column': assignment_col
                        }
                except:
                    continue
            
            # If exact 'leads' table not found, search for tables with 'lead' in name
            # Exclude 'lead_bookings' and 'lead_notes' as those are not the main leads table
            tables_query = """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE (table_name LIKE '%lead%' OR table_name LIKE '%student%')
                AND table_name NOT LIKE '%booking%'
                AND table_name NOT LIKE '%note%'
                AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY 
                    CASE WHEN table_name = 'leads' THEN 1 ELSE 2 END,
                    table_schema, table_name
            """
            
            tables = await conn.fetch(tables_query)
            
            # Common assignment column names
            assignment_columns = ['assigned_to', 'assigned_user_id', 'user_id', 'owner_id', 'assigned_by', 'created_by']
            
            for table in tables:
                schema_name = table['table_schema']
                table_name = table['table_name']
                
                # Check if this table has an id column and any assignment column
                columns_query = """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = $1
                    AND table_name = $2
                    AND (column_name = 'id' OR column_name IN ('assigned_to', 'assigned_user_id', 'user_id', 'owner_id', 'assigned_by', 'created_by'))
                """
                
                columns = await conn.fetch(columns_query, schema_name, table_name)
                
                has_id = any(col['column_name'] == 'id' for col in columns)
                assignment_col = None
                
                for col in columns:
                    if col['column_name'] in assignment_columns:
                        assignment_col = col['column_name']
                        break
                
                # Exclude lead_bookings and lead_notes - these are not the main leads table
                if has_id and assignment_col and 'note' not in table_name.lower() and 'booking' not in table_name.lower():
                    logger.info(f"Found leads table: {schema_name}.{table_name} with assignment column: {assignment_col}")
                    return {
                        'schema_name': schema_name,
                        'table_name': table_name,
                        'assignment_column': assignment_col
                    }
            
            # If not found, try common table names directly
            # Exclude lead_bookings and lead_notes as those are not the main leads table
            # Prioritize exact 'leads' table name first
            common_tables = [
                (self.SCHEMA, 'leads'),  # Try exact 'leads' table first (most likely)
                ('public', 'leads'),
                (self.SCHEMA, 'lead'),
                ('public', 'lead'),
            ]
            
            # Also try education_students as it might be the leads table
            # But exclude lead_bookings and lead_notes explicitly
            if 'lead_bookings' not in [t[1] for t in common_tables] and 'lead_notes' not in [t[1] for t in common_tables]:
                common_tables.append((self.SCHEMA, 'education_students'))
            
            for schema, table in common_tables:
                try:
                    # Try to query the table to see if it exists
                    test_query = f"SELECT id FROM {schema}.{table} LIMIT 1"
                    await conn.fetchrow(test_query)
                    
                    # If successful, try to find assignment column
                    # Skip if this is the bookings table
                    if 'booking' in table.lower():
                        continue
                    
                    for col in assignment_columns:
                        try:
                            test_col_query = f"SELECT {col} FROM {schema}.{table} LIMIT 1"
                            await conn.fetchrow(test_col_query)
                            logger.info(f"Found leads table: {schema}.{table} with assignment column: {col}")
                            return {
                                'schema_name': schema,
                                'table_name': table,
                                'assignment_column': col
                            }
                        except:
                            continue
                except:
                    continue
            
            logger.warning("Could not find leads table with assignment column")
            return None
            
        except Exception as e:
            logger.warning(f"Error finding leads table: {e}")
            return None
    
    async def _find_constraint_table(self, conn, lead_id: str) -> Optional[str]:
        """
        Try to find which table the constraint is checking by querying database constraints.
        This helps identify the correct table that needs lead assignment.
        """
        try:
            # Query to find constraints on lead_bookings table that reference leads
            constraint_query = """
                SELECT 
                    tc.table_schema,
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_schema AS foreign_table_schema,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                LEFT JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.table_schema = $1
                AND tc.table_name = $2
                AND tc.constraint_type IN ('FOREIGN KEY', 'CHECK')
            """
            
            constraints = await conn.fetch(constraint_query, self.SCHEMA, self.LEAD_BOOKINGS_TABLE)
            
            for constraint in constraints:
                if constraint['foreign_table_name']:
                    logger.info(f"Found constraint referencing table: {constraint['foreign_table_schema']}.{constraint['foreign_table_name']}")
                    return f"{constraint['foreign_table_schema']}.{constraint['foreign_table_name']}"
            
            return None
        except Exception as e:
            logger.debug(f"Error finding constraint table: {e}")
            return None
    
    async def save_booking(self, booking_data: Dict) -> Optional[str]:
        """
        Save booking to lead_bookings table in database
        
        Args:
            booking_data: Dictionary with booking data (all fields from lead_bookings table)
        
        Returns:
            Booking ID (UUID string) if successful, None otherwise
        
        Raises:
            LeadBookingsStorageError: If required fields are missing or save fails
        """
        if not DB_AVAILABLE:
            raise ImportError("asyncpg not installed")
        
        # Validate required fields before attempting to save
        lead_id = booking_data.get('lead_id')
        tenant_id = booking_data.get('tenant_id')
        assigned_user_id = booking_data.get('assigned_user_id')
        
        if not lead_id:
            error_msg = "lead_id is required but is null. Cannot save to database."
            logger.warning(error_msg)
            raise LeadBookingsStorageError(error_msg)
        
        if not tenant_id:
            logger.warning("tenant_id is null, but proceeding with save")
        
        # Note: assigned_user_id can be null - the database constraint checks if lead is assigned
        # If the database rejects due to "lead not assigned to any user", that's a database-level
        # business rule that cannot be bypassed from application code
        if not assigned_user_id:
            logger.warning("assigned_user_id is null - database may reject if lead is not assigned")
        
        conn = await self._get_connection()
        
        try:
            # Check if lead is assigned to a user, and assign if not assigned but assigned_user_id is provided
            if assigned_user_id:
                await self._ensure_lead_assigned(conn, lead_id, assigned_user_id, tenant_id)
            # Helper function to convert datetime strings to timezone-naive datetime objects
            # (PostgreSQL timestamp columns without timezone need naive datetimes)
            def parse_datetime(dt_str):
                """Parse datetime string to timezone-naive datetime object"""
                if dt_str is None:
                    return None
                if isinstance(dt_str, datetime):
                    # If it's already a datetime, convert to naive (remove timezone)
                    if dt_str.tzinfo is not None:
                        # Convert to UTC first, then remove timezone info
                        dt_utc = dt_str.astimezone(pytz.UTC)
                        return dt_utc.replace(tzinfo=None)
                    return dt_str
                try:
                    dt_obj = None
                    # Try parsing ISO format
                    if 'T' in str(dt_str):
                        dt_obj = datetime.fromisoformat(str(dt_str).replace('Z', '+00:00'))
                    else:
                        # Try parsing YYYY-MM-DD HH:MM:SS format
                        dt_obj = datetime.strptime(str(dt_str), "%Y-%m-%d %H:%M:%S")
                    
                    # Convert to timezone-naive if it's timezone-aware
                    if dt_obj and dt_obj.tzinfo is not None:
                        # Convert to UTC first, then remove timezone info
                        dt_utc = dt_obj.astimezone(pytz.UTC)
                        return dt_utc.replace(tzinfo=None)
                    
                    return dt_obj
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse datetime: {dt_str}, error: {e}")
                    return None
            
            # Insert booking into lead_bookings table
            booking_id = await conn.fetchval(f"""
                INSERT INTO {self.SCHEMA}.{self.LEAD_BOOKINGS_TABLE} (
                    id,
                    tenant_id,
                    lead_id,
                    assigned_user_id,
                    booking_type,
                    booking_source,
                    scheduled_at,
                    timezone,
                    status,
                    call_result,
                    retry_count,
                    parent_booking_id,
                    notes,
                    metadata,
                    created_by,
                    created_at,
                    updated_at,
                    is_deleted,
                    buffer_until
                ) VALUES (
                    $1::uuid,
                    $2::uuid,
                    $3::uuid,
                    $4::uuid,
                    $5,
                    $6,
                    $7::timestamp,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12::uuid,
                    $13,
                    $14::jsonb,
                    $15,
                    $16::timestamp,
                    $17::timestamp,
                    $18,
                    $19::timestamp
                )
                RETURNING id
            """,
                booking_data.get('id'),  # $1
                booking_data.get('tenant_id'),  # $2
                booking_data.get('lead_id'),  # $3
                booking_data.get('assigned_user_id'),  # $4
                booking_data.get('booking_type'),  # $5
                booking_data.get('booking_source'),  # $6
                parse_datetime(booking_data.get('scheduled_at')),  # $7
                booking_data.get('timezone'),  # $8
                booking_data.get('status'),  # $9
                booking_data.get('call_result'),  # $10
                booking_data.get('retry_count'),  # $11
                booking_data.get('parent_booking_id'),  # $12
                booking_data.get('notes'),  # $13
                json.dumps(booking_data.get('metadata')) if booking_data.get('metadata') else None,  # $14 - Convert dict to JSON string for JSONB
                booking_data.get('created_by'),  # $15
                parse_datetime(booking_data.get('created_at')),  # $16
                parse_datetime(booking_data.get('updated_at')),  # $17
                booking_data.get('is_deleted', False),  # $18
                parse_datetime(booking_data.get('buffer_until'))  # $19
            )
            
            if booking_id:
                logger.info(f"Saved booking to database: {booking_id}")
                return str(booking_id)
            else:
                logger.warning("Failed to save booking to database - no ID returned")
                return None
                
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's the "lead not assigned" error
            if "not assigned to any user" in error_msg or "Booking not allowed" in error_msg:
                logger.error(f"Database constraint error: Lead must be assigned to a user before booking can be created.")
                logger.error(f"Lead ID: {lead_id}, Assigned User ID: {assigned_user_id}")
                logger.error(f"Note: The constraint may be checking a different table than the one we updated.")
                logger.error(f"Please manually verify the lead assignment in the leads table.")
                
                # Try to find what table the constraint is checking
                try:
                    constraint_info = await self._find_constraint_table(conn, lead_id)
                    if constraint_info:
                        logger.info(f"Constraint may be checking table: {constraint_info}")
                except:
                    pass
                
                raise LeadBookingsStorageError(f"Lead {lead_id} is not assigned to any user. Please assign the lead first.")
            else:
                logger.error(f"Error saving booking to database: {e}", exc_info=True)
                raise LeadBookingsStorageError(f"Failed to save booking: {e}")
        finally:
            await self._return_connection(conn)

