"""
Knowledge Base Storage - Database layer for Google File Search stores.

Provides CRUD operations for knowledge base stores and their associations
with agents and leads for the RAG (Retrieval-Augmented Generation) feature.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from db.connection_pool import get_db_connection, USE_CONNECTION_POOLING, return_connection

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeBaseStorageError(Exception):
    """Exception raised for knowledge base storage errors."""
    pass


class KnowledgeBaseStorage:
    """
    Database storage for knowledge base (File Search) stores and associations.
    
    Manages:
    - knowledge_base_stores: Tracks Gemini FileSearchStore metadata
    - agent_knowledge_base_stores: Links agents to stores
    - lead_knowledge_base_stores: Links leads to stores for per-call customization
    """

    SCHEMA = "voice_agent"
    STORES_TABLE = "knowledge_base_stores"
    AGENT_LINKS_TABLE = "agent_knowledge_base_stores"
    LEAD_LINKS_TABLE = "lead_knowledge_base_stores"

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [env for env in required if not os.getenv(env)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _get_connection(self):
        """Get database connection (pooled or direct based on feature flag)."""
        return get_db_connection(self.db_config)

    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled."""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)

    async def ensure_tables_exist(self) -> None:
        """Create knowledge base tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Main stores table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.STORES_TABLE} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            gemini_store_name TEXT NOT NULL UNIQUE,
                            display_name TEXT NOT NULL,
                            description TEXT,
                            document_count INTEGER DEFAULT 0,
                            is_active BOOLEAN DEFAULT TRUE,
                            created_by_user_id INTEGER,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)

                    # Agent-to-store links
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.AGENT_LINKS_TABLE} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            agent_id INTEGER NOT NULL,
                            store_id UUID NOT NULL REFERENCES {self.SCHEMA}.{self.STORES_TABLE}(id) ON DELETE CASCADE,
                            is_default BOOLEAN DEFAULT FALSE,
                            priority INTEGER DEFAULT 0,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE(agent_id, store_id)
                        )
                    """)

                    # Lead-to-store links (for per-call customization)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.SCHEMA}.{self.LEAD_LINKS_TABLE} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            lead_id INTEGER NOT NULL,
                            store_id UUID NOT NULL REFERENCES {self.SCHEMA}.{self.STORES_TABLE}(id) ON DELETE CASCADE,
                            priority INTEGER DEFAULT 0,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE(lead_id, store_id)
                        )
                    """)

                    conn.commit()
                    logger.info("Knowledge base tables ensured")
        except Exception as exc:
            logger.error("Failed to create knowledge base tables: %s", exc, exc_info=True)
            raise KnowledgeBaseStorageError(f"Failed to create tables: {exc}") from exc

    # =========================================================================
    # STORE CRUD
    # =========================================================================

    async def create_store(
        self,
        gemini_store_name: str,
        display_name: str,
        description: Optional[str] = None,
        created_by_user_id: Optional[int] = None,
    ) -> str:
        """
        Create a new knowledge base store record.
        
        Args:
            gemini_store_name: The Gemini API store name (e.g., "fileSearchStores/xxx")
            display_name: Human-readable name
            description: Optional description
            created_by_user_id: User who created the store
            
        Returns:
            UUID of the created store record
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.SCHEMA}.{self.STORES_TABLE}
                            (gemini_store_name, display_name, description, created_by_user_id)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id::text
                        """,
                        (gemini_store_name, display_name, description, created_by_user_id),
                    )
                    result = cur.fetchone()
                    conn.commit()
                    store_id = result[0] if result else None
                    logger.info("Created knowledge base store: %s (%s)", display_name, store_id)
                    return store_id
        except psycopg2.IntegrityError as exc:
            logger.error("Store already exists: %s", gemini_store_name)
            raise KnowledgeBaseStorageError(f"Store already exists: {gemini_store_name}") from exc
        except Exception as exc:
            logger.error("Failed to create store: %s", exc, exc_info=True)
            raise KnowledgeBaseStorageError(f"Failed to create store: {exc}") from exc

    async def get_store_by_id(self, store_id: str) -> Optional[dict[str, Any]]:
        """Fetch a store by its database UUID."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id::text, gemini_store_name, display_name, description,
                               document_count, is_active, created_by_user_id,
                               created_at, updated_at
                        FROM {self.SCHEMA}.{self.STORES_TABLE}
                        WHERE id = %s::uuid
                        """,
                        (store_id,),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to get store %s: %s", store_id, exc, exc_info=True)
            return None

    async def get_store_by_gemini_name(self, gemini_store_name: str) -> Optional[dict[str, Any]]:
        """Fetch a store by its Gemini API name."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id::text, gemini_store_name, display_name, description,
                               document_count, is_active, created_by_user_id,
                               created_at, updated_at
                        FROM {self.SCHEMA}.{self.STORES_TABLE}
                        WHERE gemini_store_name = %s
                        """,
                        (gemini_store_name,),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as exc:
            logger.error("Failed to get store by name %s: %s", gemini_store_name, exc, exc_info=True)
            return None

    async def list_stores(
        self,
        active_only: bool = True,
        created_by_user_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """List all knowledge base stores."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    conditions = []
                    params = []

                    if active_only:
                        conditions.append("is_active = TRUE")
                    if created_by_user_id is not None:
                        conditions.append("created_by_user_id = %s")
                        params.append(created_by_user_id)

                    where_clause = " AND ".join(conditions) if conditions else "TRUE"

                    cur.execute(
                        f"""
                        SELECT id::text, gemini_store_name, display_name, description,
                               document_count, is_active, created_by_user_id,
                               created_at, updated_at
                        FROM {self.SCHEMA}.{self.STORES_TABLE}
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        """,
                        params,
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("Failed to list stores: %s", exc, exc_info=True)
            return []

    async def update_store(
        self,
        store_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        document_count: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> bool:
        """Update a store's metadata."""
        updates = []
        params = []

        if display_name is not None:
            updates.append("display_name = %s")
            params.append(display_name)
        if description is not None:
            updates.append("description = %s")
            params.append(description)
        if document_count is not None:
            updates.append("document_count = %s")
            params.append(document_count)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(is_active)

        if not updates:
            return True

        updates.append("updated_at = NOW()")
        params.append(store_id)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {self.SCHEMA}.{self.STORES_TABLE}
                        SET {", ".join(updates)}
                        WHERE id = %s::uuid
                        """,
                        params,
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error("Failed to update store %s: %s", store_id, exc, exc_info=True)
            return False

    async def delete_store(self, store_id: str) -> bool:
        """Delete a store and its associations."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {self.SCHEMA}.{self.STORES_TABLE} WHERE id = %s::uuid",
                        (store_id,),
                    )
                    conn.commit()
                    deleted = cur.rowcount > 0
                    if deleted:
                        logger.info("Deleted knowledge base store: %s", store_id)
                    return deleted
        except Exception as exc:
            logger.error("Failed to delete store %s: %s", store_id, exc, exc_info=True)
            return False

    # =========================================================================
    # AGENT-STORE LINKING
    # =========================================================================

    async def link_store_to_agent(
        self,
        agent_id: int,
        store_id: str,
        is_default: bool = False,
        priority: int = 0,
    ) -> str:
        """Link a knowledge base store to an agent."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # If setting as default, unset other defaults first
                    if is_default:
                        cur.execute(
                            f"""
                            UPDATE {self.SCHEMA}.{self.AGENT_LINKS_TABLE}
                            SET is_default = FALSE
                            WHERE agent_id = %s
                            """,
                            (agent_id,),
                        )

                    cur.execute(
                        f"""
                        INSERT INTO {self.SCHEMA}.{self.AGENT_LINKS_TABLE}
                            (agent_id, store_id, is_default, priority)
                        VALUES (%s, %s::uuid, %s, %s)
                        ON CONFLICT (agent_id, store_id) DO UPDATE
                        SET is_default = EXCLUDED.is_default,
                            priority = EXCLUDED.priority
                        RETURNING id::text
                        """,
                        (agent_id, store_id, is_default, priority),
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return result[0] if result else None
        except Exception as exc:
            logger.error("Failed to link store to agent: %s", exc, exc_info=True)
            raise KnowledgeBaseStorageError(f"Failed to link store to agent: {exc}") from exc

    async def unlink_store_from_agent(self, agent_id: int, store_id: str) -> bool:
        """Remove a store link from an agent."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM {self.SCHEMA}.{self.AGENT_LINKS_TABLE}
                        WHERE agent_id = %s AND store_id = %s::uuid
                        """,
                        (agent_id, store_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error("Failed to unlink store from agent: %s", exc, exc_info=True)
            return False

    async def get_stores_for_agent(self, agent_id: int) -> list[dict[str, Any]]:
        """Get all stores linked to an agent, ordered by priority."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT s.id::text, s.gemini_store_name, s.display_name, s.description,
                               s.document_count, s.is_active, l.is_default, l.priority
                        FROM {self.SCHEMA}.{self.STORES_TABLE} s
                        JOIN {self.SCHEMA}.{self.AGENT_LINKS_TABLE} l ON s.id = l.store_id
                        WHERE l.agent_id = %s AND s.is_active = TRUE
                        ORDER BY l.priority DESC, l.is_default DESC
                        """,
                        (agent_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("Failed to get stores for agent %s: %s", agent_id, exc, exc_info=True)
            return []

    # =========================================================================
    # LEAD-STORE LINKING
    # =========================================================================

    async def link_store_to_lead(
        self,
        lead_id: str,  # UUID
        store_id: str,
        priority: int = 0,
    ) -> str:
        """Link a knowledge base store to a lead for per-call customization."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.SCHEMA}.{self.LEAD_LINKS_TABLE}
                            (lead_id, store_id, priority)
                        VALUES (%s, %s::uuid, %s)
                        ON CONFLICT (lead_id, store_id) DO UPDATE
                        SET priority = EXCLUDED.priority
                        RETURNING id::text
                        """,
                        (lead_id, store_id, priority),
                    )
                    result = cur.fetchone()
                    conn.commit()
                    return result[0] if result else None
        except Exception as exc:
            logger.error("Failed to link store to lead: %s", exc, exc_info=True)
            raise KnowledgeBaseStorageError(f"Failed to link store to lead: {exc}") from exc

    async def unlink_store_from_lead(self, lead_id: str, store_id: str) -> bool:  # lead_id is UUID
        """Remove a store link from a lead."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM {self.SCHEMA}.{self.LEAD_LINKS_TABLE}
                        WHERE lead_id = %s AND store_id = %s::uuid
                        """,
                        (lead_id, store_id),
                    )
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as exc:
            logger.error("Failed to unlink store from lead: %s", exc, exc_info=True)
            return False

    async def get_stores_for_lead(self, lead_id: str) -> list[dict[str, Any]]:  # lead_id is UUID
        """Get all stores linked to a lead, ordered by priority."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT s.id::text, s.gemini_store_name, s.display_name, s.description,
                               s.document_count, s.is_active, l.priority
                        FROM {self.SCHEMA}.{self.STORES_TABLE} s
                        JOIN {self.SCHEMA}.{self.LEAD_LINKS_TABLE} l ON s.id = l.store_id
                        WHERE l.lead_id = %s AND s.is_active = TRUE
                        ORDER BY l.priority DESC
                        """,
                        (lead_id,),
                    )
                    return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("Failed to get stores for lead %s: %s", lead_id, exc, exc_info=True)
            return []

    # =========================================================================
    # CALL-TIME RESOLUTION
    # =========================================================================

    async def get_stores_for_call(
        self,
        agent_id: Optional[int] = None,
        lead_id: Optional[int] = None,
        store_ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Resolve Gemini store names for a call.
        
        Priority order:
        1. Explicit store_ids (if provided)
        2. Lead-specific stores
        3. Agent default stores
        
        Args:
            agent_id: Agent making the call
            lead_id: Target lead
            store_ids: Explicit store UUIDs from call parameters
            
        Returns:
            List of Gemini store names (e.g., ["fileSearchStores/xxx"])
        """
        gemini_names: list[str] = []

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Priority 1: Explicit store IDs
                    if store_ids:
                        placeholders = ",".join(["%s::uuid"] * len(store_ids))
                        cur.execute(
                            f"""
                            SELECT gemini_store_name
                            FROM {self.SCHEMA}.{self.STORES_TABLE}
                            WHERE id IN ({placeholders}) AND is_active = TRUE
                            """,
                            store_ids,
                        )
                        gemini_names = [row[0] for row in cur.fetchall()]
                        if gemini_names:
                            logger.debug("Resolved %d explicit stores for call", len(gemini_names))
                            return gemini_names

                    # Priority 2: Lead-specific stores
                    if lead_id is not None:
                        cur.execute(
                            f"""
                            SELECT s.gemini_store_name
                            FROM {self.SCHEMA}.{self.STORES_TABLE} s
                            JOIN {self.SCHEMA}.{self.LEAD_LINKS_TABLE} l ON s.id = l.store_id
                            WHERE l.lead_id = %s AND s.is_active = TRUE
                            ORDER BY l.priority DESC
                            """,
                            (lead_id,),
                        )
                        gemini_names = [row[0] for row in cur.fetchall()]
                        if gemini_names:
                            logger.debug("Resolved %d lead stores for call", len(gemini_names))
                            return gemini_names

                    # Priority 3: Agent default stores
                    if agent_id is not None:
                        cur.execute(
                            f"""
                            SELECT s.gemini_store_name
                            FROM {self.SCHEMA}.{self.STORES_TABLE} s
                            JOIN {self.SCHEMA}.{self.AGENT_LINKS_TABLE} l ON s.id = l.store_id
                            WHERE l.agent_id = %s AND s.is_active = TRUE
                            ORDER BY l.priority DESC, l.is_default DESC
                            """,
                            (agent_id,),
                        )
                        gemini_names = [row[0] for row in cur.fetchall()]
                        if gemini_names:
                            logger.debug("Resolved %d agent stores for call", len(gemini_names))

                    return gemini_names

        except Exception as exc:
            logger.error("Failed to resolve stores for call: %s", exc, exc_info=True)
            return []
