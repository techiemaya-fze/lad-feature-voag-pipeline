"""
LiveKit Configuration Storage

Database operations for voice_agent_livekit table.
Stores LiveKit server configurations with encrypted credentials.

Usage:
    storage = LiveKitConfigStorage()
    config = await storage.get_livekit_config(config_id)
"""

import os
import logging
from typing import Any, Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager
from db.connection_pool import get_db_connection, return_connection, USE_CONNECTION_POOLING
from db.db_config import get_db_config

load_dotenv()

logger = logging.getLogger(__name__)

# Schema and table constants
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")
LIVEKIT_TABLE = "voice_agent_livekit"
FULL_LIVEKIT_TABLE = f"{SCHEMA}.{LIVEKIT_TABLE}"


class LiveKitConfigStorage:
    """
    Storage operations for voice_agent_livekit table.
    
    Provides CRUD operations for LiveKit server configurations.
    """

    def __init__(self) -> None:
        self.db_config = get_db_config()

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection (pooled or direct based on feature flag)."""
        return get_db_connection(self.db_config)
    
    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled."""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)

    async def get_livekit_config(self, config_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch LiveKit configuration by UUID.
        
        Args:
            config_id: UUID string
            
        Returns:
            Dict with keys: id, name, description, livekit_url,
            livekit_api_key, livekit_api_secret (encrypted), trunk_id,
            worker_name, created_at, updated_at
            
            Returns None if not found.
        """
        if not config_id:
            logger.debug("get_livekit_config called with empty config_id")
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id, name, description, livekit_url,
                               livekit_api_key, livekit_api_secret, trunk_id,
                               worker_name, created_at, updated_at
                        FROM {FULL_LIVEKIT_TABLE}
                        WHERE id = %s::uuid
                        LIMIT 1
                        """,
                        (config_id,),
                    )
                    result = cur.fetchone()
                    
                    if result:
                        logger.debug(f"Found LiveKit config: {result['name']} (id: {config_id[:8]}...)")
                        return dict(result)
                    else:
                        logger.debug(f"LiveKit config not found: {config_id}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching LiveKit config {config_id}: {e}", exc_info=True)
            return None

    async def get_livekit_config_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """
        Fetch LiveKit configuration by name.
        
        Args:
            name: Configuration name (unique)
            
        Returns:
            Dict with config data or None if not found
        """
        if not name:
            logger.debug("get_livekit_config_by_name called with empty name")
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id, name, description, livekit_url,
                               livekit_api_key, livekit_api_secret, trunk_id,
                               worker_name, created_at, updated_at
                        FROM {FULL_LIVEKIT_TABLE}
                        WHERE name = %s
                        LIMIT 1
                        """,
                        (name,),
                    )
                    result = cur.fetchone()
                    
                    if result:
                        logger.debug(f"Found LiveKit config by name: {name}")
                        return dict(result)
                    else:
                        logger.debug(f"LiveKit config not found by name: {name}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching LiveKit config by name {name}: {e}", exc_info=True)
            return None

    async def create_livekit_config(
        self,
        name: str,
        livekit_url: str,
        livekit_api_key: str,
        livekit_api_secret: str,  # Should be encrypted before passing
        trunk_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a new LiveKit configuration.
        
        Args:
            name: Unique name for the configuration
            livekit_url: LiveKit server URL (e.g., wss://server.livekit.cloud)
            livekit_api_key: LiveKit API key
            livekit_api_secret: Encrypted LiveKit API secret (with dev-s-t- prefix)
            trunk_id: Optional SIP trunk ID
            description: Optional description
            
        Returns:
            UUID of created config, or None on error
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {FULL_LIVEKIT_TABLE}
                        (name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id),
                    )
                    config_id = cur.fetchone()[0]
                    conn.commit()
                    
                    logger.info(f"Created LiveKit config: {name} (id: {config_id})")
                    return str(config_id)
                    
        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error creating LiveKit config (name may already exist): {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating LiveKit config: {e}", exc_info=True)
            return None

    async def update_livekit_config(
        self,
        config_id: str,
        **updates
    ) -> bool:
        """
        Update LiveKit configuration.
        updated_at is automatically updated by trigger.
        
        Args:
            config_id: UUID of config to update
            **updates: Fields to update (name, description, livekit_url, 
                      livekit_api_key, livekit_api_secret, trunk_id)
            
        Returns:
            True if updated, False if not found or error
        """
        if not config_id or not updates:
            logger.debug("update_livekit_config called with empty config_id or no updates")
            return False

        # Build SET clause dynamically
        allowed_fields = {
            'name', 'description', 'livekit_url', 
            'livekit_api_key', 'livekit_api_secret', 'trunk_id',
            'worker_name'
        }
        
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not update_fields:
            logger.debug("No valid fields to update")
            return False

        set_clause = ", ".join([f"{field} = %s" for field in update_fields.keys()])
        values = list(update_fields.values())
        values.append(config_id)  # For WHERE clause

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE {FULL_LIVEKIT_TABLE}
                        SET {set_clause}
                        WHERE id = %s::uuid
                        """,
                        values,
                    )
                    rows_updated = cur.rowcount
                    conn.commit()
                    
                    if rows_updated > 0:
                        logger.info(f"Updated LiveKit config: {config_id[:8]}... (fields: {list(update_fields.keys())})")
                        return True
                    else:
                        logger.debug(f"LiveKit config not found for update: {config_id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error updating LiveKit config {config_id}: {e}", exc_info=True)
            return False

    async def delete_livekit_config(self, config_id: str) -> bool:
        """
        Delete LiveKit configuration.
        
        Args:
            config_id: UUID of config to delete
            
        Returns:
            True if deleted, False if not found or error
        """
        if not config_id:
            logger.debug("delete_livekit_config called with empty config_id")
            return False

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM {FULL_LIVEKIT_TABLE}
                        WHERE id = %s::uuid
                        """,
                        (config_id,),
                    )
                    rows_deleted = cur.rowcount
                    conn.commit()
                    
                    if rows_deleted > 0:
                        logger.info(f"Deleted LiveKit config: {config_id[:8]}...")
                        return True
                    else:
                        logger.debug(f"LiveKit config not found for deletion: {config_id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error deleting LiveKit config {config_id}: {e}", exc_info=True)
            return False

    async def list_livekit_configs(self) -> list[dict[str, Any]]:
        """
        List all LiveKit configurations.
        
        Returns:
            List of config dicts (may be empty)
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT id, name, description, livekit_url,
                               livekit_api_key, livekit_api_secret, trunk_id,
                               worker_name, created_at, updated_at
                        FROM {FULL_LIVEKIT_TABLE}
                        ORDER BY name
                        """
                    )
                    results = cur.fetchall()
                    
                    logger.debug(f"Listed {len(results)} LiveKit configs")
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Error listing LiveKit configs: {e}", exc_info=True)
            return []


# Convenience function for quick access
async def get_livekit_config(config_id: str) -> Optional[dict[str, Any]]:
    """
    Convenience function to get LiveKit config by ID.
    
    Args:
        config_id: UUID string
        
    Returns:
        Config dict or None
    """
    storage = LiveKitConfigStorage()
    return await storage.get_livekit_config(config_id)
