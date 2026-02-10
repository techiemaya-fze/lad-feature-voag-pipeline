"""
Voice storage utilities for resolving text-to-speech provider settings.
"""

import logging
import os
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager
from db.connection_pool import get_db_connection, return_connection, USE_CONNECTION_POOLING
from db.schema_constants import VOICES_FULL

load_dotenv()

logger = logging.getLogger(__name__)


class VoiceStorage:
    """Provides read access to voice configuration records."""

    def __init__(self) -> None:
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

        required_vars = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection (pooled or direct based on feature flag)"""
        return get_db_connection(self.db_config)
    
    def _return_connection(self, conn):
        """Return connection to pool if pooling is enabled"""
        if USE_CONNECTION_POOLING:
            return_connection(conn, self.db_config)

    async def get_voice_by_id(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single voice configuration by its primary key."""
        if not voice_id:
            logger.warning("Cannot fetch voice configuration without a voice_id")
            return None

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT * FROM {VOICES_FULL} WHERE id = %s LIMIT 1",
                        (voice_id,),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to fetch voice configuration for %s: %s",
                voice_id,
                exc,
                exc_info=True,
            )
            return None
