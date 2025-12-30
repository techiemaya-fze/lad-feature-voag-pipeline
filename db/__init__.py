"""
Database module.

Contains:
- config: Database configuration (local/prod toggle)
- pool: Connection pool manager
- storage/: All storage classes for DB operations
"""

from db.config import (
    get_db_config,
    get_db_config_for_schema,
    is_local_db,
    validate_db_config,
)
from db.pool import (
    get_db_connection,
    get_raw_connection,
    return_connection,
    close_all_connections,
    get_pool_stats,
    DatabaseConnection,
)

__all__ = [
    # Config
    "get_db_config",
    "get_db_config_for_schema",
    "is_local_db",
    "validate_db_config",
    # Pool
    "get_db_connection",
    "get_raw_connection",
    "return_connection",
    "close_all_connections",
    "get_pool_stats",
    "DatabaseConnection",
]
