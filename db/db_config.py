"""
Database Configuration Module
=============================

Centralized database configuration with support for local/prod environments.

Environment Variables:
    USE_LOCAL_DB: Set to 'true' to use local database (default: false)
    
    Production DB (when USE_LOCAL_DB=false):
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    
    Local DB (when USE_LOCAL_DB=true):
        LOCAL_DB_HOST, LOCAL_DB_PORT, LOCAL_DB_NAME, LOCAL_DB_USER, LOCAL_DB_PASSWORD
        (Falls back to localhost:5432/voice_agent_local if not set)

Usage:
    from db.db_config import get_db_config, is_local_db
    
    config = get_db_config()  # Returns appropriate config based on USE_LOCAL_DB
    if is_local_db():
        print("Using local database for testing")
"""

import os
import logging
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT TOGGLE
# =============================================================================

def is_local_db() -> bool:
    """
    Check if local database should be used.
    
    Returns:
        True if USE_LOCAL_DB is set to 'true', '1', or 'yes'
    """
    return os.getenv("USE_LOCAL_DB", "false").lower() in ("true", "1", "yes")


# =============================================================================
# DATABASE CONFIGURATIONS
# =============================================================================

def _get_prod_db_config() -> Dict[str, str | int]:
    """Get production database configuration from environment."""
    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def _get_local_db_config() -> Dict[str, str | int]:
    """
    Get local database configuration from environment.
    Falls back to sensible defaults for local development.
    """
    return {
        "host": os.getenv("LOCAL_DB_HOST", "localhost"),
        "port": int(os.getenv("LOCAL_DB_PORT", "5432")),
        "database": os.getenv("LOCAL_DB_NAME", "voice_agent_local"),
        "user": os.getenv("LOCAL_DB_USER", "postgres"),
        "password": os.getenv("LOCAL_DB_PASSWORD", "postgres"),
    }


def get_db_config() -> Dict[str, str | int]:
    """
    Get the active database configuration based on USE_LOCAL_DB environment variable.
    
    Returns:
        Database configuration dict with host, port, database, user, password
    """
    if is_local_db():
        config = _get_local_db_config()
        logger.debug(
            f"Using LOCAL database: {config['host']}:{config['port']}/{config['database']}"
        )
    else:
        config = _get_prod_db_config()
        logger.debug(
            f"Using PRODUCTION database: {config['host']}:{config['port']}/{config['database']}"
        )
    
    return config


def get_db_config_for_schema(schema: str = "voice_agent") -> Dict[str, str | int]:
    """
    Get database configuration with schema for search_path.
    
    Args:
        schema: Schema name to use (default: voice_agent)
        
    Returns:
        Database configuration dict with options for schema
    """
    config = get_db_config()
    config["options"] = f"-c search_path={schema}"
    return config


# =============================================================================
# VALIDATION
# =============================================================================

def validate_db_config() -> tuple[bool, str]:
    """
    Validate that required database configuration is present.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_db_config()
    
    required = ["host", "database", "user", "password"]
    missing = [key for key in required if not config.get(key)]
    
    if missing:
        db_type = "LOCAL" if is_local_db() else "PRODUCTION"
        prefix = "LOCAL_" if is_local_db() else ""
        missing_vars = [f"{prefix}DB_{k.upper()}" for k in missing]
        return False, f"{db_type} database config missing: {', '.join(missing_vars)}"
    
    return True, ""


# =============================================================================
# LOGGING ON IMPORT
# =============================================================================

if is_local_db():
    config = _get_local_db_config()
    logger.info(
        f"Database mode: LOCAL ({config['host']}:{config['port']}/{config['database']})"
    )
else:
    logger.info("Database mode: PRODUCTION (using DB_* environment variables)")
