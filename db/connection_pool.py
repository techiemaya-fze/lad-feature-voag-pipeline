"""
Database Connection Pool Manager
=================================

Provides a centralized, thread-safe connection pool for all database operations.

Features:
- Connection pooling (10 min, 50 max connections)
- Automatic retry with exponential backoff
- Feature flag control via USE_CONNECTION_POOLING env var
- Thread-safe connection management
- Connection timeout configuration
- Graceful error handling

Usage:
    from db.connection_pool import get_db_connection
    
    # In your storage class:
    def _get_connection(self):
        return get_db_connection(self.db_config)
"""

import logging
import os
import time
from contextlib import contextmanager
from threading import Lock
from typing import Optional

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature flag: Set to 'false' or '0' to disable connection pooling
USE_CONNECTION_POOLING = os.getenv("USE_CONNECTION_POOLING", "true").lower() in ("true", "1", "yes")

# Pool configuration
# Start with 1 connection to avoid overwhelming slow servers on startup
# For slow remote servers, keep pool small to avoid connection timeouts
MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN_CONNECTIONS", "1"))
MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX_CONNECTIONS", "10"))  # Reduced from 30 for slow servers

# Connection timeout (seconds) - prevents hanging on slow servers
CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))

# Retry configuration
MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
RETRY_DELAY_BASE = float(os.getenv("DB_RETRY_DELAY_BASE", "2.0"))  # Exponential backoff base


# =============================================================================
# GLOBAL POOL INSTANCE
# =============================================================================

_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = Lock()
_pool_config: Optional[dict] = None


def _initialize_pool(db_config: dict) -> pool.ThreadedConnectionPool:
    """
    Initialize the connection pool (singleton pattern).
    
    Args:
        db_config: Database configuration dict with host, port, database, user, password
        
    Returns:
        Initialized connection pool
    """
    global _pool, _pool_config
    
    with _pool_lock:
        # Check if pool already exists with same config
        if _pool is not None and _pool_config == db_config:
            return _pool
        
        # Close existing pool if config changed
        if _pool is not None:
            logger.info("Database config changed, closing existing pool")
            _pool.closeall()
            _pool = None
        
        # Create new pool with connection timeout
        pool_config = {**db_config, "connect_timeout": CONNECTION_TIMEOUT}
        
        try:
            # First test a single connection to verify connectivity
            logger.debug("Testing database connectivity before creating pool...")
            test_conn = psycopg2.connect(**pool_config)
            test_conn.close()
            logger.debug("Database connectivity verified")
            
            # Now create the pool (minconn=0 would be ideal but ThreadedConnectionPool requires at least 1)
            # Using minconn=1 since we just verified connectivity
            _pool = pool.ThreadedConnectionPool(
                MIN_CONNECTIONS,
                MAX_CONNECTIONS,
                **pool_config
            )
            _pool_config = db_config
            
            logger.info(
                "Database connection pool initialized: "
                f"min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS}, "
                f"timeout={CONNECTION_TIMEOUT}s, host={db_config.get('host')}"
            )
            
            return _pool
            
        except Exception as exc:
            logger.error(f"Failed to initialize connection pool: {exc}", exc_info=True)
            raise


def _get_pool(db_config: dict) -> pool.ThreadedConnectionPool:
    """
    Get or create the connection pool.
    
    Args:
        db_config: Database configuration dict
        
    Returns:
        Connection pool instance
    """
    global _pool
    
    if _pool is None:
        return _initialize_pool(db_config)
    
    return _pool


@contextmanager
def _get_pooled_connection(db_config: dict):
    """
    Get a connection from the pool with automatic return.
    
    Args:
        db_config: Database configuration dict
        
    Yields:
        Database connection from pool
    """
    conn = None
    connection_pool = _get_pool(db_config)
    
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn is not None:
            connection_pool.putconn(conn)


def _create_direct_connection(db_config: dict):
    """
    Create a direct connection (old behavior, no pooling).
    
    Args:
        db_config: Database configuration dict
        
    Returns:
        Direct psycopg2 connection
    """
    pool_config = {**db_config, "connect_timeout": CONNECTION_TIMEOUT}
    return psycopg2.connect(**pool_config)


def _retry_connection(db_config: dict, use_pool: bool = True):
    """
    Attempt to get a connection with retries and exponential backoff.
    
    Args:
        db_config: Database configuration dict
        use_pool: Whether to use connection pooling
        
    Returns:
        Database connection (pooled or direct)
        
    Raises:
        psycopg2.OperationalError: If all retries fail
        pool.PoolError: If pool is exhausted after retries
    """
    global _pool
    last_exception = None
    
    # More retries for pool exhaustion since connections may be returned soon
    max_attempts = MAX_RETRIES * 2 if use_pool else MAX_RETRIES
    
    for attempt in range(1, max_attempts + 1):
        try:
            if use_pool:
                # For pooled connections, we can't use context manager here
                # Caller must handle connection return
                connection_pool = _get_pool(db_config)
                return connection_pool.getconn()
            else:
                return _create_direct_connection(db_config)
        
        except pool.PoolError as exc:
            # Pool exhausted - wait and retry since connections may be returned soon
            last_exception = exc
            if attempt < max_attempts:
                delay = min(1.0 * attempt, 5.0)  # 1s, 2s, 3s, 4s, 5s (capped)
                logger.warning(
                    f"Connection pool exhausted (attempt {attempt}/{max_attempts}). "
                    f"Waiting {delay:.1f}s for connections to be returned..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Connection pool exhausted after {max_attempts} attempts: {exc}",
                    exc_info=True
                )
                
        except psycopg2.OperationalError as exc:
            last_exception = exc
            
            # If pool initialization failed, try falling back to direct connection
            if use_pool and _pool is None and attempt == max_attempts:
                logger.warning(
                    "Connection pool failed to initialize, falling back to direct connection"
                )
                try:
                    return _create_direct_connection(db_config)
                except Exception as fallback_exc:
                    logger.error(f"Direct connection fallback also failed: {fallback_exc}")
                    last_exception = fallback_exc
            
            if attempt < max_attempts:
                delay = RETRY_DELAY_BASE ** min(attempt, 3)  # Capped at 8s
                logger.warning(
                    f"Database connection failed (attempt {attempt}/{max_attempts}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Database connection failed after {max_attempts} attempts: {exc}",
                    exc_info=True
                )
    
    # All retries failed
    raise last_exception


# =============================================================================
# CONTEXT MANAGER WRAPPER (moved before PUBLIC API for proper ordering)
# =============================================================================

class DatabaseConnection:
    """
    Context manager for database connections with automatic cleanup.
    
    IMPORTANT: This properly returns connections to the pool when used with 'with' statements.
    This also validates connections before use to prevent stale connection errors.
    
    Usage:
        with DatabaseConnection(db_config) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
    """
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None
        self._use_pool = USE_CONNECTION_POOLING
    
    def _is_connection_valid(self, conn) -> bool:
        """Check if a connection is still valid by running a test query."""
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False
    
    def __enter__(self):
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                if self._use_pool:
                    self.conn = _retry_connection(self.db_config, use_pool=True)
                else:
                    self.conn = _retry_connection(self.db_config, use_pool=False)
                
                # Validate the connection is alive
                if not self._is_connection_valid(self.conn):
                    logger.warning(f"Stale connection detected (attempt {attempt + 1}/{max_attempts}), getting fresh connection")
                    
                    # Return stale connection to pool (it will be discarded)
                    if self._use_pool:
                        try:
                            connection_pool = _get_pool(self.db_config)
                            # Close the stale connection before returning
                            try:
                                self.conn.close()
                            except Exception:
                                pass
                            # Don't return it to pool - it's broken
                        except Exception:
                            pass
                    
                    # Try again with a fresh connection
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        # Last attempt - create direct connection as fallback
                        logger.warning("Creating fresh direct connection after stale pool connections")
                        self.conn = _create_direct_connection(self.db_config)
                        self._use_pool = False  # Don't return this to pool
                
                return self.conn
                
            except Exception as exc:
                if attempt < max_attempts - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {exc}, retrying...")
                    time.sleep(1)
                else:
                    raise
        
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            try:
                if self._use_pool:
                    # Return to pool - THIS IS THE CRITICAL FIX
                    connection_pool = _get_pool(self.db_config)
                    connection_pool.putconn(self.conn)
                else:
                    # Close direct connection
                    self.conn.close()
            except Exception as exc:
                logger.error(f"Error cleaning up connection: {exc}", exc_info=True)
        
        # Don't suppress exceptions
        return False


# =============================================================================
# PUBLIC API
# =============================================================================

def get_db_connection(db_config: dict):
    """
    Get a database connection context manager (pooled or direct based on feature flag).
    
    This is the main entry point for all storage classes.
    
    IMPORTANT: This returns a context manager. Use with 'with' statement:
        with get_db_connection(db_config) as conn:
            ...
    
    Args:
        db_config: Database configuration dict with host, port, database, user, password
        
    Returns:
        DatabaseConnection context manager that properly returns connections to the pool
        
    Usage:
        # In storage class:
        def _get_connection(self):
            return get_db_connection(self.db_config)
        
        # Then use as:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    return DatabaseConnection(db_config)


def get_raw_connection(db_config: dict):
    """
    Get a raw database connection (for code that manages connections manually).
    
    WARNING: You MUST call return_connection() when done, or use try/finally!
    
    Args:
        db_config: Database configuration dict
        
    Returns:
        Raw psycopg2 connection (NOT a context manager)
        
    Usage:
        conn = get_raw_connection(db_config)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
            conn.commit()
        finally:
            return_connection(conn, db_config)
    """
    if not USE_CONNECTION_POOLING:
        return _retry_connection(db_config, use_pool=False)
    return _retry_connection(db_config, use_pool=True)


def return_connection(conn, db_config: dict):
    """
    Return a connection to the pool (only needed if pooling is enabled).
    
    Args:
        conn: Connection to return
        db_config: Database configuration dict
        
    Note:
        This is automatically handled when using 'with' statements,
        but can be called manually if needed.
    """
    if not USE_CONNECTION_POOLING:
        # Direct connections are closed by the context manager
        return
    
    try:
        connection_pool = _get_pool(db_config)
        connection_pool.putconn(conn)
    except Exception as exc:
        logger.error(f"Error returning connection to pool: {exc}", exc_info=True)


def close_all_connections():
    """
    Close all connections in the pool (for graceful shutdown).
    
    Call this when your application is shutting down.
    """
    global _pool, _pool_config
    
    with _pool_lock:
        if _pool is not None:
            logger.info("Closing all database connections in pool")
            _pool.closeall()
            _pool = None
            _pool_config = None


def get_pool_stats() -> dict:
    """
    Get statistics about the connection pool.
    
    Returns:
        Dict with pool statistics (or empty if pooling disabled)
    """
    if not USE_CONNECTION_POOLING or _pool is None:
        return {
            "pooling_enabled": USE_CONNECTION_POOLING,
            "pool_initialized": False
        }
    
    # Note: ThreadedConnectionPool doesn't expose detailed stats,
    # but we can provide basic info
    return {
        "pooling_enabled": True,
        "pool_initialized": True,
        "min_connections": MIN_CONNECTIONS,
        "max_connections": MAX_CONNECTIONS,
        "connection_timeout": CONNECTION_TIMEOUT,
        "max_retries": MAX_RETRIES
    }


# =============================================================================
# LOGGING STARTUP INFO
# =============================================================================

if USE_CONNECTION_POOLING:
    logger.info(
        f"Database connection pooling ENABLED: "
        f"min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS}, timeout={CONNECTION_TIMEOUT}s"
    )
else:
    logger.info("Database connection pooling DISABLED (using direct connections)")
