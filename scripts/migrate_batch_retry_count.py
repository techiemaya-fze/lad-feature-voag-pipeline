"""
Migration: Add retry_count column to voice_call_batch_entries

This script adds the retry_count column for wave dispatch retry tracking.
Uses proper transaction handling with rollback on error.

Usage:
    uv run scripts/migrate_batch_retry_count.py
"""

import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from db.db_config import get_db_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_migration():
    """Run the migration with transaction rollback on error."""
    
    config = get_db_config()
    logger.info(f"Connecting to database: {config['host']}:{config['port']}/{config['database']}")
    
    conn = None
    try:
        # Use raw psycopg2 connection for transaction control
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
        )
        
        # Disable autocommit for explicit transaction control
        conn.autocommit = False
        
        try:
            with conn.cursor() as cur:
                # Check if column already exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'lad_dev' 
                    AND table_name = 'voice_call_batch_entries' 
                    AND column_name = 'retry_count'
                """)
                
                if cur.fetchone():
                    logger.info("Column 'retry_count' already exists. Migration skipped.")
                    return True
                
                logger.info("Adding retry_count column...")
                
                # Add column with default value
                cur.execute("""
                    ALTER TABLE lad_dev.voice_call_batch_entries 
                    ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0
                """)
                logger.info("✓ Column added successfully")
                
                # Create index for efficient wave queries
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_batch_entries_status_retry 
                    ON lad_dev.voice_call_batch_entries(batch_id, status, retry_count) 
                    WHERE is_deleted = FALSE
                """)
                logger.info("✓ Index created successfully")
                
                # Add column comment for documentation
                cur.execute("""
                    COMMENT ON COLUMN lad_dev.voice_call_batch_entries.retry_count IS 
                    'Number of times this entry was reset from dispatched to queued after wave timeout. Max 2 retries.'
                """)
                logger.info("✓ Column comment added")
                
                # Commit the transaction
                conn.commit()
                logger.info("✅ Migration completed successfully!")
                return True
                
        except Exception as e:
            # Rollback on any error
            logger.error(f"Migration failed: {e}")
            logger.info("Rolling back transaction...")
            conn.rollback()
            logger.info("✓ Rollback completed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False
        
    finally:
        if conn:
            conn.close()


def verify_migration():
    """Verify the migration was successful."""
    config = get_db_config()
    
    conn = None
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
        )
        
        with conn.cursor() as cur:
            # Check column exists with correct type
            cur.execute("""
                SELECT column_name, data_type, column_default
                FROM information_schema.columns 
                WHERE table_schema = 'lad_dev' 
                AND table_name = 'voice_call_batch_entries' 
                AND column_name = 'retry_count'
            """)
            result = cur.fetchone()
            
            if result:
                logger.info(f"✓ Column verified: {result[0]} ({result[1]}, default={result[2]})")
                return True
            else:
                logger.error("✗ Column not found!")
                return False
                
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False
        
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    success = run_migration()
    
    if success:
        verify_migration()
        sys.exit(0)
    else:
        sys.exit(1)
