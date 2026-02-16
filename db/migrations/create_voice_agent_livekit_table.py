"""
Database Migration: Create voice_agent_livekit Table

This migration creates the voice_agent_livekit table for storing LiveKit server
configurations with encrypted credentials.

Usage:
    uv run python db/migrations/create_voice_agent_livekit_table.py

Safety:
    - Uses CREATE TABLE IF NOT EXISTS (idempotent)
    - Uses CREATE OR REPLACE FUNCTION (idempotent)
    - Uses DROP TRIGGER IF EXISTS before CREATE TRIGGER (idempotent)
    - Wrapped in try-except with rollback on error
    - Prints clear success/failure messages
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get schema from environment (lad_dev or lad_prod)
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def create_voice_agent_livekit_table():
    """
    Create voice_agent_livekit table with trigger for auto-updating updated_at.
    
    This function is idempotent - safe to run multiple times.
    """
    conn = None
    try:
        conn = get_db_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            logger.info(f"Creating voice_agent_livekit table in schema: {SCHEMA}")
            
            # Create the main table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.voice_agent_livekit (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    livekit_url VARCHAR(500) NOT NULL,
                    livekit_api_key VARCHAR(255) NOT NULL,
                    livekit_api_secret TEXT NOT NULL,
                    trunk_id VARCHAR(255),
                    worker_name VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
            """)
            logger.info("✓ Table voice_agent_livekit created (or already exists)")
            
            # Add worker_name column if it doesn't exist (for existing tables)
            cur.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_schema = '{SCHEMA}' 
                        AND table_name = 'voice_agent_livekit' 
                        AND column_name = 'worker_name'
                    ) THEN
                        ALTER TABLE {SCHEMA}.voice_agent_livekit 
                        ADD COLUMN worker_name VARCHAR(255);
                    END IF;
                END $$;
            """)
            logger.info("✓ Table voice_agent_livekit created (or already exists)")
            
            # Create index on name for faster lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_voice_agent_livekit_name 
                ON {SCHEMA}.voice_agent_livekit(name)
            """)
            logger.info("✓ Index idx_voice_agent_livekit_name created (or already exists)")
            
            # Create trigger function for auto-updating updated_at
            cur.execute(f"""
                CREATE OR REPLACE FUNCTION {SCHEMA}.update_voice_agent_livekit_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            logger.info("✓ Trigger function update_voice_agent_livekit_updated_at created")
            
            # Drop existing trigger if it exists, then create new one
            cur.execute(f"""
                DROP TRIGGER IF EXISTS voice_agent_livekit_updated_at 
                ON {SCHEMA}.voice_agent_livekit
            """)
            
            cur.execute(f"""
                CREATE TRIGGER voice_agent_livekit_updated_at
                    BEFORE UPDATE ON {SCHEMA}.voice_agent_livekit
                    FOR EACH ROW
                    EXECUTE FUNCTION {SCHEMA}.update_voice_agent_livekit_updated_at()
            """)
            logger.info("✓ Trigger voice_agent_livekit_updated_at created")
            
            # Verify table was created
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = '{SCHEMA}' 
                AND table_name = 'voice_agent_livekit'
            """)
            count = cur.fetchone()[0]
            
            if count == 1:
                logger.info(f"✓ Migration completed successfully!")
                logger.info(f"  Table: {SCHEMA}.voice_agent_livekit")
                logger.info(f"  Columns: id, name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id, worker_name, created_at, updated_at")
                logger.info(f"  Trigger: auto-updates updated_at on row modification")
                return True
            else:
                logger.error("✗ Table verification failed")
                return False
                
    except Exception as e:
        logger.error(f"✗ Migration failed: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def verify_migration():
    """Verify the migration was successful by checking table structure."""
    conn = None
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cur:
            # Check table exists
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = '{SCHEMA}' 
                AND table_name = 'voice_agent_livekit'
                ORDER BY ordinal_position
            """)
            
            columns = cur.fetchall()
            
            if not columns:
                logger.warning("Table not found - migration may have failed")
                return False
            
            logger.info("\n" + "="*80)
            logger.info("Table Structure Verification:")
            logger.info("="*80)
            for col in columns:
                col_name, data_type, nullable, default = col
                logger.info(f"  {col_name:25} {data_type:20} {'NULL' if nullable == 'YES' else 'NOT NULL':10} {default or ''}")
            
            # Check trigger exists
            cur.execute(f"""
                SELECT trigger_name 
                FROM information_schema.triggers 
                WHERE event_object_schema = '{SCHEMA}' 
                AND event_object_table = 'voice_agent_livekit'
            """)
            
            triggers = cur.fetchall()
            logger.info("\nTriggers:")
            for trigger in triggers:
                logger.info(f"  ✓ {trigger[0]}")
            
            logger.info("="*80)
            return True
            
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Voice Agent LiveKit Table Migration")
    logger.info("="*80)
    logger.info(f"Target Schema: {SCHEMA}")
    logger.info(f"Database: {os.getenv('DB_NAME')}")
    logger.info(f"Host: {os.getenv('DB_HOST')}")
    logger.info("="*80)
    
    # Run migration
    success = create_voice_agent_livekit_table()
    
    if success:
        # Verify migration
        logger.info("\nVerifying migration...")
        verify_migration()
        logger.info("\n✓ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n✗ Migration failed!")
        sys.exit(1)
