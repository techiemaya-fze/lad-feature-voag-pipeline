"""
Database Migration: Expand varchar columns to TEXT in voice_call_analysis

Run with: uv run scripts/migrate_analysis_columns.py
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables from v2/.env
load_dotenv()

def get_db_config():
    """Get database configuration from environment."""
    return {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

def run_migration():
    """Run the column type migration with rollback on error."""
    config = get_db_config()
    
    print(f"Connecting to database: {config['host']}:{config['port']}/{config['database']}")
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = False  # Use transaction
        cursor = conn.cursor()
        
        # Check current column types
        print("\n--- Current column types ---")
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'lad_dev' 
              AND table_name = 'voice_call_analysis'
              AND column_name IN ('lead_category', 'engagement_level', 'disposition')
            ORDER BY column_name
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}({row[2]})" if row[2] else f"  {row[0]}: {row[1]}")
        
        # Run migration
        print("\n--- Running migration ---")
        
        migration_sql = """
            ALTER TABLE lad_dev.voice_call_analysis 
                ALTER COLUMN lead_category TYPE TEXT,
                ALTER COLUMN engagement_level TYPE TEXT,
                ALTER COLUMN disposition TYPE TEXT;
        """
        
        print("Altering columns to TEXT...")
        cursor.execute(migration_sql)
        
        # Verify changes
        print("\n--- Verifying changes ---")
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = 'lad_dev' 
              AND table_name = 'voice_call_analysis'
              AND column_name IN ('lead_category', 'engagement_level', 'disposition')
            ORDER BY column_name
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]}({row[2]})" if row[2] else f"  {row[0]}: {row[1]}")
        
        # Commit transaction
        conn.commit()
        print("\n✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if conn:
            conn.rollback()
            print("Rolled back transaction.")
        return False
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
