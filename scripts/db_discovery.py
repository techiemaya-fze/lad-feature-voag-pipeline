import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME", "vonage_agent"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "password")
    )

def inspect_db():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Get recent calls with transcripts
            print("--- Recent Calls with Transcripts ---")
            cur.execute("""
                SELECT id, status, LEFT(transcripts::text, 50) as transcript_preview 
                FROM lad_dev.voice_call_logs 
                WHERE transcripts IS NOT NULL 
                ORDER BY started_at DESC 
                LIMIT 5
            """)
            calls = cur.fetchall()
            for call in calls:
                print(f"ID: {call['id']}, Status: {call['status']}, Transcript Start: {call['transcript_preview']}")

            # 2. Check call_analysis schema (columns)
            print("\n--- call_analysis Columns ---")
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'lad_dev' AND table_name = 'call_analysis'
            """)
            columns = cur.fetchall()
            for col in columns:
                print(f"{col['column_name']} ({col['data_type']})")

            # 3. Check lead_info schema (columns)
            print("\n--- lead_info Columns ---")
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'lad_dev' AND table_name = 'lead_info'
            """)
            columns = cur.fetchall()
            for col in columns:
                print(f"{col['column_name']} ({col['data_type']})")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_db()
