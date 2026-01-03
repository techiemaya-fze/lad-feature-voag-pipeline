"""Fix sentiment column to TEXT."""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
conn.autocommit = False
cur = conn.cursor()

try:
    print("Altering sentiment column to TEXT...")
    cur.execute("ALTER TABLE lad_dev.voice_call_analysis ALTER COLUMN sentiment TYPE TEXT;")
    conn.commit()
    print("✅ Done! sentiment is now TEXT")
except Exception as e:
    print(f"❌ Error: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()
