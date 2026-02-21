"""Check existing LiveKit configs"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT', 5432),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

# List all configs
cur.execute("""
    SELECT id, name, livekit_url, trunk_id, worker_name
    FROM lad_dev.voice_agent_livekit
""")
rows = cur.fetchall()

print(f"Found {len(rows)} LiveKit config(s):\n")
for row in rows:
    print(f"ID: {row[0]}")
    print(f"Name: {row[1]}")
    print(f"URL: {row[2]}")
    print(f"Trunk ID: {row[3]}")
    print(f"Worker Name: {row[4]}")
    print("-" * 80)

conn.close()
