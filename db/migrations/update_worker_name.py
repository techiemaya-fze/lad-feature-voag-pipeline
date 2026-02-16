"""Update worker_name for UAE VM LiveKit config"""
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

# Update worker_name
cur.execute("""
    UPDATE lad_dev.voice_agent_livekit 
    SET worker_name = 'voag-staging' 
    WHERE name = 'uae-vm-selfhosted'
""")
conn.commit()

print(f'✓ Updated {cur.rowcount} row(s)')

# Verify
cur.execute("""
    SELECT name, worker_name, livekit_url 
    FROM lad_dev.voice_agent_livekit 
    WHERE name = 'uae-vm-selfhosted'
""")
row = cur.fetchone()
if row:
    print(f'✓ Verified: {row[0]}')
    print(f'  Worker Name: {row[1]}')
    print(f'  LiveKit URL: {row[2]}')

conn.close()
