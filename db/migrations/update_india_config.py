"""Update India LiveKit config with correct trunk_id and worker_name"""
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

# Update India config
print("Updating India LiveKit config...")
cur.execute("""
    UPDATE lad_dev.voice_agent_livekit
    SET trunk_id = 'ST_MmVqEuBMDNf6',
        worker_name = 'voag-dev'
    WHERE name = 'india-techiemaya-cloud'
    RETURNING id, name, trunk_id, worker_name
""")

result = cur.fetchone()
if result:
    conn.commit()
    print(f"✓ Updated config: {result[1]}")
    print(f"  - ID: {result[0]}")
    print(f"  - Trunk ID: {result[2]}")
    print(f"  - Worker Name: {result[3]}")
else:
    print("✗ Config not found!")

conn.close()
