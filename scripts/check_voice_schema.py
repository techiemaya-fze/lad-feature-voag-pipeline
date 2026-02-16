"""Insert a Sarvam test voice row into voice_agent_voices."""
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import json
import uuid

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', '5432')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor(cursor_factory=RealDictCursor)

# First, get schema of voice_agent_voices table
cur.execute("""
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_schema = 'lad_dev' AND table_name = 'voice_agent_voices'
    ORDER BY ordinal_position
""")
columns = cur.fetchall()
print("voice_agent_voices schema:")
for col in columns:
    print(f"  {col['column_name']}: {col['data_type']} (nullable={col['is_nullable']}, default={col['column_default']})")

# Get a sample row to understand existing data
cur.execute("SELECT * FROM lad_dev.voice_agent_voices LIMIT 1")
sample = cur.fetchone()
if sample:
    print("\nSample row:")
    for k, v in dict(sample).items():
        print(f"  {k}: {v}")

conn.close()
