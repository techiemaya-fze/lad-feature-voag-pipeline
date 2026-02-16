"""Insert a Sarvam test voice row into voice_agent_voices."""
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
import os
import json

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', '5432')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
conn.autocommit = False
cur = conn.cursor(cursor_factory=RealDictCursor)

# Get tenant_id from an existing voice row
cur.execute("SELECT tenant_id FROM lad_dev.voice_agent_voices WHERE tenant_id IS NOT NULL LIMIT 1")
row = cur.fetchone()
tenant_id = row['tenant_id'] if row else None
print(f"Using tenant_id: {tenant_id}")

# Check if Sarvam voice already exists
cur.execute("SELECT id FROM lad_dev.voice_agent_voices WHERE provider = 'sarvam' LIMIT 1")
existing = cur.fetchone()
if existing:
    print(f"Sarvam voice already exists: {existing['id']}")
    conn.close()
    exit(0)

provider_config = {
    "target_language_code": "hi-IN",
    "model": "bulbul:v2",
    "speaker": "anushka",
    "pitch": 0.0,
    "pace": 1.0,
    "loudness": 1.0
}

cur.execute("""
    INSERT INTO lad_dev.voice_agent_voices 
        (tenant_id, description, gender, accent, provider, provider_voice_id, provider_config)
    VALUES 
        (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id
""", (
    tenant_id,
    "Sarvam Anushka - Hindi (Test)",
    "female",
    "hi-IN",
    "sarvam",
    "anushka",
    Json(provider_config)
))

new_id = cur.fetchone()['id']
conn.commit()
print(f"Inserted Sarvam voice: id={new_id}")

# Verify
cur.execute("SELECT * FROM lad_dev.voice_agent_voices WHERE id = %s", (new_id,))
inserted = cur.fetchone()
print("\nInserted row:")
for k, v in dict(inserted).items():
    print(f"  {k}: {v}")

conn.close()
