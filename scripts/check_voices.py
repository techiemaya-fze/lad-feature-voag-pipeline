import psycopg2
from psycopg2.extras import RealDictCursor
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
cur = conn.cursor(cursor_factory=RealDictCursor)
cur.execute('SELECT id, provider, provider_voice_id, provider_config FROM lad_dev.voice_agent_voices LIMIT 5')
rows = cur.fetchall()
print('voice_agent_voices data:')
for row in rows:
    print(f"\nID: {row['id']}")
    print(f"  provider: {row['provider']}")
    print(f"  provider_voice_id: {row['provider_voice_id']}")
    print(f"  provider_config: {json.dumps(row['provider_config'], indent=2) if row['provider_config'] else 'NULL'}")
conn.close()
