"""Insert a Gemini Live realtime voice row into voice_agent_voices.

This creates a voice entry that triggers REALTIME mode in the agent worker,
using Google's Gemini Live RealtimeModel via the livekit-agents[google] plugin.

Usage:
    .venv\\Scripts\\activate
    uv run scripts/insert_gemini_realtime_voice.py
"""
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv
import os

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

SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")

# Get tenant_id from an existing voice row that has one
cur.execute(f"SELECT tenant_id FROM {SCHEMA}.voice_agent_voices WHERE tenant_id IS NOT NULL LIMIT 1")
row = cur.fetchone()
tenant_id = row['tenant_id'] if row else None
print(f"Using tenant_id: {tenant_id}")

if not tenant_id:
    print("ERROR: No tenant_id found. Cannot insert voice without tenant context.")
    cur.close()
    conn.close()
    exit(1)

# Check if Gemini realtime voice already exists
cur.execute(f"""
    SELECT id, description, provider_voice_id 
    FROM {SCHEMA}.voice_agent_voices 
    WHERE provider IN ('gemini_realtime', 'gemini-realtime', 'gemini_live', 'gemini-live') 
    LIMIT 1
""")
existing = cur.fetchone()
if existing:
    print(f"Gemini realtime voice exists: id={existing['id']}, desc={existing['description']}")
    cur.close()
    conn.close()
    exit(0)

# -----------------------------------------------------------------------
# Gemini Live provider_config — maps directly to google.realtime.RealtimeModel
#
# Available model:
#   - gemini-2.5-flash-native-audio-preview-12-2025 (default, native audio)
#
# Available voices:
#   Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr
#
# API key: GOOGLE_API_KEY environment variable (read by plugin automatically)
# -----------------------------------------------------------------------
provider_config = {
    # Core model config
    "model": "gemini-2.5-flash-native-audio-preview-12-2025",
    "voice": "Puck",                      # Natural, conversational male voice
    "temperature": 0.4,                    # Low for consistent responses
    
    # Language
    "language": "en",                      # BCP-47 language code
    
    # Generation parameters
    "top_p": 0.95,                         # Nucleus sampling
    "top_k": 40,                           # Top-K sampling
    
    # Thinking (supported by this model)
    "thinking_config": {
        "include_thoughts": False,         # Don't forward thoughts to user
    },
    
    # Affective dialog and proactivity (native audio features)
    "enable_affective_dialog": False,      # Conservative for now
    "proactivity": False,                  # Conservative for now
}

cur.execute(f"""
    INSERT INTO {SCHEMA}.voice_agent_voices
        (tenant_id, description, gender, accent, provider, provider_voice_id, provider_config)
    VALUES
        (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id, description
""", (
    tenant_id,
    "Gemini Live Puck (Realtime Native Audio)",  # description
    "male",                              # gender
    "en-US",                             # accent
    "gemini_realtime",                   # provider — triggers realtime mode detection
    "Puck",                              # provider_voice_id — Gemini voice name
    Json(provider_config),               # provider_config JSONB
))

inserted = cur.fetchone()
conn.commit()

print(f"\n✅ Inserted Gemini Live realtime voice:")
print(f"  id:                {inserted['id']}")
print(f"  description:       {inserted['description']}")
print(f"  provider:          gemini_realtime")
print(f"  provider_voice_id: Puck")
print(f"  model:             {provider_config['model']}")
print(f"  voice:             {provider_config['voice']}")
print(f"  temperature:       {provider_config['temperature']}")
print(f"  language:          {provider_config['language']}")
print(f"  top_p:             {provider_config['top_p']}")
print(f"  top_k:             {provider_config['top_k']}")
print(f"\n📋 Use this voice_id in API calls to trigger Gemini Live realtime mode.")

cur.close()
conn.close()
