"""Insert an Ultravox realtime voice row into voice_agent_voices.

This creates a voice entry that triggers REALTIME mode in the agent worker,
using the Ultravox RealtimeModel instead of separate STT/LLM/TTS pipeline.

Usage:
    .venv\\Scripts\\activate
    uv run scripts/insert_ultravox_voice.py
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

# Check if Ultravox voice already exists
cur.execute(f"SELECT id, description, provider_voice_id FROM {SCHEMA}.voice_agent_voices WHERE provider = 'ultravox' LIMIT 1")
existing = cur.fetchone()
if existing:
    print(f"Ultravox voice already exists: id={existing['id']}, desc={existing['description']}, voice={existing['provider_voice_id']}")
    cur.close()
    conn.close()
    exit(0)

# -----------------------------------------------------------------------
# Ultravox provider_config — maps directly to RealtimeModel constructor
# parameters. All values are stored as their native JSON types and
# type-coerced in realtime_builder.py before passing to the constructor.
#
# Available models (from livekit-plugins-ultravox==1.3.10):
#   - fixie-ai/ultravox              (default, balanced)
#   - fixie-ai/ultravox-llama3.3-70b (highest quality, higher latency)
#   - fixie-ai/ultravox-qwen3-32b-preview   (preview, multilingual)
#   - fixie-ai/ultravox-gemma3-27b-preview   (preview)
#
# Available voices: Mark, Jessica
# -----------------------------------------------------------------------
provider_config = {
    # Core model config
    "model": "fixie-ai/ultravox",      # Latest default model
    "voice": "Mark",                     # Built-in male voice
    "temperature": 0.4,                  # Low for consistent, focused responses
    
    # Language and duration
    "language_hint": "en",               # English language hint for better STT accuracy
    "max_duration": "30m",               # 30 minute max call duration
    "time_exceeded_message": "I'm sorry, but we've reached the maximum call duration. Thank you for your time.",
    
    # Conversation behavior
    "first_speaker": "FIRST_SPEAKER_USER",  # Wait for user to speak first
    "output_medium": "voice",            # Voice output (not text)
    "enable_greeting_prompt": False,     # Disable — our agent instructions handle greeting
    
    # Audio quality
    "input_sample_rate": 16000,          # 16kHz input (telephony standard)
    "output_sample_rate": 24000,         # 24kHz output (higher quality synthesis)
}

cur.execute(f"""
    INSERT INTO {SCHEMA}.voice_agent_voices
        (tenant_id, description, gender, accent, provider, provider_voice_id, provider_config)
    VALUES
        (%s, %s, %s, %s, %s, %s, %s)
    RETURNING id, description
""", (
    tenant_id,
    "Ultravox Mark (Realtime)",  # description
    "male",                      # gender
    "en-US",                     # accent
    "ultravox",                  # provider — triggers realtime mode detection
    "Mark",                      # provider_voice_id — Ultravox voice name
    Json(provider_config),       # provider_config JSONB
))

inserted = cur.fetchone()
conn.commit()

print(f"\n✅ Inserted Ultravox realtime voice:")
print(f"  id:                {inserted['id']}")
print(f"  description:       {inserted['description']}")
print(f"  provider:          ultravox")
print(f"  provider_voice_id: Mark")
print(f"  model:             {provider_config['model']}")
print(f"  temperature:       {provider_config['temperature']}")
print(f"  language_hint:     {provider_config['language_hint']}")
print(f"  max_duration:      {provider_config['max_duration']}")
print(f"  output_medium:     {provider_config['output_medium']}")
print(f"  first_speaker:     {provider_config['first_speaker']}")
print(f"  input_sample_rate: {provider_config['input_sample_rate']}")
print(f"  output_sample_rate:{provider_config['output_sample_rate']}")
print(f"\n📋 Use this voice_id in API calls to trigger realtime mode.")

cur.close()
conn.close()
