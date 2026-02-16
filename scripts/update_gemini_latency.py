"""Update Gemini Live voice config with optimized turn detection parameters."""
import psycopg2, os, json
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

load_dotenv()

# Optimized config for low latency
OPTIMIZED_TURN_DETECTION = {
    "start_of_speech_sensitivity": 95,  # Very sensitive to new speech
    "end_of_speech_sensitivity": 85,    # Very sensitive to pauses (end of turn)
    "silence_duration_ms": 200,         # Short silence means turn over
    "prefix_padding_ms": 50             # Less buffer
}

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', '5432')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
conn.autocommit = True
cur = conn.cursor(cursor_factory=RealDictCursor)
schema = os.getenv('DB_SCHEMA', 'lad_dev')

try:
    # Find existing Gemini Realtime voice
    cur.execute(f"SELECT id, provider_config FROM {schema}.voice_agent_voices WHERE provider = %s", ('gemini_realtime',))
    row = cur.fetchone()
    
    if not row:
        print("❌ Gemini Realtime voice row not found!")
    else:
        config = row['provider_config']
        
        # Add or update realtime_input_config
        if "realtime_input_config" not in config:
            config["realtime_input_config"] = {}
        
        config["realtime_input_config"]["automatic_activity_detection"] = OPTIMIZED_TURN_DETECTION
        
        # Remove old 'language' key if present (cleanup from previous fix)
        config.pop("language", None)

        print(f"Updating config for {row['id']}...")
        print(json.dumps(config, indent=2))
        
        cur.execute(
            f"UPDATE {schema}.voice_agent_voices SET provider_config = %s WHERE id = %s",
            (Json(config), row['id'])
        )
        print("✅ Successfully updated Gemini config with latency optimizations.")

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cur.close()
    conn.close()
