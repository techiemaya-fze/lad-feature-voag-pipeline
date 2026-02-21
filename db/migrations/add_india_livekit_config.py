"""Add India LiveKit configuration for tenant e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.en_de_crypt import encrypt_decrypt

load_dotenv()

# India LiveKit credentials
INDIA_CONFIG = {
    "name": "india-techiemaya-cloud",
    "description": "India LiveKit Cloud Server (lk.techiemaya.com)",
    "livekit_url": "wss://lk.techiemaya.com",
    "livekit_api_key": "API5QH2NJHDXQSW",
    "livekit_api_secret_plain": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN",
    "trunk_id": None,  # Will use trunk from rules or env
    "worker_name": "voag-staging",  # Same worker as UAE
}

def main():
    print("="*80)
    print("Adding India LiveKit Configuration")
    print("="*80)
    
    # Encrypt the secret
    print("\n1. Encrypting API secret...")
    encrypted_secret = encrypt_decrypt(INDIA_CONFIG["livekit_api_secret_plain"])
    print(f"   ✓ Secret encrypted: {encrypted_secret[:30]}...")
    
    # Connect to database
    print("\n2. Connecting to database...")
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT', 5432),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor()
    print(f"   ✓ Connected to {os.getenv('DB_NAME')}")
    
    # Check if config already exists
    print("\n3. Checking for existing config...")
    cur.execute("""
        SELECT id, name FROM lad_dev.voice_agent_livekit 
        WHERE name = %s
    """, (INDIA_CONFIG["name"],))
    existing = cur.fetchone()
    
    if existing:
        print(f"   ⚠ Config already exists: {existing[1]} (ID: {existing[0]})")
        print(f"   Updating existing config...")
        
        cur.execute("""
            UPDATE lad_dev.voice_agent_livekit
            SET description = %s,
                livekit_url = %s,
                livekit_api_key = %s,
                livekit_api_secret = %s,
                trunk_id = %s,
                worker_name = %s
            WHERE name = %s
            RETURNING id
        """, (
            INDIA_CONFIG["description"],
            INDIA_CONFIG["livekit_url"],
            INDIA_CONFIG["livekit_api_key"],
            encrypted_secret,
            INDIA_CONFIG["trunk_id"],
            INDIA_CONFIG["worker_name"],
            INDIA_CONFIG["name"]
        ))
        config_id = cur.fetchone()[0]
        print(f"   ✓ Updated config: {config_id}")
    else:
        print(f"   Creating new config...")
        
        cur.execute("""
            INSERT INTO lad_dev.voice_agent_livekit 
            (name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id, worker_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            INDIA_CONFIG["name"],
            INDIA_CONFIG["description"],
            INDIA_CONFIG["livekit_url"],
            INDIA_CONFIG["livekit_api_key"],
            encrypted_secret,
            INDIA_CONFIG["trunk_id"],
            INDIA_CONFIG["worker_name"]
        ))
        config_id = cur.fetchone()[0]
        print(f"   ✓ Created config: {config_id}")
    
    conn.commit()
    
    # Verify the config
    print("\n4. Verifying config...")
    cur.execute("""
        SELECT id, name, livekit_url, livekit_api_key, trunk_id, worker_name
        FROM lad_dev.voice_agent_livekit
        WHERE id = %s
    """, (config_id,))
    row = cur.fetchone()
    
    print(f"\n   Config Details:")
    print(f"   - ID: {row[0]}")
    print(f"   - Name: {row[1]}")
    print(f"   - URL: {row[2]}")
    print(f"   - API Key: {row[3]}")
    print(f"   - Trunk ID: {row[4] or '(from rules or env)'}")
    print(f"   - Worker Name: {row[5]}")
    
    # Now update the phone number rules
    print("\n5. Updating phone number rules...")
    print(f"   Looking for number: 9513456728")
    print(f"   Tenant ID: e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5")
    
    cur.execute("""
        SELECT id, base_number, rules 
        FROM lad_dev.voice_agent_numbers
        WHERE base_number = '9513456728'
        AND tenant_id = 'e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5'
    """)
    number_row = cur.fetchone()
    
    if number_row:
        print(f"   ✓ Found number: {number_row[1]}")
        
        # Update rules to add livekit_config
        cur.execute("""
            UPDATE lad_dev.voice_agent_numbers
            SET rules = jsonb_set(
                COALESCE(rules, '{}'::jsonb),
                '{livekit_config}',
                %s::jsonb
            )
            WHERE id = %s
            RETURNING rules
        """, (f'"{config_id}"', number_row[0]))
        
        updated_rules = cur.fetchone()[0]
        conn.commit()
        
        print(f"   ✓ Updated rules with livekit_config: {config_id}")
        print(f"   Rules: {updated_rules}")
    else:
        print(f"   ✗ Number not found!")
        print(f"   Please manually update the rules for number 9513456728")
        print(f"   Add this to rules JSON: \"livekit_config\": \"{config_id}\"")
    
    conn.close()
    
    print("\n" + "="*80)
    print("✓ India LiveKit Configuration Added Successfully!")
    print("="*80)
    print(f"\nConfig ID: {config_id}")
    print(f"To use this config, ensure phone number 9513456728 has:")
    print(f'  "livekit_config": "{config_id}"')
    print("\nTest with:")
    print(f"  uv run python tests/test_india_outbound_call.py")
    print("="*80)

if __name__ == "__main__":
    main()
