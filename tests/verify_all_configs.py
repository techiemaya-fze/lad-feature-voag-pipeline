"""
Verify All LiveKit Configurations

Read the table and verify all encrypted secrets decrypt to the correct values.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.connection_pool import get_db_connection
from db.db_config import get_db_config
from utils.en_de_crypt import encrypt_decrypt
from psycopg2.extras import RealDictCursor


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def main():
    print_section("VERIFY ALL LIVEKIT CONFIGURATIONS")
    
    # Expected values
    EXPECTED_SECRETS = {
        "uae-vm": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
        "india-vm": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN",
    }
    
    print("\nExpected Secrets:")
    print(f"  UAE VM: {EXPECTED_SECRETS['uae-vm']}")
    print(f"  India VM: {EXPECTED_SECRETS['india-vm']}")
    
    # Get encryption key
    encryption_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
    if not encryption_key:
        print("\n✗ LIVEKIT_SECRET_ENCRYPTION_KEY not set")
        return False
    
    print(f"\nEncryption Key: {encryption_key[:20]}...")
    
    # Read all configs
    db_config = get_db_config()
    schema = db_config.get("schema", "lad_dev")
    
    with get_db_connection(db_config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, name, description, livekit_url, livekit_api_key, 
                       livekit_api_secret, trunk_id, worker_name, created_at, updated_at
                FROM {schema}.voice_agent_livekit
                ORDER BY name
            """)
            
            configs = cur.fetchall()
    
    print_section(f"FOUND {len(configs)} CONFIGURATIONS")
    
    all_valid = True
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   {'─' * 76}")
        print(f"   ID: {config['id']}")
        print(f"   Description: {config['description']}")
        print(f"   URL: {config['livekit_url']}")
        print(f"   API Key: {config['livekit_api_key']}")
        print(f"   Trunk ID: {config['trunk_id']}")
        print(f"   Worker Name: {config['worker_name']}")
        print(f"   Created: {config['created_at']}")
        print(f"   Updated: {config['updated_at']}")
        
        # Decrypt secret
        encrypted = config['livekit_api_secret']
        print(f"   Encrypted: {encrypted[:60]}...")
        
        try:
            decrypted = encrypt_decrypt(encrypted)
            print(f"   Decrypted: {decrypted}")
            
            # Determine expected value based on URL
            if "91.74.244.94" in config['livekit_url']:
                expected = EXPECTED_SECRETS['uae-vm']
                server_type = "UAE VM"
            elif "lk.techiemaya.com" in config['livekit_url']:
                expected = EXPECTED_SECRETS['india-vm']
                server_type = "India VM"
            else:
                print(f"   ⚠ Unknown server type")
                continue
            
            # Verify
            if decrypted == expected:
                print(f"   ✓ CORRECT - Matches {server_type} secret")
            else:
                print(f"   ✗ INCORRECT - Does NOT match {server_type} secret!")
                print(f"   Expected: {expected}")
                print(f"   Got: {decrypted}")
                all_valid = False
                
        except Exception as e:
            print(f"   ✗ DECRYPTION FAILED: {e}")
            all_valid = False
    
    print_section("SUMMARY")
    
    if all_valid:
        print("\n✓ ALL CONFIGURATIONS ARE VALID")
        print("✓ All secrets decrypt correctly")
        print("✓ All secrets match expected values")
        return True
    else:
        print("\n✗ SOME CONFIGURATIONS ARE INVALID")
        print("✗ Check the output above for details")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
