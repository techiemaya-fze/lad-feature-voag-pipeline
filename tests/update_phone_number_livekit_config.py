"""
Update Phone Number with LiveKit Config

This script updates the voice_agent_numbers table to add the livekit_config UUID
to the rules JSON for phone number 545335200.

Run with: uv run python tests/update_phone_number_livekit_config.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extras import RealDictCursor
from db.db_config import get_db_config


def update_phone_number_livekit_config():
    """Update phone number 545335200 with LiveKit config UUID."""
    
    phone_number = "545335200"
    livekit_config_id = "e3ca4a84-4cc3-4be5-8240-34366fe4d0c5"
    
    print("="*80)
    print("Update Phone Number with LiveKit Config")
    print("="*80)
    print(f"Phone Number: {phone_number}")
    print(f"LiveKit Config ID: {livekit_config_id}")
    print("="*80)
    
    db_config = get_db_config()
    
    try:
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            dbname=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get current record
            print("\n1. Fetching current record...")
            cur.execute("""
                SELECT id, country_code, base_number, provider, rules
                FROM lad_dev.voice_agent_numbers
                WHERE base_number = %s
                LIMIT 1
            """, (int(phone_number),))
            
            record = cur.fetchone()
            
            if not record:
                print(f"   ✗ Phone number {phone_number} not found")
                return False
            
            print(f"   ✓ Found record:")
            print(f"     - ID: {record['id']}")
            print(f"     - Country Code: {record['country_code']}")
            print(f"     - Base Number: {record['base_number']}")
            print(f"     - Provider: {record['provider']}")
            
            # Get current rules
            current_rules = record['rules'] or {}
            print(f"\n2. Current rules:")
            print(f"   {json.dumps(current_rules, indent=2)}")
            
            # Add livekit_config to rules
            updated_rules = current_rules.copy()
            updated_rules['livekit_config'] = livekit_config_id
            
            print(f"\n3. Updated rules:")
            print(f"   {json.dumps(updated_rules, indent=2)}")
            
            # Update the record
            print(f"\n4. Updating database...")
            cur.execute("""
                UPDATE lad_dev.voice_agent_numbers
                SET rules = %s::jsonb
                WHERE base_number = %s
            """, (json.dumps(updated_rules), int(phone_number)))
            
            conn.commit()
            print(f"   ✓ Updated successfully")
            
            # Verify update
            print(f"\n5. Verifying update...")
            cur.execute("""
                SELECT rules
                FROM lad_dev.voice_agent_numbers
                WHERE base_number = %s
            """, (int(phone_number),))
            
            verified = cur.fetchone()
            verified_rules = verified['rules']
            
            if verified_rules.get('livekit_config') == livekit_config_id:
                print(f"   ✓ Verification successful!")
                print(f"   Rules now contain livekit_config: {livekit_config_id}")
            else:
                print(f"   ✗ Verification failed")
                return False
        
        conn.close()
        
        print("\n" + "="*80)
        print("✓ Phone number updated successfully!")
        print("="*80)
        print(f"\nPhone number {phone_number} now uses LiveKit config:")
        print(f"  - Config ID: {livekit_config_id}")
        print(f"  - Config Name: uae-vm-selfhosted")
        print(f"  - LiveKit URL: http://91.74.244.94:7880")
        print(f"  - Trunk ID: ST_svHE4RdTc7Ds")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = update_phone_number_livekit_config()
    sys.exit(0 if success else 1)
