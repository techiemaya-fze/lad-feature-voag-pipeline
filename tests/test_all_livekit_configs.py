"""
Test All LiveKit Configurations

This script tests connectivity to all LiveKit configurations in the database.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from livekit import api
from db.connection_pool import get_db_connection
from db.db_config import get_db_config
from utils.en_de_crypt import encrypt_decrypt
from psycopg2.extras import RealDictCursor


async def test_config_connectivity(config: dict) -> bool:
    """Test if we can connect to a LiveKit config."""
    try:
        # Decrypt secret
        decrypted_secret = encrypt_decrypt(config['livekit_api_secret'])
        
        # Test connection
        async with api.LiveKitAPI(
            config['livekit_url'],
            config['livekit_api_key'],
            decrypted_secret
        ) as lk_api:
            rooms_response = await lk_api.room.list_rooms(api.ListRoomsRequest())
            room_count = len(rooms_response.rooms) if hasattr(rooms_response, 'rooms') else 0
            print(f"    ✓ Connected! ({room_count} active rooms)")
            return True
    except Exception as e:
        print(f"    ✗ Connection failed: {e}")
        return False


async def main():
    print("="*80)
    print("  LIVEKIT CONFIGURATION CONNECTIVITY TEST")
    print("="*80)
    
    db_config = get_db_config()
    schema = db_config.get("schema", "lad_dev")
    
    with get_db_connection(db_config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, name, description, livekit_url, livekit_api_key, 
                       livekit_api_secret, trunk_id, worker_name
                FROM {schema}.voice_agent_livekit
                ORDER BY name
            """)
            
            configs = cur.fetchall()
            
            print(f"\nFound {len(configs)} configurations\n")
            
            results = []
            
            for config in configs:
                print(f"Testing: {config['name']}")
                print(f"  URL: {config['livekit_url']}")
                print(f"  Trunk: {config['trunk_id']}")
                print(f"  Worker: {config['worker_name']}")
                
                success = await test_config_connectivity(dict(config))
                results.append({
                    'name': config['name'],
                    'success': success
                })
                print()
            
            # Summary
            print("="*80)
            print("  SUMMARY")
            print("="*80)
            
            for result in results:
                status = "✓ WORKING" if result['success'] else "✗ FAILED"
                print(f"  {status}: {result['name']}")
            
            success_count = sum(1 for r in results if r['success'])
            print(f"\n  Total: {success_count}/{len(results)} configs working")


if __name__ == "__main__":
    asyncio.run(main())
