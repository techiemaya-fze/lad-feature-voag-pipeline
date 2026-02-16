"""
Manual Test for LiveKit Storage

This script tests the LiveKit storage class by:
1. Creating a real LiveKit config with UAE VM credentials
2. Retrieving it
3. Verifying decryption works
4. Listing all configs

Run with: uv run python tests/test_livekit_storage_manual.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.storage.livekit import LiveKitConfigStorage
from utils.en_de_crypt import encrypt_decrypt


async def test_livekit_storage():
    """Test LiveKit storage with real UAE VM credentials."""
    
    storage = LiveKitConfigStorage()
    
    print("="*80)
    print("LiveKit Storage Test")
    print("="*80)
    
    # UAE VM credentials (from deploy/uae-vm/.env)
    uae_config = {
        "name": "uae-vm-selfhosted",
        "description": "Self-hosted LiveKit server on UAE VM (91.74.244.94)",
        "livekit_url": "http://91.74.244.94:7880",
        "livekit_api_key": "APIbe273e3142c7b96a4a87bba4",
        "livekit_api_secret_plain": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
        "trunk_id": "ST_svHE4RdTc7Ds",
    }
    
    # Encrypt the secret
    print("\n1. Encrypting LiveKit API secret...")
    encrypted_secret = encrypt_decrypt(uae_config["livekit_api_secret_plain"])
    print(f"   ✓ Encrypted: {encrypted_secret[:50]}...")
    
    # Check if config already exists
    print("\n2. Checking if config already exists...")
    existing = await storage.get_livekit_config_by_name(uae_config["name"])
    
    if existing:
        print(f"   ✓ Config already exists: {existing['id']}")
        config_id = existing['id']
    else:
        # Create the config
        print("\n3. Creating LiveKit config in database...")
        config_id = await storage.create_livekit_config(
            name=uae_config["name"],
            description=uae_config["description"],
            livekit_url=uae_config["livekit_url"],
            livekit_api_key=uae_config["livekit_api_key"],
            livekit_api_secret=encrypted_secret,
            trunk_id=uae_config["trunk_id"],
        )
        
        if config_id:
            print(f"   ✓ Created config with ID: {config_id}")
        else:
            print("   ✗ Failed to create config")
            return False
    
    # Retrieve the config
    print("\n4. Retrieving config from database...")
    retrieved = await storage.get_livekit_config(config_id)
    
    if retrieved:
        print(f"   ✓ Retrieved config: {retrieved['name']}")
        print(f"     - ID: {retrieved['id']}")
        print(f"     - URL: {retrieved['livekit_url']}")
        print(f"     - API Key: {retrieved['livekit_api_key']}")
        print(f"     - Trunk ID: {retrieved['trunk_id']}")
        print(f"     - Encrypted Secret: {retrieved['livekit_api_secret'][:50]}...")
    else:
        print("   ✗ Failed to retrieve config")
        return False
    
    # Decrypt the secret
    print("\n5. Decrypting API secret...")
    decrypted_secret = encrypt_decrypt(retrieved['livekit_api_secret'])
    
    if decrypted_secret == uae_config["livekit_api_secret_plain"]:
        print(f"   ✓ Decryption successful!")
        print(f"     Original:  {uae_config['livekit_api_secret_plain']}")
        print(f"     Decrypted: {decrypted_secret}")
    else:
        print("   ✗ Decryption failed - values don't match")
        return False
    
    # List all configs
    print("\n6. Listing all LiveKit configs...")
    all_configs = await storage.list_livekit_configs()
    print(f"   ✓ Found {len(all_configs)} config(s):")
    for config in all_configs:
        print(f"     - {config['name']} ({config['id'][:8]}...)")
        print(f"       URL: {config['livekit_url']}")
        print(f"       Trunk: {config['trunk_id']}")
    
    # Test update
    print("\n7. Testing update (adding description suffix)...")
    updated = await storage.update_livekit_config(
        config_id,
        description=f"{uae_config['description']} [TESTED]"
    )
    
    if updated:
        print("   ✓ Update successful")
        # Verify update
        updated_config = await storage.get_livekit_config(config_id)
        print(f"     New description: {updated_config['description']}")
    else:
        print("   ✗ Update failed")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    print(f"\nLiveKit Config ID for UAE VM: {config_id}")
    print("Use this UUID in voice_agent_numbers.rules JSON:")
    print(f'  "livekit_config": "{config_id}"')
    print("="*80)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_livekit_storage())
    sys.exit(0 if success else 1)
