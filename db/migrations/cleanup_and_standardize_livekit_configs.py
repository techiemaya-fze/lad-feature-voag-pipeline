"""
Cleanup and Standardize LiveKit Configurations

This script:
1. Reads all existing configs from voice_agent_livekit table
2. Tests decryption of secrets using current encryption key
3. Removes configs with secrets that cannot be decrypted (orphaned/invalid)
4. Ensures we have exactly 3 configs:
   - UAE VM (http://91.74.244.94:7880) - ST_TihECwc9vxgj - voag-uae-workers
   - India VM Vonage (wss://lk.techiemaya.com) - ST_oCcPBvFqzMWf - voag-staging
   - India VM Sasya (wss://lk.techiemaya.com) - ST_FgSvNTqJmFcR - indian-synapse
5. Uses transaction with rollback on any error

Expected Configs:
-----------------
1. UAE VM:
   - URL: http://91.74.244.94:7880
   - API Key: APIbe273e3142c7b96a4a87bba4
   - API Secret: SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e
   - Trunk: ST_TihECwc9vxgj
   - Worker: voag-uae-workers

2. India VM - Vonage:
   - URL: wss://lk.techiemaya.com
   - API Key: API5QH2NJHDXQSW
   - API Secret: fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN
   - Trunk: ST_oCcPBvFqzMWf
   - Worker: voag-staging

3. India VM - Sasya:
   - URL: wss://lk.techiemaya.com
   - API Key: API5QH2NJHDXQSW
   - API Secret: fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN
   - Trunk: ST_FgSvNTqJmFcR
   - Worker: indian-synapse
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from db.connection_pool import get_db_connection
from db.db_config import get_db_config
from utils.en_de_crypt import encrypt_decrypt
from psycopg2.extras import RealDictCursor

# Target configurations
TARGET_CONFIGS = {
    "uae-vm": {
        "name": "uae-vm-selfhosted",
        "description": "Self-hosted LiveKit server on UAE VM (91.74.244.94)",
        "livekit_url": "http://91.74.244.94:7880",
        "livekit_api_key": "APIbe273e3142c7b96a4a87bba4",
        "livekit_api_secret_plain": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
        "trunk_id": "ST_TihECwc9vxgj",
        "worker_name": "voag-uae-workers",
    },
    "india-vm-vonage": {
        "name": "india-techiemaya-vonage",
        "description": "India LiveKit Cloud (lk.techiemaya.com) - Vonage Trunk",
        "livekit_url": "wss://lk.techiemaya.com",
        "livekit_api_key": "API5QH2NJHDXQSW",
        "livekit_api_secret_plain": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN",
        "trunk_id": "ST_oCcPBvFqzMWf",
        "worker_name": "voag-staging",
    },
    "india-vm-sasya": {
        "name": "india-techiemaya-sasya",
        "description": "India LiveKit Cloud (lk.techiemaya.com) - Sasya Local Trunk",
        "livekit_url": "wss://lk.techiemaya.com",
        "livekit_api_key": "API5QH2NJHDXQSW",
        "livekit_api_secret_plain": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN",
        "trunk_id": "ST_FgSvNTqJmFcR",
        "worker_name": "indian-synapse",
    },
}


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_step(step: str):
    """Print step."""
    print(f"\n{step}")
    print("-" * 80)


def can_decrypt(encrypted_value: str) -> tuple[bool, str]:
    """
    Test if a value can be decrypted with current key.
    
    Returns:
        (success, decrypted_value_or_error)
    """
    try:
        decrypted = encrypt_decrypt(encrypted_value)
        return True, decrypted
    except Exception as e:
        return False, str(e)


def encrypt_secret(plain_secret: str) -> str:
    """Encrypt a plain secret with dev-s-t- prefix."""
    # Import here to use the encryption function
    from cryptography.fernet import Fernet
    
    encryption_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
    if not encryption_key:
        raise RuntimeError("LIVEKIT_SECRET_ENCRYPTION_KEY not set in environment")
    
    cipher = Fernet(encryption_key.encode())
    encrypted = cipher.encrypt(plain_secret.encode()).decode()
    return f"dev-s-t-{encrypted}"


def main():
    print_section("LIVEKIT CONFIG CLEANUP AND STANDARDIZATION")
    
    # Check encryption key
    encryption_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
    if not encryption_key:
        print("✗ LIVEKIT_SECRET_ENCRYPTION_KEY not set in .env file")
        print("  Cannot proceed without encryption key")
        return False
    
    print(f"✓ Encryption key loaded: {encryption_key[:20]}...")
    
    db_config = get_db_config()
    schema = db_config.get("schema", "lad_dev")
    
    print(f"✓ Database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
    print(f"✓ Schema: {schema}")
    
    try:
        with get_db_connection(db_config) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                
                # STEP 1: Read all existing configs
                print_step("STEP 1: Read Existing Configurations")
                
                cur.execute(f"""
                    SELECT id, name, description, livekit_url, livekit_api_key, 
                           livekit_api_secret, trunk_id, worker_name, created_at
                    FROM {schema}.voice_agent_livekit
                    ORDER BY created_at
                """)
                
                existing_configs = cur.fetchall()
                print(f"Found {len(existing_configs)} existing configurations:")
                
                for config in existing_configs:
                    print(f"\n  • {config['name']}")
                    print(f"    ID: {config['id']}")
                    print(f"    URL: {config['livekit_url']}")
                    print(f"    API Key: {config['livekit_api_key']}")
                    print(f"    Trunk: {config['trunk_id']}")
                    print(f"    Worker: {config['worker_name']}")
                    print(f"    Secret: {config['livekit_api_secret'][:30]}...")
                
                # STEP 2: Test decryption and categorize
                print_step("STEP 2: Test Decryption and Categorize")
                
                valid_configs = []
                invalid_configs = []
                
                for config in existing_configs:
                    config_dict = dict(config)
                    can_decrypt_flag, result = can_decrypt(config_dict['livekit_api_secret'])
                    
                    if can_decrypt_flag:
                        config_dict['decrypted_secret'] = result
                        valid_configs.append(config_dict)
                        print(f"  ✓ {config_dict['name']}: Decryption OK")
                        print(f"    Decrypted: {result[:30]}...")
                    else:
                        invalid_configs.append(config_dict)
                        print(f"  ✗ {config_dict['name']}: Decryption FAILED")
                        print(f"    Error: {result}")
                
                print(f"\nSummary:")
                print(f"  Valid configs (can decrypt): {len(valid_configs)}")
                print(f"  Invalid configs (cannot decrypt): {len(invalid_configs)}")
                
                # STEP 3: Plan changes
                print_step("STEP 3: Plan Changes")
                
                # Identify which target configs already exist
                existing_by_key = {}
                for config in valid_configs:
                    # Match by URL + API Key + Trunk
                    key = (config['livekit_url'], config['livekit_api_key'], config['trunk_id'])
                    existing_by_key[key] = config
                
                configs_to_delete = []
                configs_to_update = []
                configs_to_create = []
                
                # Check each target config
                for target_key, target_config in TARGET_CONFIGS.items():
                    match_key = (
                        target_config['livekit_url'],
                        target_config['livekit_api_key'],
                        target_config['trunk_id']
                    )
                    
                    if match_key in existing_by_key:
                        existing = existing_by_key[match_key]
                        # Check if update needed
                        needs_update = (
                            existing['name'] != target_config['name'] or
                            existing['description'] != target_config['description'] or
                            existing['worker_name'] != target_config['worker_name'] or
                            existing['decrypted_secret'] != target_config['livekit_api_secret_plain']
                        )
                        
                        if needs_update:
                            configs_to_update.append({
                                'id': existing['id'],
                                'existing_name': existing['name'],
                                'target': target_config,
                                'target_key': target_key
                            })
                            print(f"  ↻ UPDATE: {existing['name']} -> {target_config['name']}")
                        else:
                            print(f"  ✓ KEEP: {existing['name']} (already correct)")
                    else:
                        configs_to_create.append({
                            'target': target_config,
                            'target_key': target_key
                        })
                        print(f"  + CREATE: {target_config['name']}")
                
                # Mark invalid configs for deletion
                for config in invalid_configs:
                    configs_to_delete.append(config)
                    print(f"  - DELETE: {config['name']} (cannot decrypt)")
                
                # Mark valid configs that don't match any target
                for config in valid_configs:
                    match_key = (config['livekit_url'], config['livekit_api_key'], config['trunk_id'])
                    if match_key not in [
                        (tc['livekit_url'], tc['livekit_api_key'], tc['trunk_id'])
                        for tc in TARGET_CONFIGS.values()
                    ]:
                        # This is a valid config but doesn't match our targets
                        # Don't delete it - keep it as-is
                        print(f"  ⚠ KEEP: {config['name']} (valid but not in target list)")
                
                print(f"\nPlan Summary:")
                print(f"  Configs to DELETE: {len(configs_to_delete)}")
                print(f"  Configs to UPDATE: {len(configs_to_update)}")
                print(f"  Configs to CREATE: {len(configs_to_create)}")
                
                # STEP 4: Execute changes in transaction
                print_step("STEP 4: Execute Changes (with transaction)")
                
                if not configs_to_delete and not configs_to_update and not configs_to_create:
                    print("  ✓ No changes needed - all configs are correct!")
                    return True
                
                print("  Starting transaction...")
                
                # Delete invalid configs
                for config in configs_to_delete:
                    print(f"  Deleting: {config['name']} (ID: {config['id']})")
                    cur.execute(f"""
                        DELETE FROM {schema}.voice_agent_livekit
                        WHERE id = %s
                    """, (config['id'],))
                
                # Update existing configs
                for update_info in configs_to_update:
                    target = update_info['target']
                    encrypted_secret = encrypt_secret(target['livekit_api_secret_plain'])
                    
                    print(f"  Updating: {update_info['existing_name']} -> {target['name']}")
                    cur.execute(f"""
                        UPDATE {schema}.voice_agent_livekit
                        SET name = %s,
                            description = %s,
                            livekit_url = %s,
                            livekit_api_key = %s,
                            livekit_api_secret = %s,
                            trunk_id = %s,
                            worker_name = %s
                        WHERE id = %s
                    """, (
                        target['name'],
                        target['description'],
                        target['livekit_url'],
                        target['livekit_api_key'],
                        encrypted_secret,
                        target['trunk_id'],
                        target['worker_name'],
                        update_info['id']
                    ))
                
                # Create new configs
                for create_info in configs_to_create:
                    target = create_info['target']
                    encrypted_secret = encrypt_secret(target['livekit_api_secret_plain'])
                    
                    print(f"  Creating: {target['name']}")
                    cur.execute(f"""
                        INSERT INTO {schema}.voice_agent_livekit
                        (name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id, worker_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        target['name'],
                        target['description'],
                        target['livekit_url'],
                        target['livekit_api_key'],
                        encrypted_secret,
                        target['trunk_id'],
                        target['worker_name']
                    ))
                    result = cur.fetchone()
                    new_id = result['id'] if result else None
                    print(f"    Created with ID: {new_id}")
                
                # Commit transaction
                conn.commit()
                print("\n  ✓ Transaction committed successfully!")
                
                # STEP 5: Verify final state
                print_step("STEP 5: Verify Final State")
                
                cur.execute(f"""
                    SELECT id, name, description, livekit_url, livekit_api_key, 
                           livekit_api_secret, trunk_id, worker_name
                    FROM {schema}.voice_agent_livekit
                    ORDER BY name
                """)
                
                final_configs = cur.fetchall()
                print(f"Final configuration count: {len(final_configs)}")
                
                for config in final_configs:
                    can_decrypt_flag, decrypted = can_decrypt(config['livekit_api_secret'])
                    
                    print(f"\n  • {config['name']}")
                    print(f"    ID: {config['id']}")
                    print(f"    URL: {config['livekit_url']}")
                    print(f"    API Key: {config['livekit_api_key']}")
                    print(f"    Trunk: {config['trunk_id']}")
                    print(f"    Worker: {config['worker_name']}")
                    
                    if can_decrypt_flag:
                        print(f"    Secret: ✓ Decrypts to {decrypted[:30]}...")
                    else:
                        print(f"    Secret: ✗ CANNOT DECRYPT!")
                
                print_section("SUCCESS - All Changes Applied")
                return True
                
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("  Transaction rolled back - no changes made")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
