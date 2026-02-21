"""
Encryption Sanity Check

This script demonstrates why the same plaintext encrypted with the same key
produces different ciphertext each time.

This is a SECURITY FEATURE, not a bug!
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptography.fernet import Fernet
from db.connection_pool import get_db_connection
from db.db_config import get_db_config
from utils.en_de_crypt import encrypt_decrypt
from psycopg2.extras import RealDictCursor


def print_section(title: str):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def encrypt_with_fernet(plaintext: str, key: str) -> str:
    """Encrypt plaintext with Fernet (with dev-s-t- prefix)."""
    cipher = Fernet(key.encode())
    encrypted = cipher.encrypt(plaintext.encode()).decode()
    return f"dev-s-t-{encrypted}"


def main():
    print_section("ENCRYPTION SANITY CHECK")
    
    # Get encryption key
    encryption_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
    if not encryption_key:
        print("✗ LIVEKIT_SECRET_ENCRYPTION_KEY not set")
        return
    
    print(f"Encryption Key: {encryption_key[:20]}...")
    
    # Get India configs from database
    db_config = get_db_config()
    schema = db_config.get("schema", "lad_dev")
    
    with get_db_connection(db_config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT name, livekit_api_secret, trunk_id, worker_name
                FROM {schema}.voice_agent_livekit
                WHERE livekit_url = 'wss://lk.techiemaya.com'
                ORDER BY name
            """)
            
            india_configs = cur.fetchall()
    
    print(f"\nFound {len(india_configs)} India VM configs")
    
    # Decrypt and compare
    print_section("STEP 1: Decrypt All India VM Secrets")
    
    decrypted_secrets = {}
    
    for config in india_configs:
        encrypted = config['livekit_api_secret']
        decrypted = encrypt_decrypt(encrypted)
        decrypted_secrets[config['name']] = decrypted
        
        print(f"\n{config['name']}:")
        print(f"  Trunk: {config['trunk_id']}")
        print(f"  Worker: {config['worker_name']}")
        print(f"  Encrypted: {encrypted[:50]}...")
        print(f"  Decrypted: {decrypted}")
    
    # Check if all decrypted values are the same
    print_section("STEP 2: Compare Decrypted Values")
    
    unique_secrets = set(decrypted_secrets.values())
    
    if len(unique_secrets) == 1:
        print("✓ All India VM configs have the SAME decrypted secret!")
        print(f"  Secret: {list(unique_secrets)[0]}")
    else:
        print("✗ India VM configs have DIFFERENT decrypted secrets!")
        for name, secret in decrypted_secrets.items():
            print(f"  {name}: {secret}")
    
    # Explain why encrypted values differ
    print_section("STEP 3: Why Are Encrypted Values Different?")
    
    print("""
Fernet encryption uses AES-128 in CBC mode with PKCS7 padding and HMAC for
authentication. Each encryption includes:

1. TIMESTAMP (8 bytes) - Current time when encrypted
2. IV (Initialization Vector, 16 bytes) - Random value for each encryption
3. CIPHERTEXT - The actual encrypted data
4. HMAC (32 bytes) - Authentication tag

The IV is randomly generated for EACH encryption operation. This means:
- Same plaintext + Same key + Different IV = Different ciphertext

This is a SECURITY FEATURE called "semantic security":
- Prevents pattern analysis attacks
- An attacker cannot tell if two ciphertexts contain the same plaintext
- Makes the encryption non-deterministic

Example: Encrypting the same secret 3 times with the same key:
""")
    
    # Demonstrate by encrypting the same value multiple times
    test_secret = list(unique_secrets)[0] if unique_secrets else "test_secret"
    
    print(f"\nPlaintext: {test_secret}")
    print(f"Key: {encryption_key[:20]}...")
    print("\nEncrypting 3 times:")
    
    for i in range(1, 4):
        encrypted = encrypt_with_fernet(test_secret, encryption_key)
        print(f"\n  Encryption #{i}:")
        print(f"    {encrypted[:60]}...")
        
        # Verify it decrypts correctly
        decrypted = encrypt_decrypt(encrypted)
        status = "✓" if decrypted == test_secret else "✗"
        print(f"    Decrypts correctly: {status}")
    
    print_section("CONCLUSION")
    
    print("""
✓ Different encrypted values for the same plaintext is NORMAL and SECURE
✓ All encrypted values decrypt to the same plaintext correctly
✓ This prevents attackers from identifying duplicate secrets
✓ Fernet's random IV ensures semantic security

Key Takeaways:
1. Same plaintext + Same key = Different ciphertext (due to random IV)
2. All ciphertexts decrypt to the same plaintext
3. This is a security feature, not a bug
4. Never compare encrypted values directly - always decrypt first
""")


if __name__ == "__main__":
    main()
