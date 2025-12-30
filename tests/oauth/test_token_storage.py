"""
OAuth Token Storage Tests.

Tests the UserTokenStorage class methods:
- get_user_by_user_id()
- get_identity()
- get_google_token_blob()
- get_microsoft_token_blob()
- get_connected_gmail()
- get_booking_config()

Usage:
    cd d:\vonage\vonage-voice-agent\v2
    uv run python tests/oauth/test_token_storage.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from db.storage.tokens import UserTokenStorage

# Test user UUID (the one we verified OAuth with)
TEST_USER_ID = "81f1decc-7ee5-4093-b55c-95ac9b7c9f45"


async def test_get_user_by_user_id():
    """Test fetching user record by UUID."""
    print("\n=== Test: get_user_by_user_id ===")
    storage = UserTokenStorage()
    
    user = await storage.get_user_by_user_id(TEST_USER_ID)
    if user:
        print(f"✓ User found:")
        print(f"  - id: {user.get('id')}")
        print(f"  - email: {user.get('email')}")
        print(f"  - first_name: {user.get('first_name')}")
        print(f"  - last_name: {user.get('last_name')}")
        return True
    else:
        print(f"✗ User not found: {TEST_USER_ID}")
        return False


async def test_get_google_identity():
    """Test fetching Google identity from user_identities."""
    print("\n=== Test: get_identity (google) ===")
    storage = UserTokenStorage()
    
    identity = await storage.get_identity(TEST_USER_ID, "google")
    if identity:
        print(f"✓ Google identity found:")
        print(f"  - provider: {identity.get('provider')}")
        print(f"  - provider_user_id: {identity.get('provider_user_id')}")
        print(f"  - has provider_data: {bool(identity.get('provider_data'))}")
        print(f"  - updated_at: {identity.get('updated_at')}")
        return True
    else:
        print(f"✗ Google identity not found for user: {TEST_USER_ID}")
        return False


async def test_get_microsoft_identity():
    """Test fetching Microsoft identity from user_identities."""
    print("\n=== Test: get_identity (microsoft) ===")
    storage = UserTokenStorage()
    
    identity = await storage.get_identity(TEST_USER_ID, "microsoft")
    if identity:
        print(f"✓ Microsoft identity found:")
        print(f"  - provider: {identity.get('provider')}")
        print(f"  - provider_user_id: {identity.get('provider_user_id')}")
        print(f"  - has provider_data: {bool(identity.get('provider_data'))}")
        print(f"  - updated_at: {identity.get('updated_at')}")
        return True
    else:
        print(f"✗ Microsoft identity not found for user: {TEST_USER_ID}")
        return False


async def test_get_google_token_blob():
    """Test fetching and decoding Google token blob."""
    print("\n=== Test: get_google_token_blob ===")
    storage = UserTokenStorage()
    
    blob = await storage.get_google_token_blob(TEST_USER_ID)
    if blob:
        print(f"✓ Google token blob retrieved:")
        print(f"  - size: {len(blob)} bytes")
        print(f"  - first 50 bytes: {blob[:50]}...")
        return True
    else:
        print(f"✗ Google token blob not found for user: {TEST_USER_ID}")
        return False


async def test_get_microsoft_token_blob():
    """Test fetching and decoding Microsoft token blob."""
    print("\n=== Test: get_microsoft_token_blob ===")
    storage = UserTokenStorage()
    
    blob = await storage.get_microsoft_token_blob(TEST_USER_ID)
    if blob:
        print(f"✓ Microsoft token blob retrieved:")
        print(f"  - size: {len(blob)} bytes")
        print(f"  - first 50 bytes: {blob[:50]}...")
        return True
    else:
        print(f"✗ Microsoft token blob not found for user: {TEST_USER_ID}")
        return False


async def test_get_connected_gmail():
    """Test getting connected Gmail address."""
    print("\n=== Test: get_connected_gmail ===")
    storage = UserTokenStorage()
    
    gmail = await storage.get_connected_gmail(TEST_USER_ID)
    if gmail:
        print(f"✓ Connected Gmail: {gmail}")
        return True
    else:
        print(f"✗ Connected Gmail not found for user: {TEST_USER_ID}")
        return False


async def test_decrypt_google_tokens():
    """Test actually decrypting the Google token blob."""
    print("\n=== Test: Decrypt Google Tokens ===")
    from utils.google_oauth import TokenEncryptor, get_google_oauth_settings
    
    storage = UserTokenStorage()
    blob = await storage.get_google_token_blob(TEST_USER_ID)
    
    if not blob:
        print(f"✗ No Google token blob to decrypt")
        return False
    
    try:
        settings = get_google_oauth_settings()
        encryptor = TokenEncryptor(settings.encryption_key)
        payload = encryptor.decrypt_json(blob)
        
        if payload:
            print(f"✓ Google tokens decrypted successfully:")
            print(f"  - has token: {bool(payload.get('token'))}")
            print(f"  - has refresh_token: {bool(payload.get('refresh_token'))}")
            print(f"  - scopes: {payload.get('scopes', [])}")
            print(f"  - expiry: {payload.get('expiry')}")
            return True
        else:
            print(f"✗ Decrypted payload is empty")
            return False
    except Exception as e:
        print(f"✗ Failed to decrypt: {e}")
        return False


async def test_decrypt_microsoft_tokens():
    """Test actually decrypting the Microsoft token blob."""
    print("\n=== Test: Decrypt Microsoft Tokens ===")
    from utils.google_oauth import TokenEncryptor, get_google_oauth_settings
    
    storage = UserTokenStorage()
    blob = await storage.get_microsoft_token_blob(TEST_USER_ID)
    
    if not blob:
        print(f"✗ No Microsoft token blob to decrypt")
        return False
    
    try:
        settings = get_google_oauth_settings()
        encryptor = TokenEncryptor(settings.encryption_key)
        payload = encryptor.decrypt_json(blob)
        
        if payload:
            print(f"✓ Microsoft tokens decrypted successfully:")
            print(f"  - has access_token: {bool(payload.get('access_token'))}")
            print(f"  - has refresh_token: {bool(payload.get('refresh_token'))}")
            print(f"  - scope: {payload.get('scope', '')[:100]}...")
            print(f"  - expires_at: {payload.get('expires_at')}")
            return True
        else:
            print(f"✗ Decrypted payload is empty")
            return False
    except Exception as e:
        print(f"✗ Failed to decrypt: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("OAuth Token Storage Tests")
    print(f"Test User ID: {TEST_USER_ID}")
    print("=" * 60)
    
    results = []
    
    # User lookup test
    results.append(("get_user_by_user_id", await test_get_user_by_user_id()))
    
    # Identity tests
    results.append(("get_identity (google)", await test_get_google_identity()))
    results.append(("get_identity (microsoft)", await test_get_microsoft_identity()))
    
    # Token blob tests
    results.append(("get_google_token_blob", await test_get_google_token_blob()))
    results.append(("get_microsoft_token_blob", await test_get_microsoft_token_blob()))
    
    # Connected email test
    results.append(("get_connected_gmail", await test_get_connected_gmail()))
    
    # Decryption tests
    results.append(("decrypt_google_tokens", await test_decrypt_google_tokens()))
    results.append(("decrypt_microsoft_tokens", await test_decrypt_microsoft_tokens()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
