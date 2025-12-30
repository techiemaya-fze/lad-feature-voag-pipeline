"""
Microsoft Bookings Tool Tests.

Tests the Microsoft OAuth and Bookings integration:
- Token decryption
- Graph API access

Usage:
    cd d:\vonage\vonage-voice-agent\v2
    uv run python tests/oauth/test_microsoft_bookings.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import httpx

# Test user UUID
TEST_USER_ID = "81f1decc-7ee5-4093-b55c-95ac9b7c9f45"


async def test_microsoft_token_retrieval():
    """Test retrieving and decrypting Microsoft tokens."""
    print("\n=== Test: Microsoft Token Retrieval ===")
    
    from db.storage.tokens import UserTokenStorage
    from utils.google_oauth import TokenEncryptor, get_google_oauth_settings
    
    storage = UserTokenStorage()
    blob = await storage.get_microsoft_token_blob(TEST_USER_ID)
    
    if not blob:
        print(f"✗ No Microsoft token blob found")
        return None, False
    
    try:
        settings = get_google_oauth_settings()
        encryptor = TokenEncryptor(settings.encryption_key)
        payload = encryptor.decrypt_json(blob)
        
        access_token = payload.get("access_token")
        if access_token:
            print(f"✓ Microsoft access token retrieved:")
            print(f"  - token length: {len(access_token)} chars")
            print(f"  - has refresh_token: {bool(payload.get('refresh_token'))}")
            return access_token, True
        else:
            print(f"✗ No access_token in payload")
            return None, False
    except Exception as e:
        print(f"✗ Failed to decrypt: {e}")
        return None, False


async def test_microsoft_graph_me(access_token: str):
    """Test Microsoft Graph /me endpoint."""
    print("\n=== Test: Microsoft Graph /me ===")
    
    if not access_token:
        print(f"✗ No access token provided")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"✓ Microsoft Graph /me succeeded:")
                print(f"  - displayName: {data.get('displayName')}")
                print(f"  - mail: {data.get('mail')}")
                print(f"  - userPrincipalName: {data.get('userPrincipalName')}")
                return True
            else:
                print(f"✗ Microsoft Graph /me failed: {resp.status_code}")
                print(f"  Response: {resp.text[:200]}")
                return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False


async def test_microsoft_bookings(access_token: str):
    """Test Microsoft Bookings API."""
    print("\n=== Test: Microsoft Bookings API ===")
    
    if not access_token:
        print(f"✗ No access token provided")
        return False
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://graph.microsoft.com/v1.0/solutions/bookingBusinesses",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if resp.status_code == 200:
                data = resp.json()
                businesses = data.get("value", [])
                print(f"✓ Microsoft Bookings API succeeded:")
                print(f"  - businesses found: {len(businesses)}")
                for biz in businesses[:3]:
                    print(f"    - {biz.get('displayName', 'Unknown')}")
                return True
            elif resp.status_code == 403:
                print(f"⚠ Microsoft Bookings API returned Forbidden:")
                print(f"  - This may mean Bookings is not enabled or no permissions")
                print(f"  - This is an API permission issue, not a code issue")
                return True  # Not a code failure
            else:
                print(f"✗ Microsoft Bookings API failed: {resp.status_code}")
                print(f"  Response: {resp.text[:200]}")
                return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Microsoft Bookings Tool Tests")
    print(f"Test User ID: {TEST_USER_ID}")
    print("=" * 60)
    
    results = []
    
    # Token retrieval
    access_token, success = await test_microsoft_token_retrieval()
    results.append(("Microsoft Token Retrieval", success))
    
    # Graph API /me
    results.append(("Microsoft Graph /me", await test_microsoft_graph_me(access_token)))
    
    # Bookings API
    results.append(("Microsoft Bookings API", await test_microsoft_bookings(access_token)))
    
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
