"""
Google Workspace Tool Tests.

Tests the Google Workspace integration:
- Google Calendar (create/check events)
- Gmail (send emails)

Usage:
    cd d:\vonage\vonage-voice-agent\v2
    uv run python tests/oauth/test_google_workspace.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Test user UUID
TEST_USER_ID = "81f1decc-7ee5-4093-b55c-95ac9b7c9f45"


async def test_google_credential_resolver():
    """Test that GoogleCredentialResolver can load credentials."""
    print("\n=== Test: GoogleCredentialResolver ===")
    
    from auth.google import GoogleCredentialResolver, GoogleCredentialError
    
    resolver = GoogleCredentialResolver()
    
    try:
        credentials = await resolver.load_credentials(TEST_USER_ID)
        print(f"✓ Credentials loaded successfully:")
        print(f"  - has token: {bool(credentials.token)}")
        print(f"  - has refresh_token: {bool(credentials.refresh_token)}")
        print(f"  - token valid: {credentials.valid}")
        print(f"  - scopes: {list(credentials.scopes or [])[:3]}...")
        return True
    except GoogleCredentialError as e:
        print(f"✗ Failed to load credentials: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


async def test_google_calendar_list():
    """Test listing Google Calendar events."""
    print("\n=== Test: Google Calendar - List Events ===")
    
    from auth.google import GoogleCredentialResolver
    from googleapiclient.discovery import build
    
    resolver = GoogleCredentialResolver()
    
    try:
        credentials = await resolver.load_credentials(TEST_USER_ID)
        
        # Build calendar service
        service = build("calendar", "v3", credentials=credentials)
        
        # List upcoming events
        now = datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=5,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])
        print(f"✓ Calendar events retrieved: {len(events)} upcoming events")
        
        for event in events[:3]:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(f"  - {start[:16]}: {event.get('summary', 'No title')[:40]}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to list calendar events: {e}")
        return False


async def test_google_gmail_profile():
    """Test fetching Gmail profile (doesn't send email)."""
    print("\n=== Test: Gmail - Get Profile ===")
    
    from auth.google import GoogleCredentialResolver
    from googleapiclient.discovery import build
    
    resolver = GoogleCredentialResolver()
    
    try:
        credentials = await resolver.load_credentials(TEST_USER_ID)
        
        # Build Gmail service
        service = build("gmail", "v1", credentials=credentials)
        
        # Get profile
        profile = service.users().getProfile(userId="me").execute()
        
        print(f"✓ Gmail profile retrieved:")
        print(f"  - email: {profile.get('emailAddress')}")
        print(f"  - messages total: {profile.get('messagesTotal', 0)}")
        print(f"  - threads total: {profile.get('threadsTotal', 0)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to get Gmail profile: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Google Workspace Tool Tests")
    print(f"Test User ID: {TEST_USER_ID}")
    print("=" * 60)
    
    results = []
    
    # Credential loading
    results.append(("GoogleCredentialResolver", await test_google_credential_resolver()))
    
    # Calendar API
    results.append(("Google Calendar - List Events", await test_google_calendar_list()))
    
    # Gmail API
    results.append(("Gmail - Get Profile", await test_google_gmail_profile()))
    
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
