"""
Individual Tool Tests for OAuth-Dependent Tools.

Tests each tool individually with detailed output monitoring:
1. GoogleCalendarTool - list_events, create_event
2. GmailEmailTool - send_email
3. MicrosoftBookingsTool - get_booking_businesses, get_services, etc.

Usage:
    cd d:\vonage\vonage-voice-agent\v2
    uv run python tests/oauth/test_tools_individual.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Test user UUID
TEST_USER_ID = "81f1decc-7ee5-4093-b55c-95ac9b7c9f45"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subsection(title: str):
    print(f"\n--- {title} ---")


# =============================================================================
# GOOGLE CALENDAR TOOL TESTS
# =============================================================================

async def test_google_calendar_tool_list_events():
    """Test GoogleCalendarTool.list_events()"""
    print_subsection("GoogleCalendarTool.list_events()")
    
    from auth.google import GoogleCredentialResolver
    from tools.google_calendar_tool import GoogleCalendarTool, CalendarToolError
    
    try:
        print("  [1] Loading Google credentials...")
        resolver = GoogleCredentialResolver()
        credentials = await resolver.load_credentials(TEST_USER_ID)
        print(f"      ✓ Credentials loaded (token valid: {credentials.valid})")
        
        print("  [2] Creating GoogleCalendarTool...")
        tool = GoogleCalendarTool(credentials)
        print("      ✓ Tool created")
        
        print("  [3] Calling list_events()...")
        now = datetime.now(timezone.utc)
        end = now + timedelta(days=7)
        
        events = tool.list_events(calendar_id="primary", start=now, end=end, max_results=10)
        
        print(f"      ✓ Retrieved {len(events)} events")
        for i, event in enumerate(events[:5], 1):
            start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", "Unknown"))
            summary = event.get("summary", "No title")
            print(f"        {i}. {start[:16]} - {summary[:50]}")
        
        return True, "list_events works correctly"
        
    except CalendarToolError as e:
        print(f"      ✗ CalendarToolError: {e}")
        return False, str(e)
    except Exception as e:
        print(f"      ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, str(e)


async def test_google_calendar_tool_create_event():
    """Test GoogleCalendarTool.create_event() - DRY RUN"""
    print_subsection("GoogleCalendarTool.create_event() [DRY RUN]")
    
    from auth.google import GoogleCredentialResolver
    from tools.google_calendar_tool import GoogleCalendarTool, CalendarEventRequest, CalendarToolError
    
    try:
        print("  [1] Loading Google credentials...")
        resolver = GoogleCredentialResolver()
        credentials = await resolver.load_credentials(TEST_USER_ID)
        print(f"      ✓ Credentials loaded")
        
        print("  [2] Creating GoogleCalendarTool...")
        tool = GoogleCalendarTool(credentials)
        print("      ✓ Tool created")
        
        print("  [3] Preparing event payload (validation only)...")
        start = datetime.now(timezone.utc) + timedelta(days=1)
        end = start + timedelta(hours=1)
        
        payload = CalendarEventRequest(
            summary="Test Event - Do Not Create",
            description="This is a test event",
            start=start,
            end=end,
            timezone="UTC",
            attendees=[],
            meet_required=False,
            send_updates="none"
        )
        print(f"      ✓ Payload created: {payload.summary}")
        print("  [4] Skipping actual creation (dry run mode)")
        
        return True, "create_event payload validation passed"
        
    except CalendarToolError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# =============================================================================
# GMAIL EMAIL TOOL TESTS
# =============================================================================

async def test_gmail_email_tool():
    """Test GmailEmailTool - DRY RUN"""
    print_subsection("GmailEmailTool.send_email() [DRY RUN]")
    
    from auth.google import GoogleCredentialResolver
    from tools.gmail_email_tool import GmailEmailTool, EmailPayload, GmailToolError
    
    try:
        print("  [1] Loading Google credentials...")
        resolver = GoogleCredentialResolver()
        credentials = await resolver.load_credentials(TEST_USER_ID)
        print(f"      ✓ Credentials loaded")
        
        print("  [2] Creating GmailEmailTool...")
        tool = GmailEmailTool(credentials)
        print("      ✓ Tool created")
        
        print("  [3] Preparing email payload...")
        payload = EmailPayload(
            to=["test@example.com"],
            subject="Test Email - Do Not Send",
            text_body="This is a test email body",
        )
        print(f"      ✓ Payload created: Subject='{payload.subject}'")
        
        print("  [4] Testing MIME message building...")
        mime_msg = tool._build_mime_message(payload)
        print(f"      ✓ MIME message built: To={mime_msg['To']}")
        
        print("  [5] Skipping actual send (dry run mode)")
        
        return True, "Gmail payload validation passed"
        
    except GmailToolError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# =============================================================================
# MICROSOFT BOOKINGS TOOL TESTS
# =============================================================================

async def get_microsoft_access_token():
    """Helper to get Microsoft access token."""
    from db.storage.tokens import UserTokenStorage
    from utils.google_oauth import TokenEncryptor, get_google_oauth_settings
    
    storage = UserTokenStorage()
    blob = await storage.get_microsoft_token_blob(TEST_USER_ID)
    
    if not blob:
        return None
    
    settings = get_google_oauth_settings()
    encryptor = TokenEncryptor(settings.encryption_key)
    payload = encryptor.decrypt_json(blob)
    
    return payload.get("access_token")


async def test_microsoft_bookings_tool_get_businesses():
    """Test MicrosoftBookingsTool.get_booking_businesses()"""
    print_subsection("MicrosoftBookingsTool.get_booking_businesses()")
    
    from tools.microsoft_bookings_tool import MicrosoftBookingsTool, MicrosoftBookingsToolError
    
    try:
        print("  [1] Loading Microsoft access token...")
        access_token = await get_microsoft_access_token()
        if not access_token:
            print("      ✗ No access token found")
            return False, "No Microsoft access token"
        print(f"      ✓ Access token loaded ({len(access_token)} chars)")
        
        print("  [2] Creating MicrosoftBookingsTool...")
        tool = MicrosoftBookingsTool(access_token)
        print("      ✓ Tool created")
        
        print("  [3] Calling get_booking_businesses()...")
        businesses = await tool.get_booking_businesses()  # AWAIT
        
        print(f"      ✓ Retrieved {len(businesses)} businesses")
        for i, biz in enumerate(businesses[:5], 1):
            biz_id = biz.get('id', 'N/A')
            print(f"        {i}. {biz.get('displayName', 'Unknown')} (ID: {biz_id[:20]}...)")
        
        if businesses:
            return True, f"Found {len(businesses)} booking businesses"
        else:
            print("      ⚠ No booking businesses found")
            return True, "No booking businesses but API works"
        
    except MicrosoftBookingsToolError as e:
        print(f"      ✗ MicrosoftBookingsToolError: {e}")
        return False, str(e)
    except Exception as e:
        print(f"      ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, str(e)


async def test_microsoft_bookings_tool_get_services():
    """Test MicrosoftBookingsTool.get_services()"""
    print_subsection("MicrosoftBookingsTool.get_services()")
    
    from tools.microsoft_bookings_tool import MicrosoftBookingsTool, MicrosoftBookingsToolError
    
    try:
        print("  [1] Loading Microsoft access token...")
        access_token = await get_microsoft_access_token()
        if not access_token:
            return False, "No Microsoft access token"
        print(f"      ✓ Access token loaded")
        
        print("  [2] Creating MicrosoftBookingsTool...")
        tool = MicrosoftBookingsTool(access_token)
        
        print("  [3] Getting booking businesses...")
        businesses = await tool.get_booking_businesses()  # AWAIT
        if not businesses:
            print("      ⚠ No booking businesses found")
            return True, "Skipped - no booking businesses"
        
        business_id = businesses[0]["id"]
        print(f"      ✓ Using business: {businesses[0].get('displayName')}")
        
        print("  [4] Calling get_services()...")
        services = await tool.get_services(business_id)  # AWAIT
        
        print(f"      ✓ Retrieved {len(services)} services")
        for i, svc in enumerate(services[:5], 1):
            duration = svc.get("defaultDuration", "Unknown")
            print(f"        {i}. {svc.get('displayName', 'Unknown')} ({duration})")
        
        return True, f"Found {len(services)} services"
        
    except MicrosoftBookingsToolError as e:
        print(f"      ✗ MicrosoftBookingsToolError: {e}")
        return False, str(e)
    except Exception as e:
        print(f"      ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, str(e)


async def test_microsoft_bookings_tool_get_staff():
    """Test MicrosoftBookingsTool.get_staff_members()"""
    print_subsection("MicrosoftBookingsTool.get_staff_members()")
    
    from tools.microsoft_bookings_tool import MicrosoftBookingsTool, MicrosoftBookingsToolError
    
    try:
        print("  [1] Loading Microsoft access token...")
        access_token = await get_microsoft_access_token()
        if not access_token:
            return False, "No Microsoft access token"
        print(f"      ✓ Access token loaded")
        
        print("  [2] Creating tool and getting businesses...")
        tool = MicrosoftBookingsTool(access_token)
        businesses = await tool.get_booking_businesses()  # AWAIT
        if not businesses:
            return True, "Skipped - no booking businesses"
        
        business_id = businesses[0]["id"]
        print(f"      ✓ Using business: {businesses[0].get('displayName')}")
        
        print("  [3] Calling get_staff_members()...")
        staff = await tool.get_staff_members(business_id)  # AWAIT
        
        print(f"      ✓ Retrieved {len(staff)} staff members")
        for i, member in enumerate(staff[:5], 1):
            email = member.get("emailAddress", "No email")
            print(f"        {i}. {member.get('displayName', 'Unknown')} ({email})")
        
        return True, f"Found {len(staff)} staff members"
        
    except MicrosoftBookingsToolError as e:
        print(f"      ✗ MicrosoftBookingsToolError: {e}")
        return False, str(e)
    except Exception as e:
        print(f"      ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, str(e)


async def test_microsoft_bookings_tool_get_availability():
    """Test MicrosoftBookingsTool.get_availability()"""
    print_subsection("MicrosoftBookingsTool.get_availability()")
    
    from tools.microsoft_bookings_tool import MicrosoftBookingsTool, MicrosoftBookingsToolError
    
    try:
        print("  [1] Loading Microsoft access token...")
        access_token = await get_microsoft_access_token()
        if not access_token:
            return False, "No Microsoft access token"
        print(f"      ✓ Access token loaded")
        
        print("  [2] Getting businesses and services...")
        tool = MicrosoftBookingsTool(access_token)
        businesses = await tool.get_booking_businesses()  # AWAIT
        if not businesses:
            return True, "Skipped - no booking businesses"
        
        business_id = businesses[0]["id"]
        services = await tool.get_services(business_id)  # AWAIT
        if not services:
            return True, "Skipped - no services"
        
        service_id = services[0]["id"]
        print(f"      ✓ Business: {businesses[0].get('displayName')}")
        print(f"      ✓ Service: {services[0].get('displayName')}")
        
        print("  [3] Calling get_availability()...")
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=7)
        
        slots = await tool.get_availability(  # AWAIT
            business_id=business_id,
            service_id=service_id,
            start_date=start,
            end_date=end
        )
        
        print(f"      ✓ Retrieved {len(slots)} available slots")
        for i, slot in enumerate(slots[:5], 1):
            start_dt = slot.get("startDateTime", {}).get("dateTime", "Unknown")[:16]
            end_dt = slot.get("endDateTime", {}).get("dateTime", "Unknown")[:16]
            print(f"        {i}. {start_dt} - {end_dt}")
        
        return True, f"Found {len(slots)} available slots"
        
    except MicrosoftBookingsToolError as e:
        print(f"      ✗ MicrosoftBookingsToolError: {e}")
        return False, str(e)
    except Exception as e:
        print(f"      ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, str(e)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all individual tool tests."""
    print("=" * 70)
    print(" INDIVIDUAL OAUTH TOOL TESTS")
    print(f" Test User ID: {TEST_USER_ID}")
    print("=" * 70)
    
    results = []
    
    # Google Calendar Tool
    print_section("GOOGLE CALENDAR TOOL")
    success, msg = await test_google_calendar_tool_list_events()
    results.append(("GoogleCalendarTool.list_events", success, msg))
    
    success, msg = await test_google_calendar_tool_create_event()
    results.append(("GoogleCalendarTool.create_event [DRY]", success, msg))
    
    # Gmail Email Tool
    print_section("GMAIL EMAIL TOOL")
    success, msg = await test_gmail_email_tool()
    results.append(("GmailEmailTool.send_email [DRY]", success, msg))
    
    # Microsoft Bookings Tool
    print_section("MICROSOFT BOOKINGS TOOL")
    success, msg = await test_microsoft_bookings_tool_get_businesses()
    results.append(("MicrosoftBookingsTool.get_businesses", success, msg))
    
    success, msg = await test_microsoft_bookings_tool_get_services()
    results.append(("MicrosoftBookingsTool.get_services", success, msg))
    
    success, msg = await test_microsoft_bookings_tool_get_staff()
    results.append(("MicrosoftBookingsTool.get_staff_members", success, msg))
    
    success, msg = await test_microsoft_bookings_tool_get_availability()
    results.append(("MicrosoftBookingsTool.get_availability", success, msg))
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, msg in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if not success:
            print(f"          -> {msg}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
