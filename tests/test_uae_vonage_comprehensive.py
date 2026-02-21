"""
Comprehensive Test for UAE VM LiveKit with Vonage Trunk

This test performs thorough validation of:
1. Database credential resolution and decryption
2. LiveKit server connectivity (can we reach the server?)
3. LiveKit API authentication (are credentials valid?)
4. Room creation (can we create rooms?)
5. SIP trunk configuration (is trunk_id valid?)
6. Agent dispatch (does the worker receive the job?)
7. Call initiation (does the call actually start?)

Expected Configuration:
- LiveKit URL: http://91.74.244.94:7880
- API Key: APIbe273e3142c7b96a4a87bba4
- API Secret: SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e
- Trunk ID: ST_UD8BthXGHtbp (from database config)
- Worker Name: voag-staging
- From Number: +19513456728
- To Number: +918384884150
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from livekit import api
from api.services.call_service import get_call_service
from utils.call_routing import validate_and_format_call
from utils.livekit_resolver import resolve_livekit_credentials
from db.db_config import get_db_config


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_step(step_num: int, title: str):
    """Print a formatted step header."""
    print(f"\n{step_num}. {title}")
    print("-" * 80)


def print_success(message: str, indent: int = 3):
    """Print a success message."""
    print(f"{' ' * indent}✓ {message}")


def print_error(message: str, indent: int = 3):
    """Print an error message."""
    print(f"{' ' * indent}✗ {message}")


def print_warning(message: str, indent: int = 3):
    """Print a warning message."""
    print(f"{' ' * indent}⚠ {message}")


def print_info(message: str, indent: int = 3):
    """Print an info message."""
    print(f"{' ' * indent}• {message}")


async def test_livekit_connectivity(url: str, api_key: str, api_secret: str) -> bool:
    """Test if we can connect to the LiveKit server."""
    print_info("Testing LiveKit server connectivity...")
    
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
            # Try to list rooms (this will fail if server is unreachable or auth is wrong)
            rooms_response = await lk_api.room.list_rooms(api.ListRoomsRequest())
            room_count = len(rooms_response.rooms) if hasattr(rooms_response, 'rooms') else 0
            print_success(f"Connected to LiveKit server successfully")
            print_info(f"Server has {room_count} active rooms")
            return True
    except Exception as e:
        print_error(f"Failed to connect to LiveKit server: {e}")
        print_info(f"Error type: {type(e).__name__}")
        return False


async def test_room_creation(url: str, api_key: str, api_secret: str) -> tuple[bool, str]:
    """Test if we can create a room."""
    print_info("Testing room creation...")
    
    test_room_name = f"test-room-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
            # Create a test room
            room = await lk_api.room.create_room(
                api.CreateRoomRequest(name=test_room_name)
            )
            print_success(f"Room created: {test_room_name}")
            print_info(f"Room SID: {room.sid}")
            print_info(f"Room name: {room.name}")
            return True, test_room_name
    except Exception as e:
        print_error(f"Failed to create room: {e}")
        print_info(f"Error type: {type(e).__name__}")
        return False, ""


async def test_room_deletion(url: str, api_key: str, api_secret: str, room_name: str) -> bool:
    """Test if we can delete a room."""
    print_info(f"Cleaning up test room: {room_name}")
    
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
            await lk_api.room.delete_room(
                api.DeleteRoomRequest(room=room_name)
            )
            print_success(f"Test room deleted successfully")
            return True
    except Exception as e:
        print_warning(f"Failed to delete test room: {e}")
        return False


async def test_agent_dispatch(url: str, api_key: str, api_secret: str, room_name: str, 
                              worker_name: str, metadata: dict) -> tuple[bool, str]:
    """Test if we can dispatch an agent."""
    print_info("Testing agent dispatch...")
    
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
            # Dispatch agent
            dispatch_result = await lk_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=worker_name,
                    room=room_name,
                    metadata=json.dumps(metadata),
                )
            )
            
            dispatch_id = dispatch_result.id if dispatch_result else None
            print_success(f"Agent dispatched successfully")
            print_info(f"Dispatch ID: {dispatch_id}")
            print_info(f"Agent Name: {worker_name}")
            print_info(f"Room: {room_name}")
            return True, dispatch_id
    except Exception as e:
        print_error(f"Failed to dispatch agent: {e}")
        print_info(f"Error type: {type(e).__name__}")
        return False, ""


async def check_sip_participant(url: str, api_key: str, api_secret: str, room_name: str) -> bool:
    """Check if a SIP participant was created in the room."""
    print_info("Checking for SIP participant...")
    
    try:
        async with api.LiveKitAPI(url, api_key, api_secret) as lk_api:
            # Wait a bit for SIP participant to join
            await asyncio.sleep(3)
            
            # List participants in the room
            participants_response = await lk_api.room.list_participants(
                api.ListParticipantsRequest(room=room_name)
            )
            participants = participants_response.participants if hasattr(participants_response, 'participants') else []
            
            print_info(f"Room has {len(participants)} participants")
            
            for p in participants:
                print_info(f"  - {p.identity} (kind: {p.kind})")
                if "sip" in p.identity.lower() or p.kind == "SIP":
                    print_success(f"SIP participant found: {p.identity}")
                    return True
            
            if len(participants) == 0:
                print_warning("No participants in room yet")
            else:
                print_warning("No SIP participant found")
            
            return False
    except Exception as e:
        print_warning(f"Failed to check participants: {e}")
        return False


async def run_comprehensive_test():
    """Run comprehensive LiveKit and call dispatch test."""
    
    print_section("COMPREHENSIVE UAE VM LIVEKIT TEST (VONAGE TRUNK)")
    
    # Configuration
    from_number = "+19513456728"
    to_number = "+918384884150"
    agent_id = 33
    tenant_id = "e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5"
    voice_id = "default"
    
    print_info(f"From: {from_number}")
    print_info(f"To: {to_number}")
    print_info(f"Agent: {agent_id}")
    print_info(f"Tenant: {tenant_id[:8]}...")
    
    # Step 1: Initialize services
    print_step(1, "Initialize Call Service")
    call_service = get_call_service()
    await call_service._ensure_storage()
    print_success("Call service initialized")
    
    # Step 2: Resolve voice
    print_step(2, "Resolve Voice Configuration")
    try:
        resolved_voice_id, voice_context = await call_service.resolve_voice(voice_id, agent_id)
        print_success(f"Voice resolved: {voice_context.voice_name}")
        print_info(f"Provider: {voice_context.provider}")
        print_info(f"TTS Voice ID: {voice_context.tts_voice_id}")
    except Exception as e:
        print_error(f"Voice resolution failed: {e}")
        return
    
    # Step 3: Validate call routing
    print_step(3, "Validate Call Routing")
    routing_result = validate_and_format_call(
        from_number=from_number,
        to_number=to_number,
        db_config=get_db_config(),
        tenant_id=tenant_id,
    )
    
    if not routing_result.success:
        print_error(f"Call routing failed: {routing_result.error_message}")
        return
    
    print_success("Call routing validated")
    print_info(f"Formatted number: {routing_result.formatted_to_number}")
    print_info(f"Carrier: {routing_result.carrier_name}")
    print_info(f"Trunk ID: {routing_result.outbound_trunk_id}")
    print_info(f"LiveKit Config ID: {routing_result.livekit_config_id[:8]}...")
    
    # Step 4: Resolve LiveKit credentials
    print_step(4, "Resolve LiveKit Credentials from Database")
    livekit_creds = await resolve_livekit_credentials(
        from_number=from_number,
        tenant_id=tenant_id,
        routing_result=routing_result,
    )
    
    if livekit_creds.source != "database":
        print_error("Credentials came from ENVIRONMENT, not database!")
        print_warning("Check encryption key and database configuration")
        return
    
    print_success("Credentials resolved from DATABASE")
    print_info(f"URL: {livekit_creds.url}")
    print_info(f"API Key: {livekit_creds.api_key}")
    print_info(f"API Secret: {livekit_creds.api_secret[:20]}... (decrypted)")
    print_info(f"Trunk ID: {livekit_creds.trunk_id}")
    print_info(f"Worker Name: {livekit_creds.worker_name}")
    
    # Verify expected values
    expected_url = "http://91.74.244.94:7880"
    expected_api_key = "APIbe273e3142c7b96a4a87bba4"
    expected_secret = "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e"
    
    if livekit_creds.url != expected_url:
        print_error(f"URL mismatch! Expected {expected_url}")
        return
    
    if livekit_creds.api_key != expected_api_key:
        print_error(f"API Key mismatch!")
        return
    
    if livekit_creds.api_secret != expected_secret:
        print_error(f"API Secret mismatch! Decryption may have failed")
        print_info(f"Expected: {expected_secret[:30]}...")
        print_info(f"Got: {livekit_creds.api_secret[:30]}...")
        return
    
    print_success("All credentials match expected values!")
    
    # Step 5: Test LiveKit server connectivity
    print_step(5, "Test LiveKit Server Connectivity")
    connected = await test_livekit_connectivity(
        livekit_creds.url,
        livekit_creds.api_key,
        livekit_creds.api_secret
    )
    
    if not connected:
        print_error("Cannot connect to LiveKit server!")
        print_warning("Possible issues:")
        print_info("- Server is down or unreachable")
        print_info("- Firewall blocking connection")
        print_info("- Invalid credentials")
        print_info("- Wrong URL or port")
        return
    
    # Step 6: Test room creation
    print_step(6, "Test Room Creation")
    room_created, test_room_name = await test_room_creation(
        livekit_creds.url,
        livekit_creds.api_key,
        livekit_creds.api_secret
    )
    
    if not room_created:
        print_error("Cannot create rooms on LiveKit server!")
        return
    
    # Step 7: Test agent dispatch (without SIP call)
    print_step(7, "Test Agent Dispatch")
    test_metadata = {
        "test": "true",
        "agent_id": agent_id,
        "voice_id": resolved_voice_id,
    }
    
    dispatched, dispatch_id = await test_agent_dispatch(
        livekit_creds.url,
        livekit_creds.api_key,
        livekit_creds.api_secret,
        test_room_name,
        livekit_creds.worker_name,
        test_metadata
    )
    
    if not dispatched:
        print_error("Cannot dispatch agents!")
        print_warning("Possible issues:")
        print_info(f"- Worker '{livekit_creds.worker_name}' not running")
        print_info("- Worker not connected to this LiveKit server")
        print_info("- Worker configuration mismatch")
    else:
        print_success("Agent dispatch successful!")
        print_warning("Note: Agent may not connect if worker is not running")
    
    # Clean up test room
    await test_room_deletion(
        livekit_creds.url,
        livekit_creds.api_key,
        livekit_creds.api_secret,
        test_room_name
    )
    
    # Step 8: Make real outbound call
    print_step(8, "Dispatch Real Outbound Call")
    print_warning(f"This will make a REAL call to {to_number}")
    print_info("Press Ctrl+C within 5 seconds to cancel...")
    
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print_error("Call cancelled by user")
        return
    
    print_info("Dispatching call...")
    
    try:
        import uuid
        job_id = uuid.uuid4().hex
        
        result = await call_service.dispatch_call(
            job_id=job_id,
            voice_id=resolved_voice_id,
            voice_context=voice_context,
            from_number=from_number,
            to_number=to_number,
            context="Comprehensive test call from UAE VM LiveKit with Vonage trunk.",
            initiated_by=None,
            agent_id=agent_id,
            llm_provider=None,
            llm_model=None,
            knowledge_base_store_ids=None,
            lead_name="Test Lead Comprehensive",
            lead_id_override=None,
        )
        
        if result.error:
            print_error(f"Dispatch failed: {result.error}")
            return
        
        print_success("Call dispatched successfully!")
        print_info(f"Room: {result.room_name}")
        print_info(f"Dispatch ID: {result.dispatch_id}")
        print_info(f"Call Log ID: {result.call_log_id}")
        
        # Step 9: Check if SIP participant joins
        print_step(9, "Monitor Call Connection")
        print_info("Waiting for SIP participant to join...")
        
        sip_joined = await check_sip_participant(
            livekit_creds.url,
            livekit_creds.api_key,
            livekit_creds.api_secret,
            result.room_name
        )
        
        if sip_joined:
            print_success("SIP participant joined! Call is connecting!")
        else:
            print_warning("SIP participant not detected yet")
            print_info("Possible issues:")
            print_info(f"- Trunk ID '{livekit_creds.trunk_id}' may be invalid")
            print_info("- SIP trunk not configured on LiveKit server")
            print_info("- Vonage trunk credentials not set")
            print_info("- Number format issue")
            print_info("- Carrier blocking the call")
        
        # Final summary
        print_section("TEST SUMMARY")
        print_success("Database credential resolution: WORKING")
        print_success("Encryption/Decryption: WORKING")
        print_success("LiveKit server connectivity: WORKING")
        print_success("Room creation: WORKING")
        print_success(f"Agent dispatch: {'WORKING' if dispatched else 'FAILED'}")
        print_success("Call dispatch: WORKING")
        
        if sip_joined:
            print_success("SIP connection: WORKING")
        else:
            print_warning("SIP connection: NOT CONFIRMED")
            print_info("Check LiveKit dashboard and worker logs for details")
        
        print_info(f"\nRoom name: {result.room_name}")
        print_info(f"Call log ID: {result.call_log_id}")
        print_info("\nMonitor the call in:")
        print_info(f"- LiveKit dashboard: {livekit_creds.url.replace('ws://', 'http://').replace('wss://', 'https://')}")
        print_info("- Database: lad_dev.voice_agent_call_logs")
        print_info("- Worker logs on UAE VM")
        
    except Exception as e:
        print_error(f"Call dispatch failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  ENVIRONMENT CHECK")
    print("="*80)
    print(f"  USE_SELFHOST_ROUTING_TABLE: {os.getenv('USE_SELFHOST_ROUTING_TABLE', 'true')}")
    print(f"  LIVEKIT_SECRET_ENCRYPTION_KEY: {'SET ✓' if os.getenv('LIVEKIT_SECRET_ENCRYPTION_KEY') else 'NOT SET ✗'}")
    print(f"  DB_HOST: {os.getenv('DB_HOST', 'not set')}")
    print(f"  DB_NAME: {os.getenv('DB_NAME', 'not set')}")
    print(f"  DB_SCHEMA: {os.getenv('DB_SCHEMA', 'not set')}")
    
    asyncio.run(run_comprehensive_test())
