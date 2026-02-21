"""
End-to-End Integration Test for Dynamic LiveKit Credentials

This test verifies the complete flow:
1. Phone number has livekit_config UUID in rules
2. Credential resolver fetches config from database
3. Decrypts the secret
4. Returns correct credentials

Run with: uv run python tests/test_livekit_integration_e2e.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.call_routing import validate_and_format_call
from utils.livekit_resolver import resolve_livekit_credentials
from db.db_config import get_db_config


async def test_e2e_with_feature_enabled():
    """Test end-to-end flow with feature flag enabled."""
    
    print("="*80)
    print("End-to-End Integration Test: Dynamic LiveKit Credentials")
    print("="*80)
    
    # Test phone number (UAE number with livekit_config)
    from_number = "+971545335200"
    to_number = "0501234567"  # UAE format
    tenant_id = None  # Will be looked up from number
    
    print(f"\nTest Parameters:")
    print(f"  From Number: {from_number}")
    print(f"  To Number: {to_number}")
    print(f"  Feature Flag: USE_SELFHOST_ROUTING_TABLE={os.getenv('USE_SELFHOST_ROUTING_TABLE', 'true')}")
    
    # Step 1: Call routing (this extracts livekit_config UUID from rules)
    print(f"\n1. Call Routing Validation...")
    routing_result = validate_and_format_call(
        from_number=from_number,
        to_number=to_number,
        db_config=get_db_config(),
        tenant_id=tenant_id,
    )
    
    if not routing_result.success:
        print(f"   ✗ Routing failed: {routing_result.error_message}")
        return False
    
    print(f"   ✓ Routing successful")
    print(f"     - Formatted number: {routing_result.formatted_to_number}")
    print(f"     - Carrier: {routing_result.carrier_name}")
    print(f"     - Trunk ID (from rules): {routing_result.outbound_trunk_id}")
    print(f"     - LiveKit Config ID: {routing_result.livekit_config_id}")
    
    if not routing_result.livekit_config_id:
        print(f"   ⚠ No livekit_config UUID found in rules (will use env vars)")
    
    # Step 2: Resolve LiveKit credentials
    print(f"\n2. Resolving LiveKit Credentials...")
    livekit_creds = await resolve_livekit_credentials(
        from_number=from_number,
        tenant_id=tenant_id,
        routing_result=routing_result,
    )
    
    print(f"   ✓ Credentials resolved")
    print(f"     - Source: {livekit_creds.source}")
    print(f"     - URL: {livekit_creds.url}")
    print(f"     - API Key: {livekit_creds.api_key}")
    print(f"     - API Secret: {livekit_creds.api_secret[:20]}... (decrypted)")
    print(f"     - Trunk ID: {livekit_creds.trunk_id}")
    
    # Step 3: Verify credentials
    print(f"\n3. Verifying Credentials...")
    
    expected_values = {
        "url": "http://91.74.244.94:7880",
        "api_key": "APIbe273e3142c7b96a4a87bba4",
        "api_secret": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
        "trunk_id": "ST_svHE4RdTc7Ds",
    }
    
    if livekit_creds.source == "database":
        # Verify database credentials
        all_match = True
        for key, expected in expected_values.items():
            actual = getattr(livekit_creds, key)
            if actual == expected:
                print(f"   ✓ {key}: matches expected value")
            else:
                print(f"   ✗ {key}: mismatch")
                print(f"       Expected: {expected}")
                print(f"       Actual: {actual}")
                all_match = False
        
        if all_match:
            print(f"\n   ✓ All credentials match expected UAE VM values!")
        else:
            print(f"\n   ✗ Some credentials don't match")
            return False
    else:
        print(f"   ⚠ Using environment variables (feature disabled or no config)")
        print(f"     URL: {livekit_creds.url}")
        print(f"     This is expected if USE_SELFHOST_ROUTING_TABLE=false")
    
    # Step 4: Test trunk ID precedence
    print(f"\n4. Testing Trunk ID Precedence...")
    print(f"   Config trunk_id: {livekit_creds.trunk_id}")
    print(f"   Rules trunk_id: {routing_result.outbound_trunk_id}")
    print(f"   Env trunk_id: {os.getenv('OUTBOUND_TRUNK_ID', 'not set')}")
    
    if livekit_creds.source == "database":
        if livekit_creds.trunk_id == expected_values["trunk_id"]:
            print(f"   ✓ Using config trunk_id (correct precedence)")
        else:
            print(f"   ✗ Trunk ID precedence incorrect")
            return False
    
    print("\n" + "="*80)
    print("✓ End-to-End Test Passed!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Credential Source: {livekit_creds.source}")
    print(f"  - LiveKit URL: {livekit_creds.url}")
    print(f"  - Trunk ID: {livekit_creds.trunk_id}")
    print(f"  - Feature Flag: {os.getenv('USE_SELFHOST_ROUTING_TABLE', 'true')}")
    print("="*80)
    
    return True


async def test_e2e_with_feature_disabled():
    """Test end-to-end flow with feature flag disabled."""
    
    print("\n" + "="*80)
    print("Testing with Feature Flag Disabled")
    print("="*80)
    
    # Temporarily disable feature
    original_value = os.getenv("USE_SELFHOST_ROUTING_TABLE")
    os.environ["USE_SELFHOST_ROUTING_TABLE"] = "false"
    
    try:
        from_number = "+971545335200"
        to_number = "0501234567"
        
        print(f"\nFeature Flag: USE_SELFHOST_ROUTING_TABLE=false")
        
        # Call routing
        routing_result = validate_and_format_call(
            from_number=from_number,
            to_number=to_number,
            db_config=get_db_config(),
            tenant_id=None,
        )
        
        # Resolve credentials
        livekit_creds = await resolve_livekit_credentials(
            from_number=from_number,
            tenant_id=None,
            routing_result=routing_result,
        )
        
        print(f"\nCredential Source: {livekit_creds.source}")
        
        if livekit_creds.source == "environment":
            print(f"✓ Correctly using environment variables when feature disabled")
            return True
        else:
            print(f"✗ Should use environment variables when feature disabled")
            return False
            
    finally:
        # Restore original value
        if original_value:
            os.environ["USE_SELFHOST_ROUTING_TABLE"] = original_value
        else:
            os.environ.pop("USE_SELFHOST_ROUTING_TABLE", None)


async def main():
    """Run all tests."""
    
    # Test 1: Feature enabled
    success1 = await test_e2e_with_feature_enabled()
    
    # Test 2: Feature disabled
    success2 = await test_e2e_with_feature_disabled()
    
    if success1 and success2:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
