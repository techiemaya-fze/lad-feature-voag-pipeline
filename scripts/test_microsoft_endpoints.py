#!/usr/bin/env python
"""
Test Microsoft Bookings API endpoints.
Run with: uv run python scripts/test_microsoft_endpoints.py
"""

import asyncio
import httpx
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "kMQgGRDAa8t5CvmkfqFYuGiXIXgNYC1EEGjYs5v8_NU")
FRONTEND_ID = os.getenv("FRONTEND_ID", "dev")
USER_ID = os.getenv("USER_ID", "81f1decc-7ee5-4093-b55c-95ac9b7c9f45")


async def test_endpoints():
    headers = {
        "X-Frontend-ID": FRONTEND_ID,
        "X-API-Key": API_KEY,
    }
    
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30.0) as client:
        print("=" * 60)
        print("Testing Microsoft Bookings API Endpoints")
        print("=" * 60)
        
        # Test 1: Status
        print("\n1. Testing /auth/microsoft/status...")
        resp = await client.get(f"/auth/microsoft/status?user_id={USER_ID}")
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Connected: {data.get('connected')}")
            print(f"   Business: {data.get('selected_business_id')}")
            print("   ✅ PASSED")
        else:
            print(f"   ❌ FAILED: {resp.text[:200]}")
        
        # Test 2: List Businesses
        print("\n2. Testing /auth/microsoft/list-businesses...")
        resp = await client.get(f"/auth/microsoft/list-businesses?user_id={USER_ID}")
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            businesses = resp.json()
            print(f"   Found {len(businesses)} businesses:")
            for b in businesses[:3]:
                print(f"     - {b.get('id')}: {b.get('display_name')}")
            print("   ✅ PASSED")
            business_ids = [b.get('id') for b in businesses]
        else:
            print(f"   ❌ FAILED: {resp.text[:200]}")
            business_ids = []
        
        # Test 3: List Services (for each business)
        print("\n3. Testing /auth/microsoft/list-services...")
        for bid in business_ids[:2]:  # Test first 2 businesses
            print(f"\n   Business: {bid}")
            resp = await client.get(f"/auth/microsoft/list-services?user_id={USER_ID}&business_id={bid}")
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                services = resp.json()
                print(f"   Found {len(services)} services")
                for s in services[:2]:
                    print(f"     - {s.get('id')}: {s.get('display_name')}")
                print("   ✅ PASSED")
            else:
                print(f"   ⚠️ {resp.status_code}: {resp.text[:100]}")
        
        # Test 4: List Staff (for first business)
        print("\n4. Testing /auth/microsoft/staff...")
        if business_ids:
            bid = business_ids[0]
            print(f"   Business: {bid}")
            resp = await client.get(f"/auth/microsoft/staff?user_id={USER_ID}&business_id={bid}")
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                staff = resp.json()
                print(f"   Found {len(staff)} staff members")
                for s in staff[:3]:
                    print(f"     - {s.get('id')}: {s.get('display_name')}")
                print("   ✅ PASSED")
            else:
                print(f"   ❌ FAILED: {resp.text[:200]}")
        
        print("\n" + "=" * 60)
        print("Testing Complete")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_endpoints())
