"""Delete test KB stores, keep only original Glinks KB."""
import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

FRONTEND_API_KEYS = json.loads(os.getenv("FRONTEND_API_KEYS", "{}"))
frontend_id = list(FRONTEND_API_KEYS.keys())[0]
api_key = FRONTEND_API_KEYS[frontend_id]
HEADERS = {"X-Frontend-ID": frontend_id, "X-API-Key": api_key}
TENANT_ID = "926070b5-189b-4682-9279-ea10ca090b84"

# Test stores to delete
TEST_STORE_ID = "9585f49e-4a2c-4b8a-81ca-ddc0d5af9ed4"  # Sahil Test KB

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # List stores first
        print("Current stores for Glinks tenant:")
        resp = await client.get(
            "http://localhost:8000/knowledge-base/stores",
            headers=HEADERS,
            params={"tenant_id": TENANT_ID}
        )
        stores = resp.json()
        for s in stores:
            marker = " (ORIGINAL - KEEP)" if "G_links" in s["display_name"] else " (TEST - DELETE)"
            print(f"  - {s['display_name']}{marker}")
            print(f"    ID: {s['id']}")
        
        # Delete test store
        print(f"\nDeleting test store: {TEST_STORE_ID}")
        resp = await client.delete(
            f"http://localhost:8000/knowledge-base/stores/{TEST_STORE_ID}",
            headers=HEADERS
        )
        print(f"Delete status: {resp.status_code}")
        if resp.status_code == 200:
            print("✅ Test store deleted!")
        else:
            print(f"Response: {resp.text[:200]}")
        
        # List remaining
        print("\nRemaining stores:")
        resp = await client.get(
            "http://localhost:8000/knowledge-base/stores",
            headers=HEADERS,
            params={"tenant_id": TENANT_ID}
        )
        stores = resp.json()
        for s in stores:
            print(f"  - {s['display_name']}")

asyncio.run(main())
