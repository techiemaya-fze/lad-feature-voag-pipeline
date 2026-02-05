"""List documents in the G_links_trial store."""
import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

FRONTEND_API_KEYS = json.loads(os.getenv("FRONTEND_API_KEYS", "{}"))
frontend_id = list(FRONTEND_API_KEYS.keys())[0]
api_key = FRONTEND_API_KEYS[frontend_id]
HEADERS = {"X-Frontend-ID": frontend_id, "X-API-Key": api_key}

# Original Glinks store
STORE_ID = "ff825aaf-bb5e-418c-80ea-f5059eaefab2"

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Documents in G_links_trial store:")
        resp = await client.get(
            f"http://localhost:8000/knowledge-base/stores/{STORE_ID}/documents",
            headers=HEADERS
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            docs = resp.json()
            if not docs:
                print("  (no documents)")
            for d in docs:
                name = d.get("display_name") or d.get("name", "unknown")
                print(f"  - {name}")
        else:
            print(f"Error: {resp.text[:200]}")

asyncio.run(main())
