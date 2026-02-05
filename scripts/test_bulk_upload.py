"""Test bulk upload endpoint."""
import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import httpx

FRONTEND_API_KEYS = json.loads(os.getenv("FRONTEND_API_KEYS", "{}"))
frontend_id = list(FRONTEND_API_KEYS.keys())[0]
api_key = FRONTEND_API_KEYS[frontend_id]
HEADERS = {
    "X-Frontend-ID": frontend_id, 
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}

# Use the sasya test store created earlier
STORE_ID = "38529e19-176c-49bd-b098-1030be181d06"

# Test folder with files
FOLDER_PATH = r"D:\vonage\vonage-voice-agent\G_links"

async def main():
    print("="*60)
    print("Testing Bulk Upload Endpoint")
    print("="*60)
    print(f"Store ID: {STORE_ID}")
    print(f"Folder: {FOLDER_PATH}")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"http://localhost:8000/knowledge-base/stores/{STORE_ID}/bulk-upload",
            headers=HEADERS,
            json={"folder_path": FOLDER_PATH}
        )
        
        print(f"\nStatus: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n✅ Bulk upload complete!")
            print(f"   Total files: {data['total_files']}")
            print(f"   Successful: {data['successful']}")
            print(f"   Failed: {data['failed']}")
            print("\nResults:")
            for r in data['results']:
                status = "✅" if r['success'] else "❌"
                print(f"   {status} {r['filename']}")
                if r.get('error'):
                    print(f"      Error: {r['error'][:100]}")
        else:
            print(f"\n❌ Error: {resp.text[:500]}")

asyncio.run(main())
