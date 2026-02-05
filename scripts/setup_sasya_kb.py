"""Create Sasya KB store, upload all docs, and set as default."""
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

TENANT_ID = "734cd516-e252-4728-9c52-4663ee552653"
FOLDER_PATH = r"D:\vonage\vonage-voice-agent\sasya spaces"
STORE_NAME = "Sasya Spaces KB"

async def main():
    print("="*60)
    print("Creating Sasya KB Store")
    print("="*60)
    print(f"Tenant ID: {TENANT_ID}")
    print(f"Folder: {FOLDER_PATH}")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        # 1. Create store
        print("\n[1] Creating KB store...")
        resp = await client.post(
            "http://localhost:8000/knowledge-base/stores",
            headers=HEADERS,
            json={
                "tenant_id": TENANT_ID,
                "display_name": STORE_NAME,
                "description": "Sasya Spaces knowledge base",
                "is_default": True,  # Set as default
                "priority": 1
            }
        )
        
        if resp.status_code != 200:
            print(f"❌ Failed to create store: {resp.text}")
            return
        
        store_data = resp.json()
        store_id = store_data["id"]
        gemini_store = store_data["gemini_store_name"]
        print(f"✅ Created store: {store_id}")
        print(f"   Gemini: {gemini_store}")
        print(f"   Default: {store_data.get('is_default', False)}")
        
        # 2. Bulk upload
        print(f"\n[2] Bulk uploading files from {FOLDER_PATH}...")
        resp = await client.post(
            f"http://localhost:8000/knowledge-base/stores/{store_id}/bulk-upload",
            headers=HEADERS,
            json={"folder_path": FOLDER_PATH}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Bulk upload complete!")
            print(f"   Total: {data['total_files']}")
            print(f"   Success: {data['successful']}")
            print(f"   Failed: {data['failed']}")
            for r in data['results']:
                status = "✅" if r['success'] else "❌"
                print(f"   {status} {r['filename']}")
        else:
            print(f"❌ Bulk upload failed: {resp.text[:300]}")
        
        # 3. Verify store is default
        print(f"\n[3] Verifying default store for tenant...")
        resp = await client.get(
            f"http://localhost:8000/knowledge-base/tenants/{TENANT_ID}/stores",
            headers={k:v for k,v in HEADERS.items() if k != "Content-Type"}
        )
        
        if resp.status_code == 200:
            stores = resp.json()
            for s in stores:
                print(f"   - {s['display_name']} (default={s.get('is_default', False)})")
        
        print("\n" + "="*60)
        print("Done!")
        print("="*60)

asyncio.run(main())
