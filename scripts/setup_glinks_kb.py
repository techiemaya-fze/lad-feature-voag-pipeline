"""Create Glinks KB store, upload all PDFs, and set as default."""
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

TENANT_ID = "926070b5-189b-4682-9279-ea10ca090b84"
FOLDER_PATH = r"D:\vonage\vonage-voice-agent\Glinks_pdf"
STORE_NAME = "glinks_corrected"

async def main():
    print("="*60)
    print("Creating Glinks KB Store")
    print("="*60)
    print(f"Tenant ID: {TENANT_ID}")
    print(f"Folder: {FOLDER_PATH}")
    print(f"Store Name: {STORE_NAME}")
    
    async with httpx.AsyncClient(timeout=1200.0) as client:  # 20 min timeout for bulk upload
        # 1. Create store
        print("\n[1] Creating KB store...")
        resp = await client.post(
            "http://localhost:8000/knowledge-base/stores",
            headers=HEADERS,
            json={
                "tenant_id": TENANT_ID,
                "display_name": STORE_NAME,
                "description": "Glinks International university/college data - corrected version",
                "is_default": True,
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
        print("    (This may take several minutes for 89 PDFs...)")
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
            if data['failed'] > 0:
                print("\n   Failed files:")
                for r in data['results']:
                    if not r['success']:
                        print(f"   ❌ {r['filename']}: {r.get('error', 'Unknown error')}")
        else:
            print(f"❌ Bulk upload failed: {resp.text[:500]}")
        
        # 3. Verify store is default
        print(f"\n[3] Verifying default store for tenant...")
        resp = await client.get(
            f"http://localhost:8000/knowledge-base/tenants/{TENANT_ID}/stores",
            headers={k:v for k,v in HEADERS.items() if k != "Content-Type"}
        )
        
        if resp.status_code == 200:
            stores = resp.json()
            for s in stores:
                print(f"   - {s['display_name']} (docs={s.get('document_count', 0)}, default={s.get('is_default', False)})")
        
        print("\n" + "="*60)
        print("Done! Store saved in lad_dev.knowledge_base_catalog")
        print("="*60)

asyncio.run(main())
