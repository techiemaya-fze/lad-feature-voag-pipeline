"""
Test KB API Endpoints.

Tests the actual v2 API endpoints:
1. POST /knowledge-base/stores - Create store
2. POST /knowledge-base/stores/{id}/documents - Upload document
3. GET /knowledge-base/tenants/{tenant_id}/stores - List tenant stores

Run: uv run python scripts/test_kb_api_endpoints.py
"""

import os
import sys
import asyncio
import httpx
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# API Config
BASE_URL = "http://localhost:8000"

# Check if FRONTEND_API_KEYS is configured
FRONTEND_API_KEYS_RAW = os.getenv("FRONTEND_API_KEYS", "{}")
try:
    FRONTEND_API_KEYS = json.loads(FRONTEND_API_KEYS_RAW)
except:
    FRONTEND_API_KEYS = {}

# Build headers based on security config
if FRONTEND_API_KEYS:
    # Use first configured frontend
    frontend_id = list(FRONTEND_API_KEYS.keys())[0]
    api_key = FRONTEND_API_KEYS[frontend_id]
    HEADERS = {
        "X-Frontend-ID": frontend_id,
        "X-API-Key": api_key,
    }
    print(f"Using configured frontend: {frontend_id}")
else:
    # Security disabled - still need headers but they won't be validated
    HEADERS = {
        "X-Frontend-ID": "test",
    }
    print("Security disabled (no FRONTEND_API_KEYS) - using minimal headers")

# Get tenant ID from env or use Glinks default
TENANT_ID = "926070b5-189b-4682-9279-ea10ca090b84"  # Glinks tenant


async def main():
    print("="*60)
    print("KB API Endpoints Test")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Tenant ID: {TENANT_ID}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # 0. Check security status
        print("\n[0/5] Checking API security status...")
        try:
            resp = await client.get(f"{BASE_URL}/security/status")
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                security_info = resp.json()
                print(f"   Security enabled: {security_info.get('security_enabled', 'unknown')}")
        except Exception as e:
            print(f"   Could not check security: {e}")
        
        # 1. List existing stores
        print("\n[1/5] Listing existing stores...")
        try:
            resp = await client.get(
                f"{BASE_URL}/knowledge-base/stores",
                headers=HEADERS,
                params={"tenant_id": TENANT_ID}
            )
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                stores = resp.json()
                print(f"   Found {len(stores)} store(s)")
                for s in stores:
                    print(f"      - {s['display_name']} (id={s['id'][:8]}...)")
            else:
                print(f"   Response: {resp.text[:300]}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Create a new test store
        print("\n[2/5] Creating test store...")
        store_id = None
        gemini_store_name = None
        try:
            resp = await client.post(
                f"{BASE_URL}/knowledge-base/stores",
                headers={**HEADERS, "Content-Type": "application/json"},
                json={
                    "tenant_id": TENANT_ID,
                    "display_name": "Sahil Test KB",
                    "description": "Test KB for Sahil's unique info",
                    "is_default": False,
                    "priority": 1,
                }
            )
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                store_id = data["id"]
                gemini_store_name = data["gemini_store_name"]
                print(f"   ✅ Created store!")
                print(f"      ID: {store_id}")
                print(f"      Gemini: {gemini_store_name}")
            else:
                print(f"   Response: {resp.text[:300]}")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
        
        if not store_id:
            print("\n❌ Failed to create store. See response above.")
            print("   Check server logs for details.")
            return
        
        # 3. Upload document to the store
        print("\n[3/5] Uploading test document...")
        test_doc_path = os.path.join(os.path.dirname(__file__), "sahil_test_doc.md")
        
        if not os.path.exists(test_doc_path):
            print(f"   Creating test doc...")
            with open(test_doc_path, 'w') as f:
                f.write("""# About Sahil Toshniwal - The Creator

## Personal Information
- **Full Name:** Sahil Toshniwal
- **Title:** The Creator of Gods and Architect of Digital Realms
- **Favorite Color:** Electric Violet (#8B00FF)
- **Lucky Number:** 42

## The Legend
Sahil is "The Creator of Gods" because he designed AI system "Project Prometheus" 
that generates divine consciousness patterns.
""")
        
        try:
            with open(test_doc_path, 'rb') as f:
                files = {'file': ('sahil_test_doc.md', f, 'text/markdown')}
                data = {'display_name': 'Sahil Creator Info'}
                resp = await client.post(
                    f"{BASE_URL}/knowledge-base/stores/{store_id}/documents",
                    headers=HEADERS,
                    files=files,
                    data=data,
                )
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                doc_data = resp.json()
                print(f"   ✅ Uploaded!")
                print(f"      Doc: {doc_data.get('document', {}).get('display_name')}")
                print(f"      State: {doc_data.get('document', {}).get('state')}")
            else:
                print(f"   Response: {resp.text[:300]}")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Wait for indexing
        print("\n[4/5] Waiting for indexing (30s)...")
        await asyncio.sleep(30)
        
        # 5. Query using the store
        print("\n[5/5] Testing KB query...")
        
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("   ⚠️ No GOOGLE_API_KEY set, skipping query test")
        elif not gemini_store_name:
            print("   ⚠️ No gemini store name, skipping query test")
        else:
            client_genai = genai.Client(api_key=api_key)
            
            questions = [
                ("What is Sahil's favorite color?", "violet"),
                ("Who is The Creator of Gods?", "sahil"),
            ]
            
            print(f"\n   Querying with store: {gemini_store_name}\n")
            
            for q, expected in questions:
                print(f"   Q: {q}")
                try:
                    response = client_genai.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=q,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(
                                file_search=types.FileSearch(
                                    file_search_store_names=[gemini_store_name]
                                )
                            )]
                        )
                    )
                    ans = response.text[:200] if response.text else "(none)"
                    print(f"   A: {ans}")
                    if expected.lower() in ans.lower():
                        print(f"   ✅ Found '{expected}'")
                except Exception as e:
                    print(f"   Error: {e}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
    if store_id:
        print(f"Store ID: {store_id}")
    if gemini_store_name:
        print(f"Gemini Store: {gemini_store_name}")

if __name__ == "__main__":
    asyncio.run(main())
