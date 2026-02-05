"""
Test Excel Upload to KB via API.

Tests uploading Excel file from G_links folder to verify format support.

Run: uv run python scripts/test_kb_excel_upload.py
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

# Get credentials from env
FRONTEND_API_KEYS_RAW = os.getenv("FRONTEND_API_KEYS", "{}")
try:
    FRONTEND_API_KEYS = json.loads(FRONTEND_API_KEYS_RAW)
except:
    FRONTEND_API_KEYS = {}

if FRONTEND_API_KEYS:
    frontend_id = list(FRONTEND_API_KEYS.keys())[0]
    api_key = FRONTEND_API_KEYS[frontend_id]
    HEADERS = {"X-Frontend-ID": frontend_id, "X-API-Key": api_key}
else:
    HEADERS = {"X-Frontend-ID": "test"}

# Store ID from previous test
STORE_ID = "9585f49e-4a2c-4b8a-81ca-ddc0d5af9ed4"
GEMINI_STORE = "fileSearchStores/sahil-test-kb-lcczrz60qwe4"

# Excel file path
EXCEL_FILE = r"D:\vonage\vonage-voice-agent\G_links\Glinks_Questionary.xlsx"


async def main():
    print("="*60)
    print("Excel Upload Test")
    print("="*60)
    print(f"Store ID: {STORE_ID}")
    print(f"Excel file: {EXCEL_FILE}")
    
    if not os.path.exists(EXCEL_FILE):
        print(f"ERROR: File not found: {EXCEL_FILE}")
        return
    
    file_size = os.path.getsize(EXCEL_FILE)
    print(f"File size: {file_size} bytes")
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        
        # 1. Upload Excel file via API
        print("\n[1] Uploading Excel file via API...")
        try:
            with open(EXCEL_FILE, 'rb') as f:
                files = {'file': ('Glinks_Questionary.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                data = {'display_name': 'Glinks Questionary Excel'}
                resp = await client.post(
                    f"{BASE_URL}/knowledge-base/stores/{STORE_ID}/documents",
                    headers=HEADERS,
                    files=files,
                    data=data,
                )
            print(f"    Status: {resp.status_code}")
            if resp.status_code == 200:
                doc_data = resp.json()
                print(f"    ✅ Upload successful!")
                print(f"       Name: {doc_data.get('document', {}).get('name')}")
                print(f"       Display: {doc_data.get('document', {}).get('display_name')}")
                print(f"       State: {doc_data.get('document', {}).get('state')}")
            else:
                print(f"    ❌ Upload failed: {resp.text[:500]}")
                return
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            return
    
        # 2. Wait for indexing
        print("\n[2] Waiting for indexing (30 seconds)...")
        await asyncio.sleep(30)
    
    # 3. Query the KB to verify Excel content
    print("\n[3] Querying KB to verify Excel content...")
    
    from google import genai
    from google.genai import types
    
    client_genai = genai.Client()
    
    # Ask about content that should be in the Glinks questionary
    questions = [
        "What questions are in the Glinks questionary?",
        "Tell me about the Glinks questionary content.",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        try:
            response = client_genai.models.generate_content(
                model='gemini-2.5-flash',
                contents=q,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[GEMINI_STORE]
                        )
                    )]
                )
            )
            print(f"A: {response.text[:500] if response.text else '(empty)'}")
            
            # Check sources
            if response.candidates and response.candidates[0].grounding_metadata:
                grounding = response.candidates[0].grounding_metadata
                if grounding.grounding_chunks:
                    sources = {c.retrieved_context.title for c in grounding.grounding_chunks if hasattr(c, 'retrieved_context')}
                    print(f"   Sources: {sources}")
                    
        except Exception as e:
            print(f"A: Error - {e}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
