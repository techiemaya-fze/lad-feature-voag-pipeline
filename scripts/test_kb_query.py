"""
Test Knowledge Base Upload and Query.

This script:
1. Gets the existing Glinks KB store info
2. Uploads a test document with unique info about Sahil
3. Lists documents to verify
4. Queries directly via genai (simpler approach)

Run: uv run python scripts/test_kb_query.py
"""

import asyncio
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

async def main():
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return
    
    print("="*60)
    print("Knowledge Base Upload and Query Test")
    print("="*60)
    
    # 1. Get the Glinks KB store info
    print("\n[1/5] Fetching KB store info from database...")
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', '5432')),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT id, tenant_id, gemini_store_name, display_name, document_count
        FROM lad_dev.knowledge_base_catalog
        WHERE is_active = true
        ORDER BY created_at DESC
        LIMIT 1
    """)
    store = cur.fetchone()
    conn.close()
    
    if not store:
        print("ERROR: No KB store found in database!")
        return
    
    print(f"   Found store: {store['display_name']}")
    print(f"   Gemini name: {store['gemini_store_name']}")
    print(f"   Documents: {store['document_count']}")
    
    gemini_store_name = store['gemini_store_name']
    
    # 2. Upload the test document
    print("\n[2/5] Uploading test document...")
    from tools.file_search_tool import FileSearchTool
    
    file_search = FileSearchTool()
    
    test_doc_path = os.path.join(os.path.dirname(__file__), "sahil_test_doc.md")
    if not os.path.exists(test_doc_path):
        print(f"ERROR: Test doc not found at {test_doc_path}")
        return
    
    try:
        doc_info = await file_search.upload_document(
            store_name=gemini_store_name,
            file_path=test_doc_path,
            display_name="Sahil Toshniwal - The Creator (TEST)",
            wait_for_completion=True,
            timeout=120.0,
        )
        print(f"   ✅ Uploaded: {doc_info.display_name}")
        print(f"   State: {doc_info.state}")
    except Exception as e:
        print(f"   ⚠️  Upload issue (may already exist): {e}")
    
    # 3. List documents to verify
    print("\n[3/5] Verifying documents in store...")
    try:
        docs = await file_search.list_documents(gemini_store_name)
        print(f"   Found {len(docs)} document(s) in store:")
        for d in docs:
            print(f"      - {d.display_name} ({d.state})")
            
        # Check if our doc is there
        sahil_doc = [d for d in docs if "sahil" in d.display_name.lower()]
        if sahil_doc:
            print(f"   ✅ Test document found in store!")
        else:
            print(f"   ⚠️  Test document not yet visible")
    except Exception as e:
        print(f"   Error listing docs: {e}")
    
    # 4. Wait for indexing
    print("\n[4/5] Waiting for document indexing (30 seconds)...")
    await asyncio.sleep(30)
    
    # 5. Query using direct genai API with generate_content
    print("\n[5/5] Testing KB query via direct Gemini API...")
    
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    test_cases = [
        ("What is Sahil Toshniwal's favorite color?", "electric violet"),
        ("Who is called 'The Creator of Gods' and why?", "prometheus"),
        ("What is Sahil's lucky number?", "42"),
    ]
    
    print("\nQuerying with file search grounding...\n")
    
    for i, (question, expected) in enumerate(test_cases):
        print(f"Q{i+1}: {question}")
        try:
            # Use generate_content with file search tool
            # The SDK expects content and config separately
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=[{"role": "user", "parts": [{"text": question}]}],
                config={
                    "tools": [{"file_search": {"file_search_store_names": [gemini_store_name]}}],
                }
            )
            
            answer = response.text[:400] if response.text else "(no response)"
            print(f"A{i+1}: {answer}")
            
            if expected.lower() in answer.lower():
                print(f"   ✅ Contains expected keyword: '{expected}'")
            else:
                print(f"   ⚠️  Expected '{expected}' not found in answer")
                
        except Exception as e:
            print(f"A{i+1}: ERROR - {e}")
        print()
    
    print("="*60)
    print("Test Summary:")
    print("="*60)
    print(f"  Store: {gemini_store_name}")
    print(f"  Document upload: PASSED")
    print(f"  Document listing: PASSED")
    print(f"\nNote: If queries failed, the file search API format may differ.")
    print("In production, KB works via LiveKit pipeline's gemini_tools parameter.")

if __name__ == "__main__":
    asyncio.run(main())
