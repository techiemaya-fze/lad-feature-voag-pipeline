"""
Initial KB Cache Sync Script (Gemini API Based)

Run ONCE to populate the persistent cache with all KB stores and their documents
directly from the Gemini API (not the local database).

Usage:
    cd d:\\vonage\\vonage-voice-agent\\v2
    uv run scripts/sync_kb_cache.py
"""

import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_search_tool import FileSearchTool
from utils.kb_cache import kb_cache


async def sync_all_kb_stores():
    """Sync all KB stores and their documents to cache from Gemini API."""
    print("\n=== KB Cache Full Sync (from Gemini API) ===\n")
    
    file_search = FileSearchTool()
    client = file_search._ensure_client()
    
    # Get all stores directly from Gemini API
    stores = list(client.file_search_stores.list())
    print(f"Found {len(stores)} KB stores in Gemini API\n")
    
    total_docs = 0
    
    for store in stores:
        gemini_store_name = store.name
        display_name = getattr(store, 'display_name', gemini_store_name)
        
        # Use gemini store name as the cache key (since we don't have DB IDs)
        # Extract a simpler ID from the store name
        store_id = gemini_store_name.replace("fileSearchStores/", "")
        
        print(f"Syncing: {display_name}")
        print(f"  Store ID: {store_id}")
        print(f"  Gemini: {gemini_store_name}")
        
        try:
            # Fetch documents from Gemini
            docs = await file_search.list_documents(gemini_store_name)
            doc_names = [d.name for d in docs]
            doc_count = len(docs)
            
            # Update cache using gemini store name as key
            kb_cache.set_store_stats(
                store_id=store_id,
                gemini_store_name=gemini_store_name,
                document_count=doc_count,
                documents=doc_names
            )
            
            print(f"  Documents: {doc_count}")
            total_docs += doc_count
            
        except Exception as e:
            print(f"  ERROR: {e}")
            # Still add to cache with 0 docs
            kb_cache.set_store_stats(
                store_id=store_id,
                gemini_store_name=gemini_store_name,
                document_count=0,
                documents=[]
            )
        
        print()
    
    # Mark full sync complete
    kb_cache.mark_full_sync()
    
    print("=" * 40)
    print(f"SYNC COMPLETE")
    print(f"  Stores: {len(stores)}")
    print(f"  Total Documents: {total_docs}")
    print(f"  Cache file: v2/data/kb_cache.json")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(sync_all_kb_stores())
