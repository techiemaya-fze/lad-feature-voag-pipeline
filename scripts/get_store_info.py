"""Get Glinks store info from DB."""
import os, sys, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from db.storage import KnowledgeBaseStorage

async def main():
    kb = KnowledgeBaseStorage()
    stores = await kb.get_stores_for_tenant("926070b5-189b-4682-9279-ea10ca090b84", default_only=False)
    for s in stores:
        print(f"Store: {s['display_name']}")
        print(f"  Gemini: {s['gemini_store_name']}")
        print(f"  ID: {s['id']}")
asyncio.run(main())
