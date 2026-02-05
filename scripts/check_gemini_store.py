"""List documents directly from Gemini FileSearchStore."""
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# G_links_trial gemini store name (from DB)
GEMINI_STORE = "fileSearchStores/glinkstrial-326w9dxhjmno"

print("Listing documents in Gemini store:")
print(f"Store: {GEMINI_STORE}")

try:
    # Get store info
    store = client.file_search_stores.get(name=GEMINI_STORE)
    print(f"Display name: {store.display_name}")
    
    # Try to list documents - SDK API varies
    if hasattr(store, 'documents'):
        for doc in store.documents:
            print(f"  - {doc.display_name} ({doc.name})")
    else:
        print("  (SDK doesn't expose document listing directly)")
        print("  Documents exist but can't be listed via SDK")
except Exception as e:
    print(f"Error: {e}")
