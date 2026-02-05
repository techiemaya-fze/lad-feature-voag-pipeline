"""Check Gemini API tier based on storage usage."""
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

print("=== Gemini File Search Storage Usage ===\n")

stores = list(client.file_search_stores.list())
total_bytes = 0

for store in stores:
    size = int(getattr(store, 'size_bytes', 0) or 0)
    total_bytes += size
    name = getattr(store, 'display_name', store.name)
    print(f"  {name}: {size/(1024*1024):.2f} MB")

total_gb = total_bytes / (1024*1024*1024)
print(f"\n{'='*40}")
print(f"Total: {len(stores)} stores, {total_gb:.4f} GB used")
print(f"{'='*40}")

# Tier detection
print("\nTier Limits:")
print("  Free:   1 GB  - You're on Free if usage < 1 GB works")
print("  Tier 1: 10 GB - Set via Google AI Studio billing")
print("  Tier 2: 100 GB")
print("  Tier 3: 1 TB")
print("\nNote: API doesn't expose tier info. Check https://aistudio.google.com/apikey")
