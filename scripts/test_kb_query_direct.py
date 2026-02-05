"""
Test KB Query - Exact pattern from Google's blog post.

Store: fileSearchStores/sahil-test-kb-lcczrz60qwe4

Run: uv run python scripts/test_kb_query_direct.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# The store we created via the API
STORE_NAME = "fileSearchStores/sahil-test-kb-lcczrz60qwe4"

client = genai.Client()

print("="*60)
print("KB Query Test - Exact Google Blog Pattern")
print("="*60)
print(f"Store: {STORE_NAME}")

# First verify the store exists
print("\n[1] Verifying store...")
try:
    store = client.file_search_stores.get(name=STORE_NAME)
    print(f"    ✅ Store found: {store.display_name}")
except Exception as e:
    print(f"    ❌ Store error: {e}")
    sys.exit(1)

# Query using EXACT pattern from Google blog (Nov 2025)
print("\n[2] Querying with FileSearch...")
print("    Model: gemini-2.5-flash (as per docs)")
print()

questions = [
    "What is Sahil's favorite color?",
    "Who is The Creator of Gods?",
    "What is Sahil's lucky number?",
]

for i, q in enumerate(questions):
    print(f"Q{i+1}: {q}")
    try:
        # EXACT pattern from Google blog post
        response = client.models.generate_content(
            model='gemini-2.5-flash',  # Using model from docs
            contents=q,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[STORE_NAME]
                    )
                )]
            )
        )
        print(f"A{i+1}: {response.text[:300] if response.text else '(empty)'}")
        
        # Check grounding
        grounding = response.candidates[0].grounding_metadata if response.candidates else None
        if grounding and grounding.grounding_chunks:
            sources = {c.retrieved_context.title for c in grounding.grounding_chunks if hasattr(c, 'retrieved_context')}
            print(f"    Sources: {sources}")
        
    except Exception as e:
        print(f"A{i+1}: Error - {e}")
    print()

print("="*60)
print("Test Complete")
print("="*60)
