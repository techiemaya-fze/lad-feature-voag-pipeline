"""
Test retrieval from uploaded Excel file.

Run: uv run python scripts/test_excel_retrieval.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

STORE = "fileSearchStores/sahil-test-kb-lcczrz60qwe4"

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("="*60)
print("Testing Excel Retrieval from KB")
print("="*60)
print(f"Store: {STORE}")

# Questions that should be answered from the Glinks Excel questionary
questions = [
    "What mandatory fields are in the Glinks questionary?",
    "What contact information does Glinks collect?",
    "What is the country of residence field in the questionary?",
]

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=q,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[STORE]
                    )
                )]
            )
        )
        print(f"A{i}: {response.text[:400] if response.text else '(empty)'}")
        
        # Check for grounding sources
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if grounding.grounding_chunks:
                sources = set()
                for c in grounding.grounding_chunks:
                    if hasattr(c, 'retrieved_context') and c.retrieved_context:
                        sources.add(c.retrieved_context.title)
                if sources:
                    print(f"   📚 Sources: {sources}")
                    if "Glinks Questionary Excel" in sources:
                        print(f"   ✅ Excel data retrieved!")
    except Exception as e:
        print(f"A{i}: Error - {e}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
