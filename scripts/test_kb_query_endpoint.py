"""Test the KB Query endpoint."""
import os
import sys
import httpx
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# Get credentials
FRONTEND_API_KEYS = json.loads(os.getenv("FRONTEND_API_KEYS", "{}"))
frontend_id = list(FRONTEND_API_KEYS.keys())[0] if FRONTEND_API_KEYS else "test"
api_key = FRONTEND_API_KEYS.get(frontend_id, "") if FRONTEND_API_KEYS else ""

HEADERS = {
    "X-Frontend-ID": frontend_id,
    "X-API-Key": api_key,
    "Content-Type": "application/json"
}

TENANT_ID = "926070b5-189b-4682-9279-ea10ca090b84"

print("="*60)
print("Testing KB Query Endpoint")
print("="*60)
print(f"Frontend: {frontend_id}")
print(f"Tenant: {TENANT_ID}")

questions = [
    "What mandatory fields are in the Glinks questionary?",
    "What is Sahil's favorite color?",
    "Who is the Creator of Gods?",
]

with httpx.Client(timeout=60.0) as client:
    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        
        resp = client.post(
            "http://localhost:8000/knowledge-base/query",
            headers=HEADERS,
            json={
                "tenant_id": TENANT_ID,
                "question": q
            }
        )
        
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Answer: {data['answer'][:300]}...")
            print(f"Sources: {data['sources']}")
            print(f"Stores: {len(data['store_names'])} store(s)")
        else:
            print(f"Error: {resp.text[:200]}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
