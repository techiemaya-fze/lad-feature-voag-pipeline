"""Test telephony provider resolution for known phone numbers."""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.storage.numbers import NumberStorage

SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")

# 1) Get all numbers from voice_agent_numbers with their providers
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

print("=" * 90)
print("TEST: Provider Resolution for All Voice Agent Numbers")
print("=" * 90)

cur.execute(f"""
    SELECT country_code, base_number, provider
    FROM {SCHEMA}.voice_agent_numbers
    ORDER BY provider
""")
numbers = cur.fetchall()

# 2) Get available telephony pricing providers
cur.execute(f"""
    SELECT DISTINCT provider
    FROM {SCHEMA}.billing_pricing_catalog
    WHERE category = 'telephony'
    ORDER BY provider
""")
pricing_providers = {r[0].lower() for r in cur.fetchall()}
print(f"\nAvailable telephony pricing providers: {pricing_providers}")
print()

# 3) Test get_provider_by_phone for each number
ns = NumberStorage()
print(f"{'Phone Number':<25} {'DB Provider':<25} {'Resolved':<25} {'In Pricing?':<12} {'Fallback?'}")
print("-" * 100)

for cc, base, db_provider in numbers:
    phone = f"{cc}{base}" if cc else str(base)
    resolved = ns.get_provider_by_phone(phone)
    
    # Simulate _resolve_telephony_provider normalization
    if resolved:
        normalized = resolved.strip().lower()
    else:
        normalized = "vonage"
    
    in_pricing = normalized in pricing_providers
    is_fallback = not in_pricing
    
    status = "OK" if in_pricing else "-> vonage (fallback)"
    
    print(f"  {phone:<23} {db_provider:<25} {normalized:<25} {str(in_pricing):<12} {status}")

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
providers_used = set()
fallback_count = 0
for cc, base, db_provider in numbers:
    phone = f"{cc}{base}" if cc else str(base)
    resolved = ns.get_provider_by_phone(phone)
    normalized = resolved.strip().lower() if resolved else "vonage"
    if normalized in pricing_providers:
        providers_used.add(normalized)
    else:
        fallback_count += 1
        providers_used.add("vonage (fallback)")

print(f"  Total numbers: {len(numbers)}")
print(f"  Providers matched to pricing catalog: {providers_used}")
print(f"  Numbers needing fallback to vonage: {fallback_count}")

cur.close()
conn.close()
