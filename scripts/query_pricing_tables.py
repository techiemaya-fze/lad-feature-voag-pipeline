"""Quick script to query billing_pricing_catalog and voice_agent_numbers tables."""
import os
import sys
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")

conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
)
cur = conn.cursor()

print("=" * 80)
print(f"1) {SCHEMA}.billing_pricing_catalog (all active rows)")
print("=" * 80)
cur.execute(f"""
    SELECT id, category, provider, model, unit, unit_price, is_active, description
    FROM {SCHEMA}.billing_pricing_catalog
    ORDER BY category, provider, model
""")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print(f"Columns: {cols}")
for r in rows:
    print(r)

print()
print("=" * 80)
print(f"2) {SCHEMA}.voice_agent_numbers (first 20 rows)")
print("=" * 80)
cur.execute(f"""
    SELECT *
    FROM {SCHEMA}.voice_agent_numbers
    LIMIT 20
""")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print(f"Columns: {cols}")
for r in rows:
    print(r)

print()
print("=" * 80)
print("3) Distinct providers in voice_agent_numbers")
print("=" * 80)
cur.execute(f"""
    SELECT DISTINCT provider
    FROM {SCHEMA}.voice_agent_numbers
    ORDER BY provider
""")
for r in cur.fetchall():
    print(r)

print()
print("=" * 80)
print("4) Telephony rows in billing_pricing_catalog")
print("=" * 80)
cur.execute(f"""
    SELECT id, category, provider, model, unit, unit_price, is_active, description
    FROM {SCHEMA}.billing_pricing_catalog
    WHERE category = 'telephony'
    ORDER BY provider
""")
rows = cur.fetchall()
for r in rows:
    print(r)

cur.close()
conn.close()
