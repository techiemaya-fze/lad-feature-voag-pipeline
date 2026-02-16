"""Safe query to discover actual schemas and data."""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")

# 1) voice_agent_numbers schema
print("=" * 90)
print("VOICE_AGENT_NUMBERS - COLUMN NAMES")
print("=" * 90)
cur.execute(f"SELECT * FROM {SCHEMA}.voice_agent_numbers LIMIT 1")
cols = [d[0] for d in cur.description]
for i, c in enumerate(cols):
    print(f"  [{i}] {c}")

# 2) Sample row
print("\nSAMPLE ROW:")
row = cur.fetchone()
if row:
    for i, c in enumerate(cols):
        print(f"  {c} = {row[i]!r}")

# 3) Distinct providers
print()
print("=" * 90)
print("DISTINCT PROVIDERS IN voice_agent_numbers")
print("=" * 90)
cur.execute(f"SELECT DISTINCT provider, COUNT(*) FROM {SCHEMA}.voice_agent_numbers GROUP BY provider ORDER BY provider")
for r in cur.fetchall():
    print(f"  provider={r[0]!r:<30} count={r[1]}")

# 4) billing_pricing_catalog - ALL rows
print()
print("=" * 90)
print("BILLING_PRICING_CATALOG - ALL ROWS")
print("=" * 90)
cur.execute(f"SELECT * FROM {SCHEMA}.billing_pricing_catalog ORDER BY category, provider, model")
cols2 = [d[0] for d in cur.description]
print(f"  Columns: {cols2}")
for r in cur.fetchall():
    row_dict = dict(zip(cols2, r))
    print(f"  cat={row_dict.get('category')!r:<20} provider={row_dict.get('provider')!r:<20} model={row_dict.get('model')!r:<15} unit={row_dict.get('unit')!r:<12} price={row_dict.get('unit_price')} desc={row_dict.get('description')!r}")

cur.close()
conn.close()
