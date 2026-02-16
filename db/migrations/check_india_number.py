"""Check India number configuration"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT', 5432),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

cur = conn.cursor()

# Check agent 33's tenant
print("Agent 33 Tenant:")
cur.execute("SELECT id, name, tenant_id FROM lad_dev.voice_agents WHERE id = 33")
agent = cur.fetchone()
if agent:
    print(f"  Agent ID: {agent[0]}")
    print(f"  Name: {agent[1]}")
    print(f"  Tenant ID: {agent[2]}")
    agent_tenant = agent[2]
else:
    print("  Agent not found!")
    agent_tenant = None

print("\n" + "="*80)

# Check all numbers with base_number 9513456728
print("\nAll numbers with base_number 9513456728:")
cur.execute("""
    SELECT id, tenant_id, country_code, base_number, provider, rules
    FROM lad_dev.voice_agent_numbers
    WHERE base_number = '9513456728'
""")
numbers = cur.fetchall()

if numbers:
    for num in numbers:
        print(f"\n  Number ID: {num[0]}")
        print(f"  Tenant ID: {num[1]}")
        print(f"  Full Number: {num[2]}{num[3]}")
        print(f"  Provider: {num[4]}")
        print(f"  Rules: {num[5]}")
        
        if agent_tenant and num[1] == agent_tenant:
            print(f"  ✓ MATCHES agent tenant!")
else:
    print("  No numbers found!")

print("\n" + "="*80)

# Check the India LiveKit config
print("\nIndia LiveKit Config:")
cur.execute("""
    SELECT id, name, livekit_url, worker_name
    FROM lad_dev.voice_agent_livekit
    WHERE name = 'india-techiemaya-cloud'
""")
config = cur.fetchone()
if config:
    print(f"  Config ID: {config[0]}")
    print(f"  Name: {config[1]}")
    print(f"  URL: {config[2]}")
    print(f"  Worker: {config[3]}")
    config_id = config[0]
else:
    print("  Config not found!")
    config_id = None

print("\n" + "="*80)

# If we found the right number and config, update it
if agent_tenant and config_id:
    print(f"\nUpdating number for tenant {agent_tenant[:8]}... with config {config_id[:8]}...")
    
    cur.execute("""
        UPDATE lad_dev.voice_agent_numbers
        SET rules = jsonb_set(
            COALESCE(rules, '{}'::jsonb),
            '{livekit_config}',
            %s::jsonb
        )
        WHERE base_number = '9513456728'
        AND tenant_id = %s
        RETURNING id, rules
    """, (f'"{config_id}"', agent_tenant))
    
    result = cur.fetchone()
    if result:
        conn.commit()
        print(f"  ✓ Updated number {result[0]}")
        print(f"  New rules: {result[1]}")
    else:
        print(f"  ✗ Number not found for tenant {agent_tenant}")

conn.close()
