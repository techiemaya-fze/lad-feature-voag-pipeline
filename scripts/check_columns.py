"""Check all column types in voice_call_analysis table."""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()

cur.execute("""
    SELECT column_name, data_type, character_maximum_length 
    FROM information_schema.columns 
    WHERE table_schema = 'lad_dev' AND table_name = 'voice_call_analysis' 
    ORDER BY ordinal_position
""")

print("lad_dev.voice_call_analysis columns:")
for row in cur.fetchall():
    if row[2]:
        print(f"  {row[0]}: {row[1]}({row[2]})")
    else:
        print(f"  {row[0]}: {row[1]}")

cur.close()
conn.close()
