"""Analyze user_identities and users table structure."""
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT', 5432),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

with conn.cursor(cursor_factory=RealDictCursor) as cur:
    print('=== lad_dev.user_identities COLUMNS ===')
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_schema = 'lad_dev' AND table_name = 'user_identities'
        ORDER BY ordinal_position
    """)
    for row in cur.fetchall():
        print(f"  {row['column_name']}: {row['data_type']}")
    
    print()
    print('=== lad_dev.users COLUMNS ===')
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns 
        WHERE table_schema = 'lad_dev' AND table_name = 'users'
        ORDER BY ordinal_position
    """)
    for row in cur.fetchall():
        print(f"  {row['column_name']}: {row['data_type']}")
    
    print()
    print('=== lad_dev.user_identities DATA (first 3) ===')
    cur.execute('SELECT * FROM lad_dev.user_identities LIMIT 3')
    for row in cur.fetchall():
        print(dict(row))

conn.close()
