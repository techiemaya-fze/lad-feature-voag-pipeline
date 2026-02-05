import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT', '5432')),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

# Get full schema of lad_dev.knowledge_base_catalog
cur.execute("""
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns 
    WHERE table_schema = 'lad_dev' AND table_name = 'knowledge_base_catalog'
    ORDER BY ordinal_position
""")
rows = cur.fetchall()
print('lad_dev.knowledge_base_catalog columns:')
for r in rows:
    print(f'  {r[0]}: {r[1]} (nullable={r[2]}, default={r[3]})')

conn.close()
