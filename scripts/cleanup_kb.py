"""Cleanup KB catalog - remove duplicate, update count, fix default."""
import psycopg2

conn = psycopg2.connect('dbname=salesmaya_agent user=dbadmin password=TechieMaya host=165.22.221.77 port=5432')
cur = conn.cursor()

# 1. Remove the empty duplicate glinks_corrected store
cur.execute("DELETE FROM lad_dev.knowledge_base_catalog WHERE id = '077d59f6-c049-458e-9989-efeab0b255e9'")
print(f'1. Deleted empty duplicate: {cur.rowcount} row')

# 2. Update document count to 128 (actual count from Gemini)
cur.execute("UPDATE lad_dev.knowledge_base_catalog SET document_count = 128 WHERE id = '339e1f15-3206-4450-9b6d-d1766472a866'")
print(f'2. Updated document count to 128: {cur.rowcount} row')

# 3. Unmark old G_links_trial as default
cur.execute("UPDATE lad_dev.knowledge_base_catalog SET is_default = false WHERE id = 'ff825aaf-bb5e-418c-80ea-f5059eaefab2'")
print(f'3. Unmarked G_links_trial as default: {cur.rowcount} row')

conn.commit()
print('Done!')
conn.close()
