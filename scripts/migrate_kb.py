"""Execute KB migration and dev seed insertion - v2 (fixed)."""
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT', 5432),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
conn.autocommit = False

GLINKS_TENANT_ID = '926070b5-189b-4682-9279-ea10ca090b84'
GLINKS_KB_STORE_ID = 'd2c8e01f-dc2e-4a05-9735-52bede6eb3af'

try:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        
        # =============================================
        # TASK 1: Insert Dev Seed User "Sahil Tomar"
        # =============================================
        print("Task 1: Inserting dev seed user 'Sahil Tomar'...")
        
        # Check if user already exists
        cur.execute("SELECT id FROM lad_dev.users WHERE email = %s", ('dev.sahil.tomar@gmail.com',))
        existing_user = cur.fetchone()
        
        if existing_user:
            sahil_id = existing_user['id']
            print(f"  ✓ User already exists with id: {sahil_id}")
        else:
            sahil_user_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO lad_dev.users (id, email, first_name, last_name, primary_tenant_id, password_hash)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                sahil_user_id,
                'dev.sahil.tomar@gmail.com',
                'Sahil',
                'Tomar',
                GLINKS_TENANT_ID,
                'dev_seed_no_login'  # Placeholder hash
            ))
            sahil_id = cur.fetchone()['id']
            print(f"  ✓ User inserted with id: {sahil_id}")
        
        # =============================================
        # TASK 2: Create knowledge_base_catalog table
        # =============================================
        print("\nTask 2: Creating lad_dev.knowledge_base_catalog table...")
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lad_dev.knowledge_base_catalog (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                tenant_id UUID NOT NULL REFERENCES lad_dev.tenants(id) ON DELETE CASCADE,
                gemini_store_name VARCHAR(512) NOT NULL,
                display_name VARCHAR(255) NOT NULL,
                description TEXT,
                is_default BOOLEAN DEFAULT false,
                is_active BOOLEAN DEFAULT true,
                priority INTEGER DEFAULT 0,
                document_count INTEGER DEFAULT 0,
                created_by UUID REFERENCES lad_dev.users(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(tenant_id, gemini_store_name)
            )
        """)
        print("  ✓ Table created (or already exists)")
        
        # =============================================
        # TASK 3: Migrate Glinks KB Store
        # =============================================
        print("\nTask 3: Migrating Glinks KB store...")
        
        # Check if already migrated
        cur.execute("""
            SELECT id FROM lad_dev.knowledge_base_catalog 
            WHERE tenant_id = %s AND gemini_store_name LIKE '%%glinkstrial%%'
        """, (GLINKS_TENANT_ID,))
        existing_kb = cur.fetchone()
        
        if existing_kb:
            print(f"  ✓ KB already migrated with id: {existing_kb['id']}")
        else:
            # Get the existing KB store data
            cur.execute("""
                SELECT gemini_store_name, display_name, description, is_active, document_count, created_at
                FROM voice_agent.knowledge_base_stores
                WHERE id = %s
            """, (GLINKS_KB_STORE_ID,))
            
            kb_row = cur.fetchone()
            if kb_row:
                cur.execute("""
                    INSERT INTO lad_dev.knowledge_base_catalog 
                        (tenant_id, gemini_store_name, display_name, description, is_default, is_active, 
                         document_count, created_by, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    GLINKS_TENANT_ID,
                    kb_row['gemini_store_name'],
                    kb_row['display_name'],
                    kb_row['description'],
                    True,  # is_default = true for Glinks
                    kb_row['is_active'],
                    kb_row['document_count'],
                    sahil_id,  # created_by
                    kb_row['created_at']
                ))
                kb_catalog_id = cur.fetchone()['id']
                print(f"  ✓ KB store migrated with id: {kb_catalog_id}")
                print(f"    - gemini_store_name: {kb_row['gemini_store_name']}")
                print(f"    - is_default: true")
            else:
                print("  ⚠ Glinks KB store not found in voice_agent.knowledge_base_stores")
        
        # Commit all changes
        conn.commit()
        print("\n✅ All migrations committed successfully!")
        
        # Verify
        print("\n=== VERIFICATION ===")
        cur.execute("SELECT id, email, first_name, last_name, primary_tenant_id FROM lad_dev.users WHERE email = 'dev.sahil.tomar@gmail.com'")
        user = cur.fetchone()
        print(f"User: {dict(user) if user else 'NOT FOUND'}")
        
        cur.execute("SELECT id, tenant_id, gemini_store_name, display_name, is_default, is_active FROM lad_dev.knowledge_base_catalog")
        for row in cur.fetchall():
            print(f"KB Catalog: {dict(row)}")

except Exception as e:
    conn.rollback()
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    conn.close()
