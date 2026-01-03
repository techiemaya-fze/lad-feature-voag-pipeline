"""Insert tenant_features entries for voice agent tools."""
import psycopg2

conn = psycopg2.connect(
    host='165.22.221.77',
    port=5432,
    database='salesmaya_agent',
    user='dbadmin',
    password='TechieMaya'
)
cur = conn.cursor()

tenant_id = '926070b5-189b-4682-9279-ea10ca090b84'

# All tool features: (feature_key, enabled, config_json)
features = [
    ('voice-agent-tool-google-calendar', False, None),
    ('voice-agent-tool-google-workspace', False, None),
    ('voice-agent-tool-gmail', False, None),
    ('voice-agent-tool-microsoft-bookings-auto', True, None),
    ('voice-agent-tool-microsoft-bookings-manual', True, None),
    ('voice-agent-tool-email-templates', True, None),
    ('voice-agent-tool-knowledge-base', True, None),
    ('voice-agent-tool-human-support', True, None),
]

for feature_key, enabled, config in features:
    cur.execute("""
        INSERT INTO lad_dev.tenant_features (tenant_id, feature_key, enabled, config)
        VALUES (%s, %s, %s, %s::jsonb)
        ON CONFLICT (tenant_id, feature_key)
        DO UPDATE SET enabled = EXCLUDED.enabled, config = EXCLUDED.config
    """, (tenant_id, feature_key, enabled, config))

conn.commit()
print(f'Inserted/updated {len(features)} features for tenant {tenant_id}')

# Verify
cur.execute("""
    SELECT feature_key, enabled
    FROM lad_dev.tenant_features
    WHERE tenant_id = %s AND feature_key LIKE 'voice-agent-tool-%%'
    ORDER BY feature_key
""", (tenant_id,))

print('\nCurrent features:')
for row in cur.fetchall():
    status = '✓' if row[1] else '✗'
    print(f'  {status} {row[0]}: enabled={row[1]}')

conn.close()
