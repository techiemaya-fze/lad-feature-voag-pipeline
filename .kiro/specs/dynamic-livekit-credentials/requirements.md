# Feature Requirements: Dynamic LiveKit Credentials

## 1. Overview

Enable dynamic LiveKit credentials per phone number by creating a new `voice_agent_livekit` table to store LiveKit server configurations. Phone numbers reference these configurations via UUID in their `rules` JSON, allowing different numbers to use different self-hosted LiveKit servers without requiring application restarts. Secrets are encrypted using a custom encryption utility with a recognizable prefix.

## 2. User Stories

### 2.1 As a system administrator
I want to configure different LiveKit servers for different phone numbers so that I can route calls to self-hosted LiveKit instances based on the originating number.

### 2.2 As a developer
I want the system to dynamically read LiveKit credentials from the database so that I can add new LiveKit servers without restarting the application.

### 2.3 As an operations team member
I want the worker/agent to continue reading from environment variables so that existing worker deployments remain unchanged and only the dispatch logic is affected.

### 2.4 As a security-conscious administrator
I want LiveKit API secrets to be encrypted in the database so that credentials are not stored in plain text.

### 2.5 As a CLI user
I want a simple command-line tool to encrypt/decrypt secrets so that I can manage credentials easily without writing code.

## 3. Acceptance Criteria

### 3.1 New Database Table: `voice_agent_livekit`
- A new table `voice_agent_livekit` MUST be created with the following schema:
  - `id`: UUID, Primary Key, Auto-generated
  - `name`: VARCHAR(255), NOT NULL, Unique - Human-readable identifier
  - `description`: TEXT, NULLABLE - Optional description of the LiveKit server
  - `livekit_url`: VARCHAR(500), NOT NULL - WebSocket URL (e.g., "wss://server.livekit.cloud")
  - `livekit_api_key`: VARCHAR(255), NOT NULL - LiveKit API key
  - `livekit_api_secret`: TEXT, NOT NULL - Encrypted LiveKit API secret (prefixed with "dev-s-t-")
  - `trunk_id`: VARCHAR(255), NULLABLE - SIP trunk ID for this LiveKit server
  - `created_at`: TIMESTAMP WITH TIME ZONE, NOT NULL, DEFAULT NOW() - Auto-generated, immutable
  - `updated_at`: TIMESTAMP WITH TIME ZONE, NOT NULL, DEFAULT NOW() - Auto-updated on row modification

### 3.2 Database Schema Extension: `voice_agent_numbers.rules`
- The `voice_agent_numbers.rules` JSONB column MUST support a new optional field:
  - `livekit_config`: UUID string - References `voice_agent_livekit.id`
- The existing `outbound_trunk_id` field MUST remain for backward compatibility
- If both `livekit_config` and `outbound_trunk_id` exist, `livekit_config` takes precedence

### 3.3 Feature Flag: `USE_SELFHOST_ROUTING_TABLE`
- A new environment variable `USE_SELFHOST_ROUTING_TABLE` MUST control the feature:
  - Default value: `true` (feature enabled)
  - When set to `false`: All new code paths MUST be bypassed, system behaves exactly as before
  - When set to `true`: New LiveKit config resolution logic is active
- The feature flag MUST be checked at the earliest point in the credential resolution flow
- When disabled, the system MUST:
  - Skip querying `voice_agent_livekit` table
  - Skip decryption logic
  - Use only environment variables for LiveKit credentials
  - Use only `rules.outbound_trunk_id` for trunk ID (existing behavior)

### 3.4 Encryption Utility: `utils/en_de_crypt.py`
- A new utility module `utils/en_de_crypt.py` MUST be created with the following features:
  - Single function: `encrypt_decrypt(input_string: str) -> str`
  - Auto-detects if input is encrypted (starts with "dev-s-t-")
  - If encrypted: Returns decrypted plain text
  - If not encrypted: Returns encrypted text with "dev-s-t-" prefix
  - CLI accessible: `uv run python utils/en_de_crypt.py "your_secret_here"`
  - Uses symmetric encryption (e.g., Fernet from cryptography library)
  - Encryption key MUST be read from environment variable `LIVEKIT_SECRET_ENCRYPTION_KEY`

### 3.5 Credential Resolution Logic
- When dispatching a call (single or batch), the system MUST:
  1. Check `USE_SELFHOST_ROUTING_TABLE` environment variable
  2. If `false`: Use environment variables only (skip steps 3-7, use existing behavior)
  3. Query `voice_agent_numbers` table using `from_number` and `tenant_id`
  4. Extract `livekit_config` UUID from `rules` JSON if present
  5. If UUID exists, query `voice_agent_livekit` table to get credentials
  6. Decrypt `livekit_api_secret` using `en_de_crypt.py`
  7. Fall back to environment variables if no `livekit_config` UUID or if any step fails
  8. Use the resolved credentials to create the LiveKit API client

### 3.6 Backward Compatibility
- When `USE_SELFHOST_ROUTING_TABLE=false`: System behaves exactly as it does currently (no new code executed)
- When `USE_SELFHOST_ROUTING_TABLE=true` (default):
  - If `rules` does not contain `livekit_config`, the system MUST fall back to environment variables
  - Existing phone numbers without `livekit_config` MUST continue to work without modification
  - The `outbound_trunk_id` field in `rules` MUST continue to work for legacy configurations
  - If `livekit_config` is present, its `trunk_id` overrides `rules.outbound_trunk_id`

### 3.7 Worker/Agent Unchanged
- The worker/agent code MUST NOT be modified
- Workers MUST continue reading LiveKit credentials from environment variables
- Only the dispatch request logic in `api/services/call_service.py` MUST be updated

### 3.8 Multi-Tenant Support
- Credential lookup MUST respect `tenant_id` for multi-tenant isolation
- The system MUST use the database schema prefix from environment variable (e.g., `lad_dev` or `lad_prod`)
- The `voice_agent_livekit` table is global (not tenant-specific) but access is controlled via `voice_agent_numbers`

### 3.9 Error Handling
- If `livekit_config` UUID is not found in `voice_agent_livekit` table, the system MUST fall back to environment variables
- If decryption fails, the system MUST log an error and fall back to environment variables
- If neither database nor environment variables provide credentials, the system MUST return a clear error message
- Invalid or missing credentials MUST be logged with appropriate error messages

### 3.10 Affected Components
- `utils/en_de_crypt.py`: New encryption/decryption utility (CLI + module)
- `db/storage/livekit.py`: New storage class for `voice_agent_livekit` table operations
- `api/services/call_service.py`: Update `dispatch_call()` method to resolve LiveKit config
- `utils/call_routing.py`: Update to return `livekit_config` UUID from rules
- Single call dispatch endpoint
- Batch call dispatch logic

## 4. Out of Scope

- Modifying worker/agent code to read from database
- Adding UI for managing LiveKit credentials (manual SQL inserts for now)
- Automatic credential rotation
- Validation of LiveKit credentials before storage (assumed valid)
- Multi-region LiveKit server selection logic

## 5. Technical Notes

### 5.1 Current Implementation
```python
# Current: Always reads from .env
url = os.getenv("LIVEKIT_URL")
api_key = os.getenv("LIVEKIT_API_KEY")
api_secret = os.getenv("LIVEKIT_API_SECRET")
outbound_trunk = routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
```

### 5.2 Proposed Implementation with Feature Flag
```python
# Check feature flag first
use_selfhost = os.getenv("USE_SELFHOST_ROUTING_TABLE", "true").lower() == "true"

if use_selfhost:
    # New behavior: Try database first
    rules = get_number_rules(from_number, tenant_id)
    livekit_config_id = rules.get('livekit_config')
    
    if livekit_config_id:
        livekit_config = get_livekit_config(livekit_config_id)
        url = livekit_config['livekit_url']
        api_key = livekit_config['livekit_api_key']
        api_secret = decrypt(livekit_config['livekit_api_secret'])
        outbound_trunk = livekit_config['trunk_id'] or rules.get('outbound_trunk_id')
    else:
        # Fallback to environment variables
        url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        outbound_trunk = rules.get('outbound_trunk_id') or os.getenv("OUTBOUND_TRUNK_ID")
else:
    # Legacy behavior: Use environment variables only
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    outbound_trunk = routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
```

### 5.3 Database Schema SQL
```sql
-- New table for LiveKit configurations
CREATE TABLE lad_dev.voice_agent_livekit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    livekit_url VARCHAR(500) NOT NULL,
    livekit_api_key VARCHAR(255) NOT NULL,
    livekit_api_secret TEXT NOT NULL,  -- Encrypted with "dev-s-t-" prefix
    trunk_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION lad_dev.update_voice_agent_livekit_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER voice_agent_livekit_updated_at
    BEFORE UPDATE ON lad_dev.voice_agent_livekit
    FOR EACH ROW
    EXECUTE FUNCTION lad_dev.update_voice_agent_livekit_updated_at();
```

### 5.4 Example Rules JSON (Updated)
```json
{
  "inbound": true,
  "outbound": true,
  "allowed_outbound": "global",
  "required_lead_format": ["country_code", "base_number"],
  "outbound_trunk_id": "legacy_trunk_id",
  "livekit_config": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 5.5 Example voice_agent_livekit Row
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "self-hosted-mumbai",
  "description": "Self-hosted LiveKit server in Mumbai datacenter",
  "livekit_url": "wss://mumbai.livekit.example.com",
  "livekit_api_key": "APIxxxxxxxxx",
  "livekit_api_secret": "dev-s-t-gAAAAABh1234567890abcdefghijklmnop...",
  "trunk_id": "mumbai_trunk_001",
  "created_at": "2026-02-16T10:30:00+00:00",
  "updated_at": "2026-02-16T10:30:00+00:00"
}
```

### 5.6 Encryption Utility Usage
```bash
# Encrypt a secret
uv run python utils/en_de_crypt.py "my_secret_api_key"
# Output: dev-s-t-gAAAAABh1234567890abcdefghijklmnop...

# Decrypt a secret
uv run python utils/en_de_crypt.py "dev-s-t-gAAAAABh1234567890abcdefghijklmnop..."
# Output: my_secret_api_key
```

## 6. Dependencies

- Existing database table: `lad_dev.voice_agent_numbers`
- New database table: `lad_dev.voice_agent_livekit` (to be created)
- Existing function: `get_number_rules()` in `utils/call_routing.py`
- Existing function: `dispatch_call()` in `api/services/call_service.py`
- Python library: `cryptography` (for Fernet encryption)
- Environment variable: `LIVEKIT_SECRET_ENCRYPTION_KEY` (32-byte base64 key)

## 7. Testing Requirements

### 7.1 Unit Tests
- Test feature flag behavior:
  - With `USE_SELFHOST_ROUTING_TABLE=false`: Verify only env vars are used
  - With `USE_SELFHOST_ROUTING_TABLE=true`: Verify database lookup is attempted
- Test encryption/decryption utility:
  - Encrypt plain text and verify "dev-s-t-" prefix
  - Decrypt encrypted text and verify original value
  - Handle invalid input gracefully
- Test credential resolution:
  - With valid `livekit_config` UUID
  - With invalid/missing UUID (fallback to env)
  - With no `livekit_config` (fallback to env)
  - With decryption failure (fallback to env)
- Test multi-tenant isolation

### 7.2 Integration Tests
- Test feature flag toggle:
  - Disable feature and verify system works as before
  - Enable feature and verify new behavior
- Test single call dispatch with database credentials
- Test batch call dispatch with database credentials
- Test mixed scenario (some numbers with DB config, some with env)
- Test backward compatibility with `outbound_trunk_id` only

### 7.3 Manual Testing
- Test feature flag toggle without restart (if possible)
- Verify existing calls continue to work (backward compatibility)
- Verify new calls with self-hosted LiveKit credentials work correctly
- Verify CLI encryption tool works correctly
- Verify error messages are clear when credentials are missing
- Verify `updated_at` timestamp updates automatically on row modification

## 8. Security Considerations

### 8.1 Encryption Key Management
- The `LIVEKIT_SECRET_ENCRYPTION_KEY` MUST be stored securely in environment variables
- The key MUST be a 32-byte base64-encoded string (Fernet requirement)
- Generate key using: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- The key MUST NOT be committed to version control

### 8.2 Database Access Control
- Access to `voice_agent_livekit` table MUST be restricted to authorized users
- Encrypted secrets in the database provide defense-in-depth but are not a substitute for proper access control

### 8.3 Logging
- Decrypted secrets MUST NEVER be logged
- Only log that decryption succeeded/failed, not the actual values

## 9. Migration Path

### 9.1 Phase 1: Create Infrastructure
1. Create `voice_agent_livekit` table with trigger
2. Create `utils/en_de_crypt.py` utility
3. Add `USE_SELFHOST_ROUTING_TABLE=true` to `.env.example`
4. Add `LIVEKIT_SECRET_ENCRYPTION_KEY` to `.env.example`

### 9.2 Phase 2: Update Code
1. Create `db/storage/livekit.py` storage class
2. Update `api/services/call_service.py` to resolve LiveKit config (with feature flag check)
3. Update `utils/call_routing.py` to return `livekit_config` UUID

### 9.3 Phase 3: Testing & Rollout
1. Deploy with `USE_SELFHOST_ROUTING_TABLE=false` (verify no regression)
2. Enable feature flag: `USE_SELFHOST_ROUTING_TABLE=true`
3. Test with one phone number using new config
4. Gradually migrate phone numbers to new system
5. Keep environment variable fallback for safety

### 9.4 Phase 4: Documentation
1. Document how to add new LiveKit servers
2. Document encryption key generation and rotation
3. Document feature flag usage and rollback procedure
4. Update deployment guides
