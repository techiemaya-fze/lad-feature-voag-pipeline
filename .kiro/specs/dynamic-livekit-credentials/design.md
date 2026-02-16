# Design Document: Dynamic LiveKit Credentials

## 1. Architecture Overview

This feature introduces a new database table and credential resolution system to support multiple LiveKit servers per tenant. The design follows a modular approach with minimal changes to existing large files.

### 1.1 High-Level Flow

```
Call Dispatch Request
    ↓
Check USE_SELFHOST_ROUTING_TABLE flag
    ↓
[If false] → Use environment variables (existing behavior)
    ↓
[If true] → Query voice_agent_numbers for livekit_config UUID
    ↓
[If UUID exists] → Query voice_agent_livekit table
    ↓
Decrypt livekit_api_secret
    ↓
Use resolved credentials for LiveKit API client
    ↓
[On any error] → Fallback to environment variables
```

### 1.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    api/services/call_service.py             │
│  (Minimal changes - delegates to livekit_resolver)          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              utils/livekit_resolver.py (NEW)                │
│  - resolve_livekit_credentials()                            │
│  - Feature flag check                                       │
│  - Credential resolution logic                              │
│  - Fallback handling                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────────┐  ┌────────────────────────┐
│  db/storage/livekit.py (NEW) │  │ utils/en_de_crypt.py   │
│  - LiveKitConfigStorage      │  │ (NEW)                  │
│  - get_livekit_config()      │  │ - encrypt_decrypt()    │
│  - CRUD operations           │  │ - CLI interface        │
└──────────────────────────────┘  └────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│         Database: lad_dev.voice_agent_livekit                │
│  - id (UUID PK)                                              │
│  - name, description                                         │
│  - livekit_url, livekit_api_key, livekit_api_secret         │
│  - trunk_id                                                  │
│  - created_at, updated_at (auto-managed)                     │
└──────────────────────────────────────────────────────────────┘
```

## 2. Database Design

### 2.1 New Table: `voice_agent_livekit`

```sql
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

-- Index for faster lookups
CREATE INDEX idx_voice_agent_livekit_name ON lad_dev.voice_agent_livekit(name);

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

### 2.2 Schema Extension: `voice_agent_numbers.rules`

The `rules` JSONB column will support a new optional field:

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

**Note**: No schema migration needed for `voice_agent_numbers` - JSONB is flexible.

## 3. Module Design

### 3.1 Encryption Utility: `utils/en_de_crypt.py`

**Purpose**: Symmetric encryption/decryption with auto-detection and CLI support.

**Key Features**:
- Single function handles both encryption and decryption
- Auto-detects encrypted strings by "dev-s-t-" prefix
- Uses Fernet (symmetric encryption from cryptography library)
- CLI accessible for manual credential management

**Interface**:
```python
def encrypt_decrypt(input_string: str) -> str:
    """
    Auto-detect and encrypt/decrypt a string.
    
    Args:
        input_string: Plain text or encrypted string
        
    Returns:
        - If input starts with "dev-s-t-": Returns decrypted plain text
        - Otherwise: Returns encrypted string with "dev-s-t-" prefix
        
    Raises:
        ValueError: If LIVEKIT_SECRET_ENCRYPTION_KEY not set
        cryptography.fernet.InvalidToken: If decryption fails
    """
```

**CLI Usage**:
```bash
# Encrypt
uv run python utils/en_de_crypt.py "my_secret_key"
# Output: dev-s-t-gAAAAABh1234...

# Decrypt
uv run python utils/en_de_crypt.py "dev-s-t-gAAAAABh1234..."
# Output: my_secret_key
```

**Implementation Notes**:
- Read encryption key from `LIVEKIT_SECRET_ENCRYPTION_KEY` env var
- Key must be 32-byte base64 string (Fernet requirement)
- Prefix "dev-s-t-" makes encrypted strings easily identifiable
- Never log decrypted values

### 3.2 LiveKit Config Storage: `db/storage/livekit.py`

**Purpose**: Database operations for `voice_agent_livekit` table.

**Class**: `LiveKitConfigStorage`

**Methods**:
```python
class LiveKitConfigStorage:
    """Storage operations for voice_agent_livekit table."""
    
    async def get_livekit_config(self, config_id: str) -> dict | None:
        """
        Fetch LiveKit configuration by UUID.
        
        Args:
            config_id: UUID string
            
        Returns:
            Dict with keys: id, name, description, livekit_url,
            livekit_api_key, livekit_api_secret (encrypted), trunk_id,
            created_at, updated_at
            
            Returns None if not found.
        """
    
    async def create_livekit_config(
        self,
        name: str,
        livekit_url: str,
        livekit_api_key: str,
        livekit_api_secret: str,  # Should be encrypted before passing
        trunk_id: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Create a new LiveKit configuration.
        
        Returns:
            UUID of created config
        """
    
    async def update_livekit_config(
        self,
        config_id: str,
        **updates
    ) -> bool:
        """
        Update LiveKit configuration.
        updated_at is automatically updated by trigger.
        
        Returns:
            True if updated, False if not found
        """
    
    async def delete_livekit_config(self, config_id: str) -> bool:
        """
        Delete LiveKit configuration.
        
        Returns:
            True if deleted, False if not found
        """
    
    async def list_livekit_configs(self) -> list[dict]:
        """List all LiveKit configurations."""
```

**Implementation Notes**:
- Uses async/await for consistency with existing storage classes
- Follows existing storage pattern (see `db/storage/calls.py`, `db/storage/agents.py`)
- Uses connection pool from `db/connection_pool.py`
- Returns RealDictCursor results for easy dict access

### 3.3 LiveKit Credential Resolver: `utils/livekit_resolver.py` (NEW)

**Purpose**: Centralized credential resolution logic with feature flag support.

**Key Function**:
```python
@dataclass
class LiveKitCredentials:
    """Resolved LiveKit credentials."""
    url: str
    api_key: str
    api_secret: str  # Decrypted
    trunk_id: str | None
    source: str  # "database" or "environment" for logging


async def resolve_livekit_credentials(
    from_number: str | None,
    tenant_id: str | None,
    routing_result: CallRoutingResult | None = None,
) -> LiveKitCredentials:
    """
    Resolve LiveKit credentials from database or environment variables.
    
    Args:
        from_number: Caller number (used to lookup config)
        tenant_id: Tenant ID for multi-tenant isolation
        routing_result: Optional routing result with outbound_trunk_id
        
    Returns:
        LiveKitCredentials with resolved values
        
    Raises:
        RuntimeError: If no credentials found in database or environment
        
    Flow:
        1. Check USE_SELFHOST_ROUTING_TABLE flag
        2. If false: Return environment variables
        3. If true: Try database lookup
        4. On any error: Fallback to environment variables
    """
```

**Helper Functions**:
```python
def _get_credentials_from_env(
    routing_result: CallRoutingResult | None = None
) -> LiveKitCredentials:
    """Get credentials from environment variables (existing behavior)."""


async def _get_credentials_from_database(
    from_number: str,
    tenant_id: str | None,
) -> LiveKitCredentials | None:
    """
    Get credentials from database via livekit_config UUID.
    
    Returns None if not found or on any error (triggers fallback).
    """
```

**Implementation Notes**:
- Feature flag checked first: `os.getenv("USE_SELFHOST_ROUTING_TABLE", "true").lower() == "true"`
- All database errors caught and logged, then fallback to env
- Decryption errors caught and logged, then fallback to env
- Never log decrypted secrets
- Log source of credentials (database vs environment) for debugging

## 4. Integration Points

### 4.1 Minimal Changes to `api/services/call_service.py`

**Current Code** (around line 730):
```python
url, api_key, api_secret = _validate_livekit_credentials()
```

**New Code**:
```python
from utils.livekit_resolver import resolve_livekit_credentials

# ... existing code ...

# Resolve LiveKit credentials (database or environment)
livekit_creds = await resolve_livekit_credentials(
    from_number=from_number,
    tenant_id=tenant_id,
    routing_result=routing_result,
)
url = livekit_creds.url
api_key = livekit_creds.api_key
api_secret = livekit_creds.api_secret
```

**Changes to trunk_id resolution** (around line 777):
```python
# OLD:
outbound_trunk = routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")

# NEW:
outbound_trunk = livekit_creds.trunk_id or routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
```

**Total Changes**: ~10 lines modified in `call_service.py`

### 4.2 Update to `utils/call_routing.py`

**Purpose**: Return `livekit_config` UUID from rules (if present).

**Changes to `CallRoutingResult` dataclass**:
```python
@dataclass
class CallRoutingResult:
    """Result of call routing validation and formatting."""
    success: bool
    formatted_to_number: Optional[str] = None
    error_message: Optional[str] = None
    carrier_name: Optional[str] = None
    detected_country: Optional[str] = None
    outbound_trunk_id: Optional[str] = None
    livekit_config_id: Optional[str] = None  # NEW: UUID from rules
```

**Changes to `validate_and_format_call()` function** (around line 390):
```python
carrier_name = carrier_info['carrier_name']
rules = carrier_info['rules'] or {}
outbound_trunk_id = rules.get('outbound_trunk_id')
livekit_config_id = rules.get('livekit_config')  # NEW: Extract UUID
```

**Changes to return statement** (around line 460):
```python
return CallRoutingResult(
    success=True,
    formatted_to_number=formatted_number,
    carrier_name=carrier_name,
    detected_country=detected_country,
    outbound_trunk_id=outbound_trunk_id,
    livekit_config_id=livekit_config_id,  # NEW: Include UUID
)
```

**Total Changes**: ~5 lines added to `call_routing.py`

## 5. Error Handling & Fallback Strategy

### 5.1 Fallback Hierarchy

```
1. Feature flag disabled (USE_SELFHOST_ROUTING_TABLE=false)
   → Use environment variables

2. Feature flag enabled but no livekit_config UUID in rules
   → Use environment variables

3. livekit_config UUID not found in database
   → Log warning, use environment variables

4. Decryption fails
   → Log error, use environment variables

5. Database connection error
   → Log error, use environment variables

6. Environment variables missing
   → Raise RuntimeError (existing behavior)
```

### 5.2 Error Logging

**Log Levels**:
- `INFO`: Credential source (database vs environment)
- `WARNING`: Fallback triggered (UUID not found, feature disabled)
- `ERROR`: Decryption failure, database error
- `CRITICAL`: No credentials available (database and environment)

**Example Logs**:
```python
logger.info(f"Using LiveKit credentials from database (config: {config_name})")
logger.warning(f"livekit_config UUID not found: {config_id}, falling back to environment")
logger.error(f"Failed to decrypt livekit_api_secret for config {config_id}: {e}")
```

**Security**: Never log decrypted secrets or encryption keys.

## 6. Testing Strategy

### 6.1 Unit Tests

**File**: `tests/test_en_de_crypt.py`
- Test encryption produces "dev-s-t-" prefix
- Test decryption returns original value
- Test round-trip (encrypt → decrypt)
- Test invalid input handling
- Test missing encryption key

**File**: `tests/test_livekit_storage.py`
- Test CRUD operations
- Test UUID generation
- Test updated_at trigger
- Test unique name constraint

**File**: `tests/test_livekit_resolver.py`
- Test feature flag disabled → env vars only
- Test feature flag enabled → database lookup
- Test fallback scenarios
- Test error handling

### 6.2 Integration Tests

**File**: `tests/integration/test_call_dispatch_livekit.py`
- Test single call with database credentials
- Test batch call with database credentials
- Test mixed scenario (some DB, some env)
- Test feature flag toggle

### 6.3 Manual Testing Checklist

- [ ] Deploy with `USE_SELFHOST_ROUTING_TABLE=false`, verify no regression
- [ ] Enable feature flag, verify database lookup works
- [ ] Test CLI encryption tool
- [ ] Test credential fallback (remove UUID from rules)
- [ ] Test decryption failure (corrupt encrypted secret)
- [ ] Verify `updated_at` auto-updates on row modification
- [ ] Test with multiple tenants

## 7. Security Considerations

### 7.1 Encryption Key Management

**Generation**:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Storage**:
- Store in `.env` file (not committed to git)
- Use secrets management in production (AWS Secrets Manager, HashiCorp Vault, etc.)
- Rotate keys periodically (requires re-encrypting all secrets)

**Key Rotation Process**:
1. Generate new key
2. Decrypt all secrets with old key
3. Re-encrypt with new key
4. Update database
5. Update environment variable

### 7.2 Database Security

- Restrict access to `voice_agent_livekit` table
- Use database roles and permissions
- Audit access logs
- Encrypted secrets provide defense-in-depth

### 7.3 Logging Security

**Never log**:
- Decrypted `livekit_api_secret`
- Encryption keys
- Full encrypted strings (log only first 20 chars for debugging)

**Safe to log**:
- Config UUID
- Config name
- Credential source (database vs environment)
- Success/failure status

## 8. Performance Considerations

### 8.1 Database Query Optimization

- Index on `voice_agent_livekit.name` for faster lookups
- UUID primary key for fast joins
- Connection pooling (already implemented)

### 8.2 Caching Strategy (Future Enhancement)

**Not implemented in v1, but consider for future**:
- Cache LiveKit configs in memory (TTL: 5 minutes)
- Invalidate cache on config updates
- Reduces database queries for high-volume calls

### 8.3 Decryption Performance

- Fernet decryption is fast (~0.1ms per operation)
- Negligible impact on call dispatch latency
- No caching needed for decrypted secrets (security risk)

## 9. Deployment Plan

### 9.1 Phase 1: Infrastructure Setup

1. **Database Migration**:
   ```bash
   uv run python -m db.migrations.create_voice_agent_livekit_table
   ```

2. **Environment Variables**:
   ```bash
   # Add to .env
   USE_SELFHOST_ROUTING_TABLE=false  # Start disabled
   LIVEKIT_SECRET_ENCRYPTION_KEY=<generated_key>
   ```

3. **Install Dependencies**:
   ```bash
   uv pip install cryptography
   ```

### 9.2 Phase 2: Code Deployment

1. Deploy new files:
   - `utils/en_de_crypt.py`
   - `utils/livekit_resolver.py`
   - `db/storage/livekit.py`

2. Deploy modified files:
   - `api/services/call_service.py` (minimal changes)
   - `utils/call_routing.py` (minimal changes)

3. Verify deployment with feature flag disabled

### 9.3 Phase 3: Testing & Enablement

1. **Create Test Config**:
   ```bash
   # Encrypt secret
   SECRET=$(uv run python utils/en_de_crypt.py "test_secret_key")
   
   # Insert test config
   psql -c "INSERT INTO lad_dev.voice_agent_livekit (name, livekit_url, livekit_api_key, livekit_api_secret, trunk_id) VALUES ('test-config', 'wss://test.livekit.cloud', 'test_key', '$SECRET', 'test_trunk');"
   ```

2. **Update Test Number**:
   ```sql
   UPDATE lad_dev.voice_agent_numbers
   SET rules = jsonb_set(rules, '{livekit_config}', '"<uuid_from_insert>"')
   WHERE base_number = <test_number>;
   ```

3. **Enable Feature Flag**:
   ```bash
   USE_SELFHOST_ROUTING_TABLE=true
   ```

4. **Test Call**: Make test call from configured number

5. **Monitor Logs**: Verify "Using LiveKit credentials from database"

### 9.4 Phase 4: Gradual Rollout

1. Migrate 10% of numbers to database configs
2. Monitor for 24 hours
3. Migrate 50% of numbers
4. Monitor for 48 hours
5. Migrate remaining numbers
6. Keep environment variable fallback indefinitely

## 10. Rollback Plan

### 10.1 Immediate Rollback

**If issues detected**:
```bash
# Disable feature flag
USE_SELFHOST_ROUTING_TABLE=false
```

System immediately reverts to environment variables. No code deployment needed.

### 10.2 Full Rollback

**If feature needs to be removed**:
1. Set `USE_SELFHOST_ROUTING_TABLE=false`
2. Revert code changes to `call_service.py` and `call_routing.py`
3. Remove new files (optional, they won't be called)
4. Keep database table (data preserved for future use)

## 11. Monitoring & Observability

### 11.1 Metrics to Track

- Credential resolution source (database vs environment) - percentage
- Fallback rate (how often database lookup fails)
- Decryption errors
- Database query latency for LiveKit config lookups
- Call dispatch success rate (before/after feature)

### 11.2 Alerts

- **Critical**: Decryption failure rate > 5%
- **Warning**: Fallback rate > 10%
- **Info**: Feature flag state change

### 11.3 Logging

**Key log messages**:
```python
logger.info(f"LiveKit credentials resolved from {source} for number {from_number[:4]}***")
logger.warning(f"Falling back to environment variables: {reason}")
logger.error(f"Failed to resolve LiveKit credentials from database: {error}")
```

## 12. Future Enhancements

### 12.1 Admin UI (Out of Scope for v1)

- Web interface for managing LiveKit configs
- Credential validation before save
- Audit log for config changes

### 12.2 Credential Rotation (Out of Scope for v1)

- Automated key rotation
- Graceful credential updates without downtime

### 12.3 Multi-Region Support (Out of Scope for v1)

- Geographic routing based on caller location
- Latency-based server selection

### 12.4 Caching Layer (Out of Scope for v1)

- In-memory cache for frequently used configs
- Redis-based distributed cache

## 13. Documentation Requirements

### 13.1 Developer Documentation

- How to add new LiveKit server configuration
- How to encrypt secrets using CLI tool
- How to test credential resolution locally
- How to troubleshoot fallback scenarios

### 13.2 Operations Documentation

- Deployment checklist
- Rollback procedures
- Monitoring and alerting setup
- Encryption key rotation process

### 13.3 API Documentation

- No API changes (internal feature only)
- Update internal architecture docs

## 14. Acceptance Criteria Mapping

| Requirement | Design Component | Status |
|-------------|------------------|--------|
| 3.1 New Table | Section 2.1 | ✓ Designed |
| 3.2 Schema Extension | Section 2.2 | ✓ Designed |
| 3.3 Feature Flag | Section 3.3, 4.1 | ✓ Designed |
| 3.4 Encryption Utility | Section 3.1 | ✓ Designed |
| 3.5 Credential Resolution | Section 3.3 | ✓ Designed |
| 3.6 Backward Compatibility | Section 5.1 | ✓ Designed |
| 3.7 Worker Unchanged | N/A | ✓ No changes |
| 3.8 Multi-Tenant | Section 3.3 | ✓ Designed |
| 3.9 Error Handling | Section 5 | ✓ Designed |
| 3.10 Affected Components | Section 4 | ✓ Designed |

## 15. Open Questions

1. **Encryption Key Rotation**: How often should keys be rotated? (Recommendation: Quarterly)
2. **Cache TTL**: Should we cache configs in future? (Recommendation: Yes, 5-minute TTL)
3. **Audit Logging**: Should we log all credential access? (Recommendation: Yes, for compliance)
4. **Multi-Schema Support**: Should table be in `lad_dev` and `lad_prod`? (Recommendation: Yes, use schema prefix from env)

## 16. Summary

This design introduces a clean, modular approach to dynamic LiveKit credential management:

- **New Files**: 3 files (~300 lines total)
  - `utils/en_de_crypt.py` (~80 lines)
  - `utils/livekit_resolver.py` (~150 lines)
  - `db/storage/livekit.py` (~120 lines)

- **Modified Files**: 2 files (~15 lines total changes)
  - `api/services/call_service.py` (~10 lines)
  - `utils/call_routing.py` (~5 lines)

- **Database**: 1 new table with trigger

- **Feature Flag**: Safe rollout with instant rollback capability

- **Security**: Encrypted secrets with Fernet, secure key management

- **Performance**: Minimal overhead, no caching needed in v1

- **Testing**: Comprehensive unit, integration, and manual tests

The design prioritizes:
1. **Minimal changes** to existing large files
2. **Safe deployment** with feature flag
3. **Easy rollback** without code changes
4. **Security** with encryption and proper logging
5. **Maintainability** with modular, testable code
