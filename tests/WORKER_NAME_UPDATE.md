# Worker Name Feature Update

## Summary
Added `worker_name` column to the `voice_agent_livekit` table to support dynamic worker/agent name resolution based on LiveKit configuration.

## Changes Made

### 1. Database Schema Update
- Added `worker_name VARCHAR(255)` column to `lad_dev.voice_agent_livekit` table
- Column is nullable (optional) - falls back to environment variable if not set
- Updated migration script to handle both new tables and existing tables

### 2. Code Updates

#### `utils/livekit_resolver.py`
- Updated `LiveKitCredentials` dataclass to include `worker_name` field
- Modified `_get_credentials_from_env()` to include worker_name from `VOICE_AGENT_NAME` env var
- Modified `_get_credentials_from_database()` to read `worker_name` from config with fallback to env
- Priority: config.worker_name > VOICE_AGENT_NAME env var > "inbound-agent" (default)

#### `api/services/call_service.py`
- Updated agent dispatch to use `livekit_creds.worker_name` instead of always reading from environment
- Maintains fallback to `VOICE_AGENT_NAME` env var if worker_name is None
- Line changed: `agent_name = livekit_creds.worker_name or os.getenv("VOICE_AGENT_NAME", "inbound-agent")`

### 3. Database Data Update
- Updated UAE VM LiveKit config with `worker_name = 'voag-staging'`
- Config ID: `e3ca4a84-4cc3-4be5-8240-34366fe4d0c5`
- Config Name: `uae-vm-selfhosted`

### 4. Test Updates
- Updated `tests/test_real_outbound_call.py` to display and verify worker_name
- Added verification that worker_name matches expected value ('voag-staging')

## Test Results

```
6. Resolving LiveKit Credentials...
   ✓ LiveKit credentials resolved:
     - Source: database
     - URL: http://91.74.244.94:7880
     - API Key: APIbe273e3142c7b96a4a87bba4
     - API Secret: SEC43172b2431a470ae0... (decrypted)
     - Trunk ID: ST_svHE4RdTc7Ds
     - Worker Name: voag-staging  ← NEW
   ✓ Using UAE VM URL (correct)
   ✓ Using UAE VM API key (correct)
   ✓ Using UAE VM trunk ID (correct)
   ✓ Using UAE VM worker name (correct)  ← NEW

8. Test Summary:
   ✓ Call successfully dispatched to LiveKit
   ✓ Using UAE VM server: http://91.74.244.94:7880
   ✓ Trunk ID: ST_svHE4RdTc7Ds
   ✓ Worker Name: voag-staging  ← NEW
   ✓ Agent ID: 33
   ✓ Call Log ID: 985e87fc-bc83-4ec9-9b67-fc5970ac6f0a
```

## Feature Behavior

### With Feature Flag Enabled (`USE_SELFHOST_ROUTING_TABLE=true`)
1. System queries `voice_agent_livekit` table using `livekit_config` UUID from phone number rules
2. If `worker_name` is set in the config, uses that value
3. If `worker_name` is NULL, falls back to `VOICE_AGENT_NAME` environment variable
4. If environment variable not set, uses default "inbound-agent"

### With Feature Flag Disabled (`USE_SELFHOST_ROUTING_TABLE=false`)
- System uses `VOICE_AGENT_NAME` environment variable (existing behavior)
- Falls back to "inbound-agent" if not set

## Migration Scripts

### Add Column (Idempotent)
```bash
uv run python db/migrations/create_voice_agent_livekit_table.py
```

### Update Existing Config
```bash
uv run python db/migrations/update_worker_name.py
```

### Verify Configs
```bash
uv run python db/migrations/check_livekit_configs.py
```

## Use Cases

### Different Workers for Different Regions
```sql
-- UAE VM with staging worker
UPDATE lad_dev.voice_agent_livekit 
SET worker_name = 'voag-staging' 
WHERE name = 'uae-vm-selfhosted';

-- India VM with production worker
UPDATE lad_dev.voice_agent_livekit 
SET worker_name = 'voag-production' 
WHERE name = 'india-vm-selfhosted';
```

### Testing with Different Worker Versions
```sql
-- Test worker for development
UPDATE lad_dev.voice_agent_livekit 
SET worker_name = 'voag-dev-v2' 
WHERE name = 'test-livekit-config';
```

## Backward Compatibility

- Existing configs with NULL `worker_name` continue to work (falls back to env var)
- Feature flag disabled: System behaves exactly as before
- No breaking changes to existing functionality
- Worker code unchanged (still reads from environment)

## Security Considerations

- `worker_name` is not encrypted (it's not sensitive data)
- Worker name is logged for debugging purposes
- No PII or credentials in worker name

## Next Steps

1. Update other LiveKit configs with appropriate worker names
2. Document worker naming conventions
3. Consider adding worker version tracking
4. Monitor worker dispatch logs to verify correct worker selection

## Files Modified

1. `db/migrations/create_voice_agent_livekit_table.py` - Added worker_name column
2. `utils/livekit_resolver.py` - Added worker_name to credentials resolution
3. `api/services/call_service.py` - Use worker_name from credentials
4. `tests/test_real_outbound_call.py` - Display and verify worker_name
5. `db/migrations/update_worker_name.py` - Script to update worker_name (new)
6. `db/migrations/check_livekit_configs.py` - Script to check configs (new)

## Deployment Notes

- Migration is idempotent (safe to run multiple times)
- No downtime required
- Can be deployed with feature flag disabled for safety
- Gradual rollout: Update worker_name for configs one at a time
