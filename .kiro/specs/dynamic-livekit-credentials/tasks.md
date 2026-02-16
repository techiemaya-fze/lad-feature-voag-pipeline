# Implementation Tasks: Dynamic LiveKit Credentials

## Phase 1: Database & Infrastructure Setup

### 1. Create Database Migration
- [x] 1.1 Create SQL migration file for `voice_agent_livekit` table
  - Create table with all columns (id, name, description, livekit_url, livekit_api_key, livekit_api_secret, trunk_id, created_at, updated_at)
  - Add UUID primary key with auto-generation
  - Add UNIQUE constraint on name
  - Add index on name for faster lookups
  - Create trigger function for auto-updating updated_at
  - Create trigger on UPDATE
  - Test migration on dev database

### 2. Update Environment Configuration
- [x] 2.1 Add new environment variables to `.env.example`
  - Add `USE_SELFHOST_ROUTING_TABLE=true` with comment
  - Add `LIVEKIT_SECRET_ENCRYPTION_KEY=` with generation instructions
  - Document key generation command in comments

## Phase 2: Core Utilities Implementation

### 3. Implement Encryption Utility
- [x] 3.1 Create `utils/en_de_crypt.py`
  - Import cryptography.fernet
  - Implement `encrypt_decrypt(input_string: str) -> str` function
  - Check for "dev-s-t-" prefix to determine encrypt vs decrypt
  - Read encryption key from `LIVEKIT_SECRET_ENCRYPTION_KEY` env var
  - Handle missing encryption key with clear error message
  - Add CLI interface using `if __name__ == "__main__"`
  - Add docstrings and type hints
  
- [x] 3.2 Write unit tests for encryption utility
  - Test encryption produces "dev-s-t-" prefix
  - Test decryption returns original value
  - Test round-trip (encrypt → decrypt → original)
  - Test invalid encrypted string handling
  - Test missing encryption key error
  - Test CLI interface

### 4. Implement LiveKit Config Storage
- [x] 4.1 Create `db/storage/livekit.py`
  - Create `LiveKitConfigStorage` class
  - Implement `get_livekit_config(config_id: str) -> dict | None`
  - Implement `create_livekit_config(...)` with all parameters
  - Implement `update_livekit_config(config_id: str, **updates) -> bool`
  - Implement `delete_livekit_config(config_id: str) -> bool`
  - Implement `list_livekit_configs() -> list[dict]`
  - Use async/await pattern
  - Use connection pool from `db/connection_pool.py`
  - Use RealDictCursor for dict results
  - Add proper error handling and logging
  
- [x] 4.2 Write unit tests for LiveKit storage
  - Test get_livekit_config with valid UUID
  - Test get_livekit_config with invalid UUID (returns None)
  - Test create_livekit_config returns UUID
  - Test update_livekit_config updates fields
  - Test updated_at auto-updates on modification
  - Test delete_livekit_config
  - Test list_livekit_configs
  - Test unique name constraint

### 5. Implement LiveKit Credential Resolver
- [x] 5.1 Create `utils/livekit_resolver.py`
  - Create `LiveKitCredentials` dataclass (url, api_key, api_secret, trunk_id, source)
  - Implement `resolve_livekit_credentials(from_number, tenant_id, routing_result) -> LiveKitCredentials`
  - Implement `_get_credentials_from_env(routing_result) -> LiveKitCredentials`
  - Implement `_get_credentials_from_database(from_number, tenant_id) -> LiveKitCredentials | None`
  - Check `USE_SELFHOST_ROUTING_TABLE` feature flag first
  - If flag is false, return env credentials immediately
  - If flag is true, try database lookup
  - Query `voice_agent_numbers` for `livekit_config` UUID
  - If UUID exists, query `voice_agent_livekit` table
  - Decrypt `livekit_api_secret` using `encrypt_decrypt()`
  - Implement comprehensive fallback logic (catch all exceptions)
  - Add logging for credential source (database vs environment)
  - Never log decrypted secrets
  - Add proper error handling with fallback to env vars
  
- [x] 5.2 Write unit tests for credential resolver
  - Test feature flag disabled → env vars only
  - Test feature flag enabled with valid UUID → database credentials
  - Test feature flag enabled with no UUID → env vars fallback
  - Test feature flag enabled with invalid UUID → env vars fallback
  - Test decryption failure → env vars fallback
  - Test database connection error → env vars fallback
  - Test missing env vars → RuntimeError
  - Test trunk_id precedence (config > rules > env)
  - Mock database calls and encryption

## Phase 3: Integration with Existing Code

### 6. Update Call Routing Module
- [x] 6.1 Modify `utils/call_routing.py`
  - Add `livekit_config_id: Optional[str] = None` to `CallRoutingResult` dataclass
  - In `validate_and_format_call()`, extract `livekit_config` from rules: `livekit_config_id = rules.get('livekit_config')`
  - Add `livekit_config_id=livekit_config_id` to all `CallRoutingResult` return statements
  - Verify no breaking changes to existing functionality
  
- [x] 6.2 Write tests for call routing changes
  - Test CallRoutingResult includes livekit_config_id
  - Test livekit_config_id extracted from rules
  - Test backward compatibility (rules without livekit_config)

### 7. Update Call Service Module
- [x] 7.1 Modify `api/services/call_service.py`
  - Import `resolve_livekit_credentials` from `utils.livekit_resolver`
  - Replace `url, api_key, api_secret = _validate_livekit_credentials()` with credential resolver call
  - Call `livekit_creds = await resolve_livekit_credentials(from_number, tenant_id, routing_result)`
  - Extract `url = livekit_creds.url`, `api_key = livekit_creds.api_key`, `api_secret = livekit_creds.api_secret`
  - Update trunk_id resolution: `outbound_trunk = livekit_creds.trunk_id or routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")`
  - Verify minimal changes (target ~10 lines modified)
  - Keep existing error handling intact
  
- [x] 7.2 Write integration tests for call dispatch
  - Test single call dispatch with database credentials
  - Test single call dispatch with env credentials (fallback)
  - Test batch call dispatch with database credentials
  - Test feature flag disabled → env vars only
  - Test mixed scenario (some numbers with DB config, some without)

## Phase 4: Testing & Validation

### 8. Integration Testing
- [x] 8.1 Create integration test suite
  - Test end-to-end call dispatch with database credentials
  - Test end-to-end call dispatch with feature flag disabled
  - Test credential fallback scenarios
  - Test multi-tenant isolation
  - Test encryption/decryption in full flow
  
- [x] 8.2 Manual testing checklist
  - Deploy to dev environment with `USE_SELFHOST_ROUTING_TABLE=false`
  - Verify existing calls work (no regression)
  - Create test LiveKit config in database
  - Encrypt test secret using CLI tool
  - Update test phone number with `livekit_config` UUID
  - Enable feature flag: `USE_SELFHOST_ROUTING_TABLE=true`
  - Make test call and verify database credentials used
  - Check logs for "Using LiveKit credentials from database"
  - Test fallback by removing UUID from rules
  - Test decryption failure by corrupting encrypted secret
  - Verify `updated_at` auto-updates on row modification

### 9. Security Validation
- [x] 9.1 Security review
  - Verify encryption key not logged
  - Verify decrypted secrets not logged
  - Verify encryption key not in version control
  - Test key rotation process
  - Review database access permissions
  - Verify encrypted strings have "dev-s-t-" prefix

## Phase 5: Documentation & Deployment

### 10. Documentation
- [ ] 10.1 Create developer documentation
  - Document how to add new LiveKit server configuration
  - Document CLI encryption tool usage with examples
  - Document credential resolution flow
  - Document troubleshooting guide for fallback scenarios
  - Document feature flag usage
  
- [ ] 10.2 Create operations documentation
  - Document deployment checklist
  - Document rollback procedures
  - Document encryption key generation and rotation
  - Document monitoring and alerting setup
  - Update architecture diagrams

### 11. Deployment Preparation
- [ ] 11.1 Prepare deployment scripts
  - Create database migration script
  - Create encryption key generation script
  - Create test data insertion script
  - Create rollback script
  
- [ ] 11.2 Deployment checklist
  - Run database migration on staging
  - Generate and set `LIVEKIT_SECRET_ENCRYPTION_KEY`
  - Deploy code with `USE_SELFHOST_ROUTING_TABLE=false`
  - Verify no regression in staging
  - Create test LiveKit config
  - Enable feature flag in staging
  - Test calls in staging
  - Monitor logs and metrics
  - Deploy to production with feature flag disabled
  - Gradually enable feature flag in production

## Phase 6: Monitoring & Rollout

### 12. Monitoring Setup
- [ ] 12.1 Add monitoring and metrics
  - Add metric for credential source (database vs environment)
  - Add metric for fallback rate
  - Add metric for decryption errors
  - Add metric for database query latency
  - Set up alerts for high fallback rate
  - Set up alerts for decryption failures
  
- [ ] 12.2 Gradual rollout
  - Migrate 10% of phone numbers to database configs
  - Monitor for 24 hours
  - Migrate 50% of phone numbers
  - Monitor for 48 hours
  - Migrate remaining phone numbers
  - Document rollout results

## Optional Enhancements (Future)

### 13. Admin UI (Optional)
- [ ]* 13.1 Create web interface for managing LiveKit configs
  - CRUD operations for LiveKit configurations
  - Credential validation before save
  - Audit log for config changes

### 14. Caching Layer (Optional)
- [ ]* 14.1 Implement in-memory cache for LiveKit configs
  - Cache configs with 5-minute TTL
  - Invalidate cache on updates
  - Measure performance improvement

### 15. Credential Rotation (Optional)
- [ ]* 15.1 Implement automated credential rotation
  - Scheduled key rotation
  - Graceful credential updates
  - Zero-downtime rotation process

---

## Task Execution Notes

- Tasks marked with `*` are optional and can be skipped
- Execute tasks in order (dependencies exist between phases)
- Run tests after each implementation task
- Update documentation as you implement
- Use feature flag for safe deployment
- Keep changes minimal in large files (800+ lines)
- All new code should follow existing patterns in the codebase
- Use async/await consistently
- Add proper type hints and docstrings
- Log appropriately (never log secrets)
