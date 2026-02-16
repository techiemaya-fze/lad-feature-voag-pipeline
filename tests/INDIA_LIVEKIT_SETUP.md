# India LiveKit Configuration Setup

## Summary
Successfully added India LiveKit cloud configuration for tenant `e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5` with phone number `9513456728`.

## Configuration Details

### LiveKit Config
- **Config ID**: `cf7a1d26-04aa-4623-8d4e-f4b079cdec6f`
- **Name**: `india-techiemaya-cloud`
- **URL**: `wss://lk.techiemaya.com`
- **API Key**: `API5QH2NJHDXQSW`
- **API Secret**: `fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN` (encrypted in DB)
- **Trunk ID**: `ST_MmVqEuBMDNf6`
- **Worker Name**: `voag-dev`

### Phone Number Configuration
- **Number**: `+919513456728`
- **Tenant ID**: `e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5`
- **Provider**: TechieMaya FZE
- **Rules**: Contains `livekit_config` UUID reference

## Test Results

### Test Script
`tests/test_india_outbound_call.py`

### Test Configuration
- From: +919513456728
- To: +918384884150
- Agent ID: 33
- Tenant ID: e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5 (hardcoded for testing)

### Test Output
```
✓ Call successfully dispatched to LiveKit
✓ Using India LiveKit server: wss://lk.techiemaya.com
✓ Trunk ID: ST_MmVqEuBMDNf6
✓ Worker Name: voag-dev
✓ Agent ID: 33
✓ Call Log ID: 1369951c-43f9-46f6-8cf0-5a45b44b68ff
```

**Call Dispatched:**
- Room: `call-909c0207582b492ebb27c93bbcf99271-cc8a7514`
- Dispatch ID: `AD_BKjK5mSqVN5e`
- Worker: `voag-dev` (from database config)
- Server: `wss://lk.techiemaya.com` (India cloud)

## Important Notes

### Tenant ID Resolution
The test uses a hardcoded tenant_id (`e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5`) because:
1. Agent 33 belongs to a different tenant (`734cd516-e252-4728-9c52-4663ee552653`)
2. In production, tenant_id is resolved from the `initiated_by` user ID
3. The user lookup happens in `call_service.py`: `tenant_id = await self._user_storage.get_user_tenant_id(initiated_by)`

### Multiple Numbers with Same Base
The base number `9513456728` exists for multiple tenants:
- `926070b5-189b-4682-9279-ea10ca090b84` (Vonage)
- `1ead8e68-2375-43bd-91c9-555df2521dec` (pluto travels usa)
- `05f6d939-05c7-424f-9a83-554d932bf37a` (Paras Organization)
- `734cd516-e252-4728-9c52-4663ee552653` (TechieMaya FZE - Agent 33's tenant)
- `e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5` (TechieMaya FZE - Test tenant) ✓ Has livekit_config

## Database Scripts

### Add/Update Config
```bash
uv run python db/migrations/add_india_livekit_config.py
```

### Update Trunk ID and Worker Name
```bash
uv run python db/migrations/update_india_config.py
```

### Check Configuration
```bash
uv run python db/migrations/check_india_number.py
```

### Run Test
```bash
uv run python tests/test_india_outbound_call.py
```

## Comparison: UAE vs India

| Aspect | UAE VM | India Cloud |
|--------|--------|-------------|
| **URL** | http://91.74.244.94:7880 | wss://lk.techiemaya.com |
| **Type** | Self-hosted VM | Cloud LiveKit |
| **API Key** | APIbe273e3142c7b96a4a87bba4 | API5QH2NJHDXQSW |
| **Trunk ID** | ST_svHE4RdTc7Ds | ST_MmVqEuBMDNf6 |
| **Worker** | voag-staging | voag-dev |
| **Number** | +971545335200 | +919513456728 |
| **Tenant** | 734cd516... | e0a3e9ca... |

## Production Usage

In production, calls from number `+919513456728` for tenant `e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5` will:

1. User initiates call with `initiated_by` user ID
2. System looks up user's tenant_id from `users` table
3. System queries `voice_agent_numbers` with from_number + tenant_id
4. Finds `livekit_config` UUID in rules: `cf7a1d26-04aa-4623-8d4e-f4b079cdec6f`
5. Queries `voice_agent_livekit` table for config
6. Uses India LiveKit credentials:
   - URL: wss://lk.techiemaya.com
   - Trunk: ST_MmVqEuBMDNf6
   - Worker: voag-dev
7. Dispatches call to India cloud server

## Troubleshooting

### Issue: "Using environment credentials (not database)"
**Cause**: Call routing not finding livekit_config_id in rules

**Check**:
```sql
SELECT base_number, tenant_id, rules->'livekit_config' as config_id
FROM lad_dev.voice_agent_numbers
WHERE base_number = '9513456728'
AND tenant_id = 'e0a3e9ca-3f46-4bb0-ac10-a91b5c1d20b5';
```

**Fix**: Ensure rules contain `"livekit_config": "cf7a1d26-04aa-4623-8d4e-f4b079cdec6f"`

### Issue: Wrong tenant_id used
**Cause**: Agent belongs to different tenant than the number

**Solution**: 
- In production: Use `initiated_by` user ID to get correct tenant
- In testing: Hardcode the tenant_id as shown in test script

## Next Steps

1. Test with actual user ID (`initiated_by`) instead of hardcoded tenant
2. Add carrier rules for +919513456728 to enable proper routing
3. Monitor India cloud server for call quality
4. Consider adding more India numbers with this config
5. Document user-to-tenant mapping for testing

## Files Created/Modified

1. `db/migrations/add_india_livekit_config.py` - Add India config
2. `db/migrations/update_india_config.py` - Update trunk/worker
3. `db/migrations/check_india_number.py` - Check number config
4. `tests/test_india_outbound_call.py` - Test script
5. `lad_dev.voice_agent_livekit` - New row added
6. `lad_dev.voice_agent_numbers` - Rules updated with livekit_config UUID
