# Call & Batch Cancellation API Guide

This guide explains how to cancel running calls and batches via the v2 API.

---

## Quick Reference

| Type | Endpoint | ID Required | Example |
|------|----------|-------------|---------|
| Single Call | `POST /calls/cancel` | `call_log_id` (UUID) | `"resource_id": "abc123-def456..."` |
| Batch | `POST /calls/cancel` | `job_id` with `batch-` prefix | `"resource_id": "batch-abc123..."` |
| Batch (alt) | `POST /batch/batch-cancel/{batch_id}` | `job_id` or `batch_id` | `/batch/batch-cancel/batch-abc123` |

---

## 1. Cancel a Single Call

**Endpoint:** `POST /calls/cancel`

**Request:**
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"resource_id": "CALL_LOG_ID_HERE"}'
```

**How to get the call_log_id:**
- Returned from `POST /calls/start-call` response as `results[0].call_log_id`
- From database: `SELECT id FROM call_logs_voiceagent WHERE room_name = 'RM_xxx';`
- From logs: look for `call_log_id` in log messages

**Response (success):**
```json
{
  "resource_id": "call-log-uuid",
  "resource_type": "call",
  "status": "cancelled",
  "cancelled_count": 1,
  "message": "Call cancelled. LiveKit room terminated."
}
```

**What happens:**
1. ✅ LiveKit room is deleted (disconnects all participants immediately)
2. ✅ Call status updated to `cancelled` in DB
3. ✅ If part of batch, batch entry also updated to `cancelled`

---

## 2. Stop a Batch Job

**Endpoint:** `POST /calls/cancel`

**Request (graceful - running calls complete):**
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"resource_id": "batch-JOB_ID_HERE"}'
```

**Request (force - terminate running calls too):**
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"resource_id": "batch-JOB_ID_HERE", "force": true}'
```

**How to get the batch job_id:**
- Returned from `POST /batch/trigger-batch-call` response
- Starts with `batch-` prefix (e.g., `batch-abc123def456`)

**Response (graceful):**
```json
{
  "resource_id": "batch-abc123",
  "resource_type": "batch",
  "status": "stopped",
  "cancelled_count": 5,
  "message": "Batch stopped. 5 pending entries cancelled. 2 active call(s) will complete naturally. Use force=true to terminate them."
}
```

**Response (force):**
```json
{
  "resource_id": "batch-abc123",
  "resource_type": "batch",
  "status": "stopped",
  "cancelled_count": 7,
  "message": "Batch stopped. 5 pending entries cancelled. 2 running call(s) forcefully terminated."
}
```

**What happens (graceful, default):**
1. ✅ Batch status set to `stopped`
2. ✅ All **pending** entries marked as `cancelled`
3. ⏳ Running calls are **allowed to complete** naturally

**What happens (force=true):**
1. ✅ Batch status set to `stopped`
2. ✅ All **pending** entries marked as `cancelled`
3. ✅ All **running** calls have their LiveKit rooms deleted (forcefully terminated)
4. ✅ Running entries marked as `cancelled`

---

## 3. Check Status After Cancellation

**Endpoint:** `GET /calls/status/{resource_id}`

```bash
# For call
curl "https://voag.techiemaya.com/calls/status/CALL_LOG_ID" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY"

# For batch  
curl "https://voag.techiemaya.com/calls/status/batch-JOB_ID" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## Status Values

### Call Status
| Status | Description |
|--------|-------------|
| `ringing` | Call is ringing |
| `running` | Call is active |
| `ended`, `completed` | Normal end |
| `cancelled` | Cancelled via API |
| `failed`, `error` | Call failed |
| `declined`, `rejected`, `busy`, `no_answer`, `not_reachable` | Not answered |

### Batch Status
| Status | Description |
|--------|-------------|
| `processing` | Batch is running |
| `stopped` | Cancelled via API |
| `completed` | All entries done |
| `cancelled` | Alternative cancelled state |

---

## Finding IDs from Logs

**From prod logs:**
```bash
# Find call_log_id for a room
grep "room_name" /var/log/syslog | grep "RM_xxxxx"

# Find batch job_id
grep "batch-" /var/log/syslog | grep "job_id"
```

**From database:**
```sql
-- Find call by room name
SELECT id, room_name, status FROM call_logs_voiceagent 
WHERE room_name = 'RM_xxxxx'
ORDER BY created_at DESC LIMIT 1;

-- Find active batches
SELECT id, job_id, status, total_calls, completed_calls
FROM batch_logs_voiceagent 
WHERE status = 'processing'
ORDER BY created_at DESC;
```

---

## Example: Cancel a Stuck Call

**Given logs:**
```
"job_id": "AJ_A2zYh98RDf5X", "room_id": "RM_iPMjFyQymh2b"
```

**Step 1:** Find call_log_id from database
```sql
SELECT id FROM call_logs_voiceagent 
WHERE room_name = 'RM_iPMjFyQymh2b';
```

**Step 2:** Cancel using call_log_id
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"resource_id": "THE_UUID_FROM_STEP_1"}'
```

---

## API Request Schema

### CancelRequest
```typescript
{
  resource_id: string;  // Call log UUID or batch job_id (with batch- prefix)
  force?: boolean;      // Default: false. If true, forcefully terminate running calls
}
```

### CancelResponse
```typescript
{
  resource_id: string;
  resource_type: "call" | "batch";
  status: string;
  cancelled_count: number;
  message: string;
}
```
