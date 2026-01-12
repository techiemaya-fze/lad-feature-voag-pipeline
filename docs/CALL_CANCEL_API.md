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
- Found in call logs database
- From Sentry/logs: look for `call_log_id` in log messages

**Response (success):**
```json
{
  "resource_id": "call-log-uuid",
  "resource_type": "call",
  "status": "cancelled",
  "cancelled_count": 1,
  "message": "Call cancelled and marked as terminated."
}
```

**What happens:**
1. ✅ Call status updated to `cancelled` in DB
2. ✅ If part of batch, batch entry updated to `cancelled`
3. ⚠️ **Note:** LiveKit room termination is placeholder only (doesn't actually force-disconnect the room - call may continue on network level)

---

## 2. Stop a Batch Job

**Endpoint:** `POST /calls/cancel`

**Request:**
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"resource_id": "batch-JOB_ID_HERE"}'
```

**How to get the batch job_id:**
- Returned from `POST /batch/trigger-batch-call` response
- Starts with `batch-` prefix
- Example: `batch-abc123def456`

**Response:**
```json
{
  "resource_id": "batch-abc123",
  "resource_type": "batch",
  "status": "stopped",
  "cancelled_count": 5,
  "message": "Batch stopped. 5 pending entries cancelled. 2 active call(s) will complete naturally."
}
```

**What happens:**
1. ✅ Batch status set to `stopped`
2. ✅ All **pending** entries marked as `cancelled`
3. ✅ Cancelled count incremented in batch counters
4. ⚠️ **Active calls continue** - running calls are allowed to complete naturally

---

## 3. Alternative: Cancel Batch by Path Parameter

**Endpoint:** `POST /batch/batch-cancel/{batch_id}`

**Request:**
```bash
curl -X POST "https://voag.techiemaya.com/batch/batch-cancel/batch-abc123" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Response:**
```json
{
  "batch_id": "uuid",
  "job_id": "batch-abc123",
  "status": "stopped",
  "cancelled_count": 3,
  "message": "Batch stopped. 3 pending entries cancelled."
}
```

---

## 4. Check Status After Cancellation

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
- `ringing`, `running` - Active call
- `ended`, `completed` - Normal end
- `cancelled` - Cancelled via API
- `failed`, `error` - Call failed
- `declined`, `rejected`, `busy`, `no_answer`, `not_reachable` - Not answered

### Batch Status
- `processing` - Running
- `stopped` - Cancelled via API (pending entries cancelled, active calls complete)
- `completed` - All entries done
- `cancelled` - Alternative cancelled state

---

## Important Notes

### Limitations
1. **LiveKit Room Not Force-Terminated**: Currently, cancelling a call only updates the database status. The actual LiveKit room is NOT deleted. The call may continue if network is still connected.

2. **No Retry Prevention**: There's no explicit retry prevention mechanism. If the worker checks for this call again, it should see `cancelled` status and skip.

### Recommended Usage
1. For stuck calls: Use `/calls/cancel` with the call_log_id
2. For stuck batches: Use `/calls/cancel` with the batch job_id
3. After cancelling, verify with `/calls/status/{id}` that status is `cancelled`/`stopped`

### Finding IDs from Logs

**From prod logs:**
```bash
# Find call_log_id
grep "call_log_id" /var/log/syslog | grep "room_id"

# Find batch job_id
grep "batch-" /var/log/syslog | grep "job_id"
```

**From the log message you provided:**
```
"job_id": "AJ_A2zYh98RDf5X", "room_id": "RM_iPMjFyQymh2b"
```
- This is a **LiveKit job_id/room_id**, NOT the call_log_id
- Need to find call_log_id from the database or earlier logs

---

## Example: Cancel Your Stuck Call

Based on your logs:
```
"job_id": "AJ_A2zYh98RDf5X"
"room_id": "RM_iPMjFyQymh2b"
```

**Step 1:** Find call_log_id from database
```sql
SELECT id, room_name, status FROM call_logs_voiceagent 
WHERE room_name = 'RM_iPMjFyQymh2b'
ORDER BY created_at DESC LIMIT 1;
```

**Step 2:** Cancel using call_log_id
```bash
curl -X POST "https://voag.techiemaya.com/calls/cancel" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: console" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"resource_id": "THE_UUID_FROM_STEP_1"}'
```
