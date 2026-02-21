Now I have precise line numbers for everything. Here is the full audit:

---

## Batch Status Transition Audit — Bug Report

### BUG 1 (CRITICAL): `handle_wave_timeout()` uses wrong column name — entire timeout handler is broken

**File:** [batches.py](vonage-agent/db/storage/batches.py#L1041), [L1168](vonage-agent/db/storage/batches.py#L1168), [L1185](vonage-agent/db/storage/batches.py#L1185), [L1217](vonage-agent/db/storage/batches.py#L1217), also [L748](vonage-agent/db/storage/batches.py#L748)

**Bug:** Multiple SQL statements reference `error_message` as a column name, but the actual column is `last_error`. Evidence:
- `create_batch_entry` uses `last_error` at [L206](vonage-agent/db/storage/batches.py#L206)
- `update_batch_entry_status` uses `last_error` at [L309](vonage-agent/db/storage/batches.py#L309)
- `get_batch_entries` aliases `e.last_error as error_message` at [L544](vonage-agent/db/storage/batches.py#L544)
- `recover_stale_batches` uses `last_error` at [L1297](vonage-agent/db/storage/batches.py#L1297)

But `handle_wave_timeout` at lines 1041, 1168, 1185, 1217 and `mark_pending_entries_failed` at L748 all write to `error_message`.

**Impact:** PostgreSQL raises `UndefinedColumn` error. Because `handle_wave_timeout` runs steps 1–4 in a single transaction, *step 1's reset of dispatched→queued is also rolled back*. The entire timeout mechanism is dead code. **Entries stuck in `dispatched` state after wave timeout will never be retried or failed.** The wave wait loop spins until `BATCH_WAVE_TIMEOUT` then returns with empty results, and the batch moves to the next wave leaving orphaned entries behind.

**Fix:** Change all `error_message =` to `last_error =` in `handle_wave_timeout()` (4 occurrences) and `mark_pending_entries_failed()` (1 occurrence).

---

### BUG 2 (CRITICAL): `calls.py` cancel sets batch status to `"stopped"` — invalid ENUM value, doesn't actually stop the batch

**File:** [calls.py L317](vonage-agent/api/routes/calls.py#L317)

**Bug:** `await batch_storage.update_batch_status(batch_id, "stopped")` — the `voice_call_batches.status` ENUM only allows `queued, running, completed, failed, cancelled`. `"stopped"` is not a valid value.

Additionally, `is_batch_stopped()` in [batches.py](vonage-agent/db/storage/batches.py) checks `result[0] in ("cancelled", "failed")` — it doesn't check for `"stopped"`.

And the terminal-state guard at [calls.py L307](vonage-agent/api/routes/calls.py#L307) checks `("stopped", "completed", "cancelled")` — missing `"failed"`.

**Impact:**
1. If the DB strictly enforces the ENUM, the UPDATE silently fails or throws, and the batch status is unchanged — cancel has no effect.
2. Even if the UPDATE succeeds (VARCHAR or loose ENUM), `is_batch_stopped()` returns `False`, so `_process_batch()` continues dispatching waves as if nothing happened.
3. A "failed" batch is not treated as terminal by the calls.py cancel endpoint.

**Fix:** Change `"stopped"` to `"cancelled"` at L317. Update the terminal-state check at L307 to `("completed", "cancelled", "failed")`.

---

### BUG 3 (CRITICAL): `calls.py` passes wrong arguments to `update_batch_entry_status()`

**File:** [calls.py L345-346](vonage-agent/api/routes/calls.py#L345) and [L412-414](vonage-agent/api/routes/calls.py#L412)

**Bug:** Both cancel paths call:
```python
await batch_storage.update_batch_entry_status(
    batch_id, entry["entry_index"], "cancelled"
)
```
But `update_batch_entry_status` signature is `(self, entry_id, status, error_message=None)`. So this passes `batch_id` as `entry_id`, `entry["entry_index"]` (an int) as `status`, and `"cancelled"` as `error_message`.

**Impact:** The UPDATE targets a row with `id = <batch_id>` (wrong table/row). If it matches nothing, the cancel silently fails and the entry status is never updated. If it somehow matches, it writes an integer as the status value, corrupting data. Individual call cancellation within a batch does not update the batch entry.

**Fix — batch force-cancel (L345):** Use `str(entry["id"])` as entry_id and `"cancelled"` as status:
```python
await batch_storage.update_batch_entry_status(str(entry["id"]), "cancelled")
```
**Fix — individual call cancel (L412):** Same pattern:
```python
await batch_storage.update_batch_entry_status(str(entry["id"]), "cancelled")
```

---

### BUG 4 (HIGH): `cleanup_and_save()` always reports `"ended"` to batch callback regardless of actual call outcome

**File:** [cleanup_handler.py L816](vonage-agent/agent/cleanup_handler.py#L816)

**Bug:** `await update_batch_on_call_complete(ctx, "ended")` is hardcoded. Even when the call failed (SIP TwirpError), was declined, or was cancelled, the batch callback receives `call_status="ended"`.

In `entry_completed_callback` at [batch.py](vonage-agent/api/routes/batch.py), the mapping is:
```python
entry_status = "completed" if request.call_status in ("ended", "completed") else "failed"
```

So a failed/declined/cancelled call gets `entry_status = "completed"` and increments `completed_calls` instead of `failed_calls`.

**Impact:** Batch statistics are wrong — failed calls are counted as completed. Batch reports will show inflated success rates.

**Fix:** In `cleanup_and_save()`, after `update_call_status()` resolves the final status, pass that actual status to the batch callback:
```python
# After update_call_status:
final_status = determine_final_status(existing_status)
# ...
await update_batch_on_call_complete(ctx, final_status)
```

---

### BUG 5 (HIGH): Double semaphore release in worker.py

**File:** [worker.py L1308](vonage-agent/agent/worker.py#L1308) (finally block) and [cleanup_handler.py](vonage-agent/agent/cleanup_handler.py) L810-812 (cleanup_and_save)

**Bug:** The semaphore is released in two places:
1. The `finally` block at L1306-1308 runs when `entrypoint()` returns (after greeting is sent, **while call is still active**)
2. `cleanup_and_save()` releases it again when the call actually ends

**Impact:** The asyncio Semaphore counter drifts upward. After a few calls, `_MAX_CONCURRENT_CALLS` is no longer enforced and the worker accepts unlimited concurrent calls, potentially overloading the server.

**Fix:** Remove the `finally` block at L1306-1308. The semaphore should only be released in `cleanup_and_save()`. For the edge case where the shutdown callback isn't registered (exception during early setup), add a try/except around the setup code that releases the semaphore before re-raising.

---

### BUG 6 (HIGH): Batch `failed_calls` counter not incremented for timeout-failed entries

**File:** [batch.py](vonage-agent/api/routes/batch.py) — `_process_batch()`, after `_wait_for_wave_completion()` returns

**Bug:** When `handle_wave_timeout()` marks entries as failed (`failed_max_retries`, `failed_ringing`, `failed_ongoing_stuck`), it updates the entry rows in the DB. But back in `_process_batch()`, only the `reset_to_queued` count is checked for retry logic. The failed entry counts are never used to increment `batch.failed_calls`.

**Impact:** `voice_call_batches.failed_calls` is undercounted. `completed_calls + failed_calls < total_calls` even when the batch is done. This doesn't prevent batch completion (which counts entry statuses), but the batch statistics shown to users via the API are wrong.

**Fix:** After `_wait_for_wave_completion()`, add:
```python
if timeout_results:
    failed_count = (
        timeout_results.get("failed_max_retries", 0)
        + timeout_results.get("failed_ringing", 0)
        + timeout_results.get("failed_ongoing_stuck", 0)
    )
    if failed_count > 0:
        await batch_storage.increment_batch_counters(batch_id, failed_delta=failed_count)
```

---

### BUG 7 (MEDIUM): `handle_wave_timeout()` checks for entry statuses `'ringing'` and `'ongoing'` that are never set on entries

**File:** [batches.py L1185](vonage-agent/db/storage/batches.py#L1185) and [L1199](vonage-agent/db/storage/batches.py#L1199)

**Bug:** The entry status lifecycle is: `queued → running → dispatched → completed/failed/cancelled`. The statuses `ringing` and `ongoing` belong to `voice_call_logs`, not `voice_call_batch_entries`. Steps 3 and 4 of `handle_wave_timeout()` will never match any rows.

**Impact:** Dead code. Entries whose calls are in ringing/ongoing state at the call_log level still have entry status `dispatched` — step 1 or 2 handles them. So this is a logical error but doesn't cause stuck entries **as long as Bug #1 is fixed** (currently step 2+ never executes due to the wrong column name).

**Fix:** Remove the ringing/ongoing checks, or if you want to handle them, join against `voice_call_logs` to check the *call's* status rather than the entry's.

---

### BUG 8 (MEDIUM): `_wait_for_wave_completion()` doesn't check `is_batch_stopped()`

**File:** [batch.py](vonage-agent/api/routes/batch.py) — `_wait_for_wave_completion()` function

**Bug:** The wave poll loop only checks `count_pending_entries()` and timeout. After a batch cancel, dispatched entries in the current wave that are waiting on workers will keep the loop spinning for up to `BATCH_WAVE_TIMEOUT` (20 minutes) before the timeout handler kicks in.

**Impact:** Cancelling a batch doesn't take effect promptly if a wave is in progress. The `_process_batch` task holds resources for up to 20 minutes unnecessarily.

**Fix:** Add a `is_batch_stopped()` check inside the while-loop, and break out early if cancelled:
```python
if await batch_storage.is_batch_stopped(batch_id):
    return {"completed": False, "timeout_results": None, "cancelled": True}
```

---

### BUG 9 (MEDIUM): `get_pending_entries()` filters on `'pending'` status which is never used

**File:** [batches.py L584](vonage-agent/db/storage/batches.py#L584)

**Bug:** `WHERE ... status IN ('pending', 'running')` — entries are created as `'queued'`, never `'pending'`. This method would miss all queued entries.

**Impact:** If this method is called anywhere (not currently used in the audited paths), it returns incomplete results.

**Fix:** Change to `status IN ('queued', 'running', 'dispatched')`.

---

### BUG 10 (MEDIUM): Batch status API counts `"pending"` entries but entries use `"queued"`

**File:** [batch.py L609](vonage-agent/api/routes/batch.py#L609) and [calls.py](vonage-agent/api/routes/calls.py) (same pattern in `get_call_or_batch_status`)

**Bug:**
```python
pending = sum(1 for e in entries if e["status"] == "pending")
running = sum(1 for e in entries if e["status"] == "running")
```
Entries start as `"queued"`, not `"pending"`. And after dispatch, entries are `"dispatched"`, not `"running"`. Both counts will always be 0.

**Impact:** The API always reports `pending_calls=0` and `running_calls=0` regardless of actual state. Misleading dashboard data.

**Fix:** Change to:
```python
pending = sum(1 for e in entries if e["status"] in ("queued", "dispatched"))
running = sum(1 for e in entries if e["status"] == "running")
```

---

### BUG 11 (LOW): `update_batch_entry_status()` has no guard against overwriting terminal status

**File:** [batches.py L305-313](vonage-agent/db/storage/batches.py#L305)

**Bug:** The UPDATE is unconditional: `UPDATE ... SET status = %s WHERE id = %s`. If a race occurs (e.g., worker callback arrives after timeout handler already failed the entry), the entry status can flip from `failed` back to `completed`.

**Impact:** Minor race condition. With Bug #1 fixed (correct column names), the timeout handler properly fails entries. If the worker callback arrives late, it could overwrite that to `completed`, leading to double-counted entries (both `failed_delta` from timeout and `completed_delta` from callback are incremented).

**Fix:** Add a guard: `WHERE id = %s AND status NOT IN ('completed', 'failed', 'cancelled', 'declined', 'ended')`.

---

### BUG 12 (LOW): `calls.py` cancel path doesn't trigger batch completion check or report

**File:** [calls.py](vonage-agent/api/routes/calls.py) cancel handler (~L314-370)

**Bug:** Unlike `batch.py`'s `cancel_batch()` which calls `check_and_complete_batch()` and triggers report generation, the `calls.py` cancel path just sets the status and cancels entries without checking if the batch is now fully done.

**Impact:** If a batch is cancelled via the `/calls/cancel` endpoint instead of `/batch/batch-cancel`, no completion report is generated even when all entries are terminal.

**Fix:** Add `check_and_complete_batch()` call and report trigger after cancellation, matching the logic in `batch.py cancel_batch()`.

---

### Summary Table

| # | Severity | File | Bug | Impact |
|---|----------|------|-----|--------|
| 1 | **CRITICAL** | batches.py | `error_message` vs `last_error` column | Wave timeout handler is 100% broken; entries stuck in `dispatched` forever |
| 2 | **CRITICAL** | calls.py L317 | Status `"stopped"` not in ENUM, not checked by `is_batch_stopped()` | Cancel via calls.py doesn't stop the batch |
| 3 | **CRITICAL** | calls.py L345, L412 | Wrong args to `update_batch_entry_status()` | Entry cancel is a silent no-op or data corruption |
| 4 | **HIGH** | cleanup_handler.py L816 | Hardcoded `"ended"` to batch callback | Failed/declined calls counted as completed |
| 5 | **HIGH** | worker.py L1308 | Double semaphore release | Concurrent call limit not enforced |
| 6 | **HIGH** | batch.py | Timeout-failed entries not counted | `failed_calls` counter undercount |
| 7 | **MEDIUM** | batches.py | `ringing`/`ongoing` never set on entries | Dead code (masked by Bug #1) |
| 8 | **MEDIUM** | batch.py | Wave loop doesn't check cancel | 20-min delay on cancel |
| 9 | **MEDIUM** | batches.py L584 | Filters on `'pending'` not `'queued'` | Wrong results if called |
| 10 | **MEDIUM** | batch.py L609 / calls.py | Counts `"pending"` not `"queued"` | API always shows 0 pending/running |
| 11 | **LOW** | batches.py L305 | No terminal-status guard on UPDATE | Race condition on entry status |
| 12 | **LOW** | calls.py | No completion check after cancel | Missing batch report |