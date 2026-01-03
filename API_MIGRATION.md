# API Endpoint Migration Guide (Old → V2)

> **For Frontend Developers** - Complete reference for migrating from legacy API to V2.

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Single Call API](#1-single-call-api)
3. [Batch Call API](#2-batch-call-api)
4. [Call Status API](#3-call-status-api)
5. [Agents API](#4-agents-api)
6. [OAuth (Google)](#5-oauth---google)
7. [Recordings API](#6-recordings-api)
8. [Knowledge Base API](#7-knowledge-base-api)
9. [Breaking Changes Summary](#breaking-changes-summary)

---

## Quick Reference

| Category | Old Prefix | V2 Prefix | Breaking Changes |
|----------|-----------|-----------|------------------|
| Calls | `/calls` | `/calls` | Path renamed |
| Batch | `/` | `/batch` | Added prefix |
| Agents | `/` | `/agents` | Added prefix |
| Auth | `/auth` | `/auth` | Minor path change |
| Recordings | `/recordings` | `/recordings` | ✅ No changes |
| KB | `/knowledge-base` | `/knowledge-base` | ✅ No changes |

---

## 1. Single Call API

### Endpoint Change
| Action | Old | V2 |
|--------|-----|-----|
| **Trigger Call** | `POST /calls` | `POST /calls/start-call` |
| **Get Job** | `GET /calls/{job_id}` | `GET /calls/job/{job_id}` |

### Request Payload: `SingleCallPayload`

#### Old (main.py)
```json
{
  "voice_id": "string, required",
  "to_number": "string, required (E.164)",
  "from_number": "string | null",
  "added_context": "string | null",
  "llm_provider": "string | null",
  "llm_model": "string | null",
  "initiated_by": "int | null",      // ⚠️ INTEGER
  "agent_id": "int | null",
  "lead_name": "string | null",
  "lead_id": "int | null",           // ⚠️ INTEGER
  "knowledge_base_store_ids": ["string"] | null
}
```

#### V2 (api/models.py)
```json
{
  "voice_id": "string, required",
  "to_number": "string, required (E.164)",
  "from_number": "string | null",
  "added_context": "string | null",
  "llm_provider": "string | null",
  "llm_model": "string | null",
  "initiated_by": "string | null",   // ✅ UUID STRING
  "agent_id": "int | null",
  "lead_name": "string | null",
  "lead_id": "string | null",        // ✅ UUID STRING
  "knowledge_base_store_ids": ["string"] | null
}
```

#### ⚠️ Breaking Changes
| Field | Old Type | New Type | Migration |
|-------|----------|----------|-----------|
| `initiated_by` | `int` | `string` (UUID) | Convert user ID to UUID |
| `lead_id` | `int` | `string` (UUID) | Convert lead ID to UUID |

#### Example V2 Request
```json
{
  "voice_id": "voice-abc123",
  "to_number": "+919876543210",
  "from_number": "+18001234567",
  "added_context": "Customer inquired about premium plan",
  "agent_id": 5,
  "initiated_by": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "lead_name": "John Doe",
  "lead_id": "f1e2d3c4-b5a6-7890-1234-567890abcdef",
  "knowledge_base_store_ids": ["fileSearchStores/store-id-123"]
}
```

---

## 2. Batch Call API

### Endpoint Change
| Action | Old | V2 |
|--------|-----|-----|
| **Trigger Batch** | `POST /trigger-batch-call` | `POST /batch/trigger-batch-call` |
| **Get Status** | `GET /batch-status/{id}` | `GET /batch/batch-status/{id}` |
| **Cancel Batch** | `POST /batch-stop/{id}` | `POST /batch/batch-cancel/{id}` |

### Request Payload: `BatchCallJsonRequest`

#### Old
```json
{
  "voice_id": "string, required",
  "from_number": "string | null",
  "added_context": "string | null",
  "initiated_by": "int | null",       // ⚠️ INTEGER
  "agent_id": "int | null",
  "llm_provider": "string | null",
  "llm_model": "string | null",
  "knowledge_base_store_ids": ["string"] | null,
  "entries": [BatchCallJsonEntry]     // required
}
```

#### V2
```json
{
  "voice_id": "string, required",
  "from_number": "string | null",
  "added_context": "string | null",
  "initiated_by": "string | null",    // ✅ UUID STRING
  "agent_id": "int | null",
  "llm_provider": "string | null",
  "llm_model": "string | null",
  "knowledge_base_store_ids": ["string"] | null,
  "entries": [BatchCallJsonEntry]     // required
}
```

### Entry Payload: `BatchCallJsonEntry`

#### Old
```json
{
  "to_number": "string, required (E.164)",
  "lead_name": "string | null",
  "added_context": "string | null",
  "lead_id": "int | null"            // ⚠️ INTEGER (if present)
}
```

#### V2
```json
{
  "to_number": "string, required (E.164)",
  "lead_name": "string | null",
  "added_context": "string | null",
  "lead_id": "string | null",        // ✅ UUID STRING
  "knowledge_base_store_ids": ["string"] | null  // ✅ NEW
}
```

#### ⚠️ Breaking Changes
| Field | Old Type | New Type | Notes |
|-------|----------|----------|-------|
| `initiated_by` | `int` | `string` (UUID) | Top-level |
| `lead_id` | `int` | `string` (UUID) | Per-entry |
| `knowledge_base_store_ids` | N/A | `list[str]` | Per-entry (NEW) |

#### Example V2 Request
```json
{
  "voice_id": "voice-xyz789",
  "from_number": "+18001234567",
  "agent_id": 10,
  "initiated_by": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "entries": [
    {
      "to_number": "+919876543210",
      "lead_name": "Alice Smith",
      "lead_id": "11111111-2222-3333-4444-555555555555"
    },
    {
      "to_number": "+919876543211",
      "lead_name": "Bob Jones",
      "added_context": "Follow-up from last week"
    }
  ]
}
```

---

## 3. Call Status API

### Endpoint Change
| Action | Old | V2 |
|--------|-----|-----|
| **Get Status** | `GET /calls/status/{resource_id}` | `GET /calls/status/{resource_id}` ✅ |
| **Cancel** | `POST /calls/cancel` | `POST /calls/cancel` ✅ |

### Response: `CallStatusResponse`
```json
{
  "call_log_id": "uuid-string",
  "status": "ringing|in_progress|completed|failed|cancelled",
  "call_duration": 123.45,
  "call_recording_url": "gs://bucket/path/file.ogg",
  "transcriptions": {...},
  "started_at": "2024-12-30T10:00:00Z",
  "ended_at": "2024-12-30T10:05:00Z",
  "batch_id": "uuid-string | null",
  "is_batch_call": false
}
```

### Cancel Request: `CancelRequest`
```json
{
  "resource_id": "call-uuid or batch-job-id"
}
```

**No changes** - request/response identical.

---

## 4. Agents API (beta not tested)

### Endpoint Change
| Action | Old | V2 |
|--------|-----|-----|
| **List** | `GET /voice-agents` | `GET /agents/voice-agents` |
| **Get** | `GET /voice-agents/{id}` | `GET /agents/voice-agents/{id}` |
| **Create** | `POST /voice-agents` | `POST /agents/voice-agents` |
| **Update** | `PUT /voice-agents/{id}` | `PUT /agents/voice-agents/{id}` |
| **Delete** | `DELETE /voice-agents/{id}` | `DELETE /agents/voice-agents/{id}` |

### Create Request: `AgentCreateRequest`
```json
{
  "name": "string, required (max 255)",
  "description": "string | null (max 1000)",
  "voice_id": "string | null",
  "instructions": "string | null",
  "is_active": true
}
```

### Update Request: `AgentUpdateRequest`
```json
{
  "name": "string | null",
  "description": "string | null",
  "voice_id": "string | null",
  "instructions": "string | null",
  "is_active": true | false
}
```

### Response: `AgentResponse`
```json
{
  "id": 123,
  "name": "Sales Agent",
  "description": "Handles sales calls",
  "voice_id": "voice-abc",
  "instructions": "You are a helpful sales agent...",
  "is_active": true,
  "created_at": "2024-12-30T10:00:00Z",
  "updated_at": "2024-12-30T10:05:00Z"
}
```

**No payload changes** - only path prefix changed.

---

## 5. OAuth - Google

### Endpoint Change
| Action | Old | V2 |
|--------|-----|-----|
| **Start** | `GET/POST /auth/google/start` | ✅ Same |
| **Callback** | `GET /auth/callback` | `GET /auth/google/callback` |
| **Status** | `GET /auth/status` | ✅ Same |
| **Revoke** | `POST /auth/revoke` | ✅ Same |

### Status Response: `OAuthStatusResponse`
```json
{
  "connected": true,
  "expires_at": "2024-12-31T10:00:00Z",
  "scopes": ["calendar", "gmail"],
  "has_refresh_token": true,
  "connected_gmail": "user@gmail.com"
}
```

---

## 6. Recordings API

### Endpoints (No Changes)
| Action | Endpoint |
|--------|----------|
| **Get Signed URL** | `POST /recordings/signed-url` |
| **Get by Call ID** | `GET /recordings/calls/{resource}/signed-url` |

### Request: `SignedUrlRequest`
```json
{
  "gs_url": "gs://bucket/path/to/recording.ogg"
}
```

### Response: `SignedUrlResponse`
```json
{
  "signed_url": "https://storage.googleapis.com/...",
  "gs_url": "gs://bucket/path/to/recording.ogg",
  "expires_in_hours": 24
}
```

---

## 7. Knowledge Base API

### Endpoints (No Changes)
All endpoints remain identical:

| Endpoint | Method |
|----------|--------|
| `/knowledge-base/status` | GET |
| `/knowledge-base/stores` | GET, POST |
| `/knowledge-base/stores/{store_id}` | GET, DELETE |
| `/knowledge-base/agents/{agent_id}/stores` | GET, POST |
| `/knowledge-base/agents/{agent_id}/stores/{store_id}` | DELETE |
| `/knowledge-base/leads/{lead_id}/stores` | GET, POST |
| `/knowledge-base/leads/{lead_id}/stores/{store_id}` | DELETE |

---

## Breaking Changes Summary

### Type Changes (⚠️ Breaking)

| Field | Location | Old | New |
|-------|----------|-----|-----|
| `initiated_by` | SingleCallPayload | `int` | `string` (UUID) |
| `initiated_by` | BatchCallJsonRequest | `int` | `string` (UUID) |
| `lead_id` | SingleCallPayload | `int` | `string` (UUID) |
| `lead_id` | BatchCallJsonEntry | `int` | `string` (UUID) |

### Path Changes

| Old Path | New Path |
|----------|----------|
| `POST /calls` | `POST /calls/start-call` |
| `GET /calls/{job_id}` | `GET /calls/job/{job_id}` |
| `POST /trigger-batch-call` | `POST /batch/trigger-batch-call` |
| `GET /batch-status/{id}` | `GET /batch/batch-status/{id}` |
| `POST /batch-stop/{id}` | `POST /batch/batch-cancel/{id}` |
| `GET /voice-agents` | `GET /agents/voice-agents` |
| `GET /auth/callback` | `GET /auth/google/callback` |

### New Features in V2

1. **Per-entry KB Store IDs**: Batch entries can now have individual `knowledge_base_store_ids`
2. **Multi-tenancy**: UUID-based identifiers support multi-tenant architecture

---

## Migration Checklist

- [ ] Update `initiated_by` from integer to UUID string
- [ ] Update `lead_id` from integer to UUID string
- [ ] Change `POST /calls` → `POST /calls/start-call`
- [ ] Change `GET /calls/{job_id}` → `GET /calls/job/{job_id}`
- [ ] Add `/batch` prefix to batch endpoints
- [ ] Add `/agents` prefix to voice-agents endpoints
- [ ] Update `/auth/callback` → `/auth/google/callback`
- [ ] Test all endpoints after migration
