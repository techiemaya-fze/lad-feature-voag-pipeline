# Knowledge Base API Testing Guide

This guide provides curl commands to test the Knowledge Base (KB) API endpoints for document upload, store management, and querying.

## Environment Setup

```bash
# Base URLs
export BASE_URL="http://localhost:8000"
# export BASE_URL="https://voag.techiemaya.com"

# Authentication (FRONTEND_ID and API_KEY must match - they are paired!)
export FRONTEND_ID="dev"
export API_KEY="kMQgGRDAa8t5CvmkfqFYuGiXIXgNYC1EEGjYs5v8_NU"

# Tenant IDs
export TENANT_ID="926070b5-189b-4682-9279-ea10ca090b84"  # Glinks
# export TENANT_ID="734cd516-e252-4728-9c52-4663ee552653"  # Sasya

# Common headers (for reference)
export HEADERS="-H 'Content-Type: application/json' -H 'X-Frontend-ID: $FRONTEND_ID' -H 'X-API-Key: $API_KEY'"
```

---

## Quick Start: Test KB Query

If you already have documents uploaded, test the query endpoint directly:

```bash
curl -s -X POST "$BASE_URL/knowledge-base/query" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "tenant_id": "'"$TENANT_ID"'",
    "question": "What information do you have?"
  }' | jq
```

**Expected Response:**
```json
{
  "answer": "Based on the uploaded documents, I found...",
  "sources": ["Document Name 1", "Document Name 2"],
  "store_names": ["fileSearchStores/xxx"]
}
```

---

## Full Workflow: Create Store → Upload → Query

### Step 1: Check KB Status

```bash
curl -s "$BASE_URL/knowledge-base/status" | jq
```

Expected: `{"enabled": true, ...}`

---

### Step 2: List Existing Stores (Optional)

```bash
curl -s -X GET "$BASE_URL/knowledge-base/stores?tenant_id=$TENANT_ID" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" | jq
```

---

### Step 3: Create a New KB Store

```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "tenant_id": "'"$TENANT_ID"'",
    "display_name": "My KB Store",
    "description": "Knowledge base for testing",
    "is_default": false,
    "priority": 1
  }' | jq
```

**Save the returned IDs:**
```bash
export STORE_ID="<returned-id>"
export GEMINI_STORE="<returned-gemini_store_name>"
```

---

### Step 4: Upload Documents

#### Text/Markdown File
```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores/$STORE_ID/documents" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/document.md" \
  -F "display_name=My Document" | jq
```

#### Excel File (auto-converts to text)
```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores/$STORE_ID/documents" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/spreadsheet.xlsx" \
  -F "display_name=Excel Data" | jq
```

#### PDF File
```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores/$STORE_ID/documents" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/path/to/document.pdf" \
  -F "display_name=PDF Document" | jq
```

**Wait ~30 seconds for indexing after upload.**

---

### Step 4b: Bulk Upload (All Files from Folder)

Upload all supported files from a folder at once:

```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores/$STORE_ID/bulk-upload" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "folder_path": "/absolute/path/to/your/folder"
  }'
```

**Windows example:**
```bash
curl -s -X POST "$BASE_URL/knowledge-base/stores/$STORE_ID/bulk-upload" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "folder_path": "D:/vonage/vonage-voice-agent/G_links"
  }'
```

**Response:**
```json
{
  "store_id": "xxx",
  "total_files": 5,
  "successful": 4,
  "failed": 1,
  "results": [
    {"filename": "doc1.pdf", "success": true, "display_name": "doc1"},
    {"filename": "data.xlsx", "success": true, "display_name": "data"},
    {"filename": "bad.xyz", "success": false, "error": "Unsupported format"}
  ]
}
```

---

### Step 5: Query the KB

```bash
curl -s -X POST "$BASE_URL/knowledge-base/query" \
  -H "Content-Type: application/json" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "tenant_id": "'"$TENANT_ID"'",
    "question": "What does the document say about X?"
  }' | jq
```

---

## Additional Endpoints

### List Documents in Store
```bash
curl -s -X GET "$BASE_URL/knowledge-base/stores/$STORE_ID/documents" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" | jq
```

### Get Stores for Tenant
```bash
curl -s -X GET "$BASE_URL/knowledge-base/tenants/$TENANT_ID/stores" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" | jq
```

### Get Single Store Details
```bash
curl -s -X GET "$BASE_URL/knowledge-base/stores/$STORE_ID" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" | jq
```

### Delete a Store
```bash
curl -s -X DELETE "$BASE_URL/knowledge-base/stores/$STORE_ID" \
  -H "X-Frontend-ID: $FRONTEND_ID" \
  -H "X-API-Key: $API_KEY" | jq
```

---

## Supported File Formats

| Format | Extension | Conversion |
|--------|-----------|------------|
| Plain Text | .txt | Direct upload |
| Markdown | .md | Direct upload |
| PDF | .pdf | Direct upload |
| Excel | .xlsx, .xls | Auto-converts to text |
| Word | .docx, .doc | Auto-converts to text |
| PowerPoint | .pptx, .ppt | Auto-converts to text |
| CSV | .csv | Auto-converts to text |
| HTML | .html | Direct upload |
| JSON | .json | Direct upload |

---

## Notes

1. **Model Requirement**: The query endpoint uses `gemini-2.5-flash` by default (required for file search).
2. **Indexing Time**: After upload, wait ~30 seconds for document indexing.
3. **Auto-Attach**: KB stores with `is_default=true` auto-attach to calls for that tenant.
4. **Citations**: Responses include `sources` array with document titles cited.
5. **Multi-Store**: Queries search across ALL stores linked to the tenant.
