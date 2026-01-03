# V2 Architecture Documentation

## Overview

The **v2** folder contains the modular, production-ready implementation of the Vonage Voice Agent system. This architecture separates concerns into distinct modules for maintainability, testability, and scalability.

---

## Directory Structure

```
v2/
├── .env                          # Environment variables (secrets, API keys)
├── .env.example                  # Template for environment variables
├── main.py                       # FastAPI application entry point
├── schema_analysis.txt           # Database schema documentation
├── salesmaya-yts-*.json          # GCS service account credentials
│
├── agent/                        # 🎯 Core Voice Agent Components
│   ├── __init__.py               # Package exports
│   ├── worker.py                 # Main LiveKit worker entry point
│   ├── config.py                 # Pipeline configuration (VAD, STT, TTS settings)
│   ├── pipeline.py               # TTS/LLM/STT engine builders
│   ├── tool_builder.py           # Dynamic tool attachment based on tenant_features
│   ├── instruction_builder.py    # Agent prompt/instruction generation
│   ├── cleanup_handler.py        # Post-call cleanup and cost tracking
│   └── providers/                # LLM/TTS provider factories
│       ├── __init__.py
│       ├── llm_builder.py        # LLM instance creation (Gemini, OpenAI)
│       └── tts_builder.py        # TTS engine creation (Google, ElevenLabs)
│
├── api/                          # 🌐 REST API Layer (FastAPI)
│   ├── __init__.py
│   ├── middleware.py             # Request logging, auth, error handling
│   ├── models.py                 # Pydantic request/response models
│   ├── routes/                   # API endpoint handlers
│   │   ├── __init__.py
│   │   ├── agents.py             # /agents - Agent CRUD
│   │   ├── calls.py              # /calls - Single call triggers
│   │   ├── batch.py              # /batch - Batch call campaigns
│   │   ├── recordings.py         # /recordings - Call recordings & signed URLs
│   │   ├── knowledge_base.py     # /kb - Knowledge base management
│   │   ├── oauth.py              # /auth/google - Google OAuth
│   │   └── oauth_microsoft.py    # /auth/microsoft - Microsoft OAuth + Bookings
│   └── services/                 # Business logic services
│       ├── __init__.py
│       └── call_service.py       # Call triggering logic
│
├── db/                           # 💾 Database Layer
│   ├── __init__.py
│   ├── config.py                 # DB connection settings (alias of db_config.py)
│   ├── db_config.py              # Database configuration (local vs production)
│   ├── connection_pool.py        # PostgreSQL connection pooling
│   ├── pool.py                   # Alias for connection_pool
│   ├── schema_constants.py       # Table names, column names
│   ├── migrations/               # Database migrations
│   └── storage/                  # Data access layer (DAOs)
│       ├── __init__.py
│       ├── agents.py             # Agent CRUD operations
│       ├── calls.py              # Call log storage
│       ├── batches.py            # Batch/campaign storage
│       ├── leads.py              # Lead management
│       ├── students.py           # G-Links student storage
│       ├── tokens.py             # OAuth token storage
│       ├── knowledge_base.py     # KB catalog storage
│       ├── email_templates.py    # Email template storage
│       ├── call_analysis.py      # Post-call analysis results
│       ├── numbers.py            # Phone number management
│       └── voices.py             # Custom voice configurations
│
├── tools/                        # 🔧 Agent Tools (Function Calling)
│   ├── google_workspace.py       # AgentGoogleWorkspace - OAuth wrapper
│   ├── google_calendar_tool.py   # Low-level Calendar API
│   ├── gmail_email_tool.py       # Low-level Gmail API
│   ├── microsoft_bookings.py     # AgentMicrosoftBookings - with config support
│   ├── microsoft_bookings_tool.py# Low-level MS Bookings API
│   ├── email_templates.py        # Template rendering + sending
│   ├── builtin_email_templates.py# Hardcoded fallback templates
│   ├── file_search_tool.py       # Gemini RAG document management
│   └── document_converter.py     # Document format conversion for KB
│
├── utils/                        # 🛠️ Shared Utilities
│   ├── __init__.py
│   ├── api_security.py           # API key validation, rate limiting
│   ├── google_oauth.py           # Google OAuth helpers
│   ├── microsoft_oauth.py        # Microsoft OAuth helpers
│   ├── google_credentials.py     # GCS credential management
│   ├── usage_tracker.py          # UsageCollector for cost tracking
│   ├── tenant_utils.py           # Tenant resolution utilities
│   ├── signed_url_cache.py       # GCS signed URL caching
│   ├── audio_trim.py             # Audio silence trimming
│   ├── logger.py                 # Logging configuration
│   └── logger_config.py          # Extended logging setup
│
├── analysis/                     # 📊 Post-Call Analytics
│   ├── __init__.py
│   ├── merged_analytics.py       # Main analytics orchestrator (146KB)
│   ├── call_report.py            # Single call report generation
│   ├── batch_report.py           # Batch campaign reports
│   ├── lead_extractor.py         # Lead extraction from transcripts
│   ├── lead_info_extractor.py    # Detailed lead info extraction
│   ├── student_extractor.py      # G-Links student extraction
│   ├── lad_dev.py                # LAD schema analytics
│   ├── runner.py                 # CLI analytics runner
│   ├── logs/                     # Analytics logs
│   ├── exports/                  # CSV/Excel exports
│   └── json_exports/             # JSON data exports
│
├── recording/                    # 🎙️ Call Recording Module
│   ├── __init__.py
│   ├── recorder.py               # Main CallRecorder class
│   ├── api.py                    # Recording API utilities
│   ├── audio_trim.py             # Post-call silence trimming
│   └── transcription.py          # TranscriptionTracker
│
├── auth/                         # 🔐 OAuth Handlers
│   ├── __init__.py
│   ├── google.py                 # Google OAuth flow
│   └── microsoft.py              # Microsoft OAuth flow
│
├── tts/                          # 🔊 Text-to-Speech Extensions
│   ├── __init__.py
│   └── google_chirp_streaming.py # Google Chirp streaming TTS
│
├── storage/                      # 📁 File Storage Utilities
│   ├── __init__.py
│   ├── gcs.py                    # GCS upload/download helpers
│   └── url_cache.py              # Signed URL caching
│
├── batch/                        # 📞 Batch Queue Management
│   └── queue_manager.py          # Batch call queue processing
│
├── scripts/                      # 🔧 Utility Scripts
│   ├── analyze_schema.py         # Database schema analysis
│   ├── insert_tenant_features.py # Seed tenant_features table
│   ├── migrate_kb.py             # Knowledge base migration
│   ├── benchmarks/               # Performance benchmarks
│   ├── db_tools/                 # Database utilities
│   └── setup/                    # Setup scripts
│
└── tests/                        # 🧪 Test Files
    └── oauth/
        └── test_tools_individual.py
```

---

## Detailed Module Descriptions

### 1. `agent/` - Core Voice Agent Components

The heart of the voice agent system. Handles LiveKit integration, conversation flow, and tool execution.

| File | Purpose | Used By |
|------|---------|---------|
| `worker.py` | **Main Entry Point** - LiveKit worker, creates VoiceAssistant, handles inbound/outbound calls | LiveKit runtime |
| `config.py` | Pipeline configuration - VAD, STT, TTS, endpointing settings | worker.py, pipeline.py |
| `pipeline.py` | Engine builders - creates LLM, TTS, STT instances based on config | worker.py |
| `tool_builder.py` | **Dynamic Tool Attachment** - queries tenant_features, builds @function_tool decorated functions | worker.py |
| `instruction_builder.py` | Agent prompt generation - builds system instructions from templates | worker.py |
| `cleanup_handler.py` | Post-call cleanup - saves transcripts, calculates costs, updates call status | worker.py |

#### `agent/providers/` - LLM/TTS Factories

| File | Purpose |
|------|---------|
| `llm_builder.py` | Creates LLM instances (Gemini 2.0 Flash, OpenAI GPT-4) |
| `tts_builder.py` | Creates TTS engines (Google Cloud TTS, ElevenLabs) |

---

### 2. `api/` - REST API Layer

FastAPI-based REST API for external integrations and frontend communication.

| File | Purpose |
|------|---------|
| `middleware.py` | Request logging, authentication, CORS, error handling |
| `models.py` | Pydantic models for request/response validation |

#### `api/routes/` - Endpoint Handlers

| File | Endpoints | Purpose |
|------|-----------|---------|
| `agents.py` | `/agents/*` | Agent CRUD (create, update, delete) |
| `calls.py` | `/calls/*` | Single call trigger, call status |
| `batch.py` | `/batch/*` | Batch campaign management |
| `recordings.py` | `/recordings/*` | Recording access, signed URLs |
| `knowledge_base.py` | `/kb/*` | KB store management, document upload |
| `oauth.py` | `/auth/google/*` | Google OAuth flow |
| `oauth_microsoft.py` | `/auth/microsoft/*` | Microsoft OAuth, Bookings config |

#### `api/services/` - Business Logic

| File | Purpose |
|------|---------|
| `call_service.py` | Call triggering and dispatch logic |

---

### 3. `db/` - Database Layer

PostgreSQL database access with connection pooling and modular storage classes.

| File | Purpose |
|------|---------|
| `db_config.py` | Database configuration (local vs production switch) |
| `connection_pool.py` | Connection pooling with retry logic |
| `schema_constants.py` | Table names, column definitions |

#### `db/storage/` - Data Access Objects

| File | Tables Accessed | Purpose |
|------|-----------------|---------|
| `agents.py` | `agents_voiceagent` | Agent configuration CRUD |
| `calls.py` | `call_logs_voiceagent` | Call log storage and retrieval |
| `batches.py` | `batch_logs_voiceagent`, `batch_call_entries` | Batch campaign management |
| `tokens.py` | `user_identities` | OAuth token storage (Google, Microsoft) |
| `leads.py` | `leads_voiceagent` | Lead management |
| `students.py` | `students_voiceagent` | G-Links student data |
| `knowledge_base.py` | `lad_dev.knowledge_base_catalog` | KB store metadata |
| `email_templates.py` | `lad_dev.communication_templates` | Email templates |

---

### 4. `tools/` - Agent Tools

Function calling tools that the LLM can invoke during conversations.

| File | Tool Functions | Purpose |
|------|----------------|---------|
| `google_workspace.py` | `AgentGoogleWorkspace` | High-level Google Calendar + Gmail wrapper |
| `microsoft_bookings.py` | `AgentMicrosoftBookings` | MS Bookings with tenant_features config support |
| `email_templates.py` | `create_email_template_tools()` | Template-based email sending |
| `file_search_tool.py` | `FileSearchTool` | Gemini RAG document management |
| `builtin_email_templates.py` | Fallback templates | Hardcoded email templates |

**Architecture:**
```
tool_builder.py
    ├── build_google_workspace_tools() → AgentGoogleWorkspace
    ├── build_microsoft_bookings_tools() → AgentMicrosoftBookings
    ├── build_knowledge_base_tools() → FileSearchTool
    ├── build_email_template_tools() → email_templates.py
    └── build_human_support_tools() → SIP transfer
```

---

### 5. `utils/` - Shared Utilities

Common utilities used across the application.

| File | Purpose |
|------|---------|
| `usage_tracker.py` | `UsageCollector` - tracks LLM/TTS/STT costs per call |
| `api_security.py` | API key validation, rate limiting |
| `google_oauth.py` | Token encryption, OAuth helpers |
| `microsoft_oauth.py` | Microsoft OAuth token management |
| `tenant_utils.py` | Tenant ID resolution from agent/call |
| `signed_url_cache.py` | GCS signed URL caching for recordings |
| `audio_trim.py` | Silence detection and trimming |

---

### 6. `analysis/` - Post-Call Analytics

Runs after calls complete to extract insights and generate reports.

| File | Purpose |
|------|---------|
| `merged_analytics.py` | **Main orchestrator** - runs all analysis steps |
| `call_report.py` | Single call report generation |
| `batch_report.py` | Batch campaign summary reports |
| `lead_extractor.py` | Extract lead info from transcripts |
| `student_extractor.py` | G-Links specific student extraction |
| `lad_dev.py` | LAD schema analytics |
| `runner.py` | CLI entry point for analytics |

---

### 7. `recording/` - Call Recording

Handles LiveKit egress for call recording and post-processing.

| File | Purpose |
|------|---------|
| `recorder.py` | `CallRecorder` - manages LiveKit room composite egress |
| `transcription.py` | `TranscriptionTracker` - collects transcription events |
| `audio_trim.py` | Trims silence from recordings |
| `api.py` | Recording retrieval utilities |

---

### 8. `auth/` - OAuth Handlers

OAuth 2.0 implementation for Google and Microsoft integrations.

| File | Purpose |
|------|---------|
| `google.py` | Google OAuth authorization flow |
| `microsoft.py` | Microsoft OAuth authorization flow |

---

### 9. `storage/` - File Storage Utilities

Handles GCS operations and URL caching.

| File | Purpose |
|------|---------|
| `gcs.py` | GCS upload/download helpers |
| `url_cache.py` | Signed URL caching for recordings |

---

### 10. `tts/` - Text-to-Speech Extensions

Custom TTS implementations.

| File | Purpose |
|------|---------|
| `google_chirp_streaming.py` | Google Chirp streaming TTS |

---

### 11. `batch/` - Batch Queue Management

| File | Purpose |
|------|---------|
| `queue_manager.py` | Batch call queue processing |

---

### 12. `scripts/` - Utility Scripts

| File | Purpose |
|------|---------|
| `analyze_schema.py` | Database schema analysis |
| `insert_tenant_features.py` | Seed tenant_features table |
| `migrate_kb.py` | Knowledge base migration |

---

## Key Integration Points

### Call Flow
```
1. API receives call request → api/routes/calls.py
2. LiveKit job dispatched → agent/worker.py
3. VoiceAssistant created with tools → agent/tool_builder.py
4. Conversation happens with LLM → agent/pipeline.py
5. Call ends → agent/cleanup_handler.py
6. Post-call analysis → analysis/merged_analytics.py
```

### Tool Enablement Flow
```
1. tenant_features table queried → tool_builder._get_tools_from_tenant_features()
2. ToolConfig created with enabled flags
3. build_*_tools() called for each enabled tool
4. @function_tool decorated functions returned
5. Passed to VoiceAssistant(tools=[...])
```

### Cost Tracking Flow
```
1. UsageCollector created in worker.py
2. Attached to AgentSession
3. Collects LLM/TTS/STT usage events
4. cleanup_handler calculates total cost
5. Saved to call_logs_voiceagent
```

---

## Configuration

### Environment Variables
See `.env.example` for all required variables:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `GEMINI_API_KEY` or `OPENAI_API_KEY`
- `GOOGLE_CLOUD_*` for TTS/STT
- `DATABASE_*` for PostgreSQL
- `TOOLS_DECIDED_BY_BACKEND=true` for tenant-based tool enablement

### Tenant Features
Tools are enabled per-tenant via `lad_dev.tenant_features`:
```sql
INSERT INTO lad_dev.tenant_features (tenant_id, feature_key, enabled, config)
VALUES ('uuid', 'voice-agent-tool-microsoft-bookings-auto', true, 
        '{"business_id": "...", "service_id": "...", "staff_id": "..."}'::jsonb);
```

---

*Last Updated: 2024-12-30*
