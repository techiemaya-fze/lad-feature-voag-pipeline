"""
Schema constants for voice agent tables.

Phase 13: All tables now use the lad_dev schema.
The old voice_agent schema is deprecated and no longer used.
Phase 19: Schema is now configurable via DB_SCHEMA env variable.
"""

import os
from dotenv import load_dotenv

# Load environment variables BEFORE reading DB_SCHEMA
load_dotenv()

# =============================================================================
# SCHEMA NAME - Configurable via DB_SCHEMA env variable
# =============================================================================

SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")


# =============================================================================
# TABLE NAMES - Call Logs
# =============================================================================

CALL_LOGS_TABLE = "voice_call_logs"
CALL_LOGS_FULL = f"{SCHEMA}.{CALL_LOGS_TABLE}"

# Column names (new schema)
COL_CALL_ID = "id"
COL_RECORDING_URL = "recording_url"
COL_TRANSCRIPTS = "transcripts"
COL_DURATION = "duration_seconds"
COL_DIRECTION = "direction"
COL_LEAD_ID = "lead_id"  # UUID FK to leads table


# =============================================================================
# TABLE NAMES - Call Analysis
# =============================================================================

ANALYSIS_TABLE = "voice_call_analysis"
ANALYSIS_FULL = f"{SCHEMA}.{ANALYSIS_TABLE}"


# =============================================================================
# TABLE NAMES - Leads
# =============================================================================

LEADS_TABLE = "leads"
LEADS_FULL = f"{SCHEMA}.{LEADS_TABLE}"


# =============================================================================
# TABLE NAMES - Students (Education vertical)
# =============================================================================

STUDENTS_TABLE = "education_students"
STUDENTS_FULL = f"{SCHEMA}.{STUDENTS_TABLE}"


# =============================================================================
# TABLE NAMES - Batches
# =============================================================================

BATCHES_TABLE = "voice_call_batches"
BATCHES_FULL = f"{SCHEMA}.{BATCHES_TABLE}"

BATCH_ENTRIES_TABLE = "voice_call_batch_entries"
BATCH_ENTRIES_FULL = f"{SCHEMA}.{BATCH_ENTRIES_TABLE}"


# =============================================================================
# TABLE NAMES - Agents
# =============================================================================

AGENTS_TABLE = "voice_agents"
AGENTS_FULL = f"{SCHEMA}.{AGENTS_TABLE}"


# =============================================================================
# TABLE NAMES - Email Templates
# =============================================================================

EMAIL_TEMPLATES_TABLE = "communication_templates"
EMAIL_TEMPLATES_FULL = f"{SCHEMA}.{EMAIL_TEMPLATES_TABLE}"


# =============================================================================
# TABLE NAMES - Numbers
# =============================================================================

NUMBERS_TABLE = "voice_agent_numbers"
NUMBERS_FULL = f"{SCHEMA}.{NUMBERS_TABLE}"


# =============================================================================
# TABLE NAMES - Voices
# =============================================================================

VOICES_TABLE = "voice_agent_voices"
VOICES_FULL = f"{SCHEMA}.{VOICES_TABLE}"


# =============================================================================
# TABLE NAMES - Users
# =============================================================================

USERS_TABLE = "users"
USERS_FULL = f"{SCHEMA}.{USERS_TABLE}"

USER_IDENTITIES_TABLE = "user_identities"
USER_IDENTITIES_FULL = f"{SCHEMA}.{USER_IDENTITIES_TABLE}"


# =============================================================================
# TABLE NAMES - Pricing
# =============================================================================

PRICING_TABLE = "voice_agent_pricing"
PRICING_FULL = f"{SCHEMA}.{PRICING_TABLE}"

BILLING_PRICING_CATALOG_TABLE = "billing_pricing_catalog"
BILLING_PRICING_CATALOG_FULL = f"{SCHEMA}.{BILLING_PRICING_CATALOG_TABLE}"


# =============================================================================
# TABLE NAMES - Tenants & Features
# =============================================================================

TENANTS_TABLE = "tenants"
TENANTS_FULL = f"{SCHEMA}.{TENANTS_TABLE}"

TENANT_FEATURES_TABLE = "tenant_features"
TENANT_FEATURES_FULL = f"{SCHEMA}.{TENANT_FEATURES_TABLE}"


# =============================================================================
# TABLE NAMES - Knowledge Base
# =============================================================================

KNOWLEDGE_BASE_CATALOG_TABLE = "knowledge_base_catalog"
KNOWLEDGE_BASE_CATALOG_FULL = f"{SCHEMA}.{KNOWLEDGE_BASE_CATALOG_TABLE}"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_schema_info() -> dict:
    """Return info about current schema configuration."""
    return {
        "schema": SCHEMA,
        "call_logs": CALL_LOGS_FULL,
        "analysis": ANALYSIS_FULL,
        "leads": LEADS_FULL,
        "students": STUDENTS_FULL,
        "batches": BATCHES_FULL,
        "batch_entries": BATCH_ENTRIES_FULL,
        "agents": AGENTS_FULL,
        "email_templates": EMAIL_TEMPLATES_FULL,
        "numbers": NUMBERS_FULL,
        "voices": VOICES_FULL,
        "users": USERS_FULL,
        "pricing": PRICING_FULL,
    }

