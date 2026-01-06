"""
Call Service Module (V2).

Contains core call dispatch logic migrated from main.py.
This service is used by api/routes/calls.py and api/routes/batch.py.
"""

import asyncio
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from livekit import api

# Call routing validation
from utils.call_routing import validate_and_format_call
from db.db_config import get_db_config

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

GLINKS_ORG_ID = "f6de7991-df4f-43de-9f40-298fcda5f723"
GLINKS_FRONTEND_ID = "g_links"
DEFAULT_GLINKS_KB_STORE_IDS = ["fileSearchStores/glinks-complete-documents-d-8uyr36hsxdgz"]


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class VoiceContext:
    db_voice_id: str | None
    tts_voice_id: str
    provider: str | None
    overrides: dict[str, str]
    voice_name: str | None
    accent: str | None


@dataclass
class LeadMetadata:
    lead_id: str | int | None = None
    lead_name: str | None = None
    lead_notes: str | None = None
    target_type: str = "lead"  # 'lead' or 'student' for polymorphic FK


@dataclass
class BatchCallEntry:
    to_number: str
    context: str | None = None
    lead_name: str | None = None
    lead_id: str | None = None  # UUID
    call_log_id: str | None = None
    room_name: str | None = None
    lead_notes: str | None = None
    knowledge_base_store_ids: list[str] | None = None


@dataclass
class DispatchResult:
    room_name: str
    dispatch_id: str
    lead_id: str | int | None = None
    lead_name: str | None = None
    added_context: str | None = None
    call_log_id: str | None = None
    error: str | None = None  # Error message for routing failures


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_tts_provider(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    alias_map = {
        "google-tts": "google",
        "google_tts": "google",
        "gemini-tts": "gemini",
        "gemini_tts": "gemini",
        "cartesia_tts": "cartesia",
        "elevenlabs": "elevenlabs",
        "eleven-labs": "elevenlabs",
        "eleven_labs": "elevenlabs",
        "11labs": "elevenlabs",
        "11-labs": "elevenlabs",
        "rime_tts": "rime",
        "rime-ai": "rime",
        "smallestai": "smallestai",
        "smallest_ai": "smallestai",
        "smallest-ai": "smallestai",
        "waves": "smallestai",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    compact = normalized.replace(" ", "_")
    if compact in alias_map:
        return alias_map[compact]
    return normalized


def _normalize_llm_provider(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace(" ", "_").replace("-", "_")
    alias_map = {
        "gemini": "google",
        "google_gemini": "google",
        "google-gemini": "google",
        "googleai": "google",
        "chatgpt": "openai",
        "gpt": "openai",
        "gpt4": "openai",
        "gpt-4": "openai",
        "gpt-4o": "openai",
    }
    compact = normalized.replace("__", "_")
    return alias_map.get(compact, compact)


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _coerce_uuid_string(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def _augment_context_with_lead(
    context: str | None,
    lead_name: str | None,
    lead_notes: str | None,
    first_name: str | None = None,
    last_name: str | None = None,
    email: str | None = None,
    company: str | None = None,
) -> str | None:
    """Augment context with lead information for the agent."""
    def _append_line(existing: str | None, line: str | None) -> str | None:
        if not line:
            return existing
        current = (existing or "").rstrip()
        if current and line in current.splitlines():
            return current
        if current:
            return f"{current}\n{line}"
        return line

    updated = (context or None)
    
    # Add lead name (combined or from first/last)
    name_to_use = lead_name
    if not name_to_use and (first_name or last_name):
        parts = [n for n in [first_name, last_name] if n]
        name_to_use = " ".join(parts) if parts else None
    
    if name_to_use:
        lead_line = f'The lead\'s name is "{name_to_use}".'
        updated = _append_line(updated, lead_line)
    
    # Add email if available
    if email and email.strip():
        email_line = f'Lead email: {email.strip()}'
        updated = _append_line(updated, email_line)
    
    # Add company if available
    if company and company.strip():
        company_line = f'Lead company: {company.strip()}'
        updated = _append_line(updated, company_line)
    
    # Add notes if available
    if lead_notes and lead_notes.strip():
        notes_line = f"Lead notes: {lead_notes.strip()}"
        updated = _append_line(updated, notes_line)
    
    return updated.rstrip() if isinstance(updated, str) else updated



def _combine_contexts(*contexts: Optional[str]) -> str | None:
    parts = [segment.strip() for segment in contexts if segment and segment.strip()]
    if not parts:
        return None
    return "\n".join(parts)


def _validate_livekit_credentials() -> tuple[str, str, str]:
    """Validate and return LiveKit credentials."""
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([url, api_key, api_secret]):
        raise RuntimeError("Missing LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET")
    
    return url, api_key, api_secret


def _resolve_voice_context(voice_id: str, voice_record: Optional[Mapping[str, Any]]) -> VoiceContext:
    """Resolve voice context from voice ID and optional DB record."""
    if not voice_record:
        return VoiceContext(
            db_voice_id=None,
            tts_voice_id=voice_id,
            provider=None,
            overrides={},
            voice_name=None,
            accent=None,
        )
    
    db_id = voice_record.get("id")
    provider_raw = voice_record.get("provider")
    provider = _normalize_tts_provider(provider_raw)
    
    # Determine TTS voice ID
    tts_voice_id = voice_record.get("provider_voice_id")
    if not tts_voice_id:
        tts_voice_id = str(db_id) if db_id else voice_id
    
    return VoiceContext(
        db_voice_id=str(db_id) if db_id else None,
        tts_voice_id=str(tts_voice_id),
        provider=provider,
        overrides={},  # Can be extended with _extract_tts_overrides
        voice_name=voice_record.get("description"),
        accent=voice_record.get("accent"),
    )


# =============================================================================
# CALL STORAGE INTERFACE
# =============================================================================

class CallService:
    """Service for dispatching outbound calls."""
    
    def __init__(self):
        self._storage_initialized = False
        self._call_storage = None
        self._lead_storage = None
        self._agent_storage = None
        self._voice_storage = None
        self._number_storage = None  # For from_number_id lookup
        self._user_storage = None  # For user tenant lookup
    
    async def _ensure_storage(self):
        """Lazy-initialize storage instances."""
        if not self._storage_initialized:
            from db.storage import CallStorage, LeadStorage, AgentStorage, VoiceStorage, NumberStorage
            from db.storage.tokens import UserTokenStorage
            self._call_storage = CallStorage()
            self._lead_storage = LeadStorage()
            self._agent_storage = AgentStorage()
            self._voice_storage = VoiceStorage()
            self._number_storage = NumberStorage()
            self._user_storage = UserTokenStorage()
            self._storage_initialized = True
    
    async def resolve_voice(self, voice_id: str, agent_id: int | None) -> tuple[str, VoiceContext]:
        """Resolve voice ID and return (resolved_id, voice_context)."""
        await self._ensure_storage()
        
        resolved_voice_id = voice_id
        if voice_id.lower() == "default":
            if agent_id is None:
                raise ValueError("voice_id 'default' requires agent_id in the request")
            
            agent_record = await self._agent_storage.get_agent_by_id(agent_id)
            if not agent_record:
                raise ValueError(f"Agent {agent_id} not found; cannot resolve default voice")
            
            default_voice = agent_record.get("voice_id")
            if not default_voice:
                raise ValueError(f"Agent {agent_id} does not have a default voice configured")
            
            resolved_voice_id = str(default_voice)
        
        voice_record = await self._voice_storage.get_voice_by_id(resolved_voice_id)
        voice_context = _resolve_voice_context(resolved_voice_id, voice_record)
        
        return resolved_voice_id, voice_context
    
    async def should_use_glinks_kb(
        self,
        agent_id: int | None,
        frontend_id: str | None,
        kb_store_ids: list[str] | None,
    ) -> bool:
        """Check if default Glinks KB should be auto-assigned."""
        if kb_store_ids:
            return False
        
        if frontend_id and frontend_id.lower() == GLINKS_FRONTEND_ID:
            return True
        
        if agent_id:
            await self._ensure_storage()
            # Use is_education_agent (checks vertical via tenant_id)
            is_glinks = await self._agent_storage.is_education_agent(agent_id)
            if is_glinks:
                return True
        
        return False
    
    async def create_call_log(
        self,
        *,
        tenant_id: str | None,  # Required for new schema
        to_number: str,
        voice_id: str | None,
        agent_id: int | None,
        initiated_by: str | None,  # UUID - will be passed as initiated_by_user_id
        room_name: str,
        job_id: str,
        added_context: str | None = None,
        lead_id: str | int | None = None,
        from_number_id: str | None = None,  # UUID FK to voice_agent_numbers
    ) -> str:
        """Create a call log record and return the call_log_id."""
        await self._ensure_storage()
        
        if not tenant_id:
            # Try to get tenant_id from agent if not provided
            if agent_id:
                tenant_id = await self._agent_storage.get_agent_tenant_id(agent_id)
            
            if not tenant_id:
                logger.warning("No tenant_id available for create_call_log, using fallback")
                tenant_id = "00000000-0000-0000-0000-000000000000"  # Fallback for legacy calls
        
        call_log_id = await self._call_storage.create_call_log(
            tenant_id=tenant_id,
            to_number=to_number,
            lead_id=str(lead_id) if lead_id else None,
            agent_id=agent_id,
            voice_id=_coerce_uuid_string(voice_id),
            initiated_by_user_id=initiated_by,
            from_number_id=from_number_id,  # Pass through from_number_id
            job_id=job_id,
            room_name=room_name,
            added_context=added_context,
        )
        
        return str(call_log_id) if call_log_id else None
    
    async def dispatch_call(
        self,
        *,
        job_id: str,
        voice_id: str,
        voice_context: VoiceContext,
        from_number: str | None,
        to_number: str,
        context: str | None,
        initiated_by: str | None,  # UUID
        agent_id: int | None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        knowledge_base_store_ids: list[str] | None = None,
        lead_name: str | None = None,  # For lead creation/update
        lead_id_override: str | None = None,  # UUID
        batch_id: str | None = None,  # Batch call tracking
        entry_id: str | None = None,  # Batch entry tracking
    ) -> DispatchResult:
        """
        Dispatch a single outbound call via LiveKit.
        
        Returns DispatchResult with room_name, dispatch_id, call_log_id.
        """
        await self._ensure_storage()
        
        url, api_key, api_secret = _validate_livekit_credentials()
        voice_log_id = _coerce_uuid_string(voice_context.db_voice_id)
        
        # Resolve tenant_id - prefer user's tenant, fall back to agent's tenant
        tenant_id = None
        if initiated_by:
            tenant_id = await self._user_storage.get_user_tenant_id(initiated_by)
        if not tenant_id and agent_id:
            tenant_id = await self._agent_storage.get_agent_tenant_id(agent_id)
            logger.debug("Resolved tenant_id from agent_id=%s (user tenant not found)", agent_id)
        
        lead_record = None
        request_lead_name = lead_name  # Save the name from API request
        if tenant_id:
            lead_record = await self._lead_storage.find_or_create_lead(
                tenant_id=tenant_id,
                phone_number=to_number,
                user_id=initiated_by,
                name=lead_name,  # Pass lead_name from API (will save if table empty)
            )
        else:
            logger.warning("No tenant_id found for user=%s agent=%s, skipping lead lookup", initiated_by, agent_id)
        
        lead_id = lead_id_override
        # Determine lead info for agent context
        lead_name_for_agent = None
        lead_notes = None
        lead_first_name = None
        lead_last_name = None
        lead_email = None
        lead_company = None
        
        if lead_record:
            lead_id = lead_id or lead_record.get("id")
            
            # Get fields from lead record
            lead_first_name = lead_record.get("first_name")
            lead_last_name = lead_record.get("last_name")
            lead_email = lead_record.get("email")
            lead_company = lead_record.get("company")
            
            # Get table name (combined)
            table_name_raw = lead_record.get("name")
            table_name = None
            if isinstance(table_name_raw, str) and table_name_raw.strip():
                table_name = table_name_raw.strip()
            
            # Priority: request name > table name (for agent context)
            if request_lead_name and request_lead_name.strip():
                lead_name_for_agent = request_lead_name.strip()
                if table_name and table_name != lead_name_for_agent:
                    logger.info(f"Using request name '{lead_name_for_agent}' for agent (table has '{table_name}')")
            elif table_name:
                lead_name_for_agent = table_name
            
            lead_notes_raw = lead_record.get("notes")
            if isinstance(lead_notes_raw, str) and lead_notes_raw.strip():
                lead_notes = lead_notes_raw.strip()
        
        # Augment context with lead info (name, email, company, notes)
        final_context = _augment_context_with_lead(
            context=context,
            lead_name=lead_name_for_agent,
            lead_notes=lead_notes,
            first_name=lead_first_name,
            last_name=lead_last_name,
            email=lead_email,
            company=lead_company,
        )
        
        # Generate room name
        room_name = f"call-{job_id}-{uuid.uuid4().hex[:8]}"
        
        # ===== CALL ROUTING VALIDATION =====
        # Validate and format to_number based on from_number's carrier rules
        routing_result = validate_and_format_call(
            from_number=from_number,
            to_number=to_number,
            db_config=get_db_config()
        )
        
        if not routing_result.success:
            logger.error(f"Call routing failed: {routing_result.error_message}")
            # Return error result instead of raising
            return DispatchResult(
                room_name="",
                dispatch_id="",
                call_log_id="",
                error=routing_result.error_message
            )
        
        # Use the formatted number for dialing
        dial_number = routing_result.formatted_to_number or to_number
        logger.info(f"Call routing: {to_number} -> {dial_number} (carrier: {routing_result.carrier_name})")
        
        # Lookup from_number_id from voice_agent_numbers table
        from_number_id = None
        if from_number:
            from_number_id = await self._number_storage.find_number_by_phone(from_number)
            if from_number_id:
                logger.debug(f"Resolved from_number {from_number} to from_number_id={from_number_id}")
            else:
                logger.warning(f"Could not find from_number_id for from_number={from_number}")
        
        # Create call log
        call_log_id = await self.create_call_log(
            tenant_id=tenant_id,  # Already computed above for lead lookup
            to_number=to_number,
            voice_id=voice_log_id,
            agent_id=agent_id,
            initiated_by=initiated_by,
            room_name=room_name,
            job_id=job_id,
            added_context=final_context,
            lead_id=lead_id,
            from_number_id=from_number_id,  # Pass from_number_id FK
        )
        
        # Create LiveKit room and dispatch agent (use async with to close session properly)
        async with api.LiveKitAPI(url, api_key, api_secret) as livekit_api:
            # Create room
            await livekit_api.room.create_room(
                api.CreateRoomRequest(name=room_name)
            )
            
            # Create SIP participant (outbound call)
            metadata = {
                "job_id": job_id,
                "agent_id": agent_id,
                "voice_id": voice_id,
                "tts_voice_id": voice_context.tts_voice_id,
                "voice_provider": voice_context.provider,
                "call_log_id": call_log_id,
                "lead_id": lead_id,  # For vertical routing
                "to_number": to_number,  # Original number for display/logging
                "phone_number": dial_number,  # Formatted number for SIP dialing
                "call_mode": "outbound",  # Tell worker this is outbound
                "from_number": from_number,
                "initiated_by": initiated_by,  # User ID for OAuth tools
            }
            
            if llm_provider:
                metadata["llm_provider"] = llm_provider
            if llm_model:
                metadata["llm_model"] = llm_model
            if knowledge_base_store_ids:
                metadata["knowledge_base_store_ids"] = ",".join(knowledge_base_store_ids)
            if final_context:
                metadata["added_context"] = final_context[:500]  # Truncate for safety
            
            # Use trunk ID from routing rules, or fall back to env default
            outbound_trunk = routing_result.outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
            if outbound_trunk:
                metadata["outbound_trunk_id"] = outbound_trunk
            
            # Batch call tracking - worker uses these to update entry status on completion
            if batch_id:
                metadata["batch_id"] = batch_id
            if entry_id:
                metadata["entry_id"] = entry_id
            
            import json
            participant_metadata = json.dumps(metadata)
            
            # Create agent dispatch
            dispatch_result = await livekit_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=os.getenv("VOICE_AGENT_NAME", "inbound-agent"),
                    room=room_name,
                    metadata=participant_metadata,
                )
            )
            
            dispatch_id = dispatch_result.id if dispatch_result else None
        
        logger.info(
            "Dispatched call: room=%s, dispatch_id=%s, call_log_id=%s, to=%s",
            room_name, dispatch_id, call_log_id, to_number
        )
        
        return DispatchResult(
            room_name=room_name,
            dispatch_id=dispatch_id or "",
            lead_id=lead_id,
            lead_name=lead_name,
            added_context=final_context,
            call_log_id=call_log_id,
        )


# Singleton instance
_call_service: CallService | None = None


def get_call_service() -> CallService:
    global _call_service
    if _call_service is None:
        _call_service = CallService()
    return _call_service


__all__ = [
    "VoiceContext",
    "LeadMetadata",
    "BatchCallEntry",
    "DispatchResult",
    "CallService",
    "get_call_service",
    "_normalize_llm_provider",
    "_normalize_tts_provider",
    "_clean_optional_text",
    "DEFAULT_GLINKS_KB_STORE_IDS",
]
