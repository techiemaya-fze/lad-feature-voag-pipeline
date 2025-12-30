"""
API Services Module.

Contains business logic services used by route handlers.
"""

from .call_service import (
    VoiceContext,
    LeadMetadata,
    BatchCallEntry,
    DispatchResult,
    CallService,
    get_call_service,
    _normalize_llm_provider,
    _normalize_tts_provider,
    _clean_optional_text,
    DEFAULT_GLINKS_KB_STORE_IDS,
)


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
