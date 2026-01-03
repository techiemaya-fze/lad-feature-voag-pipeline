"""
LLM Provider Builder Module.

Factory functions for creating LLM client instances.
Supports runtime switching - each call creates a fresh instance.

Providers:
- Groq: llama-3.3-70b-versatile (default)
- Google Gemini: gemini-2.5-flash
- OpenAI: gpt-4o-mini
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_TRUE_FLAG_VALUES = {"1", "true", "t", "yes", "y", "on"}

DEFAULT_PROVIDER = "groq"

PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "google_gemini": "google",
    "google-gemini": "google",
    "googleai": "google",
    "google-ai": "google",
    "chatgpt": "openai",
    "gpt": "openai",
    "gpt4": "openai",
    "gpt-4": "openai",
    "gpt-4o": "openai",
}

DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}

# Gemini thinking budgets (reasoning depth)
GEMINI_THINKING_BUDGETS: dict[str, int | None] = {
    "no": 0,
    "off": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

# Feature flag for file search
ENABLE_FILE_SEARCH = os.getenv("ENABLE_FILE_SEARCH", "false").lower() in _TRUE_FLAG_VALUES


# =============================================================================
# PROVIDER NORMALIZATION
# =============================================================================

def normalize_provider(value: str | None) -> str:
    """
    Normalize LLM provider name to canonical form.
    
    Args:
        value: Raw provider name
        
    Returns:
        Canonical provider name (groq, google, openai)
    """
    if not value:
        return DEFAULT_PROVIDER
    normalized = value.strip().lower()
    if not normalized:
        return DEFAULT_PROVIDER
    
    alias = PROVIDER_ALIASES.get(normalized)
    if alias:
        return alias
    
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return PROVIDER_ALIASES.get(normalized, normalized)


def resolve_configuration(
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str]:
    """
    Resolve provider and model from overrides and environment.
    
    Priority:
    1. Explicit override
    2. Environment variable
    3. Default
    
    Args:
        provider_override: Provider from request
        model_override: Model from request
        
    Returns:
        Tuple of (provider, model)
    """
    # Check legacy gemini flag
    legacy_flag = os.getenv("USE_GEMINI_LLM")
    legacy_prefers_gemini = bool(legacy_flag and legacy_flag.strip().lower() in _TRUE_FLAG_VALUES)
    
    env_provider = os.getenv("LLM_PROVIDER")
    provider_hint = provider_override or env_provider
    if not provider_hint and legacy_prefers_gemini:
        provider_hint = "google"
    
    provider = normalize_provider(provider_hint)
    
    # Provider-specific model env vars
    provider_models = {
        "google": os.getenv("GEMINI_MODEL"),
        "groq": os.getenv("GROQ_MODEL"),
        "openai": os.getenv("OPENAI_MODEL"),
    }
    
    base_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS[DEFAULT_PROVIDER])
    
    # Model priority: override > LLM_MODEL > provider-specific > default
    model = (
        model_override
        or os.getenv("LLM_MODEL")
        or provider_models.get(provider)
        or base_model
    )
    if not model:
        model = base_model
    
    return provider, model


# =============================================================================
# GROQ PROVIDER
# =============================================================================

def create_groq(model: str) -> Any:
    """
    Create Groq LLM instance.
    
    Uses native Groq plugin if available, falls back to OpenAI-compatible.
    
    Args:
        model: Model name (e.g., "llama-3.3-70b-versatile")
        
    Returns:
        LLM client instance
    """
    from livekit.plugins import openai
    
    try:
        from livekit.plugins import groq as groq_plugin
    except ImportError:
        groq_plugin = None
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required")
    
    if groq_plugin is not None:
        logger.info("Creating Groq LLM (native plugin): %s", model)
        return groq_plugin.LLM(model=model, api_key=api_key)
    
    # Fallback to OpenAI-compatible
    logger.info("Creating Groq LLM (OpenAI compat): %s", model)
    base_url = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    return openai.LLM(base_url=base_url, api_key=api_key, model=model)


# =============================================================================
# GOOGLE GEMINI PROVIDER
# =============================================================================

def create_gemini(
    model: str,
    file_search_store_names: list[str] | None = None,
) -> Any:
    """
    Create Google Gemini LLM instance.
    
    Supports:
    - Thinking/reasoning modes (configurable depth)
    - FileSearch (RAG from uploaded docs)
    - GoogleSearch (web search)
    - UrlContext (fetch URLs)
    
    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        file_search_store_names: Optional RAG store names
        
    Returns:
        LLM client instance
    """
    from livekit.plugins import google
    
    # Configure thinking depth
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "no").strip().lower()
    thinking_budget = GEMINI_THINKING_BUDGETS.get(thinking_level, 0)
    thinking_config = {"thinking_budget": thinking_budget} if thinking_budget is not None else None
    
    if thinking_budget == 0:
        logger.info("Creating Gemini LLM (no thinking): %s", model)
    else:
        logger.info("Creating Gemini LLM (thinking=%s/%d tokens): %s", thinking_level, thinking_budget, model)
    
    # Build native Gemini tools
    gemini_tools = []
    try:
        from google.genai import types as genai_types
        
        # FileSearch for RAG
        if file_search_store_names and ENABLE_FILE_SEARCH:
            gemini_tools.append(genai_types.Tool(
                file_search=genai_types.FileSearch(
                    file_search_store_names=file_search_store_names
                )
            ))
            logger.info("Gemini FileSearch enabled: %d stores", len(file_search_store_names))
        
        # Web search
        if os.getenv("GEMINI_ENABLE_GOOGLE_SEARCH", "false").strip().lower() in _TRUE_FLAG_VALUES:
            gemini_tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
            logger.info("Gemini GoogleSearch enabled")
        
        # URL fetching
        if os.getenv("GEMINI_ENABLE_URL_CONTEXT", "false").strip().lower() in _TRUE_FLAG_VALUES:
            gemini_tools.append(genai_types.Tool(url_context=genai_types.UrlContext()))
            logger.info("Gemini UrlContext enabled")
    except ImportError:
        logger.warning("google.genai.types not available, native tools disabled")
    
    return google.LLM(
        model=model,
        thinking_config=thinking_config,
        gemini_tools=gemini_tools if gemini_tools else None,
    )


# =============================================================================
# OPENAI PROVIDER
# =============================================================================

def create_openai(model: str) -> Any:
    """
    Create OpenAI LLM instance.
    
    Supports custom base URLs for compatible endpoints.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini")
        
    Returns:
        LLM client instance
    """
    from livekit.plugins import openai
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required")
    
    kwargs: dict[str, str] = {"model": model, "api_key": api_key}
    
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip()
        logger.info("Creating OpenAI LLM: %s @ %s", model, base_url)
    else:
        logger.info("Creating OpenAI LLM: %s", model)
    
    return openai.LLM(**kwargs)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_llm(
    provider: str | None = None,
    model: str | None = None,
    file_search_store_names: list[str] | None = None,
) -> Any:
    """
    Create LLM instance for the specified provider.
    
    This is a factory function - each call creates a new instance,
    allowing runtime provider switching.
    
    Args:
        provider: Provider name (groq, google, openai)
        model: Model override
        file_search_store_names: RAG stores for Gemini
        
    Returns:
        LLM client instance
    """
    resolved_provider, resolved_model = resolve_configuration(provider, model)
    
    if resolved_provider == "google":
        return create_gemini(resolved_model, file_search_store_names)
    elif resolved_provider == "openai":
        return create_openai(resolved_model)
    else:
        return create_groq(resolved_model)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Factory
    "create_llm",
    # Individual providers
    "create_groq",
    "create_gemini",
    "create_openai",
    # Configuration
    "normalize_provider",
    "resolve_configuration",
    # Constants
    "DEFAULT_PROVIDER",
    "DEFAULT_MODELS",
    "PROVIDER_ALIASES",
]
