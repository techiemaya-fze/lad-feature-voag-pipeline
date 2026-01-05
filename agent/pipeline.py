"""
Agent Pipeline Module.

Handles TTS engine configuration, LLM provider setup, and language mapping.
Extracted from entry.py for modular architecture.

Components:
- TTS Engine Builders: Cartesia, Google, Gemini, ElevenLabs, Rime, SmallestAI  
- LLM Configuration: Provider resolution, model selection
- Language Mapping: Accent to language code conversion
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import tokenize
from livekit.agents import tts as agent_tts
from livekit.plugins import cartesia, google, openai

# Optional TTS providers
try:
    from livekit.plugins import elevenlabs
except ImportError:
    elevenlabs = None

try:
    from livekit.plugins import rime
except ImportError:
    rime = None

try:
    from livekit.plugins import smallestai
except ImportError:
    smallestai = None

try:
    from livekit.plugins import groq as groq_plugin
except ImportError:
    groq_plugin = None

# Custom TTS
from tts.google_chirp_streaming import GoogleChirpVoiceConfig, create_google_chirp_tts

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_TRUE_FLAG_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_FLAG_VALUES = {"0", "false", "f", "no", "n", "off"}

# Cartesia voice language hints
_CARTESIA_VOICE_LANGUAGE_HINTS: dict[str, str] = {
    "791d5162-d5eb-40f0-8189-f19db44611d8": "hi",  # Hindi Male Voice
}

# LLM provider configuration
_DEFAULT_LLM_PROVIDER = "groq"
_LLM_PROVIDER_ALIASES: dict[str, str] = {
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
_DEFAULT_LLM_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}

# Gemini thinking budgets
_GEMINI_THINKING_BUDGETS: dict[str, int | None] = {
    "no": 0,
    "off": 0,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
}

# Feature flags
ENABLE_FILE_SEARCH = os.getenv("ENABLE_FILE_SEARCH", "false").lower() in _TRUE_FLAG_VALUES


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _pick(*values: str | None, default: str = "") -> str:
    """Return first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _coerce_int(value: str | None) -> int | None:
    """Parse string to int, returning None on failure."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        logger.warning("Invalid integer: '%s'", value)
        return None


def _coerce_float(value: str | None) -> float | None:
    """Parse string to float, returning None on failure."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        logger.warning("Invalid float: '%s'", value)
        return None


def _coerce_bool(value: str | None) -> bool | None:
    """Parse string to bool, returning None on failure."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text in _TRUE_FLAG_VALUES:
        return True
    if text in _FALSE_FLAG_VALUES:
        return False
    logger.warning("Invalid boolean: '%s'", value)
    return None


def _coerce_int_list(value: str | None) -> list[int] | None:
    """Parse comma-separated string to list of ints."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    result: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        converted = _coerce_int(chunk)
        if converted is not None:
            result.append(converted)
    return result or None


def _coalesce_text(*values: str | None, default: str = "") -> str:
    """Return first non-empty string from values."""
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return default


# =============================================================================
# LANGUAGE MAPPING
# =============================================================================

def derive_cartesia_language(accent: str | None, env_default: str | None) -> str | None:
    """
    Map voice accent to Cartesia language code.
    
    Args:
        accent: Accent descriptor (e.g., "hindi", "hi-IN")
        env_default: Fallback from environment
        
    Returns:
        Two-letter language code or None
    """
    if accent:
        normalized = accent.strip().lower().replace("_", "-")
        if normalized:
            mapping = {
                "hindi-in": "hi", "hi-in": "hi",
                "hindi": "hi", "hi": "hi",
            }
            if normalized in mapping:
                return mapping[normalized]
            if "hindi" in normalized:
                return "hi"
            
            parts = normalized.split("-", 1)
            if parts and len(parts[0]) in {2, 3} and parts[0].isalpha():
                return parts[0].lower()
    return env_default


def derive_stt_language(accent: str | None, default_language: str) -> str:
    """
    Map voice accent to Deepgram STT language code.
    
    Args:
        accent: Accent descriptor
        default_language: Fallback language
        
    Returns:
        Language code for STT
    """
    if not accent:
        return default_language
    
    normalized = accent.strip().lower().replace("_", "-")
    if not normalized:
        return default_language
    
    mapping = {
        "hindi-in": "hi", "hi-in": "hi",
        "hindi": "hi", "hi": "hi",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "hindi" in normalized:
        return "hi"
    
    base = normalized.split("-", 1)[0]
    if len(base) in {2, 3} and base.isalpha():
        return base
    
    return default_language


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

def normalize_llm_provider(value: str | None) -> str:
    """Normalize LLM provider name to canonical form."""
    if not value:
        return _DEFAULT_LLM_PROVIDER
    normalized = value.strip().lower()
    if not normalized:
        return _DEFAULT_LLM_PROVIDER
    
    alias = _LLM_PROVIDER_ALIASES.get(normalized)
    if alias:
        return alias
    
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return _LLM_PROVIDER_ALIASES.get(normalized, normalized)


def resolve_llm_configuration(
    provider_override: str | None,
    model_override: str | None,
) -> tuple[str, str]:
    """
    Determine LLM provider and model.
    
    Args:
        provider_override: Override from call config
        model_override: Override from call config
        
    Returns:
        Tuple of (provider, model)
    """
    legacy_flag = os.getenv("USE_GEMINI_LLM")
    legacy_prefers_gemini = bool(legacy_flag and legacy_flag.strip().lower() in _TRUE_FLAG_VALUES)
    
    env_provider = os.getenv("LLM_PROVIDER")
    provider_hint = provider_override or env_provider
    if not provider_hint and legacy_prefers_gemini:
        provider_hint = "google"
    
    provider = normalize_llm_provider(provider_hint)
    
    provider_specific_models = {
        "google": os.getenv("GEMINI_MODEL"),
        "groq": os.getenv("GROQ_MODEL"),
        "openai": os.getenv("OPENAI_MODEL"),
    }
    
    base_model = _DEFAULT_LLM_MODELS.get(provider, _DEFAULT_LLM_MODELS[_DEFAULT_LLM_PROVIDER])
    model = _coalesce_text(
        model_override,
        os.getenv("LLM_MODEL"),
        provider_specific_models.get(provider),
        base_model,
    )
    if not model:
        model = base_model
    
    return provider, model


def create_llm_instance(
    provider: str,
    model: str,
    file_search_store_names: list[str] | None = None,
) -> Any:
    """
    Create LLM client instance for provider.
    
    Args:
        provider: Normalized provider name
        model: Model identifier
        file_search_store_names: Optional FileSearch stores for Gemini RAG
        
    Returns:
        LLM client instance
    """
    normalized = normalize_llm_provider(provider)
    
    if normalized == "google":
        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "no").strip().lower()
        thinking_budget = _GEMINI_THINKING_BUDGETS.get(thinking_level, 0)
        thinking_config = {"thinking_budget": thinking_budget} if thinking_budget is not None else None
        
        if thinking_budget == 0:
            logger.info("Gemini thinking disabled")
        else:
            logger.info(f"Gemini thinking: {thinking_level} ({thinking_budget} tokens)")
        
        gemini_tools = []
        try:
            from google.genai import types as genai_types
            
            if file_search_store_names and ENABLE_FILE_SEARCH:
                gemini_tools.append(genai_types.Tool(
                    file_search=genai_types.FileSearch(
                        file_search_store_names=file_search_store_names
                    )
                ))
                logger.info("Enabled Gemini FileSearch with %d store(s)", len(file_search_store_names))
            
            if os.getenv("GEMINI_ENABLE_GOOGLE_SEARCH", "false").strip().lower() in _TRUE_FLAG_VALUES:
                gemini_tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
                logger.info("Enabled Gemini GoogleSearch")
            
            if os.getenv("GEMINI_ENABLE_URL_CONTEXT", "false").strip().lower() in _TRUE_FLAG_VALUES:
                gemini_tools.append(genai_types.Tool(url_context=genai_types.UrlContext()))
                logger.info("Enabled Gemini UrlContext")
        except ImportError:
            logger.warning("google.genai.types not available")
        
        # Try with gemini_tools first (newer versions), fallback without it
        try:
            return google.LLM(
                model=model,
                thinking_config=thinking_config,
                gemini_tools=gemini_tools if gemini_tools else None,
            )
        except TypeError:
            # Older livekit-plugins-google version doesn't support gemini_tools
            logger.warning("gemini_tools not supported in installed version, continuing without native tools")
            return google.LLM(
                model=model,
                thinking_config=thinking_config,
            )
    
    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAI")
        kwargs: dict[str, str] = {"model": model, "api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url and base_url.strip():
            kwargs["base_url"] = base_url.strip()
        return openai.LLM(**kwargs)
    
    # Default: Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required for Groq")
    
    if groq_plugin is not None:
        logger.info("Using native Groq plugin")
        return groq_plugin.LLM(model=model, api_key=api_key)
    
    logger.info("Using OpenAI plugin with Groq endpoint")
    base_url = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
    return openai.LLM(base_url=base_url, api_key=api_key, model=model)


# =============================================================================
# TTS ENGINE BUILDER
# =============================================================================

def build_tts_engine(
    provider: str | None,
    *,
    default_voice_id: str | None,
    overrides: dict[str, str] | None = None,
    accent: str | None = None,
) -> tuple[Any, dict[str, str]]:
    """
    Create TTS engine for the specified provider.
    
    Args:
        provider: TTS provider name (cartesia, google, gemini, elevenlabs, rime, smallestai)
        default_voice_id: Default voice ID
        overrides: Provider-specific configuration
        accent: Language/accent hint
        
    Returns:
        Tuple of (TTS engine, configuration details dict)
    """
    normalized = (provider or "cartesia").strip().lower()
    alias_map = {
        "google-tts": "google", "google_tts": "google",
        "gemini-tts": "gemini", "gemini_tts": "gemini",
        "cartesia_tts": "cartesia",
        "google-chirp": "google_chirp", "google_chirp": "google_chirp",
        "elevenlabs_tts": "elevenlabs", "eleven_labs": "elevenlabs",
        "11labs": "elevenlabs", "11-labs": "elevenlabs",
        "rime_tts": "rime", "rime-ai": "rime",
        "smallestai": "smallestai", "smallest_ai": "smallestai",
        "smallest-ai": "smallestai", "waves": "smallestai",
    }
    normalized = alias_map.get(normalized, normalized)
    
    overrides = overrides or {}
    details: dict[str, str] = {"provider": normalized}
    
    # Google Chirp
    if normalized == "google_chirp":
        chirp_config = GoogleChirpVoiceConfig.from_overrides(overrides, accent=accent)
        engine = create_google_chirp_tts(chirp_config)
        details.update({
            "model": "chirp3",
            "language": chirp_config.language_code,
            "voice": chirp_config.voice_name,
            "speaking_rate": f"{chirp_config.speaking_rate:.2f}",
            "pitch": str(chirp_config.pitch),
            "sample_rate": str(chirp_config.sample_rate_hz),
        })
        return engine, details
    
    # Standard Google TTS
    if normalized == "google":
        language = _pick(overrides.get("language"), os.getenv("GOOGLE_TTS_LANGUAGE"), default="en-US")
        voice_name = _pick(overrides.get("voice_name"), overrides.get("voice"), os.getenv("GOOGLE_TTS_VOICE_NAME"))
        speaking_rate = _pick(overrides.get("speaking_rate"), os.getenv("GOOGLE_TTS_SPEAKING_RATE"), default="1.0")
        pitch = _pick(overrides.get("pitch"), os.getenv("GOOGLE_TTS_PITCH"), default="0")
        
        kwargs_google: dict[str, Any] = {"use_streaming": True}
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            storage_key = os.getenv("GCS_CREDENTIALS_JSON")
            if storage_key:
                candidate = os.path.abspath(storage_key) if not os.path.isabs(storage_key) else storage_key
                if os.path.exists(candidate):
                    credentials_path = candidate
        
        if credentials_path and os.path.exists(credentials_path):
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
            kwargs_google["credentials_file"] = credentials_path
            details["credentials"] = "file"
        
        if language:
            kwargs_google["language"] = language
        if voice_name:
            kwargs_google["voice_name"] = voice_name
        
        try:
            kwargs_google["speaking_rate"] = float(speaking_rate)
        except ValueError:
            pass
        try:
            kwargs_google["pitch"] = int(pitch)
        except ValueError:
            pass
        
        details["language"] = language
        if voice_name:
            details["voice"] = voice_name
        
        return google.TTS(**kwargs_google), details
    
    # Gemini TTS
    if normalized == "gemini":
        model = _pick(overrides.get("model"), os.getenv("GEMINI_TTS_MODEL"), default="gemini-2.5-flash-preview-tts")
        voice_name = _pick(overrides.get("voice"), os.getenv("GEMINI_TTS_VOICE_NAME"))
        instructions = _pick(overrides.get("instructions"), os.getenv("GEMINI_TTS_INSTRUCTIONS"))
        
        kwargs: dict[str, str] = {"model": model}
        if voice_name:
            kwargs["voice_name"] = voice_name
            details["voice"] = voice_name
        if instructions:
            kwargs["instructions"] = instructions
        
        details["model"] = model
        return google.beta.GeminiTTS(**kwargs), details
    
    # ElevenLabs
    if normalized == "elevenlabs":
        if elevenlabs is None:
            raise RuntimeError("ElevenLabs plugin not installed")
        
        model = _pick(overrides.get("model"), os.getenv("ELEVENLABS_TTS_MODEL"), default="eleven_turbo_v2_5")
        voice_id = _pick(overrides.get("voice_id"), overrides.get("voice"), os.getenv("ELEVENLABS_TTS_VOICE_ID"))
        language = _pick(overrides.get("language"), os.getenv("ELEVENLABS_TTS_LANGUAGE"))
        
        kwargs_eleven: dict[str, Any] = {"model": model}
        if voice_id:
            kwargs_eleven["voice_id"] = voice_id
        if language:
            kwargs_eleven["language"] = language
        
        api_key = _pick(overrides.get("api_key"), os.getenv("ELEVEN_API_KEY"))
        if api_key:
            kwargs_eleven["api_key"] = api_key
        
        details.update({"model": model, "voice": voice_id or "default"})
        if language:
            details["language"] = language
        
        return elevenlabs.TTS(**kwargs_eleven), details
    
    # Rime
    if normalized == "rime":
        if rime is None:
            raise RuntimeError("Rime plugin not installed")
        
        model = _pick(overrides.get("model"), os.getenv("RIME_TTS_MODEL"), default="mist")
        speaker = _pick(overrides.get("voice"), overrides.get("speaker"), os.getenv("RIME_TTS_SPEAKER"), default="lagoon")
        
        kwargs_rime: dict[str, Any] = {"model": model}
        if speaker:
            kwargs_rime["speaker"] = speaker
            details["voice"] = speaker
        
        details["model"] = model
        return rime.TTS(**kwargs_rime), details
    
    # SmallestAI
    if normalized == "smallestai":
        if smallestai is None:
            raise RuntimeError("SmallestAI plugin not installed")
        
        model = _pick(overrides.get("model"), os.getenv("SMALLEST_TTS_MODEL"), default="lightning-large")
        voice_id = _pick(overrides.get("voice"), os.getenv("SMALLEST_TTS_VOICE_ID"), default="irisha")
        language = _pick(overrides.get("language"), os.getenv("SMALLEST_TTS_LANGUAGE"), default="en")
        
        kwargs_smallest: dict[str, Any] = {"model": model}
        if voice_id:
            kwargs_smallest["voice_id"] = voice_id
        if language:
            kwargs_smallest["language"] = language
        
        details.update({"model": model, "voice": voice_id, "language": language})
        return smallestai.TTS(**kwargs_smallest), details
    
    # Default: Cartesia
    if normalized not in {"cartesia"}:
        logger.warning("Unknown TTS provider '%s', using Cartesia", provider)
        normalized = "cartesia"
        details["provider"] = normalized
    
    model = _pick(overrides.get("model"), os.getenv("CARTESIA_TTS_MODEL"), default="sonic-2")
    voice_choice = _pick(overrides.get("voice"), default_voice_id, os.getenv("CARTESIA_TTS_VOICE_ID"))
    
    voice_lang_hint = _CARTESIA_VOICE_LANGUAGE_HINTS.get(voice_choice.strip().lower()) if voice_choice else None
    language_default = derive_cartesia_language(accent, os.getenv("CARTESIA_TTS_LANGUAGE"))
    language_override = _pick(overrides.get("language"), voice_lang_hint, language_default, default="en")
    
    accent_key = (accent or "").strip().lower()
    
    kwargs_ct: dict[str, Any] = {
        "api_key": os.getenv("CARTESIA_API_KEY"),
        "model": model,
        "language": language_override,
    }
    
    if accent_key in {"hindi", "hi", "hindi-in", "hi-in"}:
        kwargs_ct["tokenizer"] = tokenize.basic.SentenceTokenizer()
    
    api_version = _pick(overrides.get("api_version"), os.getenv("CARTESIA_TTS_API_VERSION"))
    if accent_key in {"hindi", "hi", "hindi-in", "hi-in"} and not api_version:
        api_version = "2025-04-16"
    if api_version:
        kwargs_ct["api_version"] = api_version
    
    if voice_choice:
        kwargs_ct["voice"] = voice_choice
    
    # Stream pacer configuration
    buffer_seconds = 0.35
    max_text_length = 220
    buffer_env = os.getenv("CARTESIA_TTS_BUFFER_SECONDS")
    text_env = os.getenv("CARTESIA_TTS_MAX_TEXT_LENGTH")
    
    if buffer_env:
        try:
            buffer_seconds = max(float(buffer_env), 0.0)
        except ValueError:
            pass
    if text_env:
        try:
            max_text_length = max(int(text_env), 1)
        except ValueError:
            pass
    
    if buffer_seconds > 0 and accent_key not in {"hindi", "hi", "hindi-in", "hi-in"}:
        kwargs_ct["text_pacing"] = agent_tts.SentenceStreamPacer(
            min_remaining_audio=buffer_seconds,
            max_text_length=max_text_length,
        )
        details["buffer_seconds"] = f"{buffer_seconds:.2f}"
        details["max_text_length"] = str(max_text_length)
    
    details["model"] = model
    if voice_choice:
        details["voice"] = voice_choice
    if language_override:
        details["language"] = language_override
    if accent:
        details["accent"] = accent
    if api_version:
        details["api_version"] = api_version
    
    return cartesia.TTS(**kwargs_ct), details


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # TTS
    "build_tts_engine",
    # LLM
    "resolve_llm_configuration",
    "create_llm_instance",
    "normalize_llm_provider",
    # Language
    "derive_cartesia_language",
    "derive_stt_language",
    # Constants
    "ENABLE_FILE_SEARCH",
    "_TRUE_FLAG_VALUES",
    "_FALSE_FLAG_VALUES",
]
