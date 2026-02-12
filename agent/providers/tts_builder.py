"""
TTS Provider Builder Module.

Factory functions for creating Text-to-Speech engine instances.
Supports runtime switching - each call creates a fresh instance.

Providers:
- Cartesia (default): Fast streaming, sonic-2
- Google TTS: Standard Google Cloud TTS
- Google Chirp: Natural voices
- Gemini TTS: AI-powered TTS
- ElevenLabs: Premium voices
- Rime: Ultra-low latency
- SmallestAI (Waves): Humanistic voices
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import tokenize
from livekit.agents import tts as agent_tts

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_TRUE_FLAG_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_FLAG_VALUES = {"0", "false", "f", "no", "n", "off"}

DEFAULT_PROVIDER = "cartesia"

PROVIDER_ALIASES: dict[str, str] = {
    "google-tts": "google",
    "google_tts": "google",
    "gemini-tts": "gemini",
    "gemini_tts": "gemini",
    "cartesia_tts": "cartesia",
    "google-chirp": "google_chirp",
    "google_chirp": "google_chirp",
    "elevenlabs_tts": "elevenlabs",
    "eleven_labs": "elevenlabs",
    "11labs": "elevenlabs",
    "11-labs": "elevenlabs",
    "rime_tts": "rime",
    "rime-ai": "rime",
    "smallestai": "smallestai",
    "smallest_ai": "smallestai",
    "smallest-ai": "smallestai",
    "waves": "smallestai",
    "sarvam_tts": "sarvam",
    "sarvam-ai": "sarvam",
    "sarvam_ai": "sarvam",
}

# Cartesia voice language hints
CARTESIA_VOICE_LANGUAGE_HINTS: dict[str, str] = {
    "791d5162-d5eb-40f0-8189-f19db44611d8": "hi",  # Hindi Male
}


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
    """Parse string to int."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _coerce_float(value: str | None) -> float | None:
    """Parse string to float."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_bool(value: str | None) -> bool | None:
    """Parse string to bool."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    if text in _TRUE_FLAG_VALUES:
        return True
    if text in _FALSE_FLAG_VALUES:
        return False
    return None


def normalize_provider(value: str | None) -> str:
    """Normalize TTS provider name."""
    if not value:
        return DEFAULT_PROVIDER
    normalized = value.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def derive_language(accent: str | None, env_default: str | None) -> str | None:
    """Map accent to language code."""
    if accent:
        normalized = accent.strip().lower().replace("_", "-")
        if normalized:
            mapping = {"hindi-in": "hi", "hi-in": "hi", "hindi": "hi", "hi": "hi"}
            if normalized in mapping:
                return mapping[normalized]
            if "hindi" in normalized:
                return "hi"
            parts = normalized.split("-", 1)
            if parts and len(parts[0]) in {2, 3} and parts[0].isalpha():
                return parts[0].lower()
    return env_default


# =============================================================================
# CARTESIA PROVIDER
# =============================================================================

def create_cartesia(
    voice_id: str | None = None,
    accent: str | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """
    Create Cartesia TTS instance.
    
    Args:
        voice_id: Voice ID
        accent: Language/accent hint
        overrides: Additional config
        
    Returns:
        Tuple of (TTS engine, details dict)
    """
    from livekit.plugins import cartesia
    
    overrides = overrides or {}
    details: dict[str, str] = {"provider": "cartesia"}
    
    model = _pick(overrides.get("model"), os.getenv("CARTESIA_TTS_MODEL"), default="sonic-2")
    voice_choice = _pick(overrides.get("voice"), voice_id, os.getenv("CARTESIA_TTS_VOICE_ID"))
    
    voice_lang_hint = CARTESIA_VOICE_LANGUAGE_HINTS.get(voice_choice.strip().lower()) if voice_choice else None
    language_default = derive_language(accent, os.getenv("CARTESIA_TTS_LANGUAGE"))
    language = _pick(overrides.get("language"), voice_lang_hint, language_default, default="en")
    
    accent_key = (accent or "").strip().lower()
    
    kwargs: dict[str, Any] = {
        "api_key": os.getenv("CARTESIA_API_KEY"),
        "model": model,
        "language": language,
    }
    
    if accent_key in {"hindi", "hi", "hindi-in", "hi-in"}:
        kwargs["tokenizer"] = tokenize.basic.SentenceTokenizer()
    
    api_version = _pick(overrides.get("api_version"), os.getenv("CARTESIA_TTS_API_VERSION"))
    if accent_key in {"hindi", "hi", "hindi-in", "hi-in"} and not api_version:
        api_version = "2025-04-16"
    if api_version:
        kwargs["api_version"] = api_version
    
    if voice_choice:
        kwargs["voice"] = voice_choice
    
    # Stream pacer
    buffer = float(os.getenv("CARTESIA_TTS_BUFFER_SECONDS", "0.35"))
    max_text = int(os.getenv("CARTESIA_TTS_MAX_TEXT_LENGTH", "220"))
    
    if buffer > 0 and accent_key not in {"hindi", "hi", "hindi-in", "hi-in"}:
        kwargs["text_pacing"] = agent_tts.SentenceStreamPacer(
            min_remaining_audio=buffer,
            max_text_length=max_text,
        )
        details["buffer_seconds"] = f"{buffer:.2f}"
    
    details.update({"model": model, "language": language})
    if voice_choice:
        details["voice"] = voice_choice
    if accent:
        details["accent"] = accent
    
    logger.info("Creating Cartesia TTS: %s, voice=%s, lang=%s", model, voice_choice, language)
    return cartesia.TTS(**kwargs), details


# =============================================================================
# GOOGLE TTS PROVIDER
# =============================================================================

def create_google(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create Google Cloud TTS instance."""
    from livekit.plugins import google
    
    overrides = overrides or {}
    details: dict[str, str] = {"provider": "google"}
    
    language = _pick(overrides.get("language"), os.getenv("GOOGLE_TTS_LANGUAGE"), default="en-US")
    voice_name = _pick(overrides.get("voice_name"), overrides.get("voice"), os.getenv("GOOGLE_TTS_VOICE_NAME"))
    speaking_rate = _pick(overrides.get("speaking_rate"), os.getenv("GOOGLE_TTS_SPEAKING_RATE"), default="1.0")
    pitch = _pick(overrides.get("pitch"), os.getenv("GOOGLE_TTS_PITCH"), default="0")
    
    kwargs: dict[str, Any] = {"use_streaming": True}
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        storage_key = os.getenv("GCS_CREDENTIALS_JSON")
        if storage_key and os.path.exists(storage_key):
            credentials_path = os.path.abspath(storage_key)
    
    if credentials_path and os.path.exists(credentials_path):
        kwargs["credentials_file"] = credentials_path
    
    if language:
        kwargs["language"] = language
    if voice_name:
        kwargs["voice_name"] = voice_name
    
    try:
        kwargs["speaking_rate"] = float(speaking_rate)
        kwargs["pitch"] = int(pitch)
    except ValueError:
        pass
    
    details["language"] = language
    if voice_name:
        details["voice"] = voice_name
    
    logger.info("Creating Google TTS: lang=%s, voice=%s", language, voice_name)
    return google.TTS(**kwargs), details


# =============================================================================
# GOOGLE CHIRP PROVIDER
# =============================================================================

def create_google_chirp(
    accent: str | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """Create Google Chirp TTS instance."""
    from tts.google_chirp_streaming import GoogleChirpVoiceConfig, create_google_chirp_tts
    
    overrides = overrides or {}
    
    chirp_config = GoogleChirpVoiceConfig.from_overrides(overrides, accent=accent)
    engine = create_google_chirp_tts(chirp_config)
    
    details: dict[str, str] = {
        "provider": "google_chirp",
        "model": "chirp3",
        "language": chirp_config.language_code,
        "voice": chirp_config.voice_name,
        "speaking_rate": f"{chirp_config.speaking_rate:.2f}",
        "pitch": str(chirp_config.pitch),
    }
    
    logger.info("Creating Google Chirp TTS: voice=%s, lang=%s", chirp_config.voice_name, chirp_config.language_code)
    return engine, details


# =============================================================================
# GEMINI TTS PROVIDER
# =============================================================================

def create_gemini(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create Gemini TTS instance."""
    from livekit.plugins import google
    
    overrides = overrides or {}
    
    model = _pick(overrides.get("model"), os.getenv("GEMINI_TTS_MODEL"), default="gemini-2.5-flash-preview-tts")
    voice_name = _pick(overrides.get("voice"), os.getenv("GEMINI_TTS_VOICE_NAME"))
    instructions = _pick(overrides.get("instructions"), os.getenv("GEMINI_TTS_INSTRUCTIONS"))
    
    kwargs: dict[str, str] = {"model": model}
    if voice_name:
        kwargs["voice_name"] = voice_name
    if instructions:
        kwargs["instructions"] = instructions
    
    details: dict[str, str] = {"provider": "gemini", "model": model}
    if voice_name:
        details["voice"] = voice_name
    
    logger.info("Creating Gemini TTS: %s", model)
    return google.beta.GeminiTTS(**kwargs), details


# =============================================================================
# ELEVENLABS PROVIDER
# =============================================================================

def create_elevenlabs(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create ElevenLabs TTS instance with optional speed/stability settings."""
    try:
        from livekit.plugins import elevenlabs
    except ImportError:
        raise RuntimeError("ElevenLabs plugin not installed. Run: uv add livekit-plugins-elevenlabs")
    
    overrides = overrides or {}
    
    model = _pick(overrides.get("model"), os.getenv("ELEVENLABS_TTS_MODEL"), default="eleven_turbo_v2_5")
    voice_id = _pick(overrides.get("voice_id"), overrides.get("voice"), os.getenv("ELEVENLABS_TTS_VOICE_ID"))
    language = _pick(overrides.get("language"), os.getenv("ELEVENLABS_TTS_LANGUAGE"))
    
    kwargs: dict[str, Any] = {"model": model}
    if voice_id:
        kwargs["voice_id"] = voice_id
    if language:
        kwargs["language"] = language
    
    api_key = os.getenv("ELEVEN_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    
    # Build VoiceSettings for speed/stability/similarity control
    # Speed: 0.8-1.2 (default 1.0), Stability: 0.0-1.0 (default 0.71), Similarity: 0.0-1.0 (default 0.5)
    voice_settings_kwargs: dict[str, float] = {}
    
    speed = _coerce_float(overrides.get("speed"))
    if speed is not None:
        voice_settings_kwargs["speed"] = max(0.8, min(1.2, speed))  # Clamp to valid range
    
    stability = _coerce_float(overrides.get("stability"))
    if stability is not None:
        voice_settings_kwargs["stability"] = max(0.0, min(1.0, stability))
    
    similarity = _coerce_float(overrides.get("similarity_boost")) or _coerce_float(overrides.get("similarity"))
    if similarity is not None:
        voice_settings_kwargs["similarity_boost"] = max(0.0, min(1.0, similarity))
    
    if voice_settings_kwargs:
        try:
            voice_settings = elevenlabs.VoiceSettings(**voice_settings_kwargs)
            kwargs["voice_settings"] = voice_settings
            logger.info("ElevenLabs VoiceSettings: %s", voice_settings_kwargs)
        except Exception as exc:
            logger.warning("Failed to create VoiceSettings: %s", exc)
    
    details: dict[str, str] = {"provider": "elevenlabs", "model": model, "voice": voice_id or "default"}
    if language:
        details["language"] = language
    if voice_settings_kwargs:
        details["speed"] = str(voice_settings_kwargs.get("speed", 1.0))
    
    logger.info("Creating ElevenLabs TTS: %s, voice=%s", model, voice_id)
    return elevenlabs.TTS(**kwargs), details


# =============================================================================
# RIME PROVIDER
# =============================================================================

def create_rime(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create Rime TTS instance."""
    try:
        from livekit.plugins import rime
    except ImportError:
        raise RuntimeError("Rime plugin not installed. Run: uv add livekit-plugins-rime")
    
    overrides = overrides or {}
    
    model = _pick(overrides.get("model"), os.getenv("RIME_TTS_MODEL"), default="mist")
    speaker = _pick(overrides.get("voice"), overrides.get("speaker"), os.getenv("RIME_TTS_SPEAKER"), default="lagoon")
    
    kwargs: dict[str, Any] = {"model": model}
    if speaker:
        kwargs["speaker"] = speaker
    
    details: dict[str, str] = {"provider": "rime", "model": model, "voice": speaker}
    
    logger.info("Creating Rime TTS: %s, speaker=%s", model, speaker)
    return rime.TTS(**kwargs), details


# =============================================================================
# SMALLESTAI PROVIDER
# =============================================================================

def create_smallestai(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create SmallestAI (Waves) TTS instance."""
    try:
        from livekit.plugins import smallestai
    except ImportError:
        raise RuntimeError("SmallestAI plugin not installed. Run: uv add livekit-plugins-smallestai")
    
    overrides = overrides or {}
    
    model = _pick(overrides.get("model"), os.getenv("SMALLEST_TTS_MODEL"), default="lightning-large")
    voice_id = _pick(overrides.get("voice"), os.getenv("SMALLEST_TTS_VOICE_ID"), default="irisha")
    language = _pick(overrides.get("language"), os.getenv("SMALLEST_TTS_LANGUAGE"), default="en")
    
    kwargs: dict[str, Any] = {"model": model}
    if voice_id:
        kwargs["voice_id"] = voice_id
    if language:
        kwargs["language"] = language
    
    details: dict[str, str] = {"provider": "smallestai", "model": model, "voice": voice_id, "language": language}
    
    logger.info("Creating SmallestAI TTS: %s, voice=%s", model, voice_id)
    return smallestai.TTS(**kwargs), details


# =============================================================================
# SARVAM PROVIDER
# =============================================================================

def create_sarvam(overrides: dict[str, str] | None = None) -> tuple[Any, dict[str, str]]:
    """Create Sarvam TTS instance for Indian language synthesis."""
    try:
        from livekit.plugins import sarvam
    except ImportError:
        raise RuntimeError("Sarvam plugin not installed. Run: uv add livekit-plugins-sarvam")
    
    overrides = overrides or {}
    
    model = _pick(overrides.get("model"), os.getenv("SARVAM_TTS_MODEL"), default="bulbul:v2")
    speaker = _pick(overrides.get("speaker"), overrides.get("voice"), os.getenv("SARVAM_TTS_SPEAKER"), default="anushka")
    target_language_code = _pick(
        overrides.get("target_language_code"),
        overrides.get("language"),
        os.getenv("SARVAM_TTS_LANGUAGE"),
        default="hi-IN",
    )
    
    kwargs: dict[str, Any] = {
        "model": model,
        "speaker": speaker,
        "target_language_code": target_language_code,
    }
    
    api_key = os.getenv("SARVAM_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    
    # Speech sample rate: 8000 (telephony), 16000, 22050 (default), 24000
    speech_sample_rate = _coerce_int(overrides.get("speech_sample_rate"))
    if speech_sample_rate is None:
        env_rate = os.getenv("SARVAM_TTS_SAMPLE_RATE")
        speech_sample_rate = _coerce_int(env_rate)
    if speech_sample_rate is not None and speech_sample_rate in (8000, 16000, 22050, 24000):
        kwargs["speech_sample_rate"] = speech_sample_rate
    
    # Enable text preprocessing (number normalization, code-mixed text handling)
    enable_preprocessing_raw = overrides.get("enable_preprocessing")
    if enable_preprocessing_raw is not None:
        kwargs["enable_preprocessing"] = str(enable_preprocessing_raw).lower() in ("true", "1", "yes")
    
    # Optional float parameters
    pitch = _coerce_float(overrides.get("pitch"))
    if pitch is not None:
        kwargs["pitch"] = max(-20.0, min(20.0, pitch))
    
    pace = _coerce_float(overrides.get("pace"))
    if pace is not None:
        kwargs["pace"] = max(0.5, min(2.0, pace))
    
    loudness = _coerce_float(overrides.get("loudness"))
    if loudness is not None:
        kwargs["loudness"] = max(0.5, min(2.0, loudness))
    
    details: dict[str, str] = {
        "provider": "sarvam",
        "model": model,
        "speaker": speaker,
        "language": target_language_code,
    }
    if speech_sample_rate is not None:
        details["speech_sample_rate"] = str(speech_sample_rate)
    if pitch is not None:
        details["pitch"] = str(pitch)
    if pace is not None:
        details["pace"] = str(pace)
    if loudness is not None:
        details["loudness"] = str(loudness)
    
    logger.info("Creating Sarvam TTS: %s, speaker=%s, lang=%s, sample_rate=%s", model, speaker, target_language_code, kwargs.get("speech_sample_rate", 22050))
    return sarvam.TTS(**kwargs), details


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_tts(
    provider: str | None = None,
    *,
    voice_id: str | None = None,
    accent: str | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str]]:
    """
    Create TTS engine for the specified provider.
    
    Factory function - each call creates a new instance,
    allowing runtime provider switching.
    
    Args:
        provider: TTS provider name
        voice_id: Voice ID (for Cartesia)
        accent: Language/accent hint
        overrides: Provider-specific config
        
    Returns:
        Tuple of (TTS engine, configuration details)
    """
    normalized = normalize_provider(provider)
    overrides = overrides or {}
    
    if normalized == "google_chirp":
        return create_google_chirp(accent, overrides)
    elif normalized == "google":
        return create_google(overrides)
    elif normalized == "gemini":
        return create_gemini(overrides)
    elif normalized == "elevenlabs":
        return create_elevenlabs(overrides)
    elif normalized == "rime":
        return create_rime(overrides)
    elif normalized == "smallestai":
        return create_smallestai(overrides)
    elif normalized == "sarvam":
        return create_sarvam(overrides)
    else:
        # Default to Cartesia
        if normalized not in {"cartesia"}:
            logger.warning("Unknown TTS provider '%s', using Cartesia", provider)
        return create_cartesia(voice_id, accent, overrides)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Factory
    "create_tts",
    # Individual providers
    "create_cartesia",
    "create_google",
    "create_google_chirp",
    "create_gemini",
    "create_elevenlabs",
    "create_rime",
    "create_smallestai",
    "create_sarvam",
    # Utilities
    "normalize_provider",
    "derive_language",
    # Constants
    "DEFAULT_PROVIDER",
    "PROVIDER_ALIASES",
]
