"""
Realtime Model Provider Builder Module.

Factory functions for creating Realtime Model instances that combine
STT, LLM, and TTS into a single connection.

Providers:
- Ultravox: fixie-ai/ultravox (default)
- Gemini Live: gemini-2.5-flash-native-audio-preview-12-2025
- (Future) OpenAI Realtime: gpt-4o-realtime

Architecture:
    When a voice in voice_agent_voices has provider='ultravox' or
    'gemini_realtime' (or other realtime provider), the worker creates
    a RealtimeModel instead of separate STT/LLM/TTS components.
    The RealtimeModel is passed as llm= to AgentSession, which handles
    everything internally.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Set of provider names that are realtime (not pipeline-based)
REALTIME_PROVIDERS: set[str] = {
    "ultravox",
    "gemini_realtime",
    "openai_realtime",
}

# Aliases for normalization
REALTIME_PROVIDER_ALIASES: dict[str, str] = {
    "ultravox": "ultravox",
    "ultravox_realtime": "ultravox",
    "ultravox-realtime": "ultravox",
    "fixie": "ultravox",
    "gemini_realtime": "gemini_realtime",
    "gemini-realtime": "gemini_realtime",
    "gemini_live": "gemini_realtime",
    "gemini-live": "gemini_realtime",
    "gemini_native_audio": "gemini_realtime",
    "openai_realtime": "openai_realtime",
    "openai-realtime": "openai_realtime",
}


# =============================================================================
# DETECTION / NORMALIZATION
# =============================================================================

def normalize_realtime_provider(value: str | None) -> str | None:
    """
    Normalize a provider name to its canonical realtime form.
    
    Returns the canonical name if it's a realtime provider, otherwise None.
    
    Args:
        value: Raw provider name from DB
        
    Returns:
        Canonical realtime provider name or None if not realtime
    """
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    # Check alias map first
    canonical = REALTIME_PROVIDER_ALIASES.get(normalized)
    if canonical:
        return canonical
    # Check direct membership
    if normalized in REALTIME_PROVIDERS:
        return normalized
    return None


def is_realtime_provider(provider: str | None) -> bool:
    """
    Check if a provider name refers to a realtime model.
    
    Args:
        provider: Provider name (raw or normalized)
        
    Returns:
        True if this is a realtime provider
    """
    return normalize_realtime_provider(provider) is not None


# =============================================================================
# ULTRAVOX PROVIDER
# =============================================================================

# Known Ultravox RealtimeModel constructor parameters and their Python types.
# Source: livekit-plugins-ultravox/livekit/plugins/ultravox/realtime/realtime_model.py
#
# Parameter              | Type             | Default                  | provider_config key
# -----------------------|------------------|--------------------------|--------------------
# model                  | str              | "fixie-ai/ultravox"      | model
# voice                  | str              | "Mark"                   | voice
# temperature            | float            | NOT_GIVEN                | temperature
# language_hint          | str              | NOT_GIVEN                | language_hint
# max_duration           | str              | NOT_GIVEN                | max_duration
# time_exceeded_message  | str              | NOT_GIVEN                | time_exceeded_message
# enable_greeting_prompt | bool             | NOT_GIVEN                | enable_greeting_prompt
# first_speaker          | str              | "FIRST_SPEAKER_USER"     | first_speaker
# output_medium          | "text" | "voice" | "voice"                  | output_medium
# input_sample_rate      | int              | 16000                    | input_sample_rate
# output_sample_rate     | int              | 24000                    | output_sample_rate
# system_prompt          | str              | "You are a helpful..."   | system_prompt (usually from agent instructions)

def _safe_float(val: Any, key: str) -> float | None:
    """Safely convert a value to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        logger.warning("Cannot convert %s=%r to float, ignoring", key, val)
        return None


def _safe_int(val: Any, key: str) -> int | None:
    """Safely convert a value to int."""
    if val is None:
        return None
    try:
        return int(float(val))  # handle "16000.0" etc.
    except (ValueError, TypeError):
        logger.warning("Cannot convert %s=%r to int, ignoring", key, val)
        return None


def _safe_bool(val: Any, key: str) -> bool | None:
    """Safely convert a value to bool."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return bool(val)


def create_ultravox(overrides: dict[str, Any] | None = None):
    """
    Create Ultravox RealtimeModel instance with type-safe parameter mapping.
    
    Ultravox combines STT + LLM + TTS into a single realtime connection.
    All parameters are extracted from the overrides dict (sourced from 
    voice_agent_voices.provider_config JSONB column).
    
    Args:
        overrides: Dict of parameters from provider_config. All values may be
                   strings (since they pass through metadata serialization).
        
    Returns:
        ultravox.realtime.RealtimeModel instance
    """
    from livekit.plugins import ultravox
    
    config = dict(overrides or {})
    build_kwargs: dict[str, Any] = {}
    
    # --- String parameters (pass through directly) ---
    voice = config.pop("voice", "Mark")
    build_kwargs["voice"] = str(voice)
    
    model = config.pop("model", None)
    if model:
        build_kwargs["model"] = str(model)
    
    language_hint = config.pop("language_hint", None)
    if language_hint:
        build_kwargs["language_hint"] = str(language_hint)
    
    max_duration = config.pop("max_duration", None)
    if max_duration:
        build_kwargs["max_duration"] = str(max_duration)
    
    time_exceeded_message = config.pop("time_exceeded_message", None)
    if time_exceeded_message:
        build_kwargs["time_exceeded_message"] = str(time_exceeded_message)
    
    first_speaker = config.pop("first_speaker", None)
    if first_speaker:
        build_kwargs["first_speaker"] = str(first_speaker)
    
    output_medium = config.pop("output_medium", None)
    if output_medium and str(output_medium) in ("text", "voice"):
        build_kwargs["output_medium"] = str(output_medium)
    
    system_prompt = config.pop("system_prompt", None)
    if system_prompt:
        build_kwargs["system_prompt"] = str(system_prompt)
    
    # --- Float parameters ---
    temperature = _safe_float(config.pop("temperature", None), "temperature")
    if temperature is not None:
        build_kwargs["temperature"] = temperature
    
    # --- Int parameters ---
    input_sample_rate = _safe_int(config.pop("input_sample_rate", None), "input_sample_rate")
    if input_sample_rate is not None:
        build_kwargs["input_sample_rate"] = input_sample_rate
    
    output_sample_rate = _safe_int(config.pop("output_sample_rate", None), "output_sample_rate")
    if output_sample_rate is not None:
        build_kwargs["output_sample_rate"] = output_sample_rate
    
    # --- Bool parameters ---
    enable_greeting = _safe_bool(config.pop("enable_greeting_prompt", None), "enable_greeting_prompt")
    if enable_greeting is not None:
        build_kwargs["enable_greeting_prompt"] = enable_greeting
    
    # Log any unrecognized keys (they won't break anything, but good to know)
    # Remove known non-Ultravox keys that may end up here from the metadata pipeline
    _known_passthrough = {"language", "encoding", "provider", "provider_voice_id"}
    unknown = {k: v for k, v in config.items() if k not in _known_passthrough}
    if unknown:
        logger.warning(
            "Unrecognized Ultravox config keys (ignored): %s",
            list(unknown.keys()),
        )
    
    logger.info(
        "Creating Ultravox RealtimeModel: %s",
        {k: v for k, v in build_kwargs.items() if k != "system_prompt"},  # Don't log system_prompt
    )
    
    return ultravox.realtime.RealtimeModel(**build_kwargs)


# =============================================================================
# GEMINI LIVE PROVIDER
# =============================================================================

# Known Google RealtimeModel constructor parameters and their Python types.
# Source: livekit-plugins-google/livekit/plugins/google/realtime/realtime_api.py
#
# Parameter               | Type              | Default                                          | provider_config key
# ------------------------|-------------------|--------------------------------------------------|--------------------
# model                   | str               | "gemini-2.5-flash-native-audio-preview-12-2025"  | model
# voice                   | str               | "Puck"                                           | voice
# temperature             | float             | NOT_GIVEN                                        | temperature
# instructions            | str               | NOT_GIVEN                                        | instructions
# language                | str               | NOT_GIVEN                                        | language
# top_p                   | float             | NOT_GIVEN                                        | top_p
# top_k                   | int               | NOT_GIVEN                                        | top_k
# presence_penalty        | float             | NOT_GIVEN                                        | presence_penalty
# frequency_penalty       | float             | NOT_GIVEN                                        | frequency_penalty
# max_output_tokens       | int               | NOT_GIVEN                                        | max_output_tokens
# candidate_count         | int               | 1                                                | candidate_count
# enable_affective_dialog | bool              | NOT_GIVEN                                        | enable_affective_dialog
# proactivity             | bool              | NOT_GIVEN                                        | proactivity
# thinking_config         | ThinkingConfig    | NOT_GIVEN                                        | thinking_config (dict)
# vertexai                | bool              | False                                            | vertexai
# project                 | str               | env GOOGLE_CLOUD_PROJECT                         | project
# location                | str               | "us-central1"                                    | location
#
# API key: Read from GOOGLE_API_KEY env var automatically by the plugin.
# Available voices: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr

def create_gemini_realtime(overrides: dict[str, Any] | None = None):
    """
    Create Google Gemini Live RealtimeModel instance with type-safe parameter mapping.
    
    Gemini Live combines STT + LLM + TTS into a single realtime WebSocket.
    Supports native audio output with thinking, affective dialog, and proactivity.
    
    Args:
        overrides: Dict of parameters from provider_config. All values may be
                   strings (since they pass through metadata serialization).
        
    Returns:
        google.realtime.RealtimeModel instance
    """
    from livekit.plugins import google
    
    config = dict(overrides or {})
    build_kwargs: dict[str, Any] = {}
    
    # --- String parameters ---
    voice = config.pop("voice", "Puck")
    build_kwargs["voice"] = str(voice)
    
    model = config.pop("model", None)
    if model:
        build_kwargs["model"] = str(model)
    
    instructions = config.pop("instructions", None)
    if instructions:
        build_kwargs["instructions"] = str(instructions)
    
    language = config.pop("language", None)
    if language:
        build_kwargs["language"] = str(language)
    
    project = config.pop("project", None)
    if project:
        build_kwargs["project"] = str(project)
    
    location = config.pop("location", None)
    if location:
        build_kwargs["location"] = str(location)
    
    # --- Float parameters ---
    temperature = _safe_float(config.pop("temperature", None), "temperature")
    if temperature is not None:
        build_kwargs["temperature"] = temperature
    
    top_p = _safe_float(config.pop("top_p", None), "top_p")
    if top_p is not None:
        build_kwargs["top_p"] = top_p
    
    presence_penalty = _safe_float(config.pop("presence_penalty", None), "presence_penalty")
    if presence_penalty is not None:
        build_kwargs["presence_penalty"] = presence_penalty
    
    frequency_penalty = _safe_float(config.pop("frequency_penalty", None), "frequency_penalty")
    if frequency_penalty is not None:
        build_kwargs["frequency_penalty"] = frequency_penalty
    
    # --- Int parameters ---
    top_k = _safe_int(config.pop("top_k", None), "top_k")
    if top_k is not None:
        build_kwargs["top_k"] = top_k
    
    max_output_tokens = _safe_int(config.pop("max_output_tokens", None), "max_output_tokens")
    if max_output_tokens is not None:
        build_kwargs["max_output_tokens"] = max_output_tokens
    
    candidate_count = _safe_int(config.pop("candidate_count", None), "candidate_count")
    if candidate_count is not None:
        build_kwargs["candidate_count"] = candidate_count
    
    # --- Bool parameters ---
    vertexai = _safe_bool(config.pop("vertexai", None), "vertexai")
    if vertexai is not None:
        build_kwargs["vertexai"] = vertexai
    
    enable_affective_dialog = _safe_bool(
        config.pop("enable_affective_dialog", None), "enable_affective_dialog"
    )
    if enable_affective_dialog is not None:
        build_kwargs["enable_affective_dialog"] = enable_affective_dialog
    
    proactivity = _safe_bool(config.pop("proactivity", None), "proactivity")
    if proactivity is not None:
        build_kwargs["proactivity"] = proactivity
    
    # --- ThinkingConfig (pass as types.ThinkingConfig if present) ---
    thinking_config_raw = config.pop("thinking_config", None)
    if thinking_config_raw and isinstance(thinking_config_raw, dict):
        try:
            from google.genai import types
            tc_kwargs: dict[str, Any] = {}
            if "include_thoughts" in thinking_config_raw:
                tc_kwargs["include_thoughts"] = _safe_bool(
                    thinking_config_raw["include_thoughts"], "include_thoughts"
                )
            if "thinking_budget" in thinking_config_raw:
                budget = _safe_int(thinking_config_raw["thinking_budget"], "thinking_budget")
                if budget is not None:
                    tc_kwargs["thinking_budget"] = budget
            if tc_kwargs:
                build_kwargs["thinking_config"] = types.ThinkingConfig(**tc_kwargs)
        except Exception as e:
            logger.warning("Failed to build ThinkingConfig: %s, ignoring", e)

    # --- RealtimeInputConfig (Automatic Activity Detection for Latency Tuning) ---
    ric_raw = config.pop("realtime_input_config", None)
    if ric_raw and isinstance(ric_raw, dict):
        try:
            from google.genai import types
            ric_kwargs: dict[str, Any] = {}
            
            aad_raw = ric_raw.get("automatic_activity_detection")
            if aad_raw and isinstance(aad_raw, dict):
                aad_kwargs: dict[str, Any] = {}
                
                # Boolean fields
                disable_aad = _safe_bool(aad_raw.get("disabled"), "aad.disabled")
                if disable_aad is not None:
                    aad_kwargs["disabled"] = disable_aad
                
                # Integer fields
                for field in [
                    "silence_duration_ms", 
                    "prefix_padding_ms", 
                    "start_of_speech_sensitivity", 
                    "end_of_speech_sensitivity", 
                    "trigger_tokens"
                ]:
                    val = _safe_int(aad_raw.get(field), f"aad.{field}")
                    if val is not None:
                        aad_kwargs[field] = val
                
                if aad_kwargs:
                    ric_kwargs["automatic_activity_detection"] = types.AutomaticActivityDetection(**aad_kwargs)

            # Add other RealtimeInputConfig fields if needed later (e.g. ActivityHandling)
            
            if ric_kwargs:
                build_kwargs["realtime_input_config"] = types.RealtimeInputConfig(**ric_kwargs)
        except Exception as e:
            logger.warning("Failed to build RealtimeInputConfig: %s, ignoring", e)

    
    # --- ContextWindowCompression (types.ContextWindowCompressionConfig) ---
    cwc_raw = config.pop("context_window_compression", None)
    if cwc_raw and isinstance(cwc_raw, dict):
        try:
            from google.genai import types
            cwc_kwargs: dict[str, Any] = {}
            
            trigger_tokens = _safe_int(cwc_raw.get("trigger_tokens"), "trigger_tokens")
            if trigger_tokens is not None:
                cwc_kwargs["trigger_tokens"] = trigger_tokens
            
            # sliding_window can be a dict with target_tokens
            sw_raw = cwc_raw.get("sliding_window")
            if sw_raw and isinstance(sw_raw, dict):
                target_tokens = _safe_int(sw_raw.get("target_tokens"), "target_tokens")
                if target_tokens is not None:
                    cwc_kwargs["sliding_window"] = types.SlidingWindow(
                        target_tokens=target_tokens
                    )
            
            if cwc_kwargs:
                build_kwargs["context_window_compression"] = (
                    types.ContextWindowCompressionConfig(**cwc_kwargs)
                )
        except Exception as e:
            logger.warning("Failed to build ContextWindowCompressionConfig: %s, ignoring", e)
    
    # Log unrecognized keys
    _known_passthrough = {"language_hint", "encoding", "provider", "provider_voice_id"}
    unknown = {k: v for k, v in config.items() if k not in _known_passthrough}
    if unknown:
        logger.warning(
            "Unrecognized Gemini Realtime config keys (ignored): %s",
            list(unknown.keys()),
        )
    
    logger.info(
        "Creating Gemini Live RealtimeModel: %s",
        {k: v for k, v in build_kwargs.items() if k != "instructions"},
    )
    
    return google.realtime.RealtimeModel(**build_kwargs)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_realtime_model(
    provider: str,
    overrides: dict[str, Any] | None = None,
):
    """
    Create a realtime model instance for the specified provider.
    
    Factory function - each call creates a new instance,
    allowing runtime provider switching.
    
    Args:
        provider: Realtime provider name (e.g., "ultravox", "gemini_realtime")
        overrides: Provider-specific config overrides from voice_agent_voices.provider_config
        
    Returns:
        RealtimeModel instance
        
    Raises:
        ValueError: If provider is not recognized
    """
    canonical = normalize_realtime_provider(provider)
    if not canonical:
        raise ValueError(
            f"Unknown realtime provider: {provider!r}. "
            f"Known providers: {sorted(REALTIME_PROVIDERS)}"
        )
    
    config = dict(overrides or {})
    
    if canonical == "ultravox":
        return create_ultravox(overrides=config)
    
    elif canonical == "gemini_realtime":
        return create_gemini_realtime(overrides=config)
    
    elif canonical == "openai_realtime":
        raise NotImplementedError(
            "OpenAI Realtime provider is planned but not yet implemented. "
            "Use provider='ultravox' or 'gemini_realtime' for now."
        )
    
    # Should not reach here due to normalize check above
    raise ValueError(f"Unhandled realtime provider: {canonical}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Detection
    "is_realtime_provider",
    "normalize_realtime_provider",
    # Factory
    "create_realtime_model",
    # Individual providers
    "create_ultravox",
    "create_gemini_realtime",
    # Constants
    "REALTIME_PROVIDERS",
    "REALTIME_PROVIDER_ALIASES",
]
