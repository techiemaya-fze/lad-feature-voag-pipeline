"""
Providers package.

Factory modules for LLM, TTS, and Realtime providers.
Supports runtime switching - each call creates a fresh instance.

Modules:
- llm_builder: Groq, Google Gemini, OpenAI
- tts_builder: Cartesia, Google, Chirp, Gemini, ElevenLabs, Rime, SmallestAI, Sarvam
- realtime_builder: Ultravox, Gemini Live (+ future OpenAI Realtime)
"""

from agent.providers.llm_builder import (
    create_llm,
    create_groq,
    create_gemini as create_gemini_llm,
    create_openai,
    normalize_provider as normalize_llm_provider,
    resolve_configuration as resolve_llm_configuration,
)

from agent.providers.tts_builder import (
    create_tts,
    create_cartesia,
    create_google as create_google_tts,
    create_google_chirp,
    create_gemini as create_gemini_tts,
    create_elevenlabs,
    create_rime,
    create_smallestai,
    create_sarvam,
    normalize_provider as normalize_tts_provider,
)

from agent.providers.realtime_builder import (
    create_realtime_model,
    create_ultravox,
    create_gemini_realtime,
    is_realtime_provider,
    normalize_realtime_provider,
    REALTIME_PROVIDERS,
)

__all__ = [
    # LLM
    "create_llm",
    "create_groq",
    "create_gemini_llm",
    "create_openai",
    "normalize_llm_provider",
    "resolve_llm_configuration",
    # TTS
    "create_tts",
    "create_cartesia",
    "create_google_tts",
    "create_google_chirp",
    "create_gemini_tts",
    "create_elevenlabs",
    "create_rime",
    "create_smallestai",
    "create_sarvam",
    "normalize_tts_provider",
    # Realtime
    "create_realtime_model",
    "create_ultravox",
    "create_gemini_realtime",
    "is_realtime_provider",
    "normalize_realtime_provider",
    "REALTIME_PROVIDERS",
]

