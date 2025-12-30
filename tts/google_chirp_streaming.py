"""Google Chirp streaming TTS integration for LiveKit.

This module keeps the implementation isolated so we can route outbound calls
that select the ``google-chirp`` provider through Google Cloud's bidirectional
streaming Text-to-Speech API. The helper utilities wrap the LiveKit Google TTS
plugin but also expose the raw ``StreamingSynthesizeRequest`` generator that
mirrors the official Google sample found in the Text-to-Speech quickstart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from google.cloud import texttospeech
from livekit.plugins import google as livekit_google

_DEFAULT_LANGUAGE = "en-IN"
_DEFAULT_VOICE = os.getenv("GOOGLE_CHIRP_VOICE_NAME", "en-IN-Neural2-F")
_DEFAULT_SAMPLE_RATE = 24000


@dataclass
class GoogleChirpVoiceConfig:
    """Runtime configuration for the Google Chirp streaming voice."""

    language_code: str = _DEFAULT_LANGUAGE
    voice_name: str = _DEFAULT_VOICE
    speaking_rate: float = 1.0
    pitch: int = 0
    sample_rate_hz: int = _DEFAULT_SAMPLE_RATE
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.OGG_OPUS

    @classmethod
    def from_overrides(
        cls,
        overrides: dict[str, str] | None = None,
        *,
        accent: str | None = None,
    ) -> "GoogleChirpVoiceConfig":
        overrides = overrides or {}

        voice = overrides.get("voice_name") or overrides.get("voice") or _DEFAULT_VOICE
        language_hint = overrides.get("language") or _resolve_language_from_accent(accent)
        language = _coerce_language(language_hint, voice)

        speaking_rate = _coerce_float(overrides.get("speaking_rate"), 1.0)
        pitch = _coerce_int(overrides.get("pitch"), 0)
        sample_rate = _coerce_int(overrides.get("sample_rate"), _DEFAULT_SAMPLE_RATE)

        return cls(
            language_code=language,
            voice_name=voice,
            speaking_rate=speaking_rate,
            pitch=pitch,
            sample_rate_hz=sample_rate,
        )


def create_google_chirp_tts(config: GoogleChirpVoiceConfig | None = None) -> livekit_google.TTS:
    """Instantiate a LiveKit-compatible Google TTS engine configured for Chirp streaming."""

    config = config or GoogleChirpVoiceConfig()
    credentials_path = _resolve_credentials_path()

    kwargs: dict[str, object] = {
        "language": config.language_code,
        "voice_name": config.voice_name,
        "speaking_rate": config.speaking_rate,
        "pitch": config.pitch,
        "sample_rate": config.sample_rate_hz,
        "use_streaming": True,
    }

    if credentials_path:
        kwargs["credentials_file"] = credentials_path

    # ``livekit.plugins.google.TTS`` exposes the streaming_synthesize powered
    # implementation, so handing it the tuned configuration keeps latency low
    # while preserving LiveKit's backpressure handling.
    return livekit_google.TTS(**kwargs)


def build_streaming_request_sequence(
    text_chunks: Sequence[str],
    *,
    config: GoogleChirpVoiceConfig,
) -> Iterable[texttospeech.StreamingSynthesizeRequest]:
    """Yield the request flow expected by ``TextToSpeechClient.streaming_synthesize``.

    This mirrors the structure in Google's official streaming sample and can be
    useful for targeted diagnostics or manual smoke tests outside the LiveKit
    pipeline.
    """

    streaming_config = texttospeech.StreamingSynthesizeConfig(
        voice=texttospeech.VoiceSelectionParams(
            name=config.voice_name,
            language_code=config.language_code,
        ),
        streaming_audio_config=texttospeech.StreamingAudioConfig(
            audio_encoding=config.audio_encoding,
            sample_rate_hertz=config.sample_rate_hz,
            speaking_rate=config.speaking_rate,
            pitch=config.pitch,
        ),
    )

    yield texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)
    for chunk in text_chunks:
        if not chunk:
            continue
        yield texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(text=chunk),
        )


def _resolve_language_from_accent(accent: str | None) -> str | None:
    if not accent:
        return None
    normalized = accent.lower().strip()
    mapping = {
        "en-in": "en-IN",
        "hi": "hi-IN",
        "hi-in": "hi-IN",
    }
    return mapping.get(normalized, None)


def _coerce_language(language: str | None, voice_name: str) -> str:
    """Ensure the language code matches the selected voice."""

    derived_from_voice = _language_from_voice_name(voice_name)
    if derived_from_voice:
        return derived_from_voice
    return language or _DEFAULT_LANGUAGE


def _language_from_voice_name(voice_name: str) -> str | None:
    if not voice_name:
        return None
    trimmed = voice_name.strip()
    if not trimmed:
        return None

    marker = "-Chirp"
    if marker in trimmed:
        prefix = trimmed.split(marker, 1)[0]
        canonical = prefix.strip()
        return canonical if canonical else None
    return None


def _coerce_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _coerce_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_credentials_path() -> str | None:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        return credentials_path

    storage_key = os.getenv("GCS_CREDENTIALS_JSON")
    if not storage_key:
        return None

    candidate_path = storage_key
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.abspath(candidate_path)

    return candidate_path if os.path.exists(candidate_path) else None
