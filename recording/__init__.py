"""
Call recording module.

Contains:
- recorder: CallRecorder, CallTranscription, LocalBufferManager
- api: RecordingAPI for signed URL generation
- transcription: TranscriptionTracker for speech tracking
- audio_trim: ffmpeg utilities for silence trimming
"""

from recording.recorder import (
    CallRecorder,
    CallTranscription,
    TranscriptionSegment,
    LocalBufferManager,
    attach_recorder_to_session,
)
from recording.api import RecordingAPI
from recording.transcription import (
    TranscriptionTracker,
    RecorderTextOutput,
    RecorderAudioOutput,
    attach_transcription_tracker,
)
from recording.audio_trim import (
    trim_leading_silence_ffmpeg,
    probe_duration,
)

__all__ = [
    # Recorder
    "CallRecorder",
    "CallTranscription",
    "TranscriptionSegment",
    "LocalBufferManager",
    "attach_recorder_to_session",
    # API
    "RecordingAPI",
    # Transcription
    "TranscriptionTracker",
    "RecorderTextOutput",
    "RecorderAudioOutput",
    "attach_transcription_tracker",
    # Audio
    "trim_leading_silence_ffmpeg",
    "probe_duration",
]
