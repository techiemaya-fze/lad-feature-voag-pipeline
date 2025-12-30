import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from livekit import rtc
from livekit.agents.voice import io

from recording.recorder import CallRecorder

logger = logging.getLogger(__name__)


@dataclass
class _SpeechSegment:
    intended_text: str
    finalized: bool = False


class TranscriptionTracker:
    """Coordinates agent speech text with actual audio playout events."""

    def __init__(self, recorder: CallRecorder) -> None:
        self._recorder = recorder
        self._pending: Deque[_SpeechSegment] = deque()
        self._active_segment: Optional[_SpeechSegment] = None
        self._active_text_parts: list[str] = []

    def handle_text_chunk(self, text: str) -> None:
        """Handle incremental text coming from the LLM/transcription node."""
        if not text:
            return

        self._active_text_parts.append(text)
        current_text = "".join(self._active_text_parts).strip()

        if not current_text:
            return

        if self._active_segment is None:
            segment = _SpeechSegment(intended_text=current_text, finalized=False)
            self._pending.append(segment)
            self._active_segment = segment
            self._recorder.on_agent_speech_start(current_text)
        else:
            self._active_segment.intended_text = current_text
            self._recorder.update_agent_speech_text(current_text)

    def handle_text_flush(self) -> None:
        """Called when the LLM finishes producing text for the current speech."""
        if self._active_segment is None:
            self._active_text_parts.clear()
            return

        final_text = "".join(self._active_text_parts).strip()
        if final_text:
            self._active_segment.intended_text = final_text
            self._recorder.update_agent_speech_text(final_text)
        self._active_segment.finalized = True
        self._active_segment = None
        self._active_text_parts.clear()

    def handle_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: Optional[str],
    ) -> None:
        if not self._pending and self._active_segment is not None:
            segment = self._active_segment
            segment.intended_text = "".join(self._active_text_parts).strip()
            segment.finalized = True
            self._pending.append(segment)
            self._active_segment = None
            self._active_text_parts.clear()

        if not self._pending:
            logger.warning("Playback finished event received without a pending speech segment")
            return

        segment = self._pending.popleft()
        if segment is self._active_segment:
            segment.intended_text = "".join(self._active_text_parts).strip()
            segment.finalized = True
            self._active_segment = None
            self._active_text_parts.clear()

        intended = segment.intended_text.strip()
        if not intended:
            logger.debug("Skipping agent transcription segment with empty intended text")
            return

        synchronized = (synchronized_transcript or "").strip()
        spoken_text = synchronized if interrupted else intended

        if spoken_text:
            self._recorder.on_agent_speech_chunk(spoken_text)
        else:
            logger.debug("No synchronized transcript available for interrupted speech")

        self._recorder.on_agent_speech_end(was_interrupted=interrupted)

        if interrupted and synchronized and intended:
            self._log_interruption_stats(synchronized, intended, playback_position)

    def _log_interruption_stats(self, spoken: str, intended: str, playback_position: float) -> None:
        intended_len = len(intended)
        spoken_len = len(spoken)
        ratio = (spoken_len / intended_len) * 100 if intended_len else 0.0
        logger.info(
            "Agent speech interrupted: %.1f%% of intended text spoken (%.2fs playback)",
            ratio,
            playback_position,
        )


class RecorderTextOutput(io.TextOutput):
    """Intercepts text output to track agent speech transcripts."""

    def __init__(self, tracker: TranscriptionTracker, downstream: io.TextOutput) -> None:
        super().__init__(label="RecorderTextOutput", next_in_chain=downstream)
        self._tracker = tracker
        self._downstream = downstream

    async def capture_text(self, text: str) -> None:
        self._tracker.handle_text_chunk(str(text))
        await self._downstream.capture_text(text)

    def flush(self) -> None:
        self._tracker.handle_text_flush()
        self._downstream.flush()


class RecorderAudioOutput(io.AudioOutput):
    """Intercepts audio output to determine what portion was spoken."""

    def __init__(self, tracker: TranscriptionTracker, downstream: io.AudioOutput) -> None:
        capabilities = io.AudioOutputCapabilities(pause=downstream.can_pause)
        super().__init__(
            label="RecorderAudioOutput",
            next_in_chain=downstream,
            capabilities=capabilities,
            sample_rate=downstream.sample_rate,
        )
        self._tracker = tracker
        self._downstream = downstream

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        await self._downstream.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self._downstream.flush()

    def clear_buffer(self) -> None:
        self._downstream.clear_buffer()

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: Optional[str] = None,
    ) -> None:
        self._tracker.handle_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )


def attach_transcription_tracker(session, recorder: CallRecorder) -> None:
    """Wrap the session outputs so CallRecorder receives real-time transcripts."""
    if recorder is None:
        return

    audio_sink = session.output.audio
    text_sink = session.output.transcription

    if audio_sink is None or text_sink is None:
        logger.warning("Session audio or transcription outputs are not available for tracking")
        return

    if isinstance(text_sink, RecorderTextOutput) or isinstance(audio_sink, RecorderAudioOutput):
        logger.debug("Transcription tracker already attached to session outputs")
        return

    tracker = TranscriptionTracker(recorder)
    session.output.transcription = RecorderTextOutput(tracker, text_sink)
    session.output.audio = RecorderAudioOutput(tracker, audio_sink)
