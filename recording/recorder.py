"""
LiveKit Call Recording and Transcription Manager

Handles:
1. Call recording via LiveKit Egress API
2. Real-time transcription tracking (STT + TTS)
3. TTS interruption detection (tracks what was actually spoken)
4. Database storage of recordings and transcriptions
5. Local buffer management (keeps most recent 100MB)
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Callable, Awaitable

from livekit import api, rtc
from livekit.agents import AgentSession
from google.cloud import storage

from utils.audio_trim import trim_leading_silence_ffmpeg, probe_duration

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid value for %s=%s; using default %.2f", name, value, default)
        return default


TRIM_CALL_RECORDING_SILENCE = os.getenv("TRIM_CALL_RECORDING_SILENCE", "true").lower() == "true"
TRIM_SILENCE_THRESHOLD_DB = _env_float("CALL_RECORDING_SILENCE_THRESHOLD_DB", -45.0)
TRIM_MIN_SILENCE_DURATION_SECONDS = _env_float("CALL_RECORDING_SILENCE_MIN_DURATION_SECONDS", 0.8)
TRIM_WAIT_TIMEOUT_SECONDS = _env_float("CALL_RECORDING_TRIM_MAX_WAIT_SECONDS", 60.0)
TRIM_POLL_INTERVAL_SECONDS = _env_float("CALL_RECORDING_TRIM_POLL_INTERVAL_SECONDS", 2.0)
# New: Leading buffer - how much silence to keep before the first voice
TRIM_LEADING_BUFFER_SECONDS = _env_float("CALL_RECORDING_LEADING_BUFFER_SECONDS", 0.5)
# New: Trailing silence trimming settings
TRIM_TRAILING_SILENCE = os.getenv("TRIM_CALL_RECORDING_TRAILING_SILENCE", "true").lower() == "true"
TRIM_TRAILING_SILENCE_DURATION_SECONDS = _env_float("CALL_RECORDING_TRAILING_SILENCE_MIN_DURATION_SECONDS", 0.5)

# Local buffer configuration
LOCAL_BUFFER_SIZE_MB = 100
LOCAL_BUFFER_PATH = Path("./recordings_buffer")


@dataclass
class TranscriptionSegment:
    """Represents a single segment of conversation"""
    timestamp: datetime
    speaker: str  # "user" or "agent"
    text: str
    is_complete: bool = True  # False if TTS was interrupted
    intended_text: Optional[str] = None  # Full text if interrupted


@dataclass
class CallTranscription:
    """Complete call transcription with metadata"""
    call_id: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    def add_segment(self, speaker: str, text: str, is_complete: bool = True, intended_text: Optional[str] = None):
        """Add a transcription segment"""
        segment = TranscriptionSegment(
            timestamp=datetime.now(timezone.utc),
            speaker=speaker,
            text=text,
            is_complete=is_complete,
            intended_text=intended_text
        )
        self.segments.append(segment)
        logger.info(
            "Transcription segment added: speaker=%s, complete=%s, text=%s",
            speaker,
            is_complete,
            text[:50] + "..." if len(text) > 50 else text
        )

    def to_json(self) -> str:
        """Convert to JSON string (for backward compatibility)"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSONB database storage"""
        return {
            "call_id": self.call_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "segments": [
                {
                    "timestamp": seg.timestamp.isoformat(),
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "is_complete": seg.is_complete,
                    "intended_text": seg.intended_text
                }
                for seg in self.segments
            ]
        }

    def to_readable_text(self) -> str:
        """Convert to human-readable format"""
        lines = []
        for seg in self.segments:
            time_str = seg.timestamp.strftime("%H:%M:%S")
            speaker_label = "Agent" if seg.speaker == "agent" else "User"
            
            if seg.is_complete:
                lines.append(f"[{time_str}] {speaker_label}: {seg.text}")
            else:
                lines.append(f"[{time_str}] {speaker_label}: {seg.text} [INTERRUPTED]")
                if seg.intended_text:
                    lines.append(f"    (Intended: {seg.intended_text})")
        
        return "\n".join(lines)


class CallRecorder:
    """Manages call recording and transcription for LiveKit sessions"""

    def __init__(
        self,
        room_name: str,
        call_id: str,
        livekit_api: api.LiveKitAPI,
        enable_recording: bool = True,
        storage_config: Optional[dict] = None
    ):
        self.room_name = room_name
        self.call_id = call_id
        self.livekit_api = livekit_api
        self.enable_recording = enable_recording
        self.storage_config = storage_config or self._default_storage_config()
        
        self.transcription = CallTranscription(call_id=call_id)
        self.egress_id: Optional[str] = None
        self.recording_url: Optional[str] = None
        
        # Track current TTS state
        self._current_tts_text: Optional[str] = None
        self._tts_start_time: Optional[datetime] = None
        self._accumulated_tts: str = ""
        self._tts_playout_complete = asyncio.Event()
        self._tts_playout_complete.set()
        self._active_tts_segments = 0
        self._agent_speech_end_callbacks: list[Callable[[bool], Awaitable[None] | None]] = []
        
        # Duration after trimming (set by _trim_recording_silence)
        self._trimmed_duration_seconds: Optional[float] = None

    def _default_storage_config(self) -> dict:
        """Get default GCS storage configuration from environment"""
        credentials_path = os.getenv("GCS_CREDENTIALS_JSON")
        credentials_json = None
        
        # Read the credentials file content
        if credentials_path and os.path.exists(credentials_path):
            with open(credentials_path, 'r') as f:
                credentials_json = f.read()
        
        return {
            "type": "gcs",
            "bucket": os.getenv("GCS_BUCKET", "livekit-recordings"),
            "credentials_json": credentials_json,  # JSON string content, not path
            "project_id": os.getenv("GCS_PROJECT_ID"),
            "filepath_pattern": "recordings/{room_name}/{call_id}.ogg"
        }

    @staticmethod
    def _parse_gs_url(gs_url: str) -> tuple[str, str]:
        if not gs_url or not gs_url.startswith("gs://"):
            raise ValueError(f"Invalid GCS URL: {gs_url}")
        parts = gs_url[5:].split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid GCS URL format: {gs_url}")
        return parts[0], parts[1]

    def _create_gcs_client(self) -> storage.Client:
        credentials_json = self.storage_config.get("credentials_json") if self.storage_config else None
        project_id = self.storage_config.get("project_id") if self.storage_config else None
        if credentials_json:
            try:
                info = json.loads(credentials_json)
                return storage.Client.from_service_account_info(info)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load GCS credentials from JSON for call %s: %s", self.call_id, exc, exc_info=True)
                raise
        credentials_path = os.getenv("GCS_CREDENTIALS_JSON")
        if credentials_path and os.path.exists(credentials_path):
            return storage.Client.from_service_account_json(credentials_path)
        return storage.Client(project=project_id)

    async def _wait_for_blob(self, blob: storage.Blob) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TRIM_WAIT_TIMEOUT_SECONDS
        while True:
            exists = await asyncio.to_thread(blob.exists)
            if exists:
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(TRIM_POLL_INTERVAL_SECONDS)

    async def _trim_recording_silence(self) -> None:
        if not TRIM_CALL_RECORDING_SILENCE:
            return
        if not self.recording_url:
            return
        if not self.storage_config or self.storage_config.get("type") != "gcs":
            return
        try:
            bucket_name, blob_path = self._parse_gs_url(self.recording_url)
        except ValueError as exc:
            logger.warning("Skipping silence trim due to invalid recording URL for call %s: %s", self.call_id, exc)
            return
        try:
            client = self._create_gcs_client()
        except Exception:
            logger.warning("Unable to initialize GCS client for trimming call %s; leaving audio untrimmed", self.call_id)
            return

        blob = client.bucket(bucket_name).blob(blob_path)
        available = await self._wait_for_blob(blob)
        if not available:
            logger.warning(
                "Timed out waiting for recording to appear in GCS for trimming (call_id=%s)",
                self.call_id,
            )
            return

        await asyncio.to_thread(blob.reload)
        original_metadata = blob.metadata.copy() if blob.metadata else None
        content_type = blob.content_type or "audio/ogg"

        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "raw.ogg"
            output_path = Path(tmp_dir) / "trimmed.ogg"

            await asyncio.to_thread(blob.download_to_filename, str(input_path))

            try:
                input_duration = await probe_duration(input_path)
            except FileNotFoundError:
                logger.warning(
                    "ffprobe not available; duration metrics will be skipped for trimming call %s",
                    self.call_id,
                )
                input_duration = None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to probe duration for call %s before trimming: %s", self.call_id, exc)
                input_duration = None

            try:
                await trim_leading_silence_ffmpeg(
                    input_path=input_path,
                    output_path=output_path,
                    start_duration=TRIM_MIN_SILENCE_DURATION_SECONDS,
                    start_threshold_db=TRIM_SILENCE_THRESHOLD_DB,
                    leading_buffer_seconds=TRIM_LEADING_BUFFER_SECONDS,
                    trim_trailing=TRIM_TRAILING_SILENCE,
                    stop_duration=TRIM_TRAILING_SILENCE_DURATION_SECONDS,
                    stop_threshold_db=TRIM_SILENCE_THRESHOLD_DB,
                )
            except FileNotFoundError:
                logger.warning(
                    "ffmpeg not available on PATH; skipping leading silence trim for call %s",
                    self.call_id,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to trim leading silence for call %s: %s", self.call_id, exc, exc_info=True)
                return

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.warning("Trimmed recording empty for call %s; keeping original audio", self.call_id)
                return

            try:
                output_duration = await probe_duration(output_path)
            except FileNotFoundError:
                output_duration = None
            except Exception:
                output_duration = None

            trimmed_seconds = None
            if input_duration is not None and output_duration is not None:
                trimmed_seconds = max(0.0, input_duration - output_duration)

            # Store the final duration after trimming
            if output_duration is not None:
                self._trimmed_duration_seconds = output_duration
                logger.info(
                    "Final call duration after trimming: %.2fs for call %s",
                    output_duration,
                    self.call_id,
                )
            elif input_duration is not None:
                # Fallback to input duration if output probe failed
                self._trimmed_duration_seconds = input_duration

            await asyncio.to_thread(blob.upload_from_filename, str(output_path), content_type=content_type)
            if original_metadata:
                blob.metadata = original_metadata
                await asyncio.to_thread(blob.patch)

            if trimmed_seconds is not None and trimmed_seconds > 0:
                logger.info(
                    "Trimmed %.2fs of leading silence for call %s",
                    trimmed_seconds,
                    self.call_id,
                )
            else:
                logger.info("Uploaded trimmed recording for call %s", self.call_id)

    async def start_recording(self) -> Optional[str]:
        """
        Start recording the call using LiveKit Egress API
        Returns the egress_id if successful
        """
        if not self.enable_recording:
            logger.info("Recording disabled for call %s", self.call_id)
            return None

        try:
            # Configure file output based on storage type
            if self.storage_config["type"] == "gcs":
                filepath = self.storage_config["filepath_pattern"].format(
                    room_name=self.room_name,
                    call_id=self.call_id
                )
                
                file_output = api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=filepath,
                    gcp=api.GCPUpload(
                        bucket=self.storage_config["bucket"],
                        credentials=self.storage_config["credentials_json"],
                    ),
                )
                
                # Construct the recording URL
                self.recording_url = f"gs://{self.storage_config['bucket']}/{filepath}"
                
            else:
                logger.error("Unsupported storage type: %s", self.storage_config["type"])
                return None

            # Start room composite egress (records entire room audio)
            request = api.RoomCompositeEgressRequest(
                room_name=self.room_name,
                audio_only=True,
                file_outputs=[file_output],
            )

            egress_info = await self.livekit_api.egress.start_room_composite_egress(request)
            self.egress_id = egress_info.egress_id
            
            logger.info(
                "Started recording for call %s: egress_id=%s, url=%s",
                self.call_id,
                self.egress_id,
                self.recording_url
            )
            
            self.transcription.started_at = datetime.now(timezone.utc)
            return self.egress_id

        except Exception as exc:
            logger.error("Failed to start recording for call %s: %s", self.call_id, exc, exc_info=True)
            return None

    async def stop_recording(self) -> bool:
        """Stop the recording"""
        if not self.egress_id:
            logger.warning("No active recording to stop for call %s", self.call_id)
            return False

        egress_already_complete = False
        try:
            await self.livekit_api.egress.stop_egress(
                api.StopEgressRequest(egress_id=self.egress_id)
            )
            logger.info("Stopped recording for call %s: egress_id=%s", self.call_id, self.egress_id)
        except Exception as exc:
            # Check if egress already completed (race condition: room deleted before we called stop)
            # This is normal - the recording is already in GCS, we just need to trim it
            exc_str = str(exc).lower()
            if "egress_complete" in exc_str or "already" in exc_str and "complete" in exc_str:
                logger.info(
                    "Egress already completed for call %s (egress_id=%s) - proceeding with post-processing",
                    self.call_id,
                    self.egress_id,
                )
                egress_already_complete = True
            else:
                # Truly unexpected error - but still try to trim if we have a recording URL
                logger.error("Failed to stop recording for call %s: %s", self.call_id, exc, exc_info=True)
                if not self.recording_url:
                    return False
                logger.info("Recording URL exists, attempting post-processing despite stop_egress failure")

        # Always try to trim and set duration - the recording should be in GCS
        try:
            await self._trim_recording_silence()
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error trimming recording for call %s: %s", self.call_id, exc, exc_info=True)

        self.transcription.ended_at = datetime.now(timezone.utc)
        return True

    def on_user_speech(self, transcript: str):
        """Called when user speech is transcribed"""
        if transcript.strip():
            self.transcription.add_segment(
                speaker="user",
                text=transcript.strip(),
                is_complete=True
            )

    def on_agent_speech_start(self, text: str):
        """Called when agent starts speaking"""
        cleaned = text.strip()
        self._current_tts_text = cleaned if cleaned else text
        self._tts_start_time = datetime.now(timezone.utc)
        self._accumulated_tts = ""
        if self._active_tts_segments == 0:
            self._tts_playout_complete.clear()
        self._active_tts_segments += 1
        logger.debug("Agent speech started: %s", cleaned[:100] if cleaned else text[:100])

    def on_agent_speech_chunk(self, chunk: str):
        """Called for each chunk of agent speech that's actually played"""
        self._accumulated_tts += chunk

    def on_agent_speech_end(self, was_interrupted: bool = False):
        """Called when agent finishes or is interrupted"""
        if self._active_tts_segments > 0:
            self._active_tts_segments -= 1
        if self._active_tts_segments == 0:
            self._tts_playout_complete.set()

        if not self._current_tts_text:
            return

        spoken_text = self._accumulated_tts.strip()
        intended_text = self._current_tts_text.strip()

        if was_interrupted:
            self.transcription.add_segment(
                speaker="agent",
                text=spoken_text,
                is_complete=False,
                intended_text=intended_text or self._current_tts_text
            )
            intended_len = len(intended_text)
            spoken_len = len(spoken_text)
            ratio = (spoken_len / intended_len) * 100 if intended_len else 0.0
            logger.info(
                "Agent interrupted: spoke %d/%d chars (%.1f%%)",
                spoken_len,
                intended_len,
                ratio
            )
        else:
            # Agent completed the full speech
            final_text = intended_text or self._current_tts_text.strip()
            self.transcription.add_segment(
                speaker="agent",
                text=final_text,
                is_complete=True
            )

        # Reset state
        self._current_tts_text = None
        self._tts_start_time = None
        self._accumulated_tts = ""

        for callback in self._agent_speech_end_callbacks:
            try:
                result = callback(was_interrupted)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Agent speech end callback failed: %s",
                    exc,
                    exc_info=True,
                )

    async def wait_for_tts_playout(self, timeout: Optional[float] = 8.0) -> bool:
        """Wait until the most recent agent TTS has finished playing."""
        try:
            if timeout is None:
                await self._tts_playout_complete.wait()
                return True
            await asyncio.wait_for(self._tts_playout_complete.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for agent TTS playout to finish")
            return False

    def update_agent_speech_text(self, text: str) -> None:
        """Update the intended TTS text while the agent is still speaking."""
        cleaned = text.strip()
        if not cleaned and not text:
            return

        if self._current_tts_text is None:
            self.on_agent_speech_start(text)
            return

        if cleaned and cleaned != self._current_tts_text:
            logger.debug("Agent speech intended text updated: %s", cleaned[:100])
            self._current_tts_text = cleaned

    def get_transcription_json(self) -> str:
        """Get transcription as JSON string (backward compatibility)"""
        return self.transcription.to_json()

    def get_transcription_dict(self) -> dict:
        """Get transcription as dictionary for JSONB database storage"""
        return self.transcription.to_dict()

    def get_transcription_text(self) -> str:
        """Get transcription as readable text"""
        return self.transcription.to_readable_text()

    def get_recording_url(self) -> Optional[str]:
        """Get the recording URL"""
        return self.recording_url

    def get_trimmed_duration_seconds(self) -> Optional[float]:
        """Get the call duration in seconds after trimming leading silence.
        
        This value is only available after stop_recording() has completed
        and the audio has been processed.
        
        Returns:
            float: Duration in seconds, or None if not available
        """
        return self._trimmed_duration_seconds

    def register_agent_speech_end_callback(
        self, callback: Callable[[bool], Awaitable[None] | None]
    ) -> None:
        """Register a callback to be notified when agent TTS playback ends."""
        if callback not in self._agent_speech_end_callbacks:
            self._agent_speech_end_callbacks.append(callback)


class LocalBufferManager:
    """Manages local buffer of most recent recordings (100MB limit)"""
    
    def __init__(self, buffer_path: Path = LOCAL_BUFFER_PATH, max_size_mb: int = LOCAL_BUFFER_SIZE_MB):
        self.buffer_path = buffer_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        logger.info("Local buffer initialized at %s (max: %d MB)", self.buffer_path, max_size_mb)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size in bytes"""
        total_size = 0
        try:
            for file_path in self.buffer_path.rglob("*.ogg"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as exc:
            logger.error("Error calculating buffer size: %s", exc)
        return total_size
    
    def get_oldest_files(self) -> list[tuple[Path, float]]:
        """Get list of files sorted by modification time (oldest first)"""
        files = []
        try:
            for file_path in self.buffer_path.rglob("*.ogg"):
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    files.append((file_path, mtime))
            files.sort(key=lambda x: x[1])  # Sort by mtime
        except Exception as exc:
            logger.error("Error listing buffer files: %s", exc)
        return files
    
    def cleanup_old_files(self, required_space: int = 0) -> None:
        """Remove oldest files until buffer is under limit"""
        current_size = self.get_buffer_size()
        target_size = self.max_size_bytes - required_space
        
        if current_size <= target_size:
            return
        
        logger.info(
            "Buffer cleanup needed: current=%d MB, target=%d MB",
            current_size / (1024 * 1024),
            target_size / (1024 * 1024)
        )
        
        files = self.get_oldest_files()
        removed_count = 0
        removed_size = 0
        
        for file_path, _ in files:
            if current_size <= target_size:
                break
            
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                current_size -= file_size
                removed_size += file_size
                removed_count += 1
                logger.debug("Removed old recording: %s (%d bytes)", file_path.name, file_size)
            except Exception as exc:
                logger.error("Error removing file %s: %s", file_path, exc)
        
        if removed_count > 0:
            logger.info(
                "Buffer cleanup complete: removed %d files (%d MB)",
                removed_count,
                removed_size / (1024 * 1024)
            )
    
    async def download_from_gcs(self, gcs_url: str, call_id: str) -> Optional[Path]:
        """
        Download recording from GCS to local buffer
        
        Args:
            gcs_url: GCS URL (gs://bucket/path/to/file.ogg)
            call_id: Call ID for filename
        
        Returns:
            Path to local file if successful, None otherwise
        """
        try:
            # Parse GCS URL
            if not gcs_url.startswith("gs://"):
                logger.error("Invalid GCS URL: %s", gcs_url)
                return None
            
            # Extract bucket and blob path
            parts = gcs_url[5:].split("/", 1)
            if len(parts) != 2:
                logger.error("Invalid GCS URL format: %s", gcs_url)
                return None
            
            bucket_name, blob_path = parts
            
            # Initialize GCS client
            credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
            if credentials_json and os.path.exists(credentials_json):
                client = storage.Client.from_service_account_json(credentials_json)
            else:
                # Use default credentials
                client = storage.Client()
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Check if blob exists
            if not blob.exists():
                logger.warning("Recording not found in GCS: %s", gcs_url)
                return None
            
            # Get blob size
            blob.reload()
            file_size = blob.size
            
            # Cleanup buffer if needed
            self.cleanup_old_files(required_space=file_size)
            
            # Download to local buffer
            local_path = self.buffer_path / f"{call_id}.ogg"
            blob.download_to_filename(str(local_path))
            
            logger.info(
                "Downloaded recording to local buffer: %s (%d MB)",
                local_path.name,
                file_size / (1024 * 1024)
            )
            
            return local_path
            
        except Exception as exc:
            logger.error("Error downloading from GCS: %s", exc, exc_info=True)
            return None
    
    def get_local_recording(self, call_id: str) -> Optional[Path]:
        """Get local recording path if it exists"""
        local_path = self.buffer_path / f"{call_id}.ogg"
        if local_path.exists():
            return local_path
        return None
    
    def get_buffer_stats(self) -> dict:
        """Get buffer statistics"""
        files = self.get_oldest_files()
        current_size = sum(size for _, size in [(f, f.stat().st_size) for f, _ in files])
        
        return {
            "total_files": len(files),
            "total_size_mb": current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": (current_size / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0,
            "oldest_file": files[0][0].name if files else None,
            "newest_file": files[-1][0].name if files else None,
        }


def attach_recorder_to_session(
    session: AgentSession,
    recorder: CallRecorder
) -> None:
    """
    Attach a CallRecorder to an AgentSession to track transcriptions
    
    Note: LiveKit Agents framework handles transcription internally.
    This function is kept for compatibility but transcription tracking
    needs to be implemented differently using the chat context.
    """
    logger.info("CallRecorder attached to AgentSession (transcription tracking via chat context)")
