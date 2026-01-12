"""
LiveKit Voice Agent Worker - V2 Entry Point.

This module implements the main LiveKit worker that handles inbound and outbound calls.
It uses the modular v2/agent components for:
- Pipeline (TTS/LLM configuration)
- Config (VAD/STT settings)
- Tool Builder (agent tools)
- Instruction Builder (prompt assembly)
- Cleanup Handler (post-call cleanup)

Key Features:
- Multi-provider TTS support (Cartesia, Google, Gemini, ElevenLabs, Rime, SmallestAI)
- Multi-provider LLM support (Gemini, Groq, OpenAI)
- Real-time call recording and transcription
- Silence detection and automatic hangup
- Background audio (office ambience, typing sounds)
- Post-call analysis and storage
- Configurable voice personalities and instructions
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from dotenv import load_dotenv

# Load environment variables with explicit path (works when run from any directory)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(_env_path)

# ============================================================================
# AGENT LOG LEVEL CONFIGURATION
# ============================================================================
# AGENT_LOG_LEVEL takes precedence, falls back to LOG_LEVEL
def _resolve_log_level(value):
    if value is None:
        return logging.INFO
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, 
                 "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    return level_map.get(value.strip().upper(), logging.INFO)

_agent_log_level = _resolve_log_level(
    os.getenv("AGENT_LOG_LEVEL") or os.getenv("LOG_LEVEL")
)
logging.basicConfig(level=_agent_log_level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

from livekit import agents, api
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    MetricsCollectedEvent,
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.agents import StopResponse
from livekit.plugins import deepgram, silero

# Import V2 modular components
from agent.config import PipelineConfig, get_config
from agent.pipeline import (
    build_tts_engine,
    resolve_llm_configuration,
    create_llm_instance,
    derive_stt_language,
)
from agent.tool_builder import ToolConfig, get_enabled_tools, attach_tools
from agent.instruction_builder import build_instructions_async
from agent.cleanup_handler import CleanupContext, cleanup_and_save

# Storage (v2 internal)
from db.storage.calls import CallStorage
from db.storage.agents import AgentStorage

# Recording and transcription (v2 internal)
from recording.recorder import CallRecorder
from recording.transcription import attach_transcription_tracker

# Utilities (v2 internal)
from utils.usage_tracker import UsageCollector, is_component_tracking_enabled
from utils.logger_config import configure_non_blocking_logging, stop_logging

# Load environment variables
load_dotenv()

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

_log_listener = configure_non_blocking_logging()
logger = logging.getLogger(__name__)

# Agent version
AGENT_VERSION = os.getenv("AGENT_VERSION", "2.0.0")

# Max concurrent calls
_MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "1"))
_call_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CALLS)


# =============================================================================
# WORKER LOAD MANAGEMENT
# =============================================================================

def calculate_worker_load(worker: Any) -> float:
    """
    Calculate worker load to control job distribution.
    
    Returns a value between 0.0 and 1.0 based on active jobs vs max capacity.
    When load >= 1.0, the worker rejects new jobs.
    """
    active_jobs = len(worker.active_jobs) if hasattr(worker, 'active_jobs') else 0
    if _MAX_CONCURRENT_CALLS <= 0:
        return 1.0 if active_jobs > 0 else 0.0
    return active_jobs / _MAX_CONCURRENT_CALLS


# =============================================================================
# SILENCE DETECTION AND MONITORING
# =============================================================================

@dataclass(frozen=True)
class SilenceMilestone:
    """Represents a milestone during silence detection."""
    seconds: float
    callback: Callable[[float], Awaitable[None] | None]
    label: str = ""


class SilenceMonitor:
    """
    Monitors user silence after agent speech and triggers actions.
    
    Tracks periods of silence after the agent finishes speaking. Can trigger
    warnings and automatic hangup after prolonged silence.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        on_timeout: Callable[[], Awaitable[None] | None],
        logger: logging.Logger,
        milestones: Sequence[SilenceMilestone] | None = None,
    ) -> None:
        self._timeout = max(timeout_seconds, 1.0)
        self._on_timeout = on_timeout
        self._logger = logger
        self._timer_task: asyncio.Task | None = None
        self._enabled = True
        self._triggered = False
        self._milestones = tuple(
            sorted(
                (m for m in (milestones or []) if m.seconds > 0.0),
                key=lambda item: item.seconds,
            )
        )
        self._milestones_fired: set[int] = set()
        self._remaining: float | None = None
        self._elapsed: float = 0.0
        self._last_start: float | None = None
        self._pending_callbacks: set[asyncio.Task] = set()

    def notify_agent_started(self) -> None:
        """Pause silence monitoring when agent starts speaking."""
        if not self._enabled:
            return
        self._pause_timer()

    def notify_agent_completed(self) -> None:
        """Resume silence monitoring after agent finishes speaking."""
        if not self._enabled:
            return
        self._triggered = False
        if self._remaining is None or self._remaining <= 0.0:
            self._remaining = self._timeout
            self._elapsed = 0.0
            self._milestones_fired.clear()
        self._start_timer()

    def notify_user_activity(self) -> None:
        """Reset silence monitoring when user speaks."""
        if not self._enabled:
            return
        self._triggered = False
        self._cancel_pending_callbacks()
        self._reset_state()
        self._cancel_timer()

    def disable(self) -> None:
        """Permanently disable silence monitoring."""
        if not self._enabled:
            return
        self._enabled = False
        self._triggered = True
        self._cancel_pending_callbacks()
        self._reset_state()
        self._cancel_timer()

    def _start_timer(self) -> None:
        self._cancel_timer()
        if not self._enabled:
            return
        if self._remaining is None:
            self._remaining = self._timeout
            self._elapsed = 0.0
            self._milestones_fired.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._last_start = loop.time()
        self._timer_task = loop.create_task(self._await_timeout())

    def _pause_timer(self) -> None:
        if self._timer_task is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and self._last_start is not None:
            elapsed = max(0.0, loop.time() - self._last_start)
            self._elapsed += elapsed
            self._remaining = max(self._timeout - self._elapsed, 0.0)
        self._last_start = None
        self._cancel_timer()

    def _reset_state(self) -> None:
        self._remaining = None
        self._elapsed = 0.0
        self._last_start = None
        self._milestones_fired.clear()

    def _cancel_pending_callbacks(self) -> None:
        for task in tuple(self._pending_callbacks):
            if not task.done():
                task.cancel()

    def _cancel_timer(self) -> None:
        if self._timer_task is None:
            return
        task = self._timer_task
        self._timer_task = None
        if not task.done():
            task.cancel()

    async def _await_timeout(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            while (
                self._enabled
                and not self._triggered
                and self._remaining is not None
                and self._remaining > 0.0
            ):
                next_wait = self._remaining
                for idx, milestone in enumerate(self._milestones):
                    if idx in self._milestones_fired:
                        continue
                    remaining_to_milestone = milestone.seconds - self._elapsed
                    if remaining_to_milestone <= 0.0:
                        self._milestones_fired.add(idx)
                        continue
                    next_wait = min(next_wait, remaining_to_milestone)

                if self._remaining <= 0.0:
                    break

                if next_wait <= 0.0:
                    continue

                await asyncio.sleep(next_wait)

                now = loop.time()
                if self._last_start is None:
                    self._last_start = now - next_wait

                elapsed = max(0.0, now - self._last_start)
                self._elapsed += elapsed
                self._remaining = max(self._timeout - self._elapsed, 0.0)
                self._last_start = now

            if (
                self._enabled
                and not self._triggered
                and self._remaining is not None
                and self._remaining <= 0.0
            ):
                self._triggered = True
                self._logger.info("User silence exceeded %.1fs; triggering hangup", self._timeout)
                result = self._on_timeout()
                if asyncio.iscoroutine(result):
                    await result
        except asyncio.CancelledError:
            raise
        except Exception:
            self._logger.error("Error during silence timeout handling", exc_info=True)
        finally:
            self._timer_task = None
            self._last_start = None


# =============================================================================
# VOICE ASSISTANT AGENT
# =============================================================================

class VoiceAssistant(Agent):
    """
    Main voice assistant agent that handles conversations.
    
    Extends LiveKit's Agent class with custom functionality:
    - Call recording integration
    - Transcription tracking
    - Silence monitoring
    - Hangup control
    - Custom instructions and context
    """
    
    GLINKS_ORG_ID = "f6de7991-df4f-43de-9f40-298fcda5f723"
    
    def __init__(
        self,
        call_recorder=None,
        job_context=None,
        *,
        instructions: str,
        tools: list | None = None,
        silence_monitor: SilenceMonitor | None = None,
        initiating_user_id: str | int | None = None,
        is_glinks_agent: bool = False,
        knowledge_base_store_ids: list[str] | None = None,
    ):
        self.call_recorder = call_recorder
        self.job_context = job_context
        self.silence_monitor = silence_monitor
        self._is_glinks_agent = is_glinks_agent
        
        # Hangup control flags
        self._hangup_pending = False
        self._hangup_cancelled = False
        
        if isinstance(initiating_user_id, int):
            self._initiating_user_id: str | None = str(initiating_user_id)
        elif isinstance(initiating_user_id, str):
            self._initiating_user_id = initiating_user_id.strip() or None
        else:
            self._initiating_user_id = None
            
        self._google_workspace = None
        self._human_invite_pending = False
        self._human_joined = False
        self._knowledge_base_store_ids = knowledge_base_store_ids or []
        
        # Register interruption callback to cancel pending hangup
        if self.call_recorder and hasattr(self.call_recorder, 'register_agent_speech_end_callback'):
            self.call_recorder.register_agent_speech_end_callback(self._on_agent_speech_end)
        
        # Pass tools from tool_builder to parent Agent
        # NOTE: @function_tool decorated methods (like hangup_call) are auto-registered by Agent base class
        super().__init__(instructions=instructions, tools=tools or [])
    
    def _on_agent_speech_end(self, was_interrupted: bool) -> None:
        """Callback when agent speech ends - cancels pending hangup if interrupted."""
        if was_interrupted and self._hangup_pending:
            logger.info("Agent speech interrupted during hangup - cancelling hangup")
            self._hangup_cancelled = True

    @function_tool
    async def hangup_call(self, reason: str = "call_complete") -> str:
        """
        End the current call gracefully.
        
        Waits for TTS parting words to complete before hanging up.
        If human interrupts during parting words, hangup is cancelled.
        
        Args:
            reason: Reason for ending (e.g., "call_complete", "not_interested")
        
        Returns:
            Confirmation message
        """
        import asyncio
        
        logger.info(f"Agent initiating graceful hangup (reason: {reason})")
        
        # Disable silence monitor to prevent timeout during goodbye
        if self.silence_monitor:
            self.silence_monitor.disable()
        
        # Mark hangup as pending (for interruption detection)
        self._hangup_pending = True
        self._hangup_cancelled = False
        
        try:
            # Wait for TTS parting words to complete (if recorder available)
            if self.call_recorder:
                try:
                    tts_completed = await self.call_recorder.wait_for_tts_playout(timeout=20.0)
                    if not tts_completed:
                        logger.warning("TTS playout wait timed out, proceeding with hangup")
                except Exception as e:
                    logger.warning(f"Error waiting for TTS playout: {e}")
            
            # Check if hangup was cancelled due to interruption
            if getattr(self, '_hangup_cancelled', False):
                logger.info("Hangup cancelled - human interrupted during parting words")
                self._hangup_pending = False
                return "Hangup cancelled - user is speaking"
            
            # Wait 1 second after TTS for natural pause
            await asyncio.sleep(1.0)
            
            # Final check for interruption during the pause
            if getattr(self, '_hangup_cancelled', False):
                logger.info("Hangup cancelled during post-TTS pause")
                self._hangup_pending = False
                return "Hangup cancelled - user started speaking"
            
            # Execute hangup
            if self.job_context and self.job_context.room:
                try:
                    await self.job_context.api.room.delete_room(
                        api.DeleteRoomRequest(room=self.job_context.room.name)
                    )
                    logger.info(f"Room {self.job_context.room.name} deleted after graceful goodbye")
                except Exception as e:
                    logger.error(f"Error deleting room: {e}")
                    return f"Error ending call: {e}"
            
            return f"Call ended: {reason}"
            
        finally:
            self._hangup_pending = False
    
    def cancel_pending_hangup(self) -> None:
        """Cancel a pending hangup if human interrupts during parting words."""
        if getattr(self, '_hangup_pending', False):
            self._hangup_cancelled = True
            logger.info("Pending hangup marked for cancellation")

    def set_silence_monitor(self, monitor: SilenceMonitor | None) -> None:
        self.silence_monitor = monitor
    
    def mute_for_human_handoff(self) -> None:
        """
        Prepare for human handoff by muting AI agent.
        
        - Disables silence monitor (human agent will talk)
        - Sets _human_joined flag to prevent AI from interrupting
        - hangup_call tool becomes a no-op (call ends when participants leave)
        """
        logger.info("Muting AI agent for human handoff")
        
        # Disable silence monitoring - human agent will be talking
        if self.silence_monitor:
            self.silence_monitor.disable()
        
        # Mark human as joined - prevents AI from generating responses
        self._human_joined = True
        self._human_invite_pending = False
        
        logger.info("AI agent muted: silence_monitor=disabled, _human_joined=True")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

async def entrypoint(ctx: agents.JobContext):
    """
    Main entry point for the LiveKit voice agent.
    
    Handles the complete lifecycle of a call:
    1. Parse job metadata
    2. Initialize AI services (STT, LLM, TTS)
    3. Set up call recording and transcription
    4. Configure silence monitoring
    5. Connect to LiveKit room
    6. Start agent session
    7. Handle call flow
    8. Clean up on completion
    """
    logger.info(f"Agent v{AGENT_VERSION} starting job {getattr(ctx.job, 'id', 'unknown')}")
    
    # Acquire call slot
    await _call_semaphore.acquire()
    acquired_call_slot = True
    
    # Get pipeline config
    pipeline_config = get_config()
    
    # Initialize storage
    call_storage = CallStorage()
    agent_storage = AgentStorage()
    
    # Parse metadata
    phone_number = None
    voice_id = os.getenv("DEFAULT_CARTESIA_VOICE_ID", "95d51f79-c397-46f9-b49a-23763d3eaa2d")
    job_id = None
    call_log_id = None
    call_mode = "inbound"
    from_number = None
    to_number = None
    added_context = None
    agent_id: int | None = None
    tts_provider_override = None
    tts_overrides: dict[str, str] = {}
    voice_accent = None
    llm_provider_override = None
    llm_model_override = None
    initiating_user_id: str | None = None
    lead_id: str | None = None  # For vertical routing
    knowledge_base_store_ids: list[str] = []
    outbound_trunk_id: str | None = None  # SIP trunk ID from number rules
    batch_id: str | None = None  # Batch call tracking
    entry_id: str | None = None  # Batch entry tracking
    
    try:
        if ctx.job.metadata:
            dial_info = json.loads(ctx.job.metadata)
            phone_number = dial_info.get("phone_number")
            to_number = phone_number
            job_id = dial_info.get("job_id")
            call_log_id = dial_info.get("call_log_id")
            lead_id = dial_info.get("lead_id")  # For vertical routing
            call_mode = dial_info.get("call_mode", "inbound")
            from_number = dial_info.get("from_number")
            added_context = dial_info.get("added_context")
            outbound_trunk_id = dial_info.get("outbound_trunk_id")
            batch_id = dial_info.get("batch_id")  # For batch completion tracking
            entry_id = dial_info.get("entry_id")  # For batch entry status update
            
            raw_agent_id = dial_info.get("agent_id")
            if isinstance(raw_agent_id, (int, str)):
                try:
                    agent_id = int(str(raw_agent_id).strip())
                except (ValueError, TypeError):
                    pass
            
            raw_kb_ids = dial_info.get("knowledge_base_store_ids")
            if isinstance(raw_kb_ids, list):
                knowledge_base_store_ids = [s.strip() for s in raw_kb_ids if isinstance(s, str) and s.strip()]
            
            voice_id = dial_info.get("voice_id", voice_id)
            # Prefer tts_voice_id (provider-specific) over voice_id (DB UUID)
            tts_voice_id = dial_info.get("tts_voice_id")
            if tts_voice_id:
                voice_id = tts_voice_id  # Use the Cartesia/ElevenLabs voice ID
            voice_accent = dial_info.get("voice_accent")
            # Check both tts_provider and voice_provider (call_service uses voice_provider)
            tts_provider_override = dial_info.get("tts_provider") or dial_info.get("voice_provider")
            llm_provider_override = dial_info.get("llm_provider")
            llm_model_override = dial_info.get("llm_model")
            
            # Parse TTS config overrides (speed/pitch/stability from provider_config)
            raw_tts_config = dial_info.get("tts_config")
            if isinstance(raw_tts_config, dict):
                for k, v in raw_tts_config.items():
                    if v is not None:
                        tts_overrides[k] = str(v)
                if tts_overrides:
                    logger.info(f"TTS config overrides from DB: {tts_overrides}")
            
            raw_initiated_by = dial_info.get("initiated_by")
            if isinstance(raw_initiated_by, (int, str)):
                initiating_user_id = str(raw_initiated_by).strip() or None
            
            # DEBUG: Log initiated_by parsing for OAuth tool debugging
            logger.info(
                "OAuth user context: raw_initiated_by=%s, initiating_user_id=%s",
                raw_initiated_by, initiating_user_id
            )
                
            logger.info(
                "Metadata: job_id=%s, call_log_id=%s, agent_id=%s, voice_id=%s",
                job_id, call_log_id, agent_id, voice_id
            )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse metadata: {e}")
    
    # Warn if call_log_id is missing - this should never happen for outbound calls
    if not call_log_id:
        logger.warning(
            "call_log_id is None after metadata parsing. "
            "job_id=%s, call_mode=%s, agent_id=%s, raw_metadata=%s",
            job_id, call_mode, agent_id, ctx.job.metadata[:200] if ctx.job.metadata else None
        )
    
    # Connect to room
    await ctx.connect()
    
    # Load agent configuration
    agent_record = None
    is_glinks_agent = False
    system_instructions = ""
    agent_instructions = ""
    
    if agent_id is not None:
        agent_record = await agent_storage.get_agent_by_id(agent_id)
        if agent_record:
            # Column names match the agents.py SELECT query
            system_instructions = agent_record.get("system_instructions", "") or ""
            agent_instructions = agent_record.get("agent_instructions", "") or ""
            
            # Check if Glinks agent
            org_id = agent_record.get("organization_id")
            if org_id and str(org_id) == VoiceAssistant.GLINKS_ORG_ID:
                is_glinks_agent = True
    
    # Phase 4: Get tenant_id for multi-tenancy (from USER's primary_tenant_id, not agent)
    tenant_id = None
    if initiating_user_id:
        from db.storage.tokens import UserTokenStorage
        token_storage = UserTokenStorage()
        tenant_id = await token_storage.get_user_tenant_id(initiating_user_id)
        if tenant_id:
            logger.debug(f"Resolved tenant_id={tenant_id} from user_id={initiating_user_id}")
    # Fallback: try agent's tenant if no user (inbound calls)
    if not tenant_id and agent_id:
        tenant_id = await agent_storage.get_agent_tenant_id(agent_id)
        if tenant_id:
            logger.debug(f"Resolved tenant_id={tenant_id} from agent_id={agent_id} (fallback)")
    # Resolve LLM configuration
    llm_provider, llm_model = resolve_llm_configuration(
        llm_provider_override, llm_model_override
    )
    
    # Build TTS engine
    tts_provider_final = tts_provider_override or os.getenv("TTS_PROVIDER", "cartesia")
    logger.info(f"TTS provider: {tts_provider_final} (override={tts_provider_override})")
    tts_engine, tts_details = build_tts_engine(
        tts_provider_final,
        default_voice_id=voice_id,
        overrides=tts_overrides,
        accent=voice_accent,
    )
    logger.info(f"TTS engine built: {tts_details}")
    
    # Create LLM instance
    file_search_stores = [s for s in knowledge_base_store_ids if s.startswith("fileSearchStores/")]
    llm_instance = create_llm_instance(llm_provider, llm_model, file_search_stores or None)
    
    # Build STT
    stt_language = derive_stt_language(voice_accent, pipeline_config.stt.language)
    stt_engine = deepgram.STT(
        model=pipeline_config.stt.model,
        language=stt_language,
    )
    
    # Build VAD
    vad = silero.VAD.load(
        min_silence_duration=pipeline_config.vad.min_silence_duration,
        min_speech_duration=pipeline_config.vad.min_speech_duration,
        activation_threshold=pipeline_config.vad.activation_threshold,
        force_cpu=pipeline_config.vad.force_cpu,
    )
    
    # Fetch tools BEFORE building instructions (Phase 17c: tool instructions require valid tool_config)
    # NOTE: This must happen before build_instructions_async() so hangup instruction is included!
    tool_config, tool_configs = await get_enabled_tools({}, tenant_id)
    
    # Holder for VoiceAssistant - populated after creation, used by human_support tool
    voice_assistant_holder: dict = {"assistant": None}
    
    tool_list = await attach_tools(
        None,  # Agent not needed here, just building tools
        tool_config,
        tool_configs,
        tenant_id=tenant_id,
        user_id=initiating_user_id,
        knowledge_base_store_ids=knowledge_base_store_ids,
        job_context=ctx,  # For human support SIP transfer
        sip_trunk_id=outbound_trunk_id,  # SIP trunk from call routing
        from_number=from_number,  # For number validation in human support
        voice_assistant_holder=voice_assistant_holder,  # For human handoff muting
    )
    logger.info(f"Built {len(tool_list)} tools for tenant {tenant_id}, user_id={'set' if initiating_user_id else 'None'}")
    
    # Build instructions (now with valid tool_config that includes hangup instructions)
    instructions = await build_instructions_async(
        system_instructions=system_instructions,
        agent_instructions=agent_instructions,
        added_context=added_context,
        direction=call_mode,
        tool_config=tool_config,  # Actual tool config from get_enabled_tools()
        tenant_id=tenant_id,  # Pass actual tenant_id, not None
    )
    
    # Create call recorder
    call_recorder = None
    if os.getenv("ENABLE_RECORDING", "true").lower() == "true" and call_log_id:
        call_recorder = CallRecorder(
            room_name=ctx.room.name,
            call_id=call_log_id,
            livekit_api=ctx.api,
        )
    
    # Phase 1: Create usage collector for cost tracking
    usage_collector = None
    if is_component_tracking_enabled():
        usage_collector = UsageCollector()
        usage_collector.set_tts_config(tts_provider_final, voice_id or "unknown")
        usage_collector.set_llm_config(llm_provider, llm_model)
        usage_collector.set_stt_config("deepgram", pipeline_config.stt.model)
        logger.info(
            "UsageCollector enabled: TTS=%s/%s, LLM=%s/%s, STT=deepgram/%s",
            tts_provider_final, voice_id, llm_provider, llm_model, pipeline_config.stt.model
        )
    else:
        logger.debug("Component cost tracking disabled (ENABLE_COMPONENT_COST_TRACKING=false)")
    # Create silence monitor with warning milestone
    silence_timeout = pipeline_config.silence.silence_timeout_seconds
    
    # Session holder for callbacks (populated after session creation)
    session_holder: dict = {"session": None}

    async def on_silence_warning(elapsed: float):
        """Prompt user when silent for 15s."""
        logger.info(f"Silence warning at {elapsed:.1f}s - prompting user")
        sess = session_holder.get("session")
        if sess:
            try:
                await sess.generate_reply(
                    instructions="The user has been silent for a while. "
                    "Ask if they are still there and if they need any help. "
                    "Keep it brief and friendly."
                )
            except Exception as e:
                logger.warning(f"Failed to send silence warning prompt: {e}")

    async def on_silence_timeout():
        """End call after full silence timeout."""
        logger.info("Silence timeout - ending call")
        if ctx.room:
            try:
                await ctx.api.room.delete_room(api.DeleteRoomRequest(room=ctx.room.name))
            except Exception as e:
                logger.error(f"Error ending call on timeout: {e}")

    # Warning at 15s, hangup at full timeout (typically 30s)
    silence_milestones = [
        SilenceMilestone(
            seconds=15.0,
            callback=on_silence_warning,
            label="first_warning"
        ),
    ]

    silence_monitor = SilenceMonitor(
        timeout_seconds=silence_timeout,
        on_timeout=on_silence_timeout,
        logger=logger,
        milestones=silence_milestones,
    )
    
    # Create voice assistant with tools (tools already fetched above before instruction building)
    voice_assistant = VoiceAssistant(
        call_recorder=call_recorder,
        job_context=ctx,
        instructions=instructions,
        tools=tool_list,  # Pass dynamic tools
        silence_monitor=silence_monitor,
        initiating_user_id=initiating_user_id,
        is_glinks_agent=is_glinks_agent,
        knowledge_base_store_ids=knowledge_base_store_ids,
    )
    
    # Populate holder so human_support tool can access VoiceAssistant
    voice_assistant_holder["assistant"] = voice_assistant
    
    # Create agent session
    session = AgentSession(
        llm=llm_instance,
        tts=tts_engine,
        stt=stt_engine,
        vad=vad,
        min_endpointing_delay=pipeline_config.endpointing.min_endpointing_delay,
        max_endpointing_delay=pipeline_config.endpointing.max_endpointing_delay,
    )
    
    # Populate session reference for silence warning callback
    session_holder["session"] = session
    
    # Hook up session events to silence monitor
    @session.on("agent_started_speaking")
    def _on_agent_start():
        if voice_assistant.silence_monitor:
            voice_assistant.silence_monitor.notify_agent_started()
    
    @session.on("agent_stopped_speaking")
    def _on_agent_stop():
        if voice_assistant.silence_monitor:
            voice_assistant.silence_monitor.notify_agent_completed()
    
    @session.on("user_started_speaking")
    def _on_user_start():
        if voice_assistant.silence_monitor:
            voice_assistant.silence_monitor.notify_user_activity()
    
    # Phase 2: Attach usage collector to session for metrics tracking
    if usage_collector:
        from utils.usage_tracker import attach_usage_collector
        attach_usage_collector(session, usage_collector)
        logger.debug("UsageCollector attached to AgentSession")
    
    # Phase 3: Register cleanup callback with full context
    cleanup_ctx = CleanupContext(
        call_recorder=call_recorder,
        call_log_id=call_log_id,
        tenant_id=tenant_id,  # Phase 4: Now included
        lead_id=lead_id,  # For vertical routing
        call_storage=call_storage,
        silence_monitor=silence_monitor,
        usage_collector=usage_collector,  # Uses the configured one from Phase 1
        call_semaphore=_call_semaphore,
        acquired_call_slot=acquired_call_slot,
        job_id=job_id,
        batch_id=batch_id,  # For batch completion tracking
        entry_id=entry_id,  # For batch entry status update
    )
    
    async def shutdown_callback():
        logger.info("=" * 60)
        logger.info("CALL ENDED - Running cleanup_and_save")
        logger.info("=" * 60)
        await cleanup_and_save(cleanup_ctx)
    
    ctx.add_shutdown_callback(shutdown_callback)
    
    # Start recording
    if call_recorder:
        await call_recorder.start_recording()
    
    # Start session
    room_input_options = RoomInputOptions()
    session_start_task = asyncio.create_task(
        session.start(room=ctx.room, agent=voice_assistant, room_input_options=room_input_options)
    )
    
    try:
        if call_mode == "outbound" and phone_number:
            # Outbound call - dial out, use trunk from metadata or env default
            trunk_id = outbound_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
            if trunk_id:
                try:
                    # Update status to ringing before dialing
                    if call_log_id:
                        await call_storage.update_call_status(
                            call_log_id=call_log_id,
                            status="ringing"
                        )
                        logger.info(f"Call {call_log_id} status updated to ringing")
                    
                    await ctx.api.sip.create_sip_participant(
                        api.CreateSIPParticipantRequest(
                            room_name=ctx.room.name,
                            sip_trunk_id=trunk_id,
                            sip_call_to=phone_number,
                            participant_identity=f"dial-{phone_number}",
                            wait_until_answered=True,
                            krisp_enabled=True,
                        )
                    )
                    # Call is now answered - update status to ongoing
                    if call_log_id:
                        await call_storage.update_call_status(
                            call_log_id=call_log_id,
                            status="ongoing"
                        )
                        logger.info(f"Call {call_log_id} status updated to ongoing")
                    
                    await session_start_task
                    
                    if call_recorder:
                        attach_transcription_tracker(session, call_recorder)
                    
                    # Outbound greeting - try agent starter prompt, then added_context, then default
                    greeting = agent_record.get("outbound_starter_prompt", "") if agent_record else ""
                    if greeting:
                        logger.info(f"Sending outbound greeting: {greeting[:50]}...")
                        await session.generate_reply(instructions=greeting)
                    elif added_context:
                        # Use added_context as greeting instruction
                        logger.info(f"Using added_context as greeting: {added_context[:50]}...")
                        await session.generate_reply(instructions=f"Greet the user. Context: {added_context}")
                    else:
                        # Default: just start the conversation
                        logger.info("No greeting set, starting conversation with default")
                        await session.generate_reply(instructions="Greet the user warmly and ask how you can help them today.")
                        
                except api.TwirpError as e:
                    logger.error(f"SIP dial error: {e.message}")
                    if call_log_id:
                        await call_storage.update_call_status(
                            call_log_id=call_log_id,
                            status="failed",
                            ended_at=datetime.now(timezone.utc)
                        )
                    session_start_task.cancel()
                    ctx.shutdown()
                    return
        else:
            # Inbound call
            await session_start_task
            
            # Inbound call is now active - update status to ongoing
            if call_log_id:
                await call_storage.update_call_status(
                    call_log_id=call_log_id,
                    status="ongoing"
                )
                logger.info(f"Inbound call {call_log_id} status updated to ongoing")
            
            if call_recorder:
                attach_transcription_tracker(session, call_recorder)
            
            greeting = agent_record.get("inbound_starter_prompt", "") if agent_record else ""
            if greeting:
                logger.info(f"Sending inbound greeting: {greeting[:50]}...")
                await session.generate_reply(instructions=greeting)
                
    finally:
        if acquired_call_slot:
            _call_semaphore.release()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting LiveKit agent worker - Version {AGENT_VERSION}")
    logger.info(f"Max concurrent calls: {_MAX_CONCURRENT_CALLS}")
    
    try:
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name=os.getenv("VOICE_AGENT_NAME", "inbound-agent"),
                initialize_process_timeout=60.0,
                shutdown_process_timeout=60.0,
                load_fnc=calculate_worker_load,
                load_threshold=1.0,
                num_idle_processes=0,
            )
        )
    finally:
        stop_logging()
