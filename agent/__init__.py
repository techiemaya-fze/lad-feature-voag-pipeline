"""
Agent module - Core voice agent functionality.

Extracted from entry.py into modular components:
- pipeline.py: TTS engine and LLM configuration
- instruction_builder.py: Prompt and tool guide generation
- tool_builder.py: Tool attachment and configuration
- cleanup_handler.py: Post-call cleanup operations
- config.py: Voice pipeline configuration

Entry.py contains VoiceAssistant and remains as working reference
until full migration is complete.
"""

# Pipeline - TTS and LLM
from agent.pipeline import (
    build_tts_engine,
    create_llm_instance,
    resolve_llm_configuration,
    derive_cartesia_language,
    derive_stt_language,
)

# Instructions
from agent.instruction_builder import (
    InstructionBuilder,
    build_instructions,
    build_instructions_async,
)

# Tools
from agent.tool_builder import (
    ToolConfig,
    ToolBuilder,
    attach_tools,
)

# Cleanup
from agent.cleanup_handler import (
    CleanupContext,
    cleanup_and_save,
)

__all__ = [
    # Pipeline
    "build_tts_engine",
    "create_llm_instance",
    "resolve_llm_configuration",
    "derive_cartesia_language",
    "derive_stt_language",
    # Instructions
    "InstructionBuilder",
    "build_instructions",
    "build_instructions_async",
    # Tools
    "ToolConfig",
    "ToolBuilder",
    "attach_tools",
    # Cleanup
    "CleanupContext",
    "cleanup_and_save",
]
