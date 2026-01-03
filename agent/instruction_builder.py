"""
Instruction Builder Module.

Builds agent instructions for VoiceAssistant.
Extracted from entry.py for modular architecture.

Phase 17: Systematic instruction architecture:
- direction: 'inbound' | 'outbound' (only include relevant instructions)
- tool_config: Generate tool instructions dynamically from ToolConfig
- tenant_id: For tenant-specific template instructions

Components:
- InstructionBuilder: Assembles system + agent + direction + tool + context instructions

Phase 17c: Removed all hardcoded tool guides. Tool instructions are now fully dynamic
from tool_builder.py via get_tool_instructions() and get_template_instructions_for_tenant().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# INSTRUCTION BUILDER
# =============================================================================

class InstructionBuilder:
    """
    Builds combined instructions for VoiceAssistant.
    
    Phase 17: Systematic instruction architecture
    
    Assembles (in order):
    - System instructions (behavioral rules)
    - Direction-specific instructions (inbound OR outbound, not both)
    - Agent instructions (conversation flow from DB)
    - Tool instructions (only for enabled tools, from tool_builder)
    - Template instructions (dynamic from DB for tenant)
    - Context (call-specific lead info)
    """
    
    def __init__(
        self,
        system_instructions: str | None = None,
        agent_instructions: str | None = None,
        added_context: str | None = None,
        vertical: str | None = None,
        direction: str = "outbound",  # Phase 17: 'inbound' | 'outbound'
        tool_config: Any | None = None,  # Phase 17: ToolConfig
        tenant_id: str | None = None,  # Phase 17: Multi-tenancy
    ):
        self.system_instructions = system_instructions
        self.agent_instructions = agent_instructions
        self.added_context = added_context
        self.vertical = vertical or "general"
        self.direction = direction
        self.tool_config = tool_config
        self.tenant_id = tenant_id
    
    def build(self) -> str:
        """
        Build the complete instruction set.
        
        Phase 17c: Fully dynamic - no hardcoded tool guides.
        
        Order:
        1. System instructions
        2. Direction instructions (inbound OR outbound)
        3. Agent instructions  
        4. Tool instructions (from ToolConfig via tool_builder)
        5. Template instructions (from DB for tenant)
        6. Added context
        
        Returns:
            Combined instructions string
        """
        sections: list[str] = []
        
        # 1. System instructions
        system_block = (self.system_instructions or "").strip()
        if system_block:
            sections.append("# System Instructions\n" + system_block)
        
        # 2. Direction-specific instructions (Phase 17)
        direction_block = self._get_direction_instructions()
        if direction_block:
            sections.append(
                f"####################\n# {self.direction.title()} Call Instructions\n" + direction_block
            )
        
        # 3. Agent instructions
        agent_block = (self.agent_instructions or "").strip()
        if agent_block:
            sections.append("####################\n# Agent Instructions\n" + agent_block)
        
        # 4. Tool instructions (Phase 17: from ToolConfig - REQUIRED)
        if self.tool_config:
            from agent.tool_builder import get_tool_instructions
            tool_guide = get_tool_instructions(self.tool_config)
            if tool_guide:
                sections.append(
                    "####################\n# Available Tools Reference\n" + tool_guide
                )
        else:
            # No tool_config means no tools attached - log warning
            logger.warning("No tool_config provided - no tool instructions will be included")
        
        # 5. Template instructions (Phase 17b: dynamic from DB if templates enabled)
        if self.tool_config and self.tenant_id:
            if getattr(self.tool_config, 'email_templates', False) or getattr(self.tool_config, 'glinks_email', False):
                # Import here to avoid circular imports
                import asyncio
                from agent.tool_builder import get_template_instructions_for_tenant
                
                try:
                    # Run async function synchronously (safe in this context)
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're in an async context, can't run synchronously
                        # Schedule for later - this path should be avoided
                        logger.warning("Cannot fetch template instructions in running loop")
                    else:
                        template_guide = loop.run_until_complete(
                            get_template_instructions_for_tenant(self.tenant_id)
                        )
                        if template_guide:
                            sections.append(
                                "####################\n# Email Template Guide\n" + template_guide
                            )
                except Exception as exc:
                    logger.error(f"Failed to get template instructions: {exc}")
        
        # 6. Added context
        if isinstance(self.added_context, str):
            context_block = self.added_context.strip()
            if context_block:
                sections.append(
                    "####################\n# Added Context For This Call\n" + context_block
                )
                logger.info("Added context: %d chars", len(context_block))
        
        return "\n\n".join(sections) if sections else ""
    
    async def build_async(self) -> str:
        """
        Build instructions asynchronously (for use in async contexts).
        
        Phase 17c: Use this when in an async context to properly
        await get_template_instructions_for_tenant().
        """
        sections: list[str] = []
        
        # 1. System instructions
        system_block = (self.system_instructions or "").strip()
        if system_block:
            sections.append("# System Instructions\n" + system_block)
        
        # 2. Direction-specific instructions
        direction_block = self._get_direction_instructions()
        if direction_block:
            sections.append(
                f"####################\n# {self.direction.title()} Call Instructions\n" + direction_block
            )
        
        # 3. Agent instructions
        agent_block = (self.agent_instructions or "").strip()
        if agent_block:
            sections.append("####################\n# Agent Instructions\n" + agent_block)
        
        # 4. Tool instructions
        if self.tool_config:
            from agent.tool_builder import get_tool_instructions
            tool_guide = get_tool_instructions(self.tool_config)
            if tool_guide:
                sections.append(
                    "####################\n# Available Tools Reference\n" + tool_guide
                )
        
        # 5. Template instructions (async!)
        if self.tool_config and self.tenant_id:
            if getattr(self.tool_config, 'email_templates', False) or getattr(self.tool_config, 'glinks_email', False):
                from agent.tool_builder import get_template_instructions_for_tenant
                template_guide = await get_template_instructions_for_tenant(self.tenant_id)
                if template_guide:
                    sections.append(
                        "####################\n# Email Template Guide\n" + template_guide
                    )
        
        # 6. Added context
        if isinstance(self.added_context, str):
            context_block = self.added_context.strip()
            if context_block:
                sections.append(
                    "####################\n# Added Context For This Call\n" + context_block
                )
        
        return "\n\n".join(sections) if sections else ""
    
    def _get_direction_instructions(self) -> str:
        """
        Get direction-specific instructions (inbound or outbound).
        
        Phase 17: Only include relevant direction instructions, not both.
        """
        if self.direction == "outbound":
            return """
You are making an OUTBOUND call to the customer/lead.
- Introduce yourself clearly at the start
- State the purpose of your call early
- Be respectful of their time
- If they seem busy, offer to call back later
"""
        else:  # inbound
            return """
This is an INBOUND call - the customer called you.
- Thank them for calling
- Ask how you can help them today
- Listen carefully to their request before responding
"""


def build_instructions(
    system_instructions: str | None = None,
    agent_instructions: str | None = None,
    added_context: str | None = None,
    vertical: str | None = None,
    direction: str = "outbound",
    tool_config: Any | None = None,
    tenant_id: str | None = None,
) -> str:
    """
    Convenience function to build agent instructions.
    
    Phase 17c: Updated to pass all params to InstructionBuilder.
    
    Args:
        system_instructions: High-level behavioral rules
        agent_instructions: Detailed conversation flow
        added_context: Call-specific context
        vertical: Business vertical (legacy, largely ignored now)
        direction: 'inbound' or 'outbound'
        tool_config: ToolConfig for enabled tools
        tenant_id: Tenant UUID for template lookup
        
    Returns:
        Combined instruction string
    """
    builder = InstructionBuilder(
        system_instructions=system_instructions,
        agent_instructions=agent_instructions,
        added_context=added_context,
        vertical=vertical,
        direction=direction,
        tool_config=tool_config,
        tenant_id=tenant_id,
    )
    return builder.build()


async def build_instructions_async(
    system_instructions: str | None = None,
    agent_instructions: str | None = None,
    added_context: str | None = None,
    vertical: str | None = None,
    direction: str = "outbound",
    tool_config: Any | None = None,
    tenant_id: str | None = None,
) -> str:
    """
    Async convenience function to build agent instructions.
    
    Use this when in an async context.
    """
    builder = InstructionBuilder(
        system_instructions=system_instructions,
        agent_instructions=agent_instructions,
        added_context=added_context,
        vertical=vertical,
        direction=direction,
        tool_config=tool_config,
        tenant_id=tenant_id,
    )
    return await builder.build_async()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "InstructionBuilder",
    "build_instructions",
    "build_instructions_async",
]
