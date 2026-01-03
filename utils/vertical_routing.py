"""
Vertical Routing Module
=======================

Routes extracted lead data to vertical-specific tables based on tenant slug.

Vertical Detection:
- Reads tenant slug from lad_dev.tenants
- Parses slug format: "{vertical}_{client}" or "{vertical}-{client}"
- Routes to appropriate extractor/storage

Supported Verticals:
- education: Stores in lad_dev.education_students (uses analysis.lad_dev.StudentExtractor)
- realestate: (future) lad_dev.realestate_leads  
- general: Only stores in voice_call_analysis.lead_extraction

Usage:
    from utils.vertical_routing import route_lead_extraction
    
    # After main post-call analysis
    await route_lead_extraction(
        call_log_id=call_log_id,
        tenant_id=tenant_id,
        conversation=conversation,  # List of {"role": "...", "message": "..."}
        lead_id=lead_id,
        db_config=db_config
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import vertical detection
try:
    from utils.tenant_utils import get_vertical_from_tenant_id_sync
    TENANT_UTILS_AVAILABLE = True
except ImportError:
    TENANT_UTILS_AVAILABLE = False
    logger.warning("tenant_utils not available - vertical routing disabled")

# Import student extractor for education vertical
try:
    from analysis.lad_dev import StudentExtractor
    STUDENT_EXTRACTOR_AVAILABLE = True
except ImportError:
    STUDENT_EXTRACTOR_AVAILABLE = False
    logger.debug("StudentExtractor not available - education vertical routing disabled")


class VerticalRoutingResult:
    """Result of vertical routing operation."""
    
    def __init__(
        self,
        vertical: str,
        routed: bool,
        target_table: Optional[str] = None,
        error: Optional[str] = None,
        extracted_data: Optional[Dict[str, Any]] = None
    ):
        self.vertical = vertical
        self.routed = routed
        self.target_table = target_table
        self.error = error
        self.extracted_data = extracted_data
    
    def __repr__(self):
        return f"VerticalRoutingResult(vertical={self.vertical}, routed={self.routed}, table={self.target_table})"


def _format_conversation_for_extraction(conversation: list[Dict[str, Any]]) -> str:
    """
    Format conversation list to text for extraction.
    
    Args:
        conversation: List of {"role": "agent/user", "message": "..."}
        
    Returns:
        Formatted text for LLM extraction
    """
    if not conversation:
        return ""
    
    lines = []
    for turn in conversation:
        role = turn.get("role", "unknown").capitalize()
        # Map roles to expected format
        if role.lower() == "agent":
            role = "Agent"
        elif role.lower() == "user":
            role = "User"
        
        message = turn.get("message", "")
        if message:
            lines.append(f"{role}: {message}")
    
    return "\n".join(lines)


async def route_lead_extraction(
    call_log_id: str,
    tenant_id: str,
    conversation: list[Dict[str, Any]],
    lead_id: Optional[str] = None,
    db_config: Optional[Dict[str, Any]] = None
) -> VerticalRoutingResult:
    """
    Route lead extraction to vertical-specific storage based on tenant.
    
    This function should be called AFTER the main post-call analysis has stored
    the generic lead_extraction in voice_call_analysis table.
    
    Args:
        call_log_id: UUID of the call log
        tenant_id: UUID of the tenant
        conversation: List of conversation turns [{"role": "agent/user", "message": "..."}]
        lead_id: Optional lead UUID to link with extracted data
        db_config: Database configuration dict
        
    Returns:
        VerticalRoutingResult with routing outcome
    """
    if not tenant_id:
        logger.debug("No tenant_id provided, skipping vertical routing")
        return VerticalRoutingResult(
            vertical="unknown",
            routed=False,
            error="No tenant_id provided"
        )
    
    if not TENANT_UTILS_AVAILABLE:
        logger.warning("tenant_utils not available, skipping vertical routing")
        return VerticalRoutingResult(
            vertical="unknown",
            routed=False,
            error="tenant_utils not available"
        )
    
    # Determine vertical from tenant
    vertical = get_vertical_from_tenant_id_sync(tenant_id)
    logger.info(f"Vertical routing: tenant_id={tenant_id}, vertical={vertical}")
    
    if vertical == "education":
        return await _route_education_vertical(
            call_log_id=call_log_id,
            tenant_id=tenant_id,
            conversation=conversation,
            lead_id=lead_id,
            db_config=db_config
        )
    
    elif vertical == "realestate":
        # Future: Implement realestate vertical routing
        logger.info(f"Realestate vertical detected but not yet implemented")
        return VerticalRoutingResult(
            vertical="realestate",
            routed=False,
            error="Realestate vertical not yet implemented"
        )
    
    else:
        # General vertical - no additional routing needed
        logger.debug(f"General vertical, no additional routing needed")
        return VerticalRoutingResult(
            vertical="general",
            routed=False,
            target_table=None
        )


async def _route_education_vertical(
    call_log_id: str,
    tenant_id: str,
    conversation: list[Dict[str, Any]],
    lead_id: Optional[str],
    db_config: Optional[Dict[str, Any]]
) -> VerticalRoutingResult:
    """
    Route to education vertical - extract student info and store in education_students.
    """
    if not STUDENT_EXTRACTOR_AVAILABLE:
        logger.warning("StudentExtractor not available for education vertical")
        return VerticalRoutingResult(
            vertical="education",
            routed=False,
            error="StudentExtractor not available"
        )
    
    if not db_config:
        logger.warning("No db_config provided for education vertical routing")
        return VerticalRoutingResult(
            vertical="education",
            routed=False,
            error="No db_config provided"
        )
    
    try:
        # Format conversation for extraction
        conversation_text = _format_conversation_for_extraction(conversation)
        
        if not conversation_text:
            logger.info("Empty conversation, skipping education extraction")
            return VerticalRoutingResult(
                vertical="education",
                routed=False,
                error="Empty conversation"
            )
        
        # Extract student information using Gemini (async)
        extractor = StudentExtractor()
        student_info = await extractor.extract_student_information(conversation_text)
        
        if not student_info:
            logger.info(f"No student info extracted from call {call_log_id}")
            return VerticalRoutingResult(
                vertical="education",
                routed=False,
                error="No student info extracted"
            )
        
        # Save to education_students table
        saved = extractor.save_to_database(
            student_info=student_info,
            call_log_id=call_log_id,
            lead_id=lead_id,
            tenant_id=tenant_id,
            db_config=db_config
        )
        
        if saved:
            logger.info(f"Education vertical: Saved student info for call {call_log_id}")
            return VerticalRoutingResult(
                vertical="education",
                routed=True,
                target_table="lad_dev.education_students",
                extracted_data=student_info
            )
        else:
            logger.warning(f"Failed to save student info for call {call_log_id}")
            return VerticalRoutingResult(
                vertical="education",
                routed=False,
                error="Failed to save to database"
            )
            
    except Exception as e:
        logger.error(f"Education vertical routing failed for call {call_log_id}: {e}", exc_info=True)
        return VerticalRoutingResult(
            vertical="education",
            routed=False,
            error=str(e)
        )


# Convenience function for sync contexts
def route_lead_extraction_sync(
    call_log_id: str,
    tenant_id: str,
    conversation: list[Dict[str, Any]],
    lead_id: Optional[str] = None,
    db_config: Optional[Dict[str, Any]] = None
) -> VerticalRoutingResult:
    """Synchronous wrapper for route_lead_extraction."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context - create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    route_lead_extraction(call_log_id, tenant_id, conversation, lead_id, db_config)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                route_lead_extraction(call_log_id, tenant_id, conversation, lead_id, db_config)
            )
    except RuntimeError:
        # No event loop - create one
        return asyncio.run(
            route_lead_extraction(call_log_id, tenant_id, conversation, lead_id, db_config)
        )
