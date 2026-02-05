"""
Tool Builder Module.

Attaches function tools to VoiceAssistant based on configuration.
Extracted from entry.py for modular architecture.

Phase 14: Added TOOLS_DECIDED_BY_BACKEND env flag for transition.

Tool Categories:
- Call Control: hangup_call
- Calendar: Google Calendar, Microsoft Bookings
- Email: Gmail, email templates
- Search: Knowledge base (RAG)
- Support: Human agent invitation
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Callable

from livekit.agents.llm import function_tool

# Import DB connection utilities at top level
from db.db_config import get_db_config
from db.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

# =============================================================================
# ENVIRONMENT FLAGS
# =============================================================================

# Transition flag: When ON, backend decides tools via tenant_features
# When OFF, frontend sends tools_enabled in payload
TOOLS_DECIDED_BY_BACKEND = os.getenv("TOOLS_DECIDED_BY_BACKEND", "true").lower() == "true"


# =============================================================================
# TOOL CONFIGURATION
# =============================================================================

class ToolConfig:
    """
    Configuration for which tools to enable on a call.
    
    Controls tool attachment based on call payload or tenant settings.
    """
    
    def __init__(
        self,
        google_calendar: bool = False,
        google_workspace: bool = False,
        gmail: bool = False,
        microsoft_bookings_auto: bool = False,  # Auto-book with defaults
        microsoft_bookings_manual: bool = False,  # Agent lists and user selects
        microsoft_outlook: bool = False,  # Send emails via Outlook
        email_templates: bool = False,
        email_templates_microsoft: bool = False,  # MS version of email templates
        knowledge_base: bool = False,
        human_support: bool = False,
        glinks_email: bool = False,
    ):
        self.google_calendar = google_calendar
        self.google_workspace = google_workspace
        self.gmail = gmail
        self.microsoft_bookings_auto = microsoft_bookings_auto
        self.microsoft_bookings_manual = microsoft_bookings_manual
        self.microsoft_outlook = microsoft_outlook
        self.email_templates = email_templates
        self.email_templates_microsoft = email_templates_microsoft
        self.knowledge_base = knowledge_base
        self.human_support = human_support
        self.glinks_email = glinks_email
    
    @classmethod
    def from_dict(cls, data: dict) -> "ToolConfig":
        """Create from tools_enabled dict in call payload."""
        # Support both old 'microsoft_bookings' and new split fields
        ms_auto = data.get("microsoft_bookings_auto", data.get("microsoft_bookings", False))
        ms_manual = data.get("microsoft_bookings_manual", False)
        return cls(
            google_calendar=data.get("google_calendar", False),
            google_workspace=data.get("google_workspace", False),
            gmail=data.get("gmail", False),
            microsoft_bookings_auto=ms_auto,
            microsoft_bookings_manual=ms_manual,
            microsoft_outlook=data.get("microsoft_outlook", False),
            email_templates=data.get("email_templates", False),
            email_templates_microsoft=data.get("email_templates_microsoft", False),
            knowledge_base=data.get("knowledge_base", False),
            human_support=data.get("human_support", False),
            glinks_email=data.get("glinks_email", False),
        )
    
    @classmethod
    def all_enabled(cls) -> "ToolConfig":
        """Create with all tools enabled."""
        return cls(
            google_calendar=True,
            google_workspace=True,
            gmail=True,
            microsoft_bookings_auto=True,
            microsoft_bookings_manual=False,  # Don't enable both
            microsoft_outlook=True,
            email_templates=True,
            email_templates_microsoft=True,
            knowledge_base=True,
            human_support=True,
            glinks_email=True,
        )
    
    @classmethod
    def from_glinks_defaults(cls) -> "ToolConfig":
        """Create with Glinks-specific defaults."""
        return cls(
            google_calendar=True,
            google_workspace=True,
            gmail=True,
            microsoft_bookings_auto=True,
            microsoft_bookings_manual=False,
            microsoft_outlook=True,
            email_templates=True,
            email_templates_microsoft=True,
            knowledge_base=True,
            human_support=True,
            glinks_email=True,
        )


# =============================================================================
# TOOL INSTRUCTIONS (Phase 17)
# Per-tool usage instructions returned based on ToolConfig
# =============================================================================

TOOL_INSTRUCTIONS = {
    "google_calendar": """
## Google Calendar Tools
- **create_google_calendar_event**: Create event with optional Google Meet link
  - Required: summary, start_time_iso
  - Optional: end_time_iso (default 30min), timezone_name, attendee_emails, meet_required
- **list_google_events_for_day/week/month**: Query events for time range
- **list_google_events_for_range**: Query events between two specific times
""",

    "google_workspace": """
## Google Workspace Tools
- **book_google_meeting_slot**: All-in-one calendar event + Meet link + email invite
  - Required: summary, start_time_iso
  - Optional: attendee_emails, email_recipients, timezone_name
""",

    "gmail": """
## Gmail Tools
- **send_gmail**: Send email via user's Gmail account
  - Required: to, subject, body
""",

    "microsoft_bookings": """
## Microsoft Bookings Tools
- **microsoft_check_availability**: Check available appointment slots
  - Required: date (YYYY-MM-DD format)
- **microsoft_book_appointment**: Book appointment with Teams link
  - Required: start_time (ISO format), customer_name, customer_email
  - Optional: customer_phone, notes
""",

    "microsoft_outlook": """
## Microsoft Outlook Email Tools
- **send_outlook_email**: Send email from connected Microsoft/Outlook account
  - Required: to_email, subject, body
  - Optional: cc_email
  - Use for sending emails when user has connected their Microsoft account
  - Different from Gmail - uses Microsoft 365 email
""",

    "email_templates": """
## Email Template Tools
- **send_template_email**: Send email using pre-defined template
  - Required: template_key, to, placeholder_values
  - Templates are loaded based on tenant. See template list below.
""",

    "knowledge_base": """
## Knowledge Base Search Tools
- **search_knowledge_base**: Search tenant's document store
  - Required: query (search text)
  - Returns relevant passages from uploaded documents
""",

    "human_support": """
## Connecting to a Specialist
You have an **invite_human_agent** function to bring a specialist/expert into the call.

**When to use:**
- Customer explicitly asks to speak with a real person, specialist, or someone else
- Customer says things like: "Can I talk to someone?", "I want to speak to an expert", "Transfer me"
- You cannot adequately answer their question or handle their concern
- Customer is frustrated or upset and needs personal attention

**IMPORTANT - How to use:**
1. Ask for consent FIRST: "Would you like me to connect you with one of our specialists?"
2. Only call the tool AFTER they confirm (yes/okay/please)
3. The tool returns a message - KEEP TALKING while the specialist is being called in the background
4. Ask follow-up questions: "While I'm arranging that, is there anything else you'd like me to note?"
5. The specialist may take 10-30 seconds to join
6. When they join, you'll be prompted to introduce them briefly then go SILENT

**⚠️ CRITICAL - NEVER call hangup_call during this process:**
- After calling invite_human_agent, DO NOT call hangup_call under ANY circumstances
- The call continues with the specialist - you DO NOT end it
- If the specialist fails to join, you'll be instructed to continue helping yourself
- Only the customer or specialist can end the call - NOT you
- Your job after introduction is COMPLETE SILENCE - no hangup, no talking

**Example flow:**
- User: "Can I speak to someone?"
- You: "Of course! Would you like me to connect you with one of our specialists?"
- User: "Yes please"
- [Call invite_human_agent tool]
- You: "Great! I'm arranging that now. While we wait, is there anything specific you'd like me to pass on?"
- [Continue chatting naturally until specialist joins]
- [When specialist joins, say brief intro then go completely silent - NO hangup_call]
""",

    "hangup": """
## Call Ending
- Use the **hangup_call** tool to end the call gracefully
- Always say goodbye BEFORE using the tool (e.g., "Thank you for your time. Have a great day!")
- Provide a reason: "call_complete", "not_interested", "callback_scheduled", "appointment_booked"
""",
}


def get_tool_instructions(config: "ToolConfig") -> str:
    """
    Generate tool instructions based on enabled tools in ToolConfig.
    
    Phase 17: Returns only instructions for tools that are enabled.
    
    Args:
        config: ToolConfig indicating which tools are enabled
        
    Returns:
        Combined instruction string for enabled tools
    """
    sections = []
    
    # Always include hangup instruction
    sections.append(TOOL_INSTRUCTIONS["hangup"])
    
    if config.google_calendar:
        sections.append(TOOL_INSTRUCTIONS["google_calendar"])
    
    if config.google_workspace:
        sections.append(TOOL_INSTRUCTIONS["google_workspace"])
    
    if config.gmail:
        sections.append(TOOL_INSTRUCTIONS["gmail"])
    
    if config.microsoft_bookings_auto or config.microsoft_bookings_manual:
        sections.append(TOOL_INSTRUCTIONS["microsoft_bookings"])
    
    if config.microsoft_outlook:
        sections.append(TOOL_INSTRUCTIONS["microsoft_outlook"])
    
    if config.email_templates or config.glinks_email:
        sections.append(TOOL_INSTRUCTIONS["email_templates"])
    
    if config.email_templates_microsoft:
        sections.append(TOOL_INSTRUCTIONS.get("email_templates_microsoft", TOOL_INSTRUCTIONS["email_templates"]))
    
    if config.knowledge_base:
        sections.append(TOOL_INSTRUCTIONS["knowledge_base"])
    
    if config.human_support:
        sections.append(TOOL_INSTRUCTIONS["human_support"])
    
    return "\n".join(sections) if sections else ""


async def get_template_instructions_for_tenant(tenant_id: str) -> str:
    """
    Generate dynamic template instructions from DB for a specific tenant.
    
    Phase 17b: Queries lad_dev.communication_templates for active templates
    with the given tenant_id and builds instruction text with template_key
    and description.
    
    Args:
        tenant_id: Tenant UUID to fetch templates for
        
    Returns:
        Formatted instruction string with template list
    """
    try:
        from db.storage import CallStorage
        
        storage = CallStorage()
        
        # Query active templates for this tenant
        with storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT template_key, category, description, placeholders
                    FROM lad_dev.communication_templates
                    WHERE tenant_id = %s AND is_active = true
                    ORDER BY category, template_key
                """, (tenant_id,))
                templates = cur.fetchall()
        
        if not templates:
            logger.info(f"No templates found for tenant_id={tenant_id}")
            return ""
        
        # Build instruction sections by category
        categories = {}
        for key, category, description, placeholders in templates:
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "key": key,
                "description": description or f"Template {key}",
                "placeholders": placeholders or []
            })
        
        # Format as instruction text
        lines = [
            "## Email Template Selection Guide",
            "",
            "**IMPORTANT**: Never reveal template codes to users. Describe emails naturally.",
            "",
            "Use the send_template_email function with:",
            "- to: Recipient email address (as a list)",
            "- template_key: The template code (from list below)",
            "- placeholder_values: Dictionary of placeholder values",
            "",
            "### Available Templates:",
            ""
        ]
        
        category_labels = {
            "non_responsive": "NON-RESPONSIVE LEADS",
            "responsive": "RESPONSIVE LEADS",
            "grade_9_below": "GRADE 9 OR BELOW",
            "grade_10_11": "GRADE 10-11",
            "grade_12_plus": "GRADE 12+",
            "ug_pg": "UG/PG STUDENTS",
            "follow_up": "FOLLOW-UP / RETENTION",
        }
        
        for cat, tmpl_list in categories.items():
            label = category_labels.get(cat, cat.upper().replace("_", " "))
            lines.append(f"#### {label}")
            lines.append("| Template | Category | Description | Placeholders |")
            lines.append("|----------|----------|-------------|--------------|")
            for t in tmpl_list:
                placeholders_str = ", ".join(t['placeholders']) if t['placeholders'] else "student_name"
                lines.append(f"| {t['key']} | {cat} | {t['description']} | {placeholders_str} |")
            lines.append("")
        
        logger.info(f"Built template instructions for tenant_id={tenant_id}: {len(templates)} templates")
        return "\n".join(lines)
        
    except Exception as exc:
        logger.error(f"Failed to get template instructions for tenant_id={tenant_id}: {exc}", exc_info=True)
        return ""


# =============================================================================
# TOOL RESOLUTION (Request vs Tenant-based)
# =============================================================================

async def get_enabled_tools(
    request_payload: dict, 
    tenant_id: str | None = None
) -> tuple["ToolConfig", dict]:
    """
    Determine which tools to enable based on env flag.
    
    If TOOLS_DECIDED_BY_BACKEND=true: Query tenant_features table
    If TOOLS_DECIDED_BY_BACKEND=false: Use request.tools_enabled
    
    Args:
        request_payload: Call request payload with optional tools_enabled dict
        tenant_id: Tenant UUID for tenant-based lookup
        
    Returns:
        Tuple of (ToolConfig, tool_configs_dict)
    """
    if TOOLS_DECIDED_BY_BACKEND:
        logger.debug("Tools decided by backend (tenant_features)")
        return await _get_tools_from_tenant_features(tenant_id)
    else:
        logger.debug("Tools decided by frontend (request payload)")
        tools_dict = request_payload.get("tools_enabled", {})
        if tools_dict:
            return ToolConfig.from_dict(tools_dict), {}
        # Fallback: all tools enabled if no payload
        return ToolConfig.all_enabled(), {}


async def _get_tools_from_tenant_features(tenant_id: str | None) -> tuple["ToolConfig", dict]:
    """
    Query lad_dev.tenant_features for enabled tools.
    
    Returns:
        Tuple of (ToolConfig, tool_configs_dict) where tool_configs_dict
        contains the config JSONB for each enabled tool.
    """
    if not tenant_id:
        logger.warning("No tenant_id provided, using empty ToolConfig")
        return ToolConfig(), {}
    
    # Feature key to ToolConfig field mapping
    TOOL_FEATURE_MAP = {
        "voice-agent-tool-google-calendar": "google_calendar",
        "voice-agent-tool-google-workspace": "google_workspace",
        "voice-agent-tool-gmail": "gmail",
        "voice-agent-tool-microsoft-bookings-auto": "microsoft_bookings_auto",
        "voice-agent-tool-microsoft-bookings-manual": "microsoft_bookings_manual",
        "voice-agent-tool-microsoft-outlook": "microsoft_outlook",
        "voice-agent-tool-email-templates": "email_templates",
        "voice-agent-tool-email-templates-microsoft": "email_templates_microsoft",
        "voice-agent-tool-knowledge-base": "knowledge_base",
        "voice-agent-tool-human-support": "human_support",
    }
    
    try:
        config = get_db_config()
        with get_db_connection(config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT feature_key, enabled, config
                    FROM lad_dev.tenant_features
                    WHERE tenant_id = %s
                      AND feature_key LIKE 'voice-agent-tool-%%'
                """, (tenant_id,))
                rows = cur.fetchall()
        
        # Build enabled dict and configs dict
        enabled_tools = {}
        tool_configs = {}
        
        for row in rows:
            feature_key = row[0]
            is_enabled = row[1]
            config_json = row[2] or {}
            
            if feature_key in TOOL_FEATURE_MAP and is_enabled:
                field_name = TOOL_FEATURE_MAP[feature_key]
                enabled_tools[field_name] = True
                tool_configs[feature_key] = config_json
        
        tool_config = ToolConfig(
            google_calendar=enabled_tools.get("google_calendar", False),
            google_workspace=enabled_tools.get("google_workspace", False),
            gmail=enabled_tools.get("gmail", False),
            microsoft_bookings_auto=enabled_tools.get("microsoft_bookings_auto", False),
            microsoft_bookings_manual=enabled_tools.get("microsoft_bookings_manual", False),
            microsoft_outlook=enabled_tools.get("microsoft_outlook", False),
            email_templates=enabled_tools.get("email_templates", False),
            email_templates_microsoft=enabled_tools.get("email_templates_microsoft", False),
            knowledge_base=enabled_tools.get("knowledge_base", False),
            human_support=enabled_tools.get("human_support", False),
        )
        
        logger.info(
            "Loaded tenant tools: tenant=%s, enabled=%s",
            tenant_id,
            [k for k, v in enabled_tools.items() if v]
        )
        
        return tool_config, tool_configs
        
    except Exception as e:
        logger.error(f"Failed to fetch tenant tools: {e}", exc_info=True)
        return ToolConfig(), {}


async def _get_tenant_kb_stores(tenant_id: str) -> list[str]:
    """
    Get KB store IDs configured for tenant.
    
    Queries lad_dev.knowledge_base_catalog for tenant's default KB stores.
    Only returns stores where is_default=true.
    
    Args:
        tenant_id: Tenant UUID
        
    Returns:
        List of Gemini FileSearch store names
    """
    if not tenant_id:
        return []
    
    try:
        from psycopg2.extras import RealDictCursor
        
        db_config = get_db_config()
        with get_db_connection(db_config) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT gemini_store_name
                    FROM lad_dev.knowledge_base_catalog
                    WHERE tenant_id = %s 
                      AND is_default = true
                      AND is_active = true
                    ORDER BY priority DESC
                """, (tenant_id,))
                
                rows = cur.fetchall()
                store_names = [row['gemini_store_name'] for row in rows]
                
                if store_names:
                    logger.info(f"KB stores for tenant {tenant_id}: {store_names}")
                else:
                    logger.debug(f"No default KB stores for tenant {tenant_id}")
                
                return store_names
    except Exception as e:
        logger.error(f"Failed to get KB stores for tenant {tenant_id}: {e}")
        return []


# =============================================================================
# TOOL BUILDERS
# =============================================================================

def build_google_workspace_tools(
    user_id: str | int | None,
    workspace_client: Any = None,
) -> list[Callable]:
    """
    Build Google Workspace tools (Calendar + Gmail) as @function_tool decorated functions.
    
    Args:
        user_id: User ID for OAuth token resolution
        workspace_client: AgentGoogleWorkspace instance
        
    Returns:
        List of @function_tool decorated functions
    """
    if workspace_client is None:
        from tools.google_workspace import AgentGoogleWorkspace
        workspace_client = AgentGoogleWorkspace(user_id=user_id)
    
    @function_tool
    async def check_calendar(date: str) -> str:
        """
        Check calendar availability for a specific date.
        
        Args:
            date: Date to check (YYYY-MM-DD format)
            
        Returns:
            List of available time slots or busy times
        """
        try:
            events = await workspace_client.list_events(date)
            if events:
                event_list = [f"- {e.get('summary', 'Busy')} at {e.get('start', {}).get('dateTime', 'Unknown')}" for e in events]
                return f"Calendar for {date}:\n" + "\n".join(event_list)
            return f"Calendar is clear on {date}"
        except Exception as e:
            logger.error(f"Failed to check calendar: {e}")
            return f"Sorry, I couldn't check the calendar: {str(e)}"
    
    @function_tool
    async def create_calendar_event(
        title: str,
        date: str,
        start_time: str,
        duration_minutes: int = 60,
        attendee_email: str = "",
    ) -> str:
        """
        Create a calendar event.
        
        Args:
            title: Event title/summary
            date: Event date (YYYY-MM-DD format)
            start_time: Start time (HH:MM format, 24-hour)
            duration_minutes: Duration in minutes (default 60)
            attendee_email: Optional attendee email
            
        Returns:
            Confirmation with event details
        """
        try:
            result = await workspace_client.create_event(
                summary=title,
                start_time=f"{date}T{start_time}:00",
                duration_minutes=duration_minutes,
                attendee_email=attendee_email or None,
            )
            return f"Event created: {title} on {date} at {start_time}"
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return f"Sorry, I couldn't create the event: {str(e)}"
    
    @function_tool
    async def send_email(
        to_email: str,
        subject: str,
        body: str,
    ) -> str:
        """
        Send an email via Gmail.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body text
            
        Returns:
            Confirmation message
        """
        try:
            await workspace_client.send_email(
                to=to_email,
                subject=subject,
                body=body,
            )
            return f"Email sent to {to_email}"
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return f"Sorry, I couldn't send the email: {str(e)}"
    
    tools = [check_calendar, create_calendar_event, send_email]
    logger.info(f"Google Workspace tools built: user_id={user_id}, count={len(tools)}")
    return tools


def build_microsoft_bookings_tools(
    user_id: str | int | None,
    bookings_client: Any = None,
    *,
    business_id: str | None = None,
    service_id: str | None = None,
    staff_id: str | None = None,
    is_auto: bool = True,
) -> list[Callable]:
    """
    Build Microsoft Bookings tools as @function_tool decorated functions.
    
    Args:
        user_id: User ID for OAuth token resolution
        bookings_client: AgentMicrosoftBookings instance
        business_id: Default booking business ID (from tenant_features.config)
        service_id: Default booking service ID (from tenant_features.config)
        staff_id: Default booking staff ID (from tenant_features.config)
        is_auto: If True, build auto tools; if False, build manual tools
        
    Returns:
        List of @function_tool decorated functions
    """
    if bookings_client is None:
        from tools.microsoft_bookings import AgentMicrosoftBookings
        logger.debug("build_microsoft_bookings_tools: user_id=%r, business_id=%r", user_id, business_id)
        bookings_client = AgentMicrosoftBookings(
            user_id=user_id,
            default_business_id=business_id,
            default_service_id=service_id,
            default_staff_id=staff_id,
        )
    
    tools = []
    
    if is_auto:
        # Auto tools - use defaults from config
        @function_tool
        async def auto_book_appointment(
            date: str,
            time: str,
            customer_name: str,
            customer_email: str,
            customer_phone: str = "",
            notes: str = "",
        ) -> str:
            """
            Book an appointment using the configured default business and service.
            
            Args:
                date: Date for the appointment (YYYY-MM-DD format)
                time: Time for the appointment (HH:MM format, 24-hour)
                customer_name: Full name of the customer
                customer_email: Email address of the customer
                customer_phone: Phone number of the customer (optional)
                notes: Additional notes for the appointment (optional)
                
            Returns:
                Confirmation message with appointment details
            """
            try:
                start_time = f"{date} {time}"
                result = await bookings_client.auto_book_appointment(
                    start_time=start_time,
                    customer_name=customer_name,
                    customer_email=customer_email,
                    customer_phone=customer_phone or None,
                    notes=notes or None,
                )
                return f"Appointment booked! ID: {result.get('appointment_id')}, Time: {result.get('start_time')}"
            except Exception as e:
                logger.error(f"Failed to book appointment: {e}")
                return f"Sorry, I couldn't book the appointment: {str(e)}"
        
        @function_tool
        async def check_availability(date: str) -> str:
            """
            Check available appointment slots for a specific date.
            
            Args:
                date: Date to check (YYYY-MM-DD format)
                
            Returns:
                List of available time slots
            """
            try:
                slots = await bookings_client.auto_check_availability(date)
                if slots:
                    return f"Available slots on {date}: {', '.join(slots)}"
                return f"No available slots on {date}"
            except Exception as e:
                logger.error(f"Failed to check availability: {e}")
                return f"Sorry, I couldn't check availability: {str(e)}"
        
        tools = [auto_book_appointment, check_availability]
    else:
        # Manual tools - agent lists options for user to choose
        @function_tool
        async def list_booking_services() -> str:
            """List all available booking services."""
            try:
                business_id = await bookings_client._get_default_business_id()
                services = await bookings_client.list_services(business_id)
                if services:
                    service_list = [f"- {s['name']} ({s['duration_minutes']} min)" for s in services]
                    return "Available services:\n" + "\n".join(service_list)
                return "No services available"
            except Exception as e:
                return f"Error listing services: {str(e)}"
        
        tools = [list_booking_services]
    
    logger.info(
        "Microsoft Bookings tools built: user_id=%s, is_auto=%s, count=%d",
        user_id, is_auto, len(tools)
    )
    return tools


def build_microsoft_outlook_tools(
    user_id: str | int | None,
    outlook_client: Any = None,
) -> list[Callable]:
    """
    Build Microsoft Outlook email tools as @function_tool decorated functions.
    
    Args:
        user_id: User ID for OAuth token resolution
        outlook_client: AgentMicrosoftOutlook instance (optional, created if not provided)
        
    Returns:
        List of @function_tool decorated functions
    """
    if outlook_client is None:
        from tools.microsoft_outlook import AgentMicrosoftOutlook
        logger.debug("build_microsoft_outlook_tools: user_id=%r", user_id)
        outlook_client = AgentMicrosoftOutlook(user_id=user_id)
    
    @function_tool
    async def send_outlook_email(
        to_email: str,
        subject: str,
        body: str,
        cc_email: str = "",
    ) -> str:
        """
        Send an email using Microsoft Outlook.
        
        Args:
            to_email: Recipient email address (comma-separated for multiple)
            subject: Email subject line
            body: Email body text
            cc_email: CC recipient email address (optional, comma-separated for multiple)
            
        Returns:
            Confirmation message or error description
        """
        try:
            # Parse comma-separated emails
            to_emails = [e.strip() for e in to_email.split(",") if e.strip()]
            cc_emails = [e.strip() for e in cc_email.split(",") if e.strip()] if cc_email else None
            
            result = await outlook_client.send_email(
                to_emails=to_emails,
                subject=subject,
                body=body,
                cc_emails=cc_emails,
            )
            
            if result.get("success"):
                return f"Email sent successfully to {', '.join(to_emails)}"
            else:
                return f"Failed to send email: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Failed to send Outlook email: {e}")
            return f"Sorry, I couldn't send the email: {str(e)}"
    
    tools = [send_outlook_email]
    logger.info("Microsoft Outlook tools built: user_id=%s, count=%d", user_id, len(tools))
    return tools


async def build_knowledge_base_tools(
    tenant_id: str | None = None,
    store_ids: list[str] | None = None,
) -> list[Callable]:
    """
    Build knowledge base search tools as @function_tool decorated functions.
    
    Args:
        tenant_id: Tenant UUID for auto-resolution
        store_ids: Explicit list of KB store IDs
        
    Returns:
        List of @function_tool decorated functions
    """
    # Auto-resolve KB stores from tenant if not provided
    if not store_ids and tenant_id:
        store_ids = await _get_tenant_kb_stores(tenant_id)
    
    if not store_ids:
        logger.debug("No knowledge base stores configured")
        return []
    
    # Import KB client
    try:
        from tools.file_search_tool import FileSearchTool
        kb_tool = FileSearchTool()
    except ImportError:
        logger.warning("FileSearchTool not available")
        return []
    
    @function_tool
    async def search_knowledge_base(query: str) -> str:
        """
        Search the knowledge base for relevant information.
        
        Use this tool to find information from the organization's documents
        and FAQs. This includes company policies, product details, procedures, etc.
        
        Args:
            query: Search query describing what information you need
            
        Returns:
            Relevant information from the knowledge base
        """
        # Note: Actual RAG search happens via Gemini grounding tool configured in agent
        # This tool is a placeholder that logs the query
        # The LLM has access to KB content via Gemini's built-in file search grounding
        logger.info(f"KB search requested: {query[:100]}... (stores={store_ids})")
        return f"Searching knowledge base for: {query}"
    
    tools = [search_knowledge_base]
    logger.info(f"Knowledge Base tools built: stores={len(store_ids)}, count={len(tools)}")
    return tools


def build_email_template_tools(
    tenant_id: str | None = None,
    user_id: str | None = None,  # UUID
) -> list[Callable]:
    """
    Build email template tools based on tenant.
    
    Phase 18: Templates loaded from DB based on tenant_id.
    Uses generic create_email_template_tools which queries
    lad_dev.communication_templates for tenant's templates.
    
    Args:
        tenant_id: Tenant UUID for template lookup
        user_id: User ID for OAuth email sending
        
    Returns:
        List of tool functions [send_template_email, list_templates]
    """
    if not tenant_id:
        logger.warning("No tenant_id for email templates - templates will not be available")
        return []
    
    try:
        from tools.email_templates import create_email_template_tools
        
        tools = create_email_template_tools(tenant_id, user_id)
        logger.info(f"Email template tools created for tenant {tenant_id}")
        return tools
    except Exception as exc:
        logger.error(f"Failed to create email template tools: {exc}")
        return []


def build_microsoft_email_template_tools(
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> list[Callable]:
    """
    Build Microsoft email template tools based on tenant.
    
    Uses Microsoft OAuth to send template emails instead of Google.
    
    Args:
        tenant_id: Tenant UUID for template lookup
        user_id: User ID for Microsoft OAuth token resolution
        
    Returns:
        List of tool functions [send_template_email_ms, list_templates]
    """
    if not tenant_id:
        logger.warning("No tenant_id for MS email templates - templates will not be available")
        return []
    
    try:
        from tools.email_templates import create_microsoft_email_template_tools
        
        tools = create_microsoft_email_template_tools(tenant_id, user_id)
        logger.info(f"Microsoft email template tools created for tenant {tenant_id}")
        return tools
    except Exception as exc:
        logger.error(f"Failed to create Microsoft email template tools: {exc}")
        return []


def build_human_support_tools(
    phone_number: str,
    job_context: Any = None,
    sip_trunk_id: str | None = None,
    from_number: str | None = None,
    voice_assistant: Any = None,  # VoiceAssistant instance for muting
    audit_trail: Any = None,  # ToolAuditTrail for logging events
    tenant_id: str | None = None,  # Tenant ID for multi-tenant number lookup
) -> list[Callable]:
    """
    Build human support tools for transferring calls to a human agent.
    
    Args:
        phone_number: Phone number for human support (from tenant_features config)
        job_context: LiveKit job context for SIP transfer API access
        sip_trunk_id: SIP trunk ID from current call (from voice_agent_numbers.rules)
        from_number: From number of current call (for validation/logging)
        voice_assistant: VoiceAssistant instance for muting AI after human joins
        audit_trail: ToolAuditTrail for logging human handoff events
        tenant_id: Tenant UUID for multi-tenant number routing isolation
        
    Returns:
        List of @function_tool decorated functions
    """
    # Import here to avoid circular imports
    from livekit import api
    
    # State tracking for the transfer
    _transfer_pending = False
    _transfer_complete = False
    _transfer_failed = False
    _last_attempt_time: float = 0.0  # Timestamp of last attempt completion
    _RETRY_COOLDOWN = 10.0  # Seconds before retry is allowed after failure
    
    # Log initial configuration
    logger.info(f"[HumanSupport] Config: human_number={phone_number}, sip_trunk={'from_call' if sip_trunk_id else 'from_env'}, from_number={from_number[:4] if from_number else 'None'}***, voice_assistant={'set' if voice_assistant else 'None'}")
    
    # Reference to session for background task (set by worker.py)
    _session_ref: dict = {"session": None}
    
    def set_session(session):
        """Called by worker.py to set session reference for background task."""
        _session_ref["session"] = session
    
    @function_tool
    async def invite_human_agent(reason: str = "customer_request") -> str:
        """
        Connect the caller with a specialist/expert.
        
        Use this tool ONLY AFTER getting customer consent (e.g., they said "yes" 
        when you asked "Would you like me to connect you with a specialist?").
        
        This is NON-BLOCKING - it dials in the background while you continue
        the conversation naturally. Do NOT announce "connecting" or "transferring".
        
        Args:
            reason: Reason for the transfer (e.g., "customer_request", "complex_issue")
            
        Returns:
            Empty string (continue conversation normally)
        """
        nonlocal _transfer_pending, _transfer_complete, _transfer_failed, _last_attempt_time
        
        import time
        current_time = time.monotonic()
        
        # Detailed state logging at entry
        logger.info(f"[HumanSupport] ========== TOOL CALLED ==========")
        logger.info(f"[HumanSupport] reason={reason}")
        logger.info(f"[HumanSupport] State: pending={_transfer_pending}, complete={_transfer_complete}, failed={_transfer_failed}")
        logger.info(f"[HumanSupport] Session ref available: {_session_ref.get('session') is not None}")
        
        # If transfer is already pending, return status
        if _transfer_pending:
            logger.info("[HumanSupport] Transfer already pending")
            return (
                "I'm already connecting you with a specialist - they'll join very soon. "
                "Continue our conversation while we wait. DO NOT call hangup_call."
            )
        
        # If transfer already completed successfully, agent is muted
        if _transfer_complete:
            logger.info("[HumanSupport] Transfer already complete")
            # Agent should already be muted, just return silently
            return "The specialist is already on the line. Let them handle the conversation."
        
        # If transfer failed, check if cooldown has passed
        if _transfer_failed:
            time_since_failure = current_time - _last_attempt_time
            if time_since_failure < _RETRY_COOLDOWN:
                wait_remaining = int(_RETRY_COOLDOWN - time_since_failure)
                logger.info(f"[HumanSupport] In cooldown period, {wait_remaining}s remaining")
                return (
                    f"I just tried to reach a specialist but they weren't available. "
                    f"I can try again in {wait_remaining} seconds. "
                    "For now, continue helping the customer yourself. DO NOT call hangup_call."
                )
            else:
                # Cooldown passed, reset and allow retry
                logger.info("[HumanSupport] Cooldown passed, allowing retry")
                _transfer_failed = False
        
        # Validate we have required context
        if not job_context:
            logger.error("[HumanSupport] FAILED: No job_context available")
            return "I'm sorry, I cannot transfer you at this time. Please call back and ask for a human agent."
        
        # Get trunk ID - prefer call-specific, fallback to env
        trunk_id = sip_trunk_id or os.getenv("OUTBOUND_TRUNK_ID")
        if not trunk_id:
            logger.error("[HumanSupport] FAILED: No SIP trunk ID available (not in metadata, not in OUTBOUND_TRUNK_ID env)")
            return "I'm sorry, I cannot transfer you at this time due to a configuration issue."
        
        logger.info(f"[HumanSupport] Using SIP trunk: {trunk_id[:16]}... (source={'call_metadata' if sip_trunk_id else 'env'})")
        
        # Validate and format the human agent number using same rules as outbound calls
        dial_number = phone_number
        try:
            # Get db_config for number validation
            db_config = {
                "host": os.getenv("DB_HOST"),
                "port": int(os.getenv("DB_PORT", 5432)),
                "database": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
            }
            
            # Use from_number routing rules if available, otherwise just use direct number
            if from_number and db_config.get("host"):
                logger.info(f"[HumanSupport] Validating number using from_number={from_number[:4] if from_number else 'None'}*** routing rules")
                # Import here to avoid circular imports
                from utils.call_routing import validate_and_format_call
                routing_result = validate_and_format_call(from_number, phone_number, db_config, tenant_id=tenant_id)
                if not routing_result.success:
                    logger.error(f"[HumanSupport] FAILED: Number validation failed: {routing_result.error_message}")
                    return f"I'm sorry, I cannot transfer you to that number: {routing_result.error_message}"
                dial_number = routing_result.formatted_to_number
                logger.info(f"[HumanSupport] Number normalized: {phone_number} -> {dial_number}")
            else:
                logger.info(f"[HumanSupport] Using direct number (no from_number routing): {phone_number}")
        except Exception as e:
            logger.warning(f"[HumanSupport] Number validation error, using direct: {e}")
        
        # Mark as pending
        _transfer_pending = True
        
        # Background task to dial human and handle success/failure
        async def _invite_human_in_background():
            nonlocal _transfer_pending, _transfer_complete, _transfer_failed, _last_attempt_time
            
            import time
            
            try:
                logger.info(f"[HumanSupport] Background: Dialing human agent: {dial_number}")
                
                # Create SIP participant - wait_until_answered=True blocks until answered
                await job_context.api.sip.create_sip_participant(
                    api.CreateSIPParticipantRequest(
                        room_name=job_context.room.name,
                        sip_trunk_id=trunk_id,
                        sip_call_to=dial_number,
                        participant_identity=f"support-{dial_number}",
                        participant_name="Human Support Agent",
                        wait_until_answered=True,  # Block until human answers
                        krisp_enabled=True,
                    )
                )
                
                # Human answered!
                _transfer_complete = True
                _transfer_pending = False
                logger.info("[HumanSupport] Background: Human agent answered and joined")
                
                # CRITICAL: Mute the AI agent IMMEDIATELY - before intro injection
                # This disables silence monitor, stops TTS/LLM, and prevents hangup
                # voice_assistant is a getter function, so we need to call it
                assistant = voice_assistant() if callable(voice_assistant) else voice_assistant
                if assistant:
                    try:
                        assistant.mute_for_human_handoff()
                        logger.info("[HumanSupport] Background: AI agent muted IMMEDIATELY - silence_monitor disabled, TTS/LLM stopped")
                    except Exception as e:
                        logger.error(f"[HumanSupport] Background: Failed to mute AI agent: {e}")
                else:
                    logger.warning("[HumanSupport] Background: No voice_assistant available - cannot mute!")
                
                # Inject SIMPLE introduction - no [Specialist Name] placeholder, just a brief handoff
                session = _session_ref.get("session")
                if session:
                    try:
                        # Simple, direct intro - no placeholders
                        session.generate_reply(
                            instructions=(
                                "Say ONLY this brief introduction and NOTHING ELSE: "
                                "'Great news! Our specialist is now on the line to help you!' "
                                "After saying this, STOP COMPLETELY. Do not speak again. Do not call any tools."
                            )
                        )
                        logger.info("[HumanSupport] Background: Introduction instruction injected")
                    except Exception as e:
                        logger.warning(f"[HumanSupport] Background: Failed to inject introduction: {e}")
                
                # NO MORE WAITING - muting was already done above
                
                # Log to audit trail
                if audit_trail:
                    audit_trail.log_human_handoff_joined()
                    
            except Exception as e:
                # Human didn't answer or call failed
                _transfer_failed = True
                _transfer_pending = False  # Reset to allow retry after cooldown
                _last_attempt_time = time.monotonic()  # Record time for cooldown
                logger.error(f"[HumanSupport] Background: SIP call failed: {e}")
                logger.info("[HumanSupport] Background: Attempting to inject failure instruction NOW")
                
                # Inject instruction to handle failure
                session = _session_ref.get("session")
                if session:
                    try:
                        failure_instruction = (
                            "URGENT: The specialist could NOT be reached - the connection FAILED. "
                            "You MUST apologize immediately and continue helping them yourself. "
                            "Say: 'I apologize, unfortunately our specialist wasn't available right now. "
                            "Let me continue helping you myself - what would you like to know?'"
                        )
                        logger.info(f"[HumanSupport] Background: Injecting failure instruction (len={len(failure_instruction)})")
                        # generate_reply returns SpeechHandle synchronously, not a coroutine
                        handle = session.generate_reply(instructions=failure_instruction)
                        logger.info(f"[HumanSupport] Background: Failure instruction injected, SpeechHandle: {type(handle).__name__}")
                    except Exception as e2:
                        logger.warning(f"[HumanSupport] Background: Failed to inject failure instruction: {e2}")
                else:
                    logger.error("[HumanSupport] Background: No session ref - cannot inject failure instruction!")
                
                # Log to audit trail
                if audit_trail:
                    audit_trail.log_human_handoff_failed(str(e))
        
        # Start background task - returns immediately
        asyncio.create_task(_invite_human_in_background())
        
        logger.info("[HumanSupport] Background task started, returning to conversation")
        
        # CRITICAL: Inject instruction IMMEDIATELY to override any goodbye speech
        # the LLM might be generating in parallel with this tool call
        session = _session_ref.get("session")
        if session:
            try:
                # generate_reply returns a SpeechHandle synchronously, not a coroutine
                # This injects an instruction that should override any pending goodbye speech
                override_instruction = (
                    "URGENT OVERRIDE: You just initiated a specialist connection. "
                    "DO NOT say goodbye or end the call! "
                    "Say something like: 'Great, I'm arranging that connection now. "
                    "While we wait for the specialist, is there anything specific you'd like me to pass along to them?' "
                    "KEEP THE CONVERSATION GOING. Never call hangup_call during this process."
                )
                logger.info(f"[HumanSupport] Injecting override instruction (len={len(override_instruction)})")
                handle = session.generate_reply(instructions=override_instruction)
                logger.info(f"[HumanSupport] Override instruction injected, SpeechHandle: {type(handle).__name__}")
            except Exception as e:
                logger.warning(f"[HumanSupport] Failed to inject override instruction: {e}")
        else:
            logger.warning("[HumanSupport] No session ref - cannot inject override instruction!")
        
        # Log to audit trail
        if audit_trail:
            audit_trail.log_tool_call(
                "invite_human_agent",
                input_data={"reason": reason},
                output="Dialing in background",
                status="pending"
            )
            audit_trail.log_human_handoff_started(dial_number)
        
        # Return instruction telling agent to CONTINUE conversation
        # Note: This may not be seen if LLM already generated parallel speech,
        # which is why we also inject via generate_reply above
        return_msg = (
            "I'm connecting you with a specialist now. "
            "IMPORTANT: Continue the conversation naturally while I arrange this. "
            "Ask a follow-up question like 'While I'm arranging that, is there anything else you'd like me to note for the specialist?' "
            "DO NOT call hangup_call - the specialist will join when ready and I will tell you. "
            "Keep talking until you hear that the specialist has joined."
        )
        logger.info(f"[HumanSupport] Tool returning message (length={len(return_msg)})")
        logger.info(f"[HumanSupport] ========== TOOL COMPLETE ==========")
        return return_msg
    
    # Attach session setter to the tool for worker.py to use
    invite_human_agent.set_session = set_session
    
    tools = [invite_human_agent]
    logger.info(f"[HumanSupport] Tool built: invite_human_agent (phone={phone_number[:4]}***, trunk={'call' if sip_trunk_id else 'env'})")
    return tools



# =============================================================================
# MAIN TOOL ATTACHMENT
# =============================================================================

class ToolBuilder:
    """
    Builds and attaches tools to agent session.
    
    Reads configuration and attaches appropriate tools.
    """
    
    def __init__(
        self,
        config: ToolConfig | None = None,
        tenant_id: str | None = None,
        user_id: str | int | None = None,
        vertical: str | None = None,
        knowledge_base_store_ids: list[str] | None = None,
    ):
        self.config = config or ToolConfig()
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.vertical = vertical or "general"
        self.knowledge_base_store_ids = knowledge_base_store_ids or []
    
    def get_enabled_tools_summary(self) -> dict[str, bool]:
        """Get summary of which tools are enabled."""
        return {
            "google_calendar": self.config.google_calendar,
            "google_workspace": self.config.google_workspace,
            "gmail": self.config.gmail,
            "microsoft_bookings": self.config.microsoft_bookings_auto or self.config.microsoft_bookings_manual,
            "microsoft_outlook": self.config.microsoft_outlook,
            "email_templates": self.config.email_templates,
            "email_templates_microsoft": self.config.email_templates_microsoft,
            "knowledge_base": self.config.knowledge_base,
            "human_support": self.config.human_support,
            "glinks_email": self.config.glinks_email,
        }
    
    def build_tool_instructions(self) -> str:
        """
        Generate instructions for enabled tools.
        
        Phase 17c: Now uses get_tool_instructions with ToolConfig.
        
        Returns:
            Tool-specific instruction additions
        """
        # Build ToolConfig from self
        config = ToolConfig(
            google_calendar=self.google_calendar,
            google_workspace=self.google_workspace,
            gmail=self.gmail,
            microsoft_bookings=self.microsoft_bookings,
            email_templates=self.email_templates,
            knowledge_base=self.knowledge_base,
            human_support=self.human_support,
            glinks_email=getattr(self, 'glinks_email', False),
        )
        return get_tool_instructions(config)
    
    def log_configuration(self) -> None:
        """Log tool configuration for debugging."""
        summary = self.get_enabled_tools_summary()
        enabled = [k for k, v in summary.items() if v]
        logger.info("Tools enabled: %s (Vertical: %s)", enabled if enabled else "none", self.vertical)
        if self.knowledge_base_store_ids:
            logger.info("KB stores: %s", self.knowledge_base_store_ids)


async def attach_tools(
    agent: Any,
    config: ToolConfig,
    tool_configs: dict | None = None,
    *,
    tenant_id: str | None = None,
    user_id: str | int | None = None,
    vertical: str | None = None,
    knowledge_base_store_ids: list[str] | None = None,
    job_context: Any = None,
    sip_trunk_id: str | None = None,
    from_number: str | None = None,
    voice_assistant_holder: dict | None = None,  # {"assistant": VoiceAssistant} - populated after creation
    audit_trail: Any = None,  # ToolAuditTrail for logging tool calls
) -> list[Any]:
    """
    Attach tools to VoiceAssistant based on configuration.
    
    Args:
        agent: VoiceAssistant agent to attach tools to
        config: ToolConfig with enabled tools
        tool_configs: Dict of {feature_key: config_json} for tool settings
        tenant_id: Tenant UUID for resource resolution
        user_id: User ID for OAuth
        vertical: Tenant vertical (e.g. 'education')
        knowledge_base_store_ids: KB store IDs
        job_context: LiveKit job context for SIP tools (human support)
        sip_trunk_id: SIP trunk ID from current call routing
        from_number: From number of current call
        
    Returns:
        List of attached tool functions
    """
    tool_configs = tool_configs or {}
    attached_tools = []
    
    # NOTE: hangup_call is provided by VoiceAssistant class as @function_tool method
    # Do NOT add it here to avoid duplicate function name error
    logger.info("hangup_call: Provided by VoiceAssistant class (always available)")
    
    # Google Workspace tools (Calendar + Gmail)
    if config.google_workspace or config.google_calendar or config.gmail:
        try:
            tools = await build_google_workspace_tools(user_id)
            attached_tools.extend(tools)
            logger.info(f"Attached Google Workspace tools: {len(tools)} functions")
        except Exception as e:
            logger.error(f"Failed to build Google Workspace tools: {e}")
    
    # Microsoft Bookings Auto (with defaults from config)
    if config.microsoft_bookings_auto:
        try:
            cfg = tool_configs.get("voice-agent-tool-microsoft-bookings-auto", {})
            tools = build_microsoft_bookings_tools(
                user_id,
                business_id=cfg.get("business_id"),
                service_id=cfg.get("service_id"),
                staff_id=cfg.get("staff_id"),
                is_auto=True,
            )
            attached_tools.extend(tools)
            logger.info(f"Attached Microsoft Bookings Auto tools: {len(tools)} functions")
        except Exception as e:
            logger.error(f"Failed to build Microsoft Bookings Auto tools: {e}")
    
    # Microsoft Bookings Manual (agent lists options for user)
    if config.microsoft_bookings_manual:
        try:
            tools = build_microsoft_bookings_tools(user_id, is_auto=False)
            attached_tools.extend(tools)
            logger.info(f"Attached Microsoft Bookings Manual tools: {len(tools)} functions")
        except Exception as e:
            logger.error(f"Failed to build Microsoft Bookings Manual tools: {e}")
    
    # Microsoft Outlook (send emails via Outlook)
    if config.microsoft_outlook:
        try:
            tools = build_microsoft_outlook_tools(user_id)
            attached_tools.extend(tools)
            logger.info(f"Attached Microsoft Outlook tools: {len(tools)} functions")
        except Exception as e:
            logger.error(f"Failed to build Microsoft Outlook tools: {e}")
    
    # Knowledge Base tools
    if config.knowledge_base:
        try:
            store_ids = knowledge_base_store_ids or await _get_tenant_kb_stores(tenant_id) if tenant_id else []
            if store_ids:
                tools = await build_knowledge_base_tools(tenant_id, store_ids)
                attached_tools.extend(tools)
                logger.info(f"Attached Knowledge Base tools: {len(tools)} functions")
        except Exception as e:
            logger.error(f"Failed to build Knowledge Base tools: {e}")
    
    # Email Templates (Google)
    if config.email_templates:
        try:
            tools = build_email_template_tools(tenant_id, user_id) if tenant_id else []
            attached_tools.extend(tools)
            logger.info(f"Attached Email Template tools: {len(tools) if tools else 0} functions (user_id={'set' if user_id else 'None'})")
        except Exception as e:
            logger.error(f"Failed to build Email Template tools: {e}")
    
    # Email Templates (Microsoft)
    if config.email_templates_microsoft:
        try:
            tools = build_microsoft_email_template_tools(tenant_id, user_id) if tenant_id else []
            attached_tools.extend(tools)
            logger.info(f"Attached Microsoft Email Template tools: {len(tools) if tools else 0} functions")
        except Exception as e:
            logger.error(f"Failed to build Microsoft Email Template tools: {e}")
    
    # Human Support (phone from config with env fallback)
    logger.info(f"[HumanSupport] config.human_support={config.human_support}")
    if config.human_support:
        cfg = tool_configs.get("voice-agent-tool-human-support", {})
        logger.info(f"[HumanSupport] tool_configs entry: {cfg}")
        # Check both potential key names - trace the source
        phone_from_cfg1 = cfg.get("human_agent_number")
        phone_from_cfg2 = cfg.get("phone_number")
        phone_from_env = os.getenv("HUMAN_SUPPORT_NUMBER")
        phone = phone_from_cfg1 or phone_from_cfg2 or phone_from_env
        
        # Log source tracing
        if phone_from_cfg1:
            phone_source = "tool_configs.human_agent_number (DB)"
        elif phone_from_cfg2:
            phone_source = "tool_configs.phone_number (DB)"
        elif phone_from_env:
            phone_source = "HUMAN_SUPPORT_NUMBER (ENV)"
        else:
            phone_source = "NONE FOUND"
        logger.info(f"[HumanSupport] ========== PHONE SOURCE TRACE ==========")
        logger.info(f"[HumanSupport] cfg.human_agent_number: {phone_from_cfg1}")
        logger.info(f"[HumanSupport] cfg.phone_number: {phone_from_cfg2}")
        logger.info(f"[HumanSupport] env.HUMAN_SUPPORT_NUMBER: {phone_from_env}")
        logger.info(f"[HumanSupport] RESOLVED phone: {phone} (source: {phone_source})")
        logger.info(f"[HumanSupport] =========================================")
        if phone:
            try:
                # Create a getter that retrieves VoiceAssistant from holder at call time
                # This allows the tool to be built before VoiceAssistant is created
                def _get_assistant():
                    if voice_assistant_holder:
                        return voice_assistant_holder.get("assistant")
                    return None
                
                tools = build_human_support_tools(
                    phone,
                    job_context=job_context,
                    sip_trunk_id=sip_trunk_id,
                    from_number=from_number,
                    voice_assistant=_get_assistant,  # Pass getter function
                    audit_trail=audit_trail,  # For logging handoff events
                    tenant_id=tenant_id,  # Multi-tenant number routing
                )
                attached_tools.extend(tools)
                logger.info(f"Attached Human Support tools: {len(tools)} functions (trunk={'set' if sip_trunk_id else 'env'})")
            except Exception as e:
                logger.error(f"Failed to build Human Support tools: {e}", exc_info=True)
        else:
            logger.warning("[HumanSupport] SKIPPED: No phone number found in config or HUMAN_SUPPORT_NUMBER env")
    else:
        logger.info("[HumanSupport] SKIPPED: config.human_support is False")

    
    logger.info(f"Total tools attached: {len(attached_tools)}")
    return attached_tools


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Environment flag
    "TOOLS_DECIDED_BY_BACKEND",
    # Classes
    "ToolConfig",
    "ToolBuilder",
    # Instruction generation (Phase 17)
    "TOOL_INSTRUCTIONS",
    "get_tool_instructions",
    "get_template_instructions_for_tenant",  # Phase 17b: Dynamic from DB
    # Functions
    "get_enabled_tools",
    "attach_tools",
    "build_google_workspace_tools",
    "build_microsoft_bookings_tools",
    "build_microsoft_outlook_tools",
    "build_knowledge_base_tools",
    "build_email_template_tools",
]
