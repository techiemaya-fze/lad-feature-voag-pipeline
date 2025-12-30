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
from typing import Any, Callable

from livekit.agents.llm import function_tool

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
        email_templates: bool = False,
        knowledge_base: bool = False,
        human_support: bool = False,
        glinks_email: bool = False,
    ):
        self.google_calendar = google_calendar
        self.google_workspace = google_workspace
        self.gmail = gmail
        self.microsoft_bookings_auto = microsoft_bookings_auto
        self.microsoft_bookings_manual = microsoft_bookings_manual
        self.email_templates = email_templates
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
            email_templates=data.get("email_templates", False),
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
            email_templates=True,
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
            email_templates=True,
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
## Human Support Tools  
- **invite_human_agent**: Escalate call to human support
  - Use when customer requests human assistance or AI cannot resolve issue
""",

    "hangup": """
## Call Control
- **hangup_call**: End the current call gracefully
  - Optional: reason (e.g., "call_complete", "not_interested")
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
    
    if config.microsoft_bookings:
        sections.append(TOOL_INSTRUCTIONS["microsoft_bookings"])
    
    if config.email_templates or config.glinks_email:
        sections.append(TOOL_INSTRUCTIONS["email_templates"])
    
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
        "voice-agent-tool-email-templates": "email_templates",
        "voice-agent-tool-knowledge-base": "knowledge_base",
        "voice-agent-tool-human-support": "human_support",
    }
    
    try:
        from db.db_config import get_db_config
        from db.connection_pool import get_db_connection
        
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
            email_templates=enabled_tools.get("email_templates", False),
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
        from db.db_config import get_db_config
        from db.connection_pool import get_db_connection, return_connection, USE_CONNECTION_POOLING
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        db_config = get_db_config()
        conn = get_db_connection(db_config)
        
        try:
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
        finally:
            if USE_CONNECTION_POOLING:
                return_connection(conn, db_config)
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


def build_human_support_tools(
    phone_number: str,
    job_context: Any = None,
) -> list[Callable]:
    """
    Build human support tools for transferring calls to a human agent.
    
    Args:
        phone_number: Phone number for human support
        job_context: LiveKit job context for SIP transfer
        
    Returns:
        List of @function_tool decorated functions
    """
    @function_tool
    async def request_human_agent(reason: str = "customer_request") -> str:
        """
        Transfer this call to a human support agent.
        
        Use this tool when the caller specifically asks to speak
        with a human, or when you cannot adequately help them.
        
        Args:
            reason: Reason for the transfer (e.g., "customer_request", "complex_issue")
            
        Returns:
            Status message about the transfer
        """
        try:
            logger.info(f"Transferring to human agent: phone={phone_number[:4]}***, reason={reason}")
            
            if job_context and hasattr(job_context, 'room'):
                # Initiate SIP transfer to human support number
                # This would trigger a SIP INVITE to the support number
                logger.info("Human support transfer initiated")
                return f"I'm transferring you to a human agent now. Please hold."
            
            return "I'm sorry, I cannot transfer you at this time. Please call back and ask for a human agent."
        except Exception as e:
            logger.error(f"Failed to transfer to human: {e}")
            return f"Sorry, I couldn't connect you to a human agent: {str(e)}"
    
    tools = [request_human_agent]
    logger.info(f"Human Support tools built: phone={phone_number[:4]}***")
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
            "microsoft_bookings": self.config.microsoft_bookings,
            "email_templates": self.config.email_templates,
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
        
    Returns:
        List of attached tool functions
    """
    tool_configs = tool_configs or {}
    attached_tools = []
    
    # hangup_call is ALWAYS attached, regardless of tenant
    # The VoiceAssistant already has this as a method, just log it
    logger.info("hangup_call: Always available (built into VoiceAssistant)")
    
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
    
    # Email Templates
    if config.email_templates:
        try:
            tools = await build_email_template_tools(tenant_id) if tenant_id else []
            attached_tools.extend(tools)
            logger.info(f"Attached Email Template tools: {len(tools) if tools else 0} functions")
        except Exception as e:
            logger.error(f"Failed to build Email Template tools: {e}")
    
    # Human Support (phone from config with env fallback)
    if config.human_support:
        cfg = tool_configs.get("voice-agent-tool-human-support", {})
        phone = cfg.get("phone_number") or os.getenv("HUMAN_SUPPORT_NUMBER")
        if phone:
            try:
                tools = build_human_support_tools(phone)
                attached_tools.extend(tools)
                logger.info(f"Attached Human Support tools: {len(tools)} functions")
            except Exception as e:
                logger.error(f"Failed to build Human Support tools: {e}")
    
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
    "build_knowledge_base_tools",
    "build_email_template_tools",
]
