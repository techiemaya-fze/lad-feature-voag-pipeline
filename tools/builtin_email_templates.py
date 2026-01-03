"""
Built-in Email Templates
========================

This module provides hard-coded email templates that are always available,
independent of the database. These serve as fallbacks and quick-access templates.

Template Format:
- Placeholders use {{placeholder_name}} syntax
- All templates have a unique key for reference
- Templates include both text and optional HTML versions

Usage in Agent:
    The agent can reference templates by their key:
    - "meeting_confirmation" - For calendar booking confirmations
    - "follow_up_basic" - Simple follow-up after a call
    - "custom" - Agent composes the email freely

The agent should:
1. Identify which template fits the situation
2. Gather required placeholder values from the conversation
3. Call the email tool with template_key and placeholder values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BuiltinTemplate:
    """Represents a built-in email template."""
    key: str
    name: str
    category: str
    subject_template: str
    text_body_template: str
    html_body_template: str | None = None
    placeholders: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


# ============================================================================
# MEETING INVITATION TEMPLATES
# ============================================================================

MEETING_CONFIRMATION = BuiltinTemplate(
    key="meeting_confirmation",
    name="Meeting Confirmation",
    category="meeting_invite",
    subject_template="Meeting Confirmed: {{meeting_title}}",
    text_body_template="""Hi {{recipient_name}},

This is to confirm your meeting has been scheduled.

Meeting Details:
- Title: {{meeting_title}}
- Date & Time: {{meeting_datetime}}
- Duration: {{meeting_duration}}
- Join Link: {{meeting_link}}

Looking forward to speaking with you!

Best regards,
{{sender_name}}""",
    html_body_template="""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<p>Hi {{recipient_name}},</p>

<p>This is to confirm your meeting has been scheduled.</p>

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
<h3 style="margin-top: 0; color: #2c5282;">Meeting Details</h3>
<p><strong>Title:</strong> {{meeting_title}}</p>
<p><strong>Date & Time:</strong> {{meeting_datetime}}</p>
<p><strong>Duration:</strong> {{meeting_duration}}</p>
<p><strong>Join Link:</strong> <a href="{{meeting_link}}" style="color: #2b6cb0;">{{meeting_link}}</a></p>
</div>

<p>Looking forward to speaking with you!</p>

<p>Best regards,<br>
<strong>{{sender_name}}</strong></p>
</body>
</html>""",
    placeholders=(
        "recipient_name", "meeting_title", "meeting_datetime",
        "meeting_duration", "meeting_link", "sender_name"
    ),
    description="Standard meeting confirmation with Google Meet link. Use after booking a calendar slot.",
)


QUICK_INVITE = BuiltinTemplate(
    key="quick_invite",
    name="Quick Meeting Invite",
    category="meeting_invite",
    subject_template="You're invited: {{meeting_title}}",
    text_body_template="""Hi {{recipient_name}},

You're invited to join a meeting.

When: {{meeting_datetime}}
Join here: {{meeting_link}}

See you there!

{{sender_name}}""",
    placeholders=(
        "recipient_name", "meeting_title", "meeting_datetime",
        "meeting_link", "sender_name"
    ),
    description="Short, concise meeting invitation. Use for quick scheduling.",
)


# ============================================================================
# FOLLOW-UP TEMPLATES
# ============================================================================

FOLLOW_UP_BASIC = BuiltinTemplate(
    key="follow_up_basic",
    name="Basic Follow-Up",
    category="follow_up",
    subject_template="Following up on our conversation",
    text_body_template="""Hi {{recipient_name}},

Thank you for taking the time to speak with me today.

{{message_body}}

Please don't hesitate to reach out if you have any questions.

Best regards,
{{sender_name}}""",
    placeholders=("recipient_name", "message_body", "sender_name"),
    description="Simple follow-up email. Agent fills in the message_body based on conversation.",
)


FOLLOW_UP_WITH_SUMMARY = BuiltinTemplate(
    key="follow_up_summary",
    name="Follow-Up with Summary",
    category="follow_up",
    subject_template="Summary: Our conversation about {{topic}}",
    text_body_template="""Hi {{recipient_name}},

Thank you for your time today. Here's a quick summary of what we discussed:

{{call_summary}}

Next Steps:
{{next_steps}}

If you have any questions or need clarification on anything, please let me know.

Best regards,
{{sender_name}}
{{company_name}}""",
    html_body_template="""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
<p>Hi {{recipient_name}},</p>

<p>Thank you for your time today. Here's a quick summary of what we discussed:</p>

<div style="background-color: #f0f4f8; padding: 15px; border-left: 4px solid #4a90d9; margin: 15px 0;">
{{call_summary}}
</div>

<h4 style="color: #2c5282;">Next Steps:</h4>
<p>{{next_steps}}</p>

<p>If you have any questions or need clarification on anything, please let me know.</p>

<p>Best regards,<br>
<strong>{{sender_name}}</strong><br>
{{company_name}}</p>
</body>
</html>""",
    placeholders=(
        "recipient_name", "topic", "call_summary",
        "next_steps", "sender_name", "company_name"
    ),
    description="Detailed follow-up with call summary and next steps. Use after substantive conversations.",
)


# ============================================================================
# INFORMATION & RESOURCES TEMPLATES
# ============================================================================

SEND_INFORMATION = BuiltinTemplate(
    key="send_info",
    name="Send Information",
    category="information",
    subject_template="Information you requested: {{topic}}",
    text_body_template="""Hi {{recipient_name}},

As promised, here's the information you requested about {{topic}}:

{{information_content}}

{{#if resource_links}}
You may also find these resources helpful:
{{resource_links}}
{{/if}}

Let me know if you need anything else!

Best regards,
{{sender_name}}""",
    placeholders=(
        "recipient_name", "topic", "information_content",
        "resource_links", "sender_name"
    ),
    description="Send requested information or documents. Use when prospect asks for details.",
)


# ============================================================================
# REGISTRY OF ALL BUILTIN TEMPLATES
# ============================================================================

BUILTIN_TEMPLATES: dict[str, BuiltinTemplate] = {
    t.key: t for t in [
        MEETING_CONFIRMATION,
        QUICK_INVITE,
        FOLLOW_UP_BASIC,
        FOLLOW_UP_WITH_SUMMARY,
        SEND_INFORMATION,
    ]
}


def get_builtin_template(key: str) -> BuiltinTemplate | None:
    """
    Get a built-in template by its key.
    
    Args:
        key: Template key (e.g., 'meeting_confirmation')
        
    Returns:
        BuiltinTemplate or None if not found
    """
    return BUILTIN_TEMPLATES.get(key)


def list_builtin_templates(category: str | None = None) -> list[BuiltinTemplate]:
    """
    List all available built-in templates.
    
    Args:
        category: Optional filter by category
        
    Returns:
        List of BuiltinTemplate objects
    """
    templates = list(BUILTIN_TEMPLATES.values())
    if category:
        templates = [t for t in templates if t.category == category]
    return templates


def render_builtin_template(
    key: str,
    placeholders: dict[str, str],
    use_html: bool = False,
) -> tuple[str, str] | None:
    """
    Render a built-in template with provided placeholders.
    
    Args:
        key: Template key
        placeholders: Dictionary of placeholder values
        use_html: Whether to return HTML body (if available)
        
    Returns:
        Tuple of (subject, body) or None if template not found
    """
    import re
    
    template = get_builtin_template(key)
    if not template:
        return None
    
    def _render(template_str: str) -> str:
        result = template_str
        pattern = r"\{\{(\w+)\}\}"
        found = set(re.findall(pattern, template_str))
        for name in found:
            if name in placeholders:
                result = result.replace(f"{{{{{name}}}}}", str(placeholders[name]))
        # Handle conditional blocks (simple version)
        # Remove unfilled conditionals
        result = re.sub(r"\{\{#if \w+\}\}.*?\{\{/if\}\}", "", result, flags=re.DOTALL)
        return result
    
    subject = _render(template.subject_template)
    
    if use_html and template.html_body_template:
        body = _render(template.html_body_template)
    else:
        body = _render(template.text_body_template)
    
    return subject, body


def get_template_info_for_agent() -> str:
    """
    Get a formatted string describing available templates for the agent's context.
    
    Returns:
        Markdown-formatted template descriptions
    """
    lines = ["## Available Email Templates\n"]
    
    for template in BUILTIN_TEMPLATES.values():
        lines.append(f"### `{template.key}` - {template.name}")
        lines.append(f"*Category: {template.category}*")
        lines.append(f"{template.description}")
        lines.append(f"Required placeholders: {', '.join(template.placeholders)}")
        lines.append("")
    
    lines.append("### `custom` - Custom Email")
    lines.append("*Use when no template fits*")
    lines.append("Compose the email freely with full subject and body.")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# TENANT-AWARE TEMPLATE FUNCTIONS (Phase 14)
# =============================================================================

import logging
_logger = logging.getLogger(__name__)


async def get_templates_for_tenant(tenant_id: str | None = None) -> list[BuiltinTemplate]:
    """
    Get templates for a specific tenant.
    
    Phase 14: Combines builtin templates with tenant-specific DB templates.
    Future: Will query lad_dev.communication_templates table.
    
    Args:
        tenant_id: Tenant UUID for tenant-specific templates
        
    Returns:
        List of available templates (builtin + tenant-specific)
    """
    # Start with builtin templates
    templates = list(BUILTIN_TEMPLATES.values())
    
    if not tenant_id:
        _logger.debug("No tenant_id, returning builtin templates only")
        return templates
    
    # TODO: Phase 15+ - Query lad_dev.communication_templates for tenant
    # db_templates = await _load_tenant_templates_from_db(tenant_id)
    # templates.extend(db_templates)
    
    _logger.debug(f"Templates for tenant {tenant_id}: {len(templates)} builtin")
    return templates


def get_template_tools(tenant_id: str | None = None) -> list:
    """
    Get email template tool functions for agent attachment.
    
    This is called by tool_builder.py to get callable tools.
    
    Args:
        tenant_id: Tenant UUID for template resolution
        
    Returns:
        List of tool functions (placeholder - actual tools in future)
    """
    _logger.info(f"Email template tools for tenant {tenant_id}")
    # TODO: Return actual callable tool functions
    return []


__all__ = [
    "BuiltinTemplate",
    "BUILTIN_TEMPLATES",
    "get_builtin_template",
    "list_builtin_templates",
    "render_builtin_template",
    "get_template_info_for_agent",
    # Tenant-aware functions (Phase 14)
    "get_templates_for_tenant",
    "get_template_tools",
    # Individual templates for direct import
    "MEETING_CONFIRMATION",
    "QUICK_INVITE",
    "FOLLOW_UP_BASIC",
    "FOLLOW_UP_WITH_SUMMARY",
    "SEND_INFORMATION",
]
