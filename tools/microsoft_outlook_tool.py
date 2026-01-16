"""Microsoft Outlook API tool for sending emails via Microsoft Graph.

This module provides a low-level wrapper for Microsoft Graph API email operations.
It follows the same pattern as microsoft_bookings_tool.py.

Usage:
    tool = MicrosoftOutlookTool(access_token)
    await tool.send_email(
        to_recipients=[EmailRecipient(email="user@example.com", name="User")],
        subject="Hello",
        body_content="Email body text",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


@dataclass
class EmailRecipient:
    """Email recipient information."""
    email: str
    name: str | None = None


class MicrosoftOutlookToolError(RuntimeError):
    """Raised when Microsoft Outlook operations fail."""
    pass


class MicrosoftOutlookTool:
    """
    Microsoft Outlook API wrapper for voice agent.
    
    Provides methods for sending emails via Microsoft Graph API.
    Requires a valid access token with Mail.Send permission.
    """

    def __init__(self, access_token: str) -> None:
        """
        Initialize the Outlook tool.
        
        Args:
            access_token: Valid Microsoft Graph API access token with Mail.Send scope
        """
        self._access_token = access_token
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def send_email(
        self,
        to_recipients: list[EmailRecipient],
        subject: str,
        body_content: str,
        content_type: str = "Text",  # "Text" or "HTML"
        cc_recipients: list[EmailRecipient] | None = None,
        bcc_recipients: list[EmailRecipient] | None = None,
        save_to_sent_items: bool = True,
    ) -> dict[str, Any]:
        """
        Send an email via Microsoft Graph API.
        
        Args:
            to_recipients: List of primary recipients
            subject: Email subject
            body_content: The actual text/html content
            content_type: "Text" or "HTML"
            cc_recipients: Optional CC list
            bcc_recipients: Optional BCC list
            save_to_sent_items: Whether to save copy in Sent folder
            
        Returns:
            Dict with success status
            
        Raises:
            MicrosoftOutlookToolError: If email sending fails
        """
        
        # Helper to format recipients for Graph API
        def format_recipients(recipients: list[EmailRecipient]) -> list[dict]:
            return [
                {
                    "emailAddress": {
                        "address": r.email,
                        "name": r.name or r.email
                    }
                }
                for r in recipients
            ]

        payload = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": content_type,
                    "content": body_content
                },
                "toRecipients": format_recipients(to_recipients),
            },
            "saveToSentItems": save_to_sent_items
        }

        if cc_recipients:
            payload["message"]["ccRecipients"] = format_recipients(cc_recipients)
        
        if bcc_recipients:
            payload["message"]["bccRecipients"] = format_recipients(bcc_recipients)

        logger.info(f"Sending email to {len(to_recipients)} recipients, subject: {subject[:50]}...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{GRAPH_BASE_URL}/me/sendMail",
                headers=self._headers,
                json=payload,
            )
            
            # 202 Accepted is the standard success code for sendMail
            if resp.status_code == 202:
                logger.info("Email sent successfully")
                return {"success": True, "status_code": 202}
            else:
                error_text = resp.text[:500] if resp.text else "No error details"
                logger.error(f"Failed to send email: {resp.status_code} - {error_text}")
                raise MicrosoftOutlookToolError(
                    f"Failed to send email: {resp.status_code} - {error_text}"
                )

    async def get_user_info(self) -> dict[str, Any]:
        """
        Get current user's profile info (email, display name).
        
        Returns:
            User profile dict with mail, displayName, etc.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/me",
                headers=self._headers,
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                raise MicrosoftOutlookToolError(
                    f"Failed to get user info: {resp.status_code}"
                )


__all__ = [
    "EmailRecipient",
    "MicrosoftOutlookTool",
    "MicrosoftOutlookToolError",
]
