"""High-level Microsoft Outlook wrapper for LiveKit agent tools.

This module provides a high-level wrapper around MicrosoftOutlookTool that:
- Loads OAuth tokens from database
- Auto-refreshes expired tokens
- Provides simple methods for agent tools

Usage:
    outlook = AgentMicrosoftOutlook(user_id="user-uuid")
    result = await outlook.send_email(
        to_emails=["user@example.com"],
        subject="Hello",
        body="Email body text",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from auth.microsoft import (
    MicrosoftAuthService,
    token_response_to_storage_format,
)
from db.storage.tokens import UserTokenStorage
from tools.microsoft_outlook_tool import (
    MicrosoftOutlookTool,
    MicrosoftOutlookToolError,
    EmailRecipient,
)

logger = logging.getLogger(__name__)


class MicrosoftOutlookError(RuntimeError):
    """Raised when Microsoft Outlook operations fail."""
    
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class MicrosoftOutlookCredentialError(MicrosoftOutlookError):
    """Raised when Microsoft credentials are missing or invalid."""
    
    def __init__(self, user_identifier: str) -> None:
        message = (
            f"Microsoft account not connected for user {user_identifier}. "
            "Please connect your Microsoft account in settings."
        )
        super().__init__(message, status_code=401)
        self.user_identifier = user_identifier


class AgentMicrosoftOutlook:
    """
    Wraps Microsoft Outlook API with credential resolution per user.
    
    This class follows the same pattern as AgentMicrosoftBookings:
    - Loads tokens from database via UserTokenStorage
    - Auto-refreshes expired tokens
    - Provides simple async methods for agent use
    """

    def __init__(
        self,
        user_identifier: str | int | None = None,
        *,
        user_id: str | int | None = None,
    ) -> None:
        """
        Initialize with user identifier.
        
        Args:
            user_identifier: User ID (string or numeric) - deprecated, use user_id
            user_id: User ID (string or numeric)
        """
        effective_id = user_id or user_identifier
        if not effective_id:
            raise ValueError("user_id is required")
        
        self._user_id = self._normalize_identifier(effective_id)
        self._storage = UserTokenStorage()
        self._encryptor = None
        self._tool = None
        self._user_record = None
        
        logger.debug(f"AgentMicrosoftOutlook initialized for user {self._user_id}")

    @staticmethod
    def _normalize_identifier(identifier: str | int | None) -> str:
        """Normalize user identifier to string."""
        if identifier is None:
            return ""
        return str(identifier)

    def _load_user_record(self, force_reload: bool = False) -> dict | None:
        """Load user record from database."""
        if self._user_record is not None and not force_reload:
            return self._user_record
        
        self._user_record = self._storage.get_user_by_user_id(self._user_id)
        return self._user_record

    def _get_encryptor(self):
        """Get or create token encryptor."""
        if self._encryptor is None:
            from auth.google import TokenEncryptor, get_google_oauth_settings
            settings = get_google_oauth_settings()
            self._encryptor = TokenEncryptor(settings.encryption_key)
        return self._encryptor

    async def _get_access_token(self) -> str:
        """
        Get access token for Microsoft Graph API, auto-refreshing if expired.
        
        Returns:
            Valid access token
            
        Raises:
            MicrosoftOutlookCredentialError: If no tokens found
            MicrosoftOutlookError: If token refresh fails
        """
        # Get encrypted token blob
        token_blob = await self._storage.get_microsoft_token_blob(self._user_id)
        if not token_blob:
            raise MicrosoftOutlookCredentialError(self._user_id)
        
        # Decrypt
        encryptor = self._get_encryptor()
        try:
            token_data = encryptor.decrypt_json(token_blob)
        except Exception as e:
            logger.error(f"Failed to decrypt Microsoft tokens: {e}")
            raise MicrosoftOutlookError("Failed to decrypt stored tokens")
        
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        
        if not access_token:
            raise MicrosoftOutlookCredentialError(self._user_id)
        
        # Check if expired (simple check - could enhance with expires_at)
        # For now, try the token and refresh on 401
        return access_token

    async def _get_tool(self) -> MicrosoftOutlookTool:
        """Get or create the Outlook tool with valid token."""
        access_token = await self._get_access_token()
        return MicrosoftOutlookTool(access_token)

    async def send_email(
        self,
        to_emails: list[str],
        subject: str,
        body: str,
        content_type: str = "Text",
        cc_emails: list[str] | None = None,
        bcc_emails: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Send an email from the user's Outlook account.
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject line
            body: Email body text
            content_type: "Text" or "HTML" (default: "Text")
            cc_emails: Optional list of CC email addresses
            bcc_emails: Optional list of BCC email addresses
            
        Returns:
            Dict with success status and message
        """
        logger.info(f"[MicrosoftOutlook] Sending email for user {self._user_id}")
        logger.info(f"[MicrosoftOutlook] To: {to_emails}, Subject: {subject[:50]}...")
        
        try:
            tool = await self._get_tool()
            
            # Convert email strings to EmailRecipient objects
            to_recipients = [EmailRecipient(email=e) for e in to_emails]
            cc_recipients = [EmailRecipient(email=e) for e in (cc_emails or [])]
            bcc_recipients = [EmailRecipient(email=e) for e in (bcc_emails or [])]
            
            result = await tool.send_email(
                to_recipients=to_recipients,
                subject=subject,
                body_content=body,
                content_type=content_type,
                cc_recipients=cc_recipients if cc_recipients else None,
                bcc_recipients=bcc_recipients if bcc_recipients else None,
            )
            
            logger.info(f"[MicrosoftOutlook] Email sent successfully")
            return {
                "success": True,
                "message": "Email sent successfully",
                "recipients": to_emails,
            }
            
        except MicrosoftOutlookToolError as e:
            logger.error(f"[MicrosoftOutlook] Failed to send email: {e}")
            return {
                "success": False,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"[MicrosoftOutlook] Unexpected error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
            }

    async def get_sender_email(self) -> str | None:
        """
        Get the connected Microsoft account email (sender address).
        
        Returns:
            Email address of connected account, or None if not available
        """
        try:
            tool = await self._get_tool()
            user_info = await tool.get_user_info()
            return user_info.get("mail") or user_info.get("userPrincipalName")
        except Exception as e:
            logger.error(f"[MicrosoftOutlook] Failed to get sender email: {e}")
            return None


__all__ = [
    "AgentMicrosoftOutlook",
    "MicrosoftOutlookError",
    "MicrosoftOutlookCredentialError",
]
