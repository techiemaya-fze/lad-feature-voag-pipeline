"""Gmail sending helper for the Vonage voice agent."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Sequence

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class GmailToolError(RuntimeError):
    """Raised when Gmail operations fail."""


@dataclass(frozen=True)
class EmailPayload:
    to: Sequence[str]
    subject: str
    text_body: str | None = None
    html_body: str | None = None
    cc: Sequence[str] = field(default_factory=tuple)
    bcc: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.to:
            raise GmailToolError("At least one recipient is required")
        if not (self.text_body or self.html_body):
            raise GmailToolError("Provide text_body and/or html_body")


class GmailEmailTool:
    """Minimal Gmail API wrapper for sending agent follow-up emails."""

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    def _ensure_valid_token(self) -> None:
        if self._credentials.expired and self._credentials.refresh_token:
            self._credentials.refresh(Request())

    def send_email(self, payload: EmailPayload) -> dict:
        self._ensure_valid_token()
        message = self._build_mime_message(payload)
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode()
        try:
            service = build("gmail", "v1", credentials=self._credentials, cache_discovery=False)
            response = service.users().messages().send(userId="me", body={"raw": encoded}).execute()
            return response
        except HttpError as exc:  # pragma: no cover - network call
            raise GmailToolError(f"Gmail send failed: {exc}") from exc

    def _build_mime_message(self, payload: EmailPayload) -> MIMEMultipart:
        root = MIMEMultipart("alternative") if payload.html_body and payload.text_body else MIMEMultipart()
        root["To"] = ", ".join(payload.to)
        if payload.cc:
            root["Cc"] = ", ".join(payload.cc)
        if payload.bcc:
            root["Bcc"] = ", ".join(payload.bcc)
        root["Subject"] = payload.subject

        if payload.text_body:
            root.attach(MIMEText(payload.text_body, "plain"))
        if payload.html_body:
            root.attach(MIMEText(payload.html_body, "html"))
        return root


async def send_email_oauth(
    user_id: str | int,
    to: str | list[str],
    subject: str,
    text_body: str | None = None,
    html_body: str | None = None,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
) -> str:
    """
    Send email using user's OAuth credentials.
    
    Args:
        user_id: User ID with Google OAuth tokens
        to: Recipient email(s)
        subject: Email subject
        text_body: Plain text body
        html_body: HTML body
        cc: CC recipients
        bcc: BCC recipients
        
    Returns:
        Success message with message ID
    """
    from utils.google_credentials import GoogleCredentialResolver
    
    resolver = GoogleCredentialResolver()
    credentials = await resolver.load_credentials(str(user_id))
    
    recipients = [to] if isinstance(to, str) else list(to)
    
    payload = EmailPayload(
        to=recipients,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        cc=cc or [],
        bcc=bcc or [],
    )
    
    tool = GmailEmailTool(credentials)
    result = tool.send_email(payload)
    
    return f"Email sent successfully. Message ID: {result.get('id', 'unknown')}"


__all__ = ["GmailEmailTool", "EmailPayload", "GmailToolError", "send_email_oauth"]

