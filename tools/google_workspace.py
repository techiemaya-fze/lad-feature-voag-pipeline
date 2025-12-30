"""High-level Google Workspace helpers for LiveKit agent tools."""

from __future__ import annotations

import asyncio
import calendar
import logging
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Any, Callable, Sequence, TypeVar

from zoneinfo import ZoneInfo

from google.auth.exceptions import RefreshError

from tools.google_calendar_tool import (
    CalendarAttendee,
    CalendarEventRequest,
    CalendarToolError,
    GoogleCalendarTool,
)
from tools.gmail_email_tool import EmailPayload, GmailEmailTool, GmailToolError
from utils.google_credentials import GoogleCredentialError, GoogleCredentialResolver
from db.storage.tokens import UserTokenStorage

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TokenExpiredError(GoogleCredentialError):
    """Raised when Google OAuth tokens have expired and cannot be refreshed."""

    def __init__(self, user_identifier: str, original_error: Exception | None = None) -> None:
        message = (
            "Your Google account connection has expired. Please reconnect your Google account "
            "in the settings to continue using calendar and email features."
        )
        super().__init__(message, status_code=401)
        self.user_identifier = user_identifier
        self.original_error = original_error


class AgentGoogleWorkspace:
    """Wraps Calendar and Gmail helpers with credential resolution per user."""

    def __init__(self, user_identifier: str | int) -> None:
        normalized = self._normalize_identifier(user_identifier)
        if not normalized:
            raise ValueError("user_identifier is required for Google Workspace tools")
        self._user_identifier = normalized
        self._resolver = GoogleCredentialResolver()
        self._credentials = None
        self._lock = asyncio.Lock()
        self._token_storage = UserTokenStorage()

    @staticmethod
    def _normalize_identifier(identifier: str | int | None) -> str:
        if identifier is None:
            return ""
        if isinstance(identifier, int):
            return str(identifier)
        return identifier.strip()

    async def _get_credentials(self):
        if self._credentials is not None:
            return self._credentials
        async with self._lock:
            if self._credentials is None:
                self._credentials = await self._resolver.load_credentials(self._user_identifier)
        return self._credentials

    async def _clear_expired_tokens(self, original_error: Exception | None = None) -> None:
        """Clear expired/invalid tokens from the database for the current user."""
        try:
            # Get the canonical user_id from the resolver
            canonical_id, _ = await self._resolver.resolve_user(self._user_identifier)
            await self._token_storage.remove_tokens(canonical_id)
            logger.warning(
                "Cleared expired Google OAuth tokens for user_id=%s due to refresh failure",
                canonical_id,
            )
        except Exception as clear_exc:  # noqa: BLE001
            logger.error(
                "Failed to clear expired tokens for user %s: %s",
                self._user_identifier,
                clear_exc,
            )

    def _is_token_refresh_error(self, exc: Exception) -> bool:
        """Check if the exception indicates a token refresh failure."""
        if isinstance(exc, RefreshError):
            return True
        # Check for common OAuth error messages
        error_text = str(exc).lower()
        refresh_indicators = [
            "invalid_grant",
            "token has been expired or revoked",
            "token expired",
            "refresh token",
            "invalid credentials",
            "the oauth client was disabled",
            "access_denied",
        ]
        return any(indicator in error_text for indicator in refresh_indicators)

    async def _handle_api_error(self, exc: Exception) -> None:
        """Handle API errors, clearing tokens if refresh failed."""
        if self._is_token_refresh_error(exc):
            await self._clear_expired_tokens(exc)
            raise TokenExpiredError(self._user_identifier, exc) from exc
        # Re-raise the original exception for non-refresh errors
        raise

    async def _calendar(self) -> GoogleCalendarTool:
        credentials = await self._get_credentials()
        return GoogleCalendarTool(credentials)

    async def _gmail(self) -> GmailEmailTool:
        credentials = await self._get_credentials()
        return GmailEmailTool(credentials)

    async def create_event(
        self,
        *,
        summary: str,
        description: str | None,
        start: datetime,
        end: datetime,
        timezone_name: str,
        attendee_emails: Sequence[str] | None = None,
        attendee_names: Sequence[str] | None = None,
        location: str | None = None,
        meet_required: bool = True,
        send_updates: str = "all",
        calendar_id: str = "primary",
        reminders: Sequence[dict[str, str | int]] | None = None,
    ) -> dict[str, Any]:
        tool = await self._calendar()
        attendees = self._build_attendees(attendee_emails, attendee_names)
        calendar_id = (calendar_id or "primary").strip() or "primary"
        request = CalendarEventRequest(
            summary=summary,
            description=description,
            start=start,
            end=end,
            timezone=timezone_name,
            attendees=attendees,
            location=location,
            meet_required=meet_required,
            send_updates=send_updates,
            calendar_id=calendar_id,
            reminders=list(reminders) if reminders is not None else None,
        )
        try:
            event = tool.create_event(request)
        except Exception as exc:  # noqa: BLE001
            await self._handle_api_error(exc)
            raise  # Fallback if _handle_api_error doesn't raise
        return self._simplify_event(event)

    async def list_events_for_range(
        self,
        *,
        start: datetime,
        end: datetime,
        calendar_id: str = "primary",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        tool = await self._calendar()
        try:
            raw_events = tool.list_events(calendar_id=calendar_id, start=start, end=end, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            await self._handle_api_error(exc)
            raise  # Fallback if _handle_api_error doesn't raise
        return [self._simplify_event(item) for item in raw_events]

    async def list_events_for_day(
        self,
        *,
        target_date: date,
        timezone_name: str,
        calendar_id: str = "primary",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        tz = self._resolve_zone(timezone_name)
        start = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
        end = start + timedelta(days=1)
        return await self.list_events_for_range(start=start, end=end, calendar_id=calendar_id, max_results=max_results)

    async def list_events_for_week(
        self,
        *,
        start_date: date,
        timezone_name: str,
        calendar_id: str = "primary",
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        tz = self._resolve_zone(timezone_name)
        start = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
        end = start + timedelta(days=7)
        return await self.list_events_for_range(start=start, end=end, calendar_id=calendar_id, max_results=max_results)

    async def list_events_for_month(
        self,
        *,
        year: int,
        month: int,
        timezone_name: str,
        calendar_id: str = "primary",
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        if month < 1 or month > 12:
            raise CalendarToolError("month must be between 1 and 12")
        _, days_in_month = calendar.monthrange(year, month)
        tz = self._resolve_zone(timezone_name)
        start = datetime(year, month, 1, tzinfo=tz)
        end = start + timedelta(days=days_in_month)
        return await self.list_events_for_range(start=start, end=end, calendar_id=calendar_id, max_results=max_results)

    async def send_email(
        self,
        *,
        to: Sequence[str],
        subject: str,
        text_body: str | None = None,
        html_body: str | None = None,
        cc: Sequence[str] | None = None,
        bcc: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        tool = await self._gmail()
        payload = EmailPayload(
            to=list(to),
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            cc=list(cc) if cc else tuple(),
            bcc=list(bcc) if bcc else tuple(),
        )
        try:
            return tool.send_email(payload)
        except Exception as exc:  # noqa: BLE001
            await self._handle_api_error(exc)
            raise  # Fallback if _handle_api_error doesn't raise

    async def book_slot(
        self,
        *,
        summary: str,
        description: str | None,
        start: datetime,
        end: datetime,
        timezone_name: str,
        attendee_emails: Sequence[str] | None,
        attendee_names: Sequence[str] | None,
        calendar_id: str,
        send_updates: str,
        meet_required: bool,
        email_recipients: Sequence[str] | None,
        email_subject: str | None,
        email_body: str | None,
        location: str | None = None,
    ) -> dict[str, Any]:
        event = await self.create_event(
            summary=summary,
            description=description,
            start=start,
            end=end,
            timezone_name=timezone_name,
            attendee_emails=attendee_emails,
            attendee_names=attendee_names,
            location=location,
            meet_required=meet_required,
            send_updates=send_updates,
            calendar_id=calendar_id,
        )
        meet_link = event.get("meet_link")
        html_link = event.get("html_link")
        default_subject = email_subject or f"Meeting invite: {summary}"
        default_body = email_body or self._compose_default_invite(event, timezone_name)
        recipients = list(email_recipients or [])
        if not recipients:
            raise GmailToolError("email_recipients cannot be empty for slot booking")
        email_result = await self.send_email(
            to=recipients,
            subject=default_subject,
            text_body=default_body,
        )
        return {"event": event, "email": email_result, "meet_link": meet_link, "calendar_link": html_link}

    def _build_attendees(
        self,
        emails: Sequence[str] | None,
        names: Sequence[str] | None,
    ) -> list[CalendarAttendee]:
        attendees: list[CalendarAttendee] = []
        name_list = list(names or [])
        for idx, raw_email in enumerate(emails or []):
            email = raw_email.strip()
            if not email:
                continue
            display_name = None
            if idx < len(name_list):
                candidate = name_list[idx].strip()
                display_name = candidate or None
            attendees.append(CalendarAttendee(email=email, display_name=display_name))
        return attendees

    @staticmethod
    def _resolve_zone(name: str) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except Exception as exc:  # noqa: BLE001
            raise CalendarToolError(f"Unknown timezone '{name}'") from exc

    @staticmethod
    def _simplify_event(event: dict[str, Any]) -> dict[str, Any]:
        start = AgentGoogleWorkspace._extract_datetime(event.get("start"))
        end = AgentGoogleWorkspace._extract_datetime(event.get("end"))
        simplified = {
            "id": event.get("id"),
            "summary": event.get("summary") or "Untitled Event",
            "description": event.get("description"),
            "start": start,
            "end": end,
            "html_link": event.get("htmlLink"),
            "meet_link": AgentGoogleWorkspace._extract_meet_link(event),
            "location": event.get("location"),
            "attendees": [att.get("email") for att in event.get("attendees", []) if isinstance(att, dict)],
            "raw_event": event,
        }
        return simplified

    @staticmethod
    def _extract_datetime(block: Any) -> str | None:
        if not isinstance(block, dict):
            return None
        if block.get("dateTime"):
            return str(block["dateTime"])
        if block.get("date"):
            return str(block["date"])
        return None

    @staticmethod
    def _extract_meet_link(event: dict[str, Any]) -> str | None:
        hangout_link = event.get("hangoutLink")
        if isinstance(hangout_link, str) and hangout_link:
            return hangout_link
        conference = event.get("conferenceData")
        if isinstance(conference, dict):
            entry_points = conference.get("entryPoints")
            if isinstance(entry_points, list):
                for entry in entry_points:
                    if isinstance(entry, dict) and entry.get("entryPointType") == "video":
                        uri = entry.get("uri")
                        if isinstance(uri, str) and uri:
                            return uri
        return None

    @staticmethod
    def _compose_default_invite(event: dict[str, Any], timezone_name: str) -> str:
        summary = event.get("summary") or "Meeting"
        start = event.get("start")
        end = event.get("end")
        meet_link = event.get("meet_link")
        html_link = event.get("html_link")
        lines = [f"You're invited to {summary}."]
        if start:
            lines.append(f"Starts: {start} ({timezone_name})")
        if end:
            lines.append(f"Ends: {end} ({timezone_name})")
        if meet_link:
            lines.append(f"Join via Google Meet: {meet_link}")
        if html_link:
            lines.append(f"Calendar link: {html_link}")
        return "\n".join(lines)


__all__ = ["AgentGoogleWorkspace", "TokenExpiredError"]
