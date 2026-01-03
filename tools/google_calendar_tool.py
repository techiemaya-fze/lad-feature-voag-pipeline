"""Google Calendar helper used by the Vonage voice agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Sequence

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class CalendarToolError(RuntimeError):
    """Raised when Google Calendar operations fail."""


@dataclass(frozen=True)
class CalendarAttendee:
    email: str
    display_name: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class CalendarEventRequest:
    summary: str
    description: str | None
    start: datetime
    end: datetime
    timezone: str
    attendees: Sequence[CalendarAttendee] = field(default_factory=tuple)
    location: str | None = None
    meet_required: bool = True
    send_updates: str = "all"  # all | externalOnly | none
    calendar_id: str = "primary"
    reminders: list[dict[str, str | int]] | None = None


class GoogleCalendarTool:
    """Thin wrapper around googleapiclient to create Calendar events."""

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    def _ensure_valid_token(self) -> None:
        if self._credentials.expired and self._credentials.refresh_token:
            self._credentials.refresh(Request())

    def create_event(self, payload: CalendarEventRequest) -> dict:
        """Create a calendar event and optionally provision a Meet link."""

        if payload.start >= payload.end:
            raise CalendarToolError("Event end time must be after start time")

        self._ensure_valid_token()
        try:
            service = build("calendar", "v3", credentials=self._credentials, cache_discovery=False)
            body: dict[str, object] = {
                "summary": payload.summary,
                "start": {"dateTime": payload.start.isoformat(), "timeZone": payload.timezone},
                "end": {"dateTime": payload.end.isoformat(), "timeZone": payload.timezone},
            }
            if payload.description:
                body["description"] = payload.description
            if payload.attendees:
                body["attendees"] = [
                    {
                        "email": attendee.email,
                        **({"displayName": attendee.display_name} if attendee.display_name else {}),
                        **({"optional": True} if attendee.optional else {}),
                    }
                    for attendee in payload.attendees
                ]
            if payload.location:
                body["location"] = payload.location
            if payload.reminders is not None:
                body["reminders"] = {"useDefault": False, "overrides": payload.reminders}

            conference_version = 0
            if payload.meet_required:
                conference_version = 1
                body["conferenceData"] = {
                    "createRequest": {
                        "conferenceSolutionKey": {"type": "hangoutsMeet"},
                        "requestId": f"meet-{uuid.uuid4().hex}",
                    }
                }

            event = (
                service.events()
                .insert(
                    calendarId=payload.calendar_id,
                    body=body,
                    sendUpdates=payload.send_updates,
                    conferenceDataVersion=conference_version,
                )
                .execute()
            )
            return event
        except HttpError as exc:  # pragma: no cover - network call
            raise CalendarToolError(f"Google Calendar error: {exc}") from exc

    def list_events(
        self,
        *,
        calendar_id: str = "primary",
        start: datetime,
        end: datetime,
        max_results: int = 50,
        single_events: bool = True,
        order_by: str = "startTime",
    ) -> list[dict]:
        """Fetch events between the supplied window."""

        if start >= end:
            raise CalendarToolError("timeMin must be before timeMax for event listing")
        if max_results <= 0:
            raise CalendarToolError("max_results must be positive")
        self._ensure_valid_token()
        try:
            service = build("calendar", "v3", credentials=self._credentials, cache_discovery=False)
            response = (
                service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start.isoformat(),
                    timeMax=end.isoformat(),
                    singleEvents=single_events,
                    orderBy=order_by,
                    maxResults=max_results,
                )
                .execute()
            )
            return list(response.get("items", []))
        except HttpError as exc:  # pragma: no cover - network call
            raise CalendarToolError(f"Google Calendar list error: {exc}") from exc


__all__ = [
    "GoogleCalendarTool",
    "CalendarEventRequest",
    "CalendarAttendee",
    "CalendarToolError",
]
