"""High-level Microsoft Bookings wrapper for LiveKit agent tools.

This module provides a two-layer architecture:
- Layer 1 (Manual): Methods requiring explicit business_id/service_id
- Layer 2 (Auto): Methods using defaults from database
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from db.storage.tokens import UserTokenStorage
from utils.google_oauth import TokenEncryptor, get_google_oauth_settings
from utils.microsoft_oauth import (
    MicrosoftAuthService,
    token_response_to_storage_format,
)
from tools.microsoft_bookings_tool import (
    MicrosoftBookingsTool,
    MicrosoftBookingsToolError,
    BookingCustomer,
)

logger = logging.getLogger(__name__)


class MicrosoftBookingError(RuntimeError):
    """Raised when Microsoft Booking operations fail."""
    
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class MicrosoftCredentialError(MicrosoftBookingError):
    """Raised when Microsoft credentials are missing or invalid."""
    
    def __init__(self, user_identifier: str) -> None:
        message = (
            f"Microsoft account not connected for user {user_identifier}. "
            "Please connect your Microsoft account in settings."
        )
        super().__init__(message, status_code=401)
        self.user_identifier = user_identifier


class MicrosoftConfigError(MicrosoftBookingError):
    """Raised when booking configuration is missing."""
    
    def __init__(self, user_identifier: str, missing: str) -> None:
        message = (
            f"Microsoft Bookings not configured: {missing}. "
            "Please select a booking calendar in settings."
        )
        super().__init__(message, status_code=400)
        self.user_identifier = user_identifier
        self.missing = missing


class AgentMicrosoftBookings:
    """
    Wraps Microsoft Bookings API with credential resolution per user.
    
    This class follows the same pattern as AgentGoogleWorkspace:
    - Loads tokens and configuration from database
    - Provides high-level methods for agent tools
    - Handles token refresh automatically
    """

    def __init__(
        self,
        user_identifier: str | int = None,
        *,
        user_id: str | int | None = None,  # Alias for user_identifier
        default_business_id: str | None = None,
        default_service_id: str | None = None,
        default_staff_id: str | None = None,
    ) -> None:
        """
        Initialize with user identifier and optional config overrides.
        
        Args:
            user_identifier: User ID (string or numeric) - deprecated, use user_id
            user_id: User ID (string or numeric)
            default_business_id: Override from tenant_features.config
            default_service_id: Override from tenant_features.config
            default_staff_id: Override from tenant_features.config
        """
        # Support both user_identifier and user_id for backwards compatibility
        actual_user_id = user_id or user_identifier
        self._user_id = self._normalize_identifier(actual_user_id)
        self._storage = UserTokenStorage()
        self._user_record: dict[str, Any] | None = None
        self._encryptor: TokenEncryptor | None = None
        self._tool: MicrosoftBookingsTool | None = None
        self._access_token: str | None = None
        
        # Config overrides from tenant_features (take precedence over DB)
        self._config_business_id = default_business_id
        self._config_service_id = default_service_id
        self._config_staff_id = default_staff_id

    @staticmethod
    def _normalize_identifier(identifier: str | int | None) -> str:
        """Normalize user identifier to string."""
        if identifier is None:
            return ""
        return str(identifier).strip()

    async def _load_user_record(self, force_reload: bool = False) -> dict[str, Any]:
        """Load user record from database."""
        if self._user_record is not None and not force_reload:
            return self._user_record

        # Try numeric ID first
        record = None
        if self._user_id.isdigit():
            record = await self._storage.get_user_by_primary_id(int(self._user_id))
        if record is None:
            record = await self._storage.get_user_by_user_id(self._user_id)

        if not record:
            raise MicrosoftCredentialError(self._user_id)

        self._user_record = record
        return record

    def _get_encryptor(self) -> TokenEncryptor:
        """Get or create token encryptor."""
        if self._encryptor is None:
            settings = get_google_oauth_settings()
            self._encryptor = TokenEncryptor(settings.encryption_key)
        return self._encryptor

    async def _get_access_token(self) -> str:
        """Get access token for Microsoft Graph API, auto-refreshing if expired."""
        record = await self._load_user_record()
        blob = record.get("microsoft_oauth_tokens")

        if not blob:
            raise MicrosoftCredentialError(self._user_id)

        encryptor = self._get_encryptor()
        if isinstance(blob, memoryview):
            blob = blob.tobytes()

        try:
            token_payload = encryptor.decrypt_json(blob)
        except ValueError as exc:
            logger.error("Failed to decrypt Microsoft tokens: %s", exc)
            raise MicrosoftCredentialError(self._user_id) from exc

        # Check if token is expired (with 5 min buffer)
        import time
        expires_at = token_payload.get("expires_at", 0)
        is_expired = time.time() > (expires_at - 300)  # 5 min buffer
        
        refresh_token = token_payload.get("refresh_token")
        
        if is_expired and refresh_token:
            logger.info("Microsoft token expired for user %s, refreshing...", self._user_id)
            try:
                ms_service = MicrosoftAuthService()
                new_tokens = ms_service.refresh_token(refresh_token)
                
                # Convert and store new tokens - use record's user_id column, not PK
                actual_user_id = record.get("user_id") or self._user_id
                new_payload = token_response_to_storage_format(new_tokens)
                encrypted = encryptor.encrypt_json(new_payload)
                await self._storage.store_microsoft_token_blob(actual_user_id, encrypted)
                
                token_payload = new_payload
                self._user_record = None  # Clear cache
                logger.info("Successfully refreshed Microsoft token for user %s", self._user_id)
            except ValueError as exc:
                logger.warning("Token refresh failed for user %s: %s", self._user_id, exc)
                # Continue with existing token, let API call fail with proper error

        access_token = token_payload.get("access_token")
        if not access_token:
            raise MicrosoftCredentialError(self._user_id)

        return access_token

    async def _get_tool(self) -> MicrosoftBookingsTool:
        """Get or create the low-level bookings tool."""
        if self._tool is None:
            access_token = await self._get_access_token()
            self._tool = MicrosoftBookingsTool(access_token)
        return self._tool

    async def _get_default_business_id(self) -> str:
        """Get default business ID from config override or database."""
        # Config override takes precedence (from tenant_features.config)
        if self._config_business_id:
            return self._config_business_id
        
        record = await self._load_user_record()
        business_id = record.get("selected_booking_business_id")
        if not business_id:
            raise MicrosoftConfigError(self._user_id, "no default booking business selected")
        return business_id


    async def _get_default_service_id(self) -> str | None:
        """Get default service ID from config override or database."""
        # Config override takes precedence (from tenant_features.config)
        if self._config_service_id:
            return self._config_service_id
        
        record = await self._load_user_record()
        return record.get("default_service_id")

    async def _get_default_staff_member_id(self) -> str | None:
        """Get default staff member ID from config override or database."""
        # Config override takes precedence (from tenant_features.config)
        if self._config_staff_id:
            return self._config_staff_id
        
        record = await self._load_user_record()
        return record.get("default_staff_member_id")

    def _extract_available_slots(self, availability: list[dict]) -> list[str]:
        """
        Extract human-readable slot strings from availability data.
        
        Args:
            availability: Raw availability data from get_availability
        
        Returns:
            List of slot strings like "09:00 AM", "09:30 AM", etc.
        """
        slots = []
        for slot in availability:
            start_info = slot.get("startDateTime", {})
            start_dt_str = start_info.get("dateTime", "")
            if start_dt_str:
                try:
                    # Parse datetime
                    dt = datetime.fromisoformat(start_dt_str.replace("Z", ""))
                    # Format as readable time
                    slot_str = dt.strftime("%I:%M %p").lstrip("0")
                    slots.append(slot_str)
                except ValueError:
                    pass
        return slots

    # =========================================================================
    # LAYER 1: MANUAL TOOLS (Explicit IDs)
    # =========================================================================

    async def list_businesses(self) -> list[dict[str, Any]]:
        """
        List all booking businesses the user manages.
        
        Returns:
            List of businesses with id, name, email
        """
        tool = await self._get_tool()
        try:
            businesses = await tool.get_booking_businesses()
            return [
                {
                    "id": b.get("id", ""),
                    "name": b.get("displayName", "Unknown"),
                    "email": b.get("email"),
                }
                for b in businesses
            ]
        except MicrosoftBookingsToolError as exc:
            logger.error("Failed to list businesses: %s", exc)
            raise MicrosoftBookingError(str(exc)) from exc

    async def list_services(self, business_id: str) -> list[dict[str, Any]]:
        """
        List all services for a booking business.
        
        Args:
            business_id: The booking business ID
            
        Returns:
            List of services with id, name, duration_minutes
        """
        tool = await self._get_tool()
        try:
            services = await tool.get_services(business_id)
            result = []
            for s in services:
                duration = s.get("defaultDuration", "PT30M")
                # Parse ISO 8601 duration (e.g., "PT30M" -> 30)
                minutes = self._parse_duration_to_minutes(duration)
                result.append({
                    "id": s.get("id", ""),
                    "name": s.get("displayName", "Unknown"),
                    "duration_minutes": minutes,
                })
            return result
        except MicrosoftBookingsToolError as exc:
            logger.error("Failed to list services: %s", exc)
            raise MicrosoftBookingError(str(exc)) from exc

    async def list_staff_members(self, business_id: str) -> list[dict[str, Any]]:
        """
        List all staff members for a booking business.
        
        Args:
            business_id: The booking business ID
            
        Returns:
            List of staff members with id, name, email
        """
        tool = await self._get_tool()
        try:
            staff = await tool.get_staff_members(business_id)
            return [
                {
                    "id": s.get("id", ""),
                    "name": s.get("displayName", "Unknown"),
                    "email": s.get("emailAddress"),
                    "role": s.get("role", ""),
                }
                for s in staff
            ]
        except MicrosoftBookingsToolError as exc:
            logger.error("Failed to list staff: %s", exc)
            raise MicrosoftBookingError(str(exc)) from exc

    async def check_explicit_availability(
        self,
        business_id: str,
        service_id: str,
        date: str,
    ) -> list[str]:
        """
        Check available slots for a specific business and service.
        
        Args:
            business_id: The booking business ID
            service_id: The service ID
            date: Date to check (YYYY-MM-DD format)
            
        Returns:
            List of available time slots (e.g., ["09:00", "09:30", "14:00"])
        """
        tool = await self._get_tool()
        
        # Parse date and set range
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise MicrosoftBookingError(f"Invalid date format: {date}. Use YYYY-MM-DD.")
        
        start_date = target_date.replace(hour=0, minute=0, second=0)
        end_date = target_date.replace(hour=23, minute=59, second=59)
        
        try:
            # Get service details to know staff members
            services = await tool.get_services(business_id)
            service = next((s for s in services if s.get("id") == service_id), None)
            
            if not service:
                raise MicrosoftBookingError(f"Service {service_id} not found")
            
            staff_ids = service.get("staffMemberIds", [])
            
            # Get availability
            availability = await tool.get_availability(
                business_id=business_id,
                service_id=service_id,
                start_date=start_date,
                end_date=end_date,
                staff_member_ids=staff_ids if staff_ids else None,
            )
            
            # Extract available slots
            slots = self._extract_available_slots(availability)
            return slots
            
        except MicrosoftBookingsToolError as exc:
            logger.error("Failed to check availability: %s", exc)
            raise MicrosoftBookingError(str(exc)) from exc

    async def book_explicit_appointment(
        self,
        business_id: str,
        service_id: str,
        start_time: str,
        customer_name: str,
        customer_email: str,
        customer_phone: str | None = None,
        staff_member_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Book an appointment with explicit IDs.
        
        Args:
            business_id: The booking business ID
            service_id: The service ID
            start_time: Start time (ISO format or "YYYY-MM-DD HH:MM")
            customer_name: Customer's full name
            customer_email: Customer's email
            customer_phone: Customer's phone (optional)
            staff_member_id: Specific staff member to assign (optional)
            notes: Appointment notes (optional)
            
        Returns:
            Booking confirmation with id, meeting_link, etc.
        """
        tool = await self._get_tool()
        
        # Parse start time
        start_dt = self._parse_datetime(start_time)
        
        # Get service duration
        services = await tool.get_services(business_id)
        service = next((s for s in services if s.get("id") == service_id), None)
        if not service:
            raise MicrosoftBookingError(f"Service {service_id} not found")
        
        duration_str = service.get("defaultDuration", "PT30M")
        duration_minutes = self._parse_duration_to_minutes(duration_str)
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        
        customer = BookingCustomer(
            email=customer_email,
            display_name=customer_name,
            phone=customer_phone,
        )
        
        # Prepare staff member IDs
        staff_ids = [staff_member_id] if staff_member_id else None
        
        try:
            result = await tool.create_appointment(
                business_id=business_id,
                service_id=service_id,
                start_time=start_dt,
                end_time=end_dt,
                customer=customer,
                notes=notes,
                staff_member_ids=staff_ids,
            )
            
            return {
                "status": "booked",
                "appointment_id": result.get("id"),
                "start_time": result.get("startDateTime", {}).get("dateTime"),
                "end_time": result.get("endDateTime", {}).get("dateTime"),
                "meeting_link": result.get("onlineMeetingUrl") or result.get("joinWebUrl"),
                "service_name": result.get("serviceName"),
                "staff_assigned": result.get("staffMemberIds", []),
            }
            
        except MicrosoftBookingsToolError as exc:
            logger.error("Failed to book appointment: %s", exc)
            raise MicrosoftBookingError(str(exc)) from exc

    # =========================================================================
    # LAYER 2: AUTO TOOLS (Use DB Defaults)
    # =========================================================================

    async def auto_check_availability(self, date: str) -> list[str]:
        """
        Check available slots using default business and service from database.
        
        Args:
            date: Date to check (YYYY-MM-DD format)
            
        Returns:
            List of available time slots (e.g., ["09:00", "09:30", "14:00"])
        """
        business_id = await self._get_default_business_id()
        service_id = await self._get_default_service_id()
        
        if not service_id:
            # If no default service, get the first one
            services = await self.list_services(business_id)
            if not services:
                raise MicrosoftConfigError(self._user_id, "no services found for business")
            service_id = services[0]["id"]
        
        return await self.check_explicit_availability(business_id, service_id, date)

    async def auto_book_appointment(
        self,
        start_time: str,
        customer_name: str,
        customer_email: str,
        customer_phone: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Book an appointment using default business and service from database.
        
        Automatically selects a staff member if available (picks any one).
        
        Args:
            start_time: Start time (ISO format or "YYYY-MM-DD HH:MM")
            customer_name: Customer's full name
            customer_email: Customer's email
            customer_phone: Customer's phone (optional)
            notes: Appointment notes (optional)
            
        Returns:
            Booking confirmation with id, meeting_link, etc.
        """
        logger.info("auto_book_appointment: Starting for user %s", self._user_id)
        
        business_id = await self._get_default_business_id()
        logger.debug("auto_book_appointment: business_id=%s", business_id)
        
        service_id = await self._get_default_service_id()
        logger.debug("auto_book_appointment: service_id=%s", service_id)
        
        if not service_id:
            services = await self.list_services(business_id)
            if not services:
                raise MicrosoftConfigError(self._user_id, "no services found for business")
            service_id = services[0]["id"]
            logger.debug("auto_book_appointment: auto-selected service_id=%s", service_id)
        
        # Get default staff member from DB (skip API call to avoid delays)
        staff_member_id = await self._get_default_staff_member_id()
        logger.debug("auto_book_appointment: staff_member_id from DB=%s", staff_member_id)
        # Note: We no longer auto-fetch staff from API to avoid slowdowns
        # Staff will be auto-assigned by Microsoft if not specified
        
        logger.info("auto_book_appointment: Calling book_explicit_appointment")
        return await self.book_explicit_appointment(
            business_id=business_id,
            service_id=service_id,
            start_time=start_time,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            staff_member_id=staff_member_id,
            notes=notes,
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @staticmethod
    def _parse_duration_to_minutes(duration: str) -> int:
        """Parse ISO 8601 duration to minutes (e.g., 'PT30M' -> 30)."""
        if not duration:
            return 30
        
        minutes = 0
        duration = duration.upper()
        
        if "H" in duration:
            h_idx = duration.index("H")
            h_start = duration.index("T") + 1 if "T" in duration else 0
            hours = int(duration[h_start:h_idx])
            minutes += hours * 60
        
        if "M" in duration and "T" in duration:
            m_idx = duration.index("M")
            m_start = duration.index("H") + 1 if "H" in duration else duration.index("T") + 1
            mins = int(duration[m_start:m_idx])
            minutes += mins
        
        return minutes if minutes > 0 else 30

    @staticmethod
    def _parse_datetime(time_str: str) -> datetime:
        """Parse various datetime formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise MicrosoftBookingError(f"Invalid datetime format: {time_str}")


__all__ = [
    "AgentMicrosoftBookings",
    "MicrosoftBookingError",
    "MicrosoftCredentialError",
    "MicrosoftConfigError",
]
