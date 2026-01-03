"""Microsoft Bookings API tool for voice agent integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


@dataclass
class BookingCustomer:
    """Customer information for a booking appointment."""
    email: str
    display_name: str
    phone: str | None = None


@dataclass
class BookingSlot:
    """An available booking slot."""
    start_time: datetime
    end_time: datetime
    service_id: str
    staff_member_id: str | None = None


class MicrosoftBookingsToolError(RuntimeError):
    """Raised when Microsoft Bookings operations fail."""
    pass


class MicrosoftBookingsTool:
    """
    Microsoft Bookings API wrapper for voice agent.
    
    Unlike Google Calendar where you work with a personal calendar,
    Microsoft Bookings requires:
    1. A Booking Business ID (the business page)
    2. A Service ID (what type of appointment)
    3. Availability is calculated automatically by Microsoft
    """

    def __init__(self, access_token: str) -> None:
        """
        Initialize the bookings tool.
        
        Args:
            access_token: Valid Microsoft Graph API access token
        """
        self._access_token = access_token
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def get_booking_businesses(self) -> list[dict[str, Any]]:
        """
        Get all booking businesses the user has access to.
        
        Returns:
            List of booking business objects with id, displayName, etc.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch booking businesses: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch businesses: {resp.status_code}")
            
            data = resp.json()
            return data.get("value", [])

    async def get_services(self, business_id: str) -> list[dict[str, Any]]:
        """
        Get all services for a booking business.
        
        Args:
            business_id: The booking business ID
        
        Returns:
            List of service objects with id, displayName, defaultDuration, etc.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/services",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch services: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch services: {resp.status_code}")
            
            data = resp.json()
            return data.get("value", [])

    async def get_staff_members(self, business_id: str) -> list[dict[str, Any]]:
        """
        Get all staff members for a booking business.
        
        Args:
            business_id: The booking business ID
        
        Returns:
            List of staff member objects
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/staffMembers",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch staff members: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch staff: {resp.status_code}")
            
            data = resp.json()
            return data.get("value", [])

    async def get_business_details(self, business_id: str) -> dict[str, Any]:
        """
        Get booking business details including business hours.
        
        Args:
            business_id: The booking business ID
        
        Returns:
            Business object with displayName, businessHours, etc.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch business details: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch business details: {resp.status_code}")
            
            return resp.json()

    async def get_service_details(self, business_id: str, service_id: str) -> dict[str, Any]:
        """
        Get service details including duration.
        
        Args:
            business_id: The booking business ID
            service_id: The service ID
        
        Returns:
            Service object with displayName, defaultDuration, etc.
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/services/{service_id}",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch service details: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch service details: {resp.status_code}")
            
            return resp.json()

    async def get_calendar_view(
        self,
        business_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """
        Get existing appointments in a date range.
        
        Args:
            business_id: The booking business ID
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            List of appointment objects
        """
        # Format dates for query params (ISO 8601 with Z suffix for UTC)
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/calendarView",
                headers=self._headers,
                params={"start": start_str, "end": end_str},
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch calendar view: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch calendar view: {resp.status_code}")
            
            data = resp.json()
            return data.get("value", [])

    def _parse_iso_duration(self, duration: str) -> int:
        """Parse ISO 8601 duration to minutes (e.g., 'PT30M' -> 30)."""
        import re
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
        if not match:
            return 30  # Default to 30 min
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        return hours * 60 + minutes

    def _get_day_of_week(self, date: datetime) -> str:
        """Get day of week in Microsoft Bookings format."""
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        return days[date.weekday()]

    async def get_availability(
        self,
        business_id: str,
        service_id: str,
        start_date: datetime,
        end_date: datetime,
        staff_member_ids: list[str] | None = None,  # Kept for API compat
    ) -> list[dict[str, Any]]:
        """
        Get available time slots using manual calculation.
        
        Instead of using getStaffAvailability (which requires Application Permissions),
        this method:
        1. Gets business hours for the date
        2. Gets existing appointments from calendarView
        3. Calculates free slots = business hours - existing appointments
        
        Args:
            business_id: The booking business ID
            service_id: The service to book
            start_date: Start of date range to check
            end_date: End of date range to check
            staff_member_ids: Ignored (kept for API compatibility)
        
        Returns:
            List of available slot objects with start/end times
        """
        from datetime import timedelta
        
        # Step 1: Get business details (for business hours)
        business = await self.get_business_details(business_id)
        business_hours = business.get("businessHours", [])
        
        # Step 2: Get service details (for duration)
        service = await self.get_service_details(business_id, service_id)
        duration_str = service.get("defaultDuration", "PT30M")
        slot_duration_mins = self._parse_iso_duration(duration_str)
        
        # Step 3: Get existing appointments
        existing_appointments = await self.get_calendar_view(business_id, start_date, end_date)
        
        # Step 4: Build list of busy time ranges
        busy_ranges = []
        for appt in existing_appointments:
            appt_start = appt.get("start", {}).get("dateTime")
            appt_end = appt.get("end", {}).get("dateTime")
            if appt_start and appt_end:
                try:
                    # Parse datetime (remove 'Z' if present)
                    start_dt = datetime.fromisoformat(appt_start.replace("Z", ""))
                    end_dt = datetime.fromisoformat(appt_end.replace("Z", ""))
                    busy_ranges.append((start_dt, end_dt))
                except ValueError:
                    pass
        
        # Step 5: Calculate free slots for each day in range
        available_slots = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_check = end_date.replace(hour=23, minute=59, second=59)
        
        while current_date <= end_check:
            day_name = self._get_day_of_week(current_date)
            
            # Find business hours for this day
            day_hours = None
            for bh in business_hours:
                if bh.get("day", "").lower() == day_name:
                    day_hours = bh
                    break
            
            if day_hours and day_hours.get("timeSlots"):
                for time_slot in day_hours["timeSlots"]:
                    # Parse business hours (format: "09:00:00.0000000")
                    start_time_str = time_slot.get("startTime", "09:00:00")
                    end_time_str = time_slot.get("endTime", "17:00:00")
                    
                    try:
                        # Parse time components
                        start_parts = start_time_str.split(":")
                        end_parts = end_time_str.split(":")
                        
                        slot_start = current_date.replace(
                            hour=int(start_parts[0]),
                            minute=int(start_parts[1]),
                            second=0,
                            microsecond=0,
                        )
                        slot_end = current_date.replace(
                            hour=int(end_parts[0]),
                            minute=int(end_parts[1]),
                            second=0,
                            microsecond=0,
                        )
                        
                        # Generate slots at service duration intervals
                        current_slot = slot_start
                        while current_slot + timedelta(minutes=slot_duration_mins) <= slot_end:
                            slot_end_time = current_slot + timedelta(minutes=slot_duration_mins)
                            
                            # Check if this slot conflicts with any busy range
                            is_free = True
                            for busy_start, busy_end in busy_ranges:
                                # Conflict if slot overlaps with busy range
                                if current_slot < busy_end and slot_end_time > busy_start:
                                    is_free = False
                                    break
                            
                            if is_free:
                                available_slots.append({
                                    "startDateTime": {
                                        "dateTime": current_slot.isoformat(),
                                        "timeZone": "UTC",
                                    },
                                    "endDateTime": {
                                        "dateTime": slot_end_time.isoformat(),
                                        "timeZone": "UTC",
                                    },
                                    "serviceId": service_id,
                                })
                            
                            current_slot = slot_end_time
                    except (ValueError, IndexError) as e:
                        logger.warning("Failed to parse business hours: %s", e)
            
            current_date += timedelta(days=1)
        
        return available_slots

    async def create_appointment(
        self,
        business_id: str,
        service_id: str,
        start_time: datetime,
        end_time: datetime,
        customer: BookingCustomer,
        timezone: str = "UTC",
        notes: str | None = None,
        staff_member_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a booking appointment.
        
        Args:
            business_id: The booking business ID
            service_id: The service to book
            start_time: Appointment start time (should be in UTC)
            end_time: Appointment end time (should be in UTC)
            customer: Customer information
            timezone: Timezone for the appointment (default UTC)
            notes: Optional notes for the appointment
            staff_member_ids: Staff members to assign (often required)
        
        Returns:
            Created appointment object from Microsoft
        """
        # Format datetime with Z suffix for UTC (required by Microsoft API)
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
        
        # Build customer info with required @odata.type
        customer_info = {
            "@odata.type": "#microsoft.graph.bookingCustomerInformation",
            "name": customer.display_name,
            "emailAddress": customer.email,
            "timeZone": "",
            "notes": notes or "",
        }
        if customer.phone:
            customer_info["phone"] = customer.phone

        payload = {
            "serviceId": service_id,
            "startDateTime": {
                "dateTime": start_str,
                "timeZone": "UTC",
            },
            "endDateTime": {
                "dateTime": end_str,
                "timeZone": "UTC",
            },
            "customers": [customer_info],
            "isLocationOnline": True,  # Enable Teams meeting
            "optOutOfCustomerEmail": False,
        }
        
        if staff_member_ids:
            payload["staffMemberIds"] = staff_member_ids

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/appointments",
                headers=self._headers,
                json=payload,
            )
            
            if resp.status_code not in (200, 201):
                logger.error("Failed to create appointment: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to create appointment: {resp.status_code}")
            
            return resp.json()

    async def get_appointment(self, business_id: str, appointment_id: str) -> dict[str, Any]:
        """
        Get details of a specific appointment.
        
        Args:
            business_id: The booking business ID
            appointment_id: The appointment ID
        
        Returns:
            Appointment object
        """
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/appointments/{appointment_id}",
                headers=self._headers,
            )
            
            if resp.status_code != 200:
                logger.error("Failed to fetch appointment: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to fetch appointment: {resp.status_code}")
            
            return resp.json()

    async def cancel_appointment(
        self,
        business_id: str,
        appointment_id: str,
        cancellation_message: str | None = None,
    ) -> None:
        """
        Cancel a booking appointment.
        
        Args:
            business_id: The booking business ID
            appointment_id: The appointment ID
            cancellation_message: Optional message to send to customer
        """
        payload = {}
        if cancellation_message:
            payload["cancellationMessage"] = cancellation_message

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{GRAPH_BASE_URL}/solutions/bookingBusinesses/{business_id}/appointments/{appointment_id}/cancel",
                headers=self._headers,
                json=payload,
            )
            
            if resp.status_code not in (200, 204):
                logger.error("Failed to cancel appointment: %s", resp.text[:500])
                raise MicrosoftBookingsToolError(f"Failed to cancel appointment: {resp.status_code}")


__all__ = [
    "MicrosoftBookingsTool",
    "MicrosoftBookingsToolError",
    "BookingCustomer",
    "BookingSlot",
]
