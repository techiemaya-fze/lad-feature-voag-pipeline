"""
Schedule Calculator - Implements GLINKS Follow-up Process Rules
Based on GLINKS operational parameters and follow-up documentation
Calculates next call time considering working hours, working days, and stage-based schedules
"""

import logging
import re
from datetime import datetime, timedelta, time
from typing import Dict, Optional, List
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Default timezone (GST - Gulf Standard Time, UTC+4, same as Asia/Dubai)
DEFAULT_TIMEZONE = pytz.timezone('Asia/Dubai')  # GST (Gulf Standard Time)

# Working hours (GST - Gulf Standard Time)
WORKING_HOUR_START = time(10, 0)  # 10:00 AM GST
WORKING_HOUR_END = time(18, 30)    # 6:30 PM GST

# UAE Public Holidays (add more as needed)
# Format: (month, day) - these are common holidays, should be updated annually
UAE_PUBLIC_HOLIDAYS = [
    (1, 1),   # New Year's Day
    (5, 1),   # Labour Day
    (12, 2),  # National Day
    (12, 3),  # National Day (2nd day)
    # Add Islamic holidays based on lunar calendar (these vary each year)
    # You may want to use a library like 'hijri-converter' for accurate Islamic dates
]


class ScheduleCalculator:
    """Calculate next call time based on GLINKS follow-up rules"""
    
    def __init__(self, timezone_str: str = 'Asia/Dubai', public_holidays: Optional[List[tuple]] = None):
        """
        Initialize with timezone and public holidays
        
        Args:
            timezone_str: Timezone string (default: 'Asia/Dubai' for GST - Gulf Standard Time)
            public_holidays: List of (month, day) tuples for public holidays
        """
        try:
            self.timezone = pytz.timezone(timezone_str)
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {timezone_str}, using default GST (Asia/Dubai)")
            self.timezone = DEFAULT_TIMEZONE
        
        self.public_holidays = public_holidays or UAE_PUBLIC_HOLIDAYS
    
    def is_working_day(self, date: datetime) -> bool:
        """
        Check if a date is a working day
        
        Rules:
        - Exclude Mondays
        - Exclude public holidays
        - All other days are working days
        
        Args:
            date: datetime object to check
        
        Returns:
            True if working day, False otherwise
        """
        # Check if Monday (0 = Monday)
        if date.weekday() == 0:
            return False
        
        # Check if public holiday
        date_month = date.month
        date_day = date.day
        if (date_month, date_day) in self.public_holidays:
            return False
        
        return True
    
    def is_working_hour(self, dt: datetime) -> bool:
        """
        Check if datetime is within working hours (10:00 AM - 6:30 PM)
        
        Args:
            dt: datetime object to check
        
        Returns:
            True if within working hours, False otherwise
        """
        dt_time = dt.time()
        return WORKING_HOUR_START <= dt_time <= WORKING_HOUR_END
    
    def get_next_working_datetime(self, start_datetime: datetime, preserve_time: bool = False, allow_callback_override: bool = False) -> datetime:
        """
        Get next working datetime (working day + working hours)
        
        Args:
            start_datetime: Starting datetime
            preserve_time: If True, preserve the requested time when moving to next working day (for callbacks)
            allow_callback_override: If True, allow callback on non-working days (like Monday) if explicitly requested
        
        Returns:
            Next working datetime
        """
        current = start_datetime
        original_time = current.time()  # Preserve original time
        original_date = current.date()  # Preserve original date
        
        # For callback requests, check if the requested day is a non-working day
        if allow_callback_override and preserve_time:
            # Check if it's Monday (non-working day) but explicitly requested
            if current.weekday() == 0:  # Monday
                # Allow Monday if it's not a public holiday
                if (current.month, current.day) not in self.public_holidays:
                    # Monday is allowed for explicit callback requests
                    # Just ensure time is within working hours
                    if current.time() < WORKING_HOUR_START:
                        current = current.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                    elif current.time() > WORKING_HOUR_END:
                        current = current.replace(hour=WORKING_HOUR_END.hour, minute=WORKING_HOUR_END.minute, second=0, microsecond=0)
                    return current
        
        # First, ensure we're on a working day
        while not self.is_working_day(current):
            current = current + timedelta(days=1)
            if preserve_time:
                # Preserve the requested time when moving to next working day
                current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
            else:
                # Reset to start of working hours
                current = current.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
        
        # If time is before working hours, set to start of working hours
        if current.time() < WORKING_HOUR_START:
            if preserve_time and original_time >= WORKING_HOUR_START:
                # If original time was valid, try to preserve it on next day
                current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
            else:
                current = current.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
        
        # If time is after working hours, move to next working day
        if current.time() > WORKING_HOUR_END:
            current = current + timedelta(days=1)
            if preserve_time:
                # Preserve the requested time
                current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
            else:
                current = current.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
            # Ensure it's a working day
            while not self.is_working_day(current):
                current = current + timedelta(days=1)
                if preserve_time:
                    current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
        
        return current
    
    def add_working_days(self, start_date: datetime, working_days: int, preserve_original_time: bool = False) -> datetime:
        """
        Add working days to a date (excluding Mondays and holidays)
        
        Args:
            start_date: Starting date
            working_days: Number of working days to add
            preserve_original_time: If True, preserve the original time from start_date instead of using working hour start
        
        Returns:
            Date after adding working days
        """
        current = start_date
        days_added = 0
        original_time = start_date.time()
        
        while days_added < working_days:
            current = current + timedelta(days=1)
            if self.is_working_day(current):
                days_added += 1
        
        # Either preserve original time or set to start of working hours
        if preserve_original_time:
            # Keep the original time from the start_date
            current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
        else:
            # Set to start of working hours (legacy behavior)
            current = current.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
        
        return current
    
    def calculate_next_call(self, outcome: str, outcome_details: Dict, lead_info: Optional[Dict] = None, reference_time: Optional[datetime] = None, allow_outside_hours: bool = False) -> Optional[datetime]:
        """
        Main function - calculates next call time based on outcome
        
        Args:
            outcome: Outcome type ("no_answer", "responsive", "meeting_booked", "not_interested", "callback_requested", "event_followup")
            outcome_details: Dictionary with outcome details (student_grade, callback_time, etc.)
            lead_info: Optional lead information from students_information table
        
        Returns:
            datetime object for next call time (in timezone, within working hours) or None
        """
        if not outcome:
            logger.warning("No outcome provided")
            return None
        
        # Get current time in timezone (or use provided reference_time)
        if reference_time:
            # Use provided reference time (from transcription)
            if reference_time.tzinfo is None:
                now = self.timezone.localize(reference_time)
            else:
                now = reference_time.astimezone(self.timezone)
        else:
            # Default to current time
            now = datetime.now(self.timezone)
        
        # If lead_info is None, create a default one
        if lead_info is None:
            lead_info = {
                "stage": 1,
                "status": "active",
                "student_grade": outcome_details.get("student_grade"),
                "created_at": now,
                "last_call_time": now,
                "call_count": 0
            }
        
        # Handle callback_requested - use exact requested time (can be any day, just ensure working hours)
        if outcome == "callback_requested":
            callback_time = outcome_details.get("callback_time")
            if callback_time:
                parsed_time = self.parse_callback_time(callback_time, now)
                if parsed_time:
                    # Check if this is a relative time request (within/after/in X mins)
                    callback_lower = callback_time.lower().strip()
                    is_relative_time = bool(re.search(r'(?:within|after|in)\s+(\d+)\s*(?:mins?|minutes?)', callback_lower))
                    
                    if is_relative_time:
                        # For relative times, use the parsed time as-is (it's already calculated correctly)
                        # Only adjust if outside working hours, but try to preserve the relative time
                        if parsed_time.time() < WORKING_HOUR_START:
                            # Before working hours - move to start of working hours on same day
                            parsed_time = parsed_time.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                        elif parsed_time.time() > WORKING_HOUR_END:
                            # After working hours - move to start of working hours next day
                            parsed_time = parsed_time + timedelta(days=1)
                            parsed_time = parsed_time.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                        
                        # Ensure timezone awareness
                        if parsed_time.tzinfo is None:
                            parsed_time = self.timezone.localize(parsed_time)
                        
                        # Return as-is (relative times should be honored - don't call get_next_working_datetime)
                        logger.info(f"Callback relative time '{callback_time}' parsed to: {parsed_time}")
                        return parsed_time
                    else:
                        # For absolute times (like "tomorrow at 7 PM", "Monday at 12"), honor the EXACT time requested
                        # If allow_outside_hours is True (explicit agent commitment), use exact time without adjustment
                        if allow_outside_hours:
                            # Agent explicitly committed to this time - honor it regardless of working hours
                            logger.info(f"Using exact requested time: {parsed_time.time()} (agent commitment, overriding working hours)")
                        else:
                            # Only adjust if significantly outside working hours (more than 30 mins)
                            original_time = parsed_time.time()
                            time_diff_start = (parsed_time.time().hour * 60 + parsed_time.time().minute) - (WORKING_HOUR_START.hour * 60 + WORKING_HOUR_START.minute)
                            time_diff_end = (WORKING_HOUR_END.hour * 60 + WORKING_HOUR_END.minute) - (parsed_time.time().hour * 60 + parsed_time.time().minute)
                            
                            if time_diff_start < -30:  # More than 30 mins before working hours
                                # Before working hours - move to start of working hours
                                parsed_time = parsed_time.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                                logger.info(f"Adjusted time from {original_time} to {WORKING_HOUR_START} (more than 30 mins before working hours)")
                            elif time_diff_end < -30:  # More than 30 mins after working hours
                                # After working hours - move to end of working hours
                                parsed_time = parsed_time.replace(hour=WORKING_HOUR_END.hour, minute=WORKING_HOUR_END.minute, second=0, microsecond=0)
                                logger.info(f"Adjusted time from {original_time} to {WORKING_HOUR_END} (more than 30 mins after working hours)")
                            else:
                                # Within 30 mins of working hours - use exact time requested
                                logger.info(f"Using exact requested time: {original_time} (within acceptable range)")
                        
                        # Ensure timezone awareness
                        if parsed_time.tzinfo is None:
                            parsed_time = self.timezone.localize(parsed_time)
                        
                        # Return as-is (any day is allowed for callbacks, exact time is preserved)
                        logger.info(f"Callback absolute time '{callback_time}' parsed to: {parsed_time}")
                        return parsed_time
            else:
                logger.warning("Callback requested but no callback_time provided - will use Stage 2 grade-based schedule")
                # Treat like responsive outcome - use Stage 2 schedule
                if lead_info.get("stage", 1) == 1:
                    lead_info["stage"] = 2
                    lead_info["stage2_start"] = now
                next_call = self.calculate_stage2_schedule(lead_info, now)
                return self.get_next_working_datetime(next_call)
        
        # Handle different outcomes
        if outcome == "no_answer":
            next_call = self.calculate_stage1_schedule(lead_info, now)
        
        elif outcome == "responsive":
            # Move to Stage 2 if not already
            if lead_info.get("stage", 1) == 1:
                lead_info["stage"] = 2
                lead_info["stage2_start"] = now
            
            next_call = self.calculate_stage2_schedule(lead_info, now)
        
        elif outcome == "meeting_booked":
            # Move to Stage 3
            lead_info["stage"] = 3
            if not lead_info.get("stage3_start"):
                lead_info["stage3_start"] = now
            
            # For meeting booking, check if there's a followup_time/callback_time
            # Meeting bookings must be on working days only (real-time appointment)
            followup_time = outcome_details.get("callback_time") or outcome_details.get("followup_time")
            if followup_time:
                parsed_time = self.parse_callback_time(followup_time, now)
                if parsed_time:
                    # Meeting booking: Must be on working days only (strict)
                    # Adjust to next working day if needed, but preserve time
                    next_call = self.get_next_working_datetime(parsed_time, preserve_time=True, allow_callback_override=False)
                    # Don't call get_next_working_datetime again - already done above
                    return next_call
                else:
                    next_call = self.calculate_stage3_schedule(lead_info, now)
            else:
                next_call = self.calculate_stage3_schedule(lead_info, now)
        
        elif outcome == "not_interested":
            next_call = self.calculate_not_interested_schedule(lead_info, now)
        
        elif outcome == "event_followup":
            # Event follow-up: Valid for 3 years, periodic contact
            next_call = self.calculate_event_followup_schedule(lead_info, now)
        
        else:
            logger.warning(f"Unknown outcome: {outcome}, using default Stage 1 schedule")
            next_call = self.calculate_stage1_schedule(lead_info, now)
        
        if not next_call:
            return None
        
        # For callback_requested outcomes, we already handled working hours in the callback_requested section above
        # Don't apply get_next_working_datetime again as it might override relative time calculations
        if outcome == "callback_requested":
            # Already processed in callback_requested section above - return as-is
            return next_call
        
        # For other outcomes, ensure next call is within working hours and on a working day
        return self.get_next_working_datetime(next_call)
    
    def calculate_stage1_schedule(self, lead_info: Dict, now: datetime) -> datetime:
        """
        Stage 1: Non-responsive lead follow-up schedule
        
        Rules:
        - Week 1: Every alternate day (2 working days)
        - Week 2: Every third day (2 working days gap)
        - Week 3: Every fourth day (3 working days gap)
        - Week 4-8: Once per week (5 working days)
        - Month 3-4: Every alternate week (10 working days)
        - Month 5-48: Once per month (20 working days)
        """
        created_at = lead_info.get("created_at", now)
        last_call_time = lead_info.get("last_call_time", created_at)
        
        # Parse dates if strings
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if created_at.tzinfo is None:
            created_at = self.timezone.localize(created_at)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        # Calculate working days since first contact
        working_days_since_first = self.count_working_days(created_at, now)
        
        if working_days_since_first <= 5:  # Week 1 (approximately 5 working days)
            # Every alternate day (2 working days)
            next_call = self.add_working_days(last_call_time, 2)
        elif working_days_since_first <= 10:  # Week 2 (approximately 10 working days)
            # Every third day (2 working days gap)
            next_call = self.add_working_days(last_call_time, 2)
        elif working_days_since_first <= 15:  # Week 3 (approximately 15 working days)
            # Every fourth day (3 working days gap)
            next_call = self.add_working_days(last_call_time, 3)
        elif working_days_since_first <= 40:  # Week 4-8 (approximately 40 working days)
            # Once per week (5 working days)
            next_call = self.add_working_days(last_call_time, 5)
        elif working_days_since_first <= 80:  # Month 3-4 (approximately 80 working days)
            # Every alternate week (10 working days)
            next_call = self.add_working_days(last_call_time, 10)
        else:  # Month 5-48 (up to 4 years)
            # Once per month (approximately 20 working days)
            next_call = self.add_working_days(last_call_time, 20)
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1)
        
        return next_call
    
    def calculate_stage2_schedule(self, lead_info: Dict, now: datetime) -> datetime:
        """
        Stage 2: Responsive lead follow-up (pre-meeting) - Grade-based schedule
        
        Rules by Grade:
        - Grade 9 or below: Weekly for 6 months, then monthly
        - Grade 10-11: Twice per week for first year, then Grade 12 timeline
        - Grade 12+: 3x/week (6 months) → 2x/week (6 months) → weekly (year 2) → bi-weekly (year 3-4) → monthly (year 5-7)
        """
        student_grade = lead_info.get("student_grade")
        stage2_start = lead_info.get("stage2_start", now)
        last_call_time = lead_info.get("last_call_time", now)
        
        # Parse dates if strings
        if isinstance(stage2_start, str):
            stage2_start = datetime.fromisoformat(stage2_start.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if stage2_start.tzinfo is None:
            stage2_start = self.timezone.localize(stage2_start)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        working_days_in_stage = self.count_working_days(stage2_start, now)
        
        if not student_grade or student_grade <= 9:
            # Grade 9 or below
            # First 6 months: Once per week (5 working days)
            # After 6 months: Once per month (20 working days)
            if working_days_in_stage <= 120:  # Approximately 6 months of working days
                next_call = self.add_working_days(last_call_time, 5, preserve_original_time=True)
            else:
                next_call = self.add_working_days(last_call_time, 20, preserve_original_time=True)
        
        elif student_grade in [10, 11]:
            # Grade 10-11
            # First year: Twice per week (2-3 working days)
            # After first year: Follow Grade 12 timeline
            if working_days_in_stage <= 240:  # Approximately 1 year of working days
                next_call = self.add_working_days(last_call_time, 2, preserve_original_time=True)
            else:
                next_call = self.calculate_grade12_timeline(lead_info, now)
        
        else:  # Grade 12 and above (UG/PG/Masters)
            next_call = self.calculate_grade12_timeline(lead_info, now)
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1, preserve_original_time=True)
        
        return next_call
    
    def calculate_grade12_timeline(self, lead_info: Dict, now: datetime) -> datetime:
        """
        Grade 12+ timeline (UG/PG/Masters)
        
        Rules:
        - First 6 months: 3 times per week (2 working days)
        - Month 7-12: 2 times per week (2-3 working days)
        - Year 2: Once per week (5 working days)
        - Year 3-4: Every alternate week (10 working days)
        - Year 5-7: Once per month (20 working days)
        """
        stage2_start = lead_info.get("stage2_start", now)
        last_call_time = lead_info.get("last_call_time", now)
        
        # Parse dates if strings
        if isinstance(stage2_start, str):
            stage2_start = datetime.fromisoformat(stage2_start.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if stage2_start.tzinfo is None:
            stage2_start = self.timezone.localize(stage2_start)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        working_days_in_timeline = self.count_working_days(stage2_start, now)
        
        if working_days_in_timeline <= 120:  # First 6 months
            # 3 times per week (2 working days)
            next_call = self.add_working_days(last_call_time, 2, preserve_original_time=True)
        elif working_days_in_timeline <= 240:  # Month 7-12
            # 2 times per week (2-3 working days)
            next_call = self.add_working_days(last_call_time, 2, preserve_original_time=True)
        elif working_days_in_timeline <= 480:  # Year 2
            # Once per week (5 working days)
            next_call = self.add_working_days(last_call_time, 5, preserve_original_time=True)
        elif working_days_in_timeline <= 960:  # Year 3-4
            # Every alternate week (10 working days)
            next_call = self.add_working_days(last_call_time, 10, preserve_original_time=True)
        else:  # Year 5-7
            # Once per month (20 working days)
            next_call = self.add_working_days(last_call_time, 20, preserve_original_time=True)
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1, preserve_original_time=True)
        
        return next_call
    
    def calculate_stage3_schedule(self, lead_info: Dict, now: datetime) -> datetime:
        """
        Stage 3: Post-counselling retention follow-up
        
        Rules:
        - First 2 months: 2 times per week (2-3 working days)
        - Month 3-6: Once per week (5 working days)
        - Month 7-10: Every alternate week (10 working days)
        - Month 11-48: Once per month (20 working days)
        """
        stage3_start = lead_info.get("stage3_start", now)
        last_call_time = lead_info.get("last_call_time", now)
        
        # Parse dates if strings
        if isinstance(stage3_start, str):
            stage3_start = datetime.fromisoformat(stage3_start.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if stage3_start.tzinfo is None:
            stage3_start = self.timezone.localize(stage3_start)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        working_days_in_stage = self.count_working_days(stage3_start, now)
        
        if working_days_in_stage <= 40:  # First 2 months
            # 2 times per week (2-3 working days)
            next_call = self.add_working_days(last_call_time, 2)
        elif working_days_in_stage <= 120:  # Month 3-6
            # Once per week (5 working days)
            next_call = self.add_working_days(last_call_time, 5)
        elif working_days_in_stage <= 200:  # Month 7-10
            # Every alternate week (10 working days)
            next_call = self.add_working_days(last_call_time, 10)
        else:  # Month 11-48
            # Once per month (20 working days)
            next_call = self.add_working_days(last_call_time, 20)
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1)
        
        return next_call
    
    def calculate_not_interested_schedule(self, lead_info: Dict, now: datetime) -> Optional[datetime]:
        """
        Not interested - still follow up monthly for 4 years
        
        Rules:
        - Monthly follow-up for 4 years (approximately 960 working days)
        - After 4 years, close lead (return None)
        """
        created_at = lead_info.get("created_at", now)
        last_call_time = lead_info.get("last_call_time", now)
        
        # Parse dates if strings
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if created_at.tzinfo is None:
            created_at = self.timezone.localize(created_at)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        working_days_since_first = self.count_working_days(created_at, now)
        
        if working_days_since_first < 960:  # Less than 4 years (approximately 960 working days)
            # Monthly follow-up (20 working days)
            next_call = self.add_working_days(last_call_time, 20)
        else:
            # 4 years passed, don't schedule (lead should be closed)
            logger.info("Lead has been not_interested for 4+ years, not scheduling next call")
            return None
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1)
        
        return next_call
    
    def calculate_event_followup_schedule(self, lead_info: Dict, now: datetime) -> Optional[datetime]:
        """
        Event follow-up - Valid for 3 years from interest expression
        
        Rules:
        - Periodic contact for 3 years (approximately 720 working days)
        - Frequency: Monthly or as per event schedule
        """
        event_interest_date = lead_info.get("event_interest_date", now)
        last_call_time = lead_info.get("last_call_time", now)
        
        # Parse dates if strings
        if isinstance(event_interest_date, str):
            event_interest_date = datetime.fromisoformat(event_interest_date.replace('Z', '+00:00'))
        if isinstance(last_call_time, str):
            last_call_time = datetime.fromisoformat(last_call_time.replace('Z', '+00:00'))
        
        # Ensure timezone awareness
        if event_interest_date.tzinfo is None:
            event_interest_date = self.timezone.localize(event_interest_date)
        if last_call_time.tzinfo is None:
            last_call_time = self.timezone.localize(last_call_time)
        
        working_days_since_interest = self.count_working_days(event_interest_date, now)
        
        if working_days_since_interest < 720:  # Less than 3 years
            # Monthly follow-up (20 working days)
            next_call = self.add_working_days(last_call_time, 20)
        else:
            # 3 years passed, stop event follow-up
            logger.info("Event follow-up period (3 years) expired")
            return None
        
        # Ensure next call is in the future
        if next_call <= now:
            next_call = self.add_working_days(now, 1)
        
        return next_call
    
    def count_working_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Count working days between two dates (excluding Mondays and holidays)
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Number of working days
        """
        if end_date < start_date:
            return 0
        
        count = 0
        current = start_date.date()
        end = end_date.date()
        
        while current <= end:
            # Create datetime for checking
            dt = datetime.combine(current, time(12, 0))
            if self.timezone:
                dt = self.timezone.localize(dt)
            
            if self.is_working_day(dt):
                count += 1
            
            current += timedelta(days=1)
        
        return count
    
    def parse_callback_time(self, callback_string: str, reference_time: datetime) -> datetime:
        """
        Parse callback time from lead's words
        
        Examples:
        - "tomorrow 5 PM" → datetime tomorrow at 17:00
        - "Monday 6 PM" → next Monday at 18:00
        - "2025-12-27 17:00:00" → parsed datetime
        
        Args:
            callback_string: String with callback time
            reference_time: Reference time (usually now)
        
        Returns:
            datetime object (will be adjusted to working hours later)
        """
        if not callback_string:
            return None
        
        callback_lower = callback_string.lower().strip()
        
        # Try to parse as ISO format first
        try:
            parsed = datetime.fromisoformat(callback_string.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = self.timezone.localize(parsed)
            return parsed
        except (ValueError, AttributeError):
            pass
        
        # Parse "within X mins/minutes", "in X minutes", or "after X mins" format
        import re
        within_patterns = [
            r'within\s+(\d+)\s*(?:mins?|minutes?)',
            r'in\s+(\d+)\s*(?:mins?|minutes?)',
            r'after\s+(\d+)\s*(?:mins?|minutes?)',  # Add support for "after X mins"
            r'(\d+)\s*(?:mins?|minutes?)\s*(?:from now|later)'
        ]
        
        for pattern in within_patterns:
            within_match = re.search(pattern, callback_lower)
            if within_match:
                minutes = int(within_match.group(1))
                # Ensure reference_time is timezone-aware
                if reference_time.tzinfo is None:
                    ref_time = self.timezone.localize(reference_time)
                else:
                    ref_time = reference_time
                
                # Add minutes
                result_time = ref_time + timedelta(minutes=minutes)
                
                # For relative times (within/after/in X mins), use the exact calculated time
                # Only adjust if outside working hours, but preserve the relative time calculation
                if result_time.time() < WORKING_HOUR_START:
                    # Before working hours - move to start of working hours on same day
                    result_time = result_time.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                elif result_time.time() > WORKING_HOUR_END:
                    # After working hours - move to start of working hours next day
                    result_time = result_time + timedelta(days=1)
                    result_time = result_time.replace(hour=WORKING_HOUR_START.hour, minute=WORKING_HOUR_START.minute, second=0, microsecond=0)
                
                # For relative times, don't enforce working day restriction - use the calculated time
                # The time is already calculated relative to reference_time, so honor it
                # Only ensure timezone awareness
                if result_time.tzinfo is None:
                    result_time = self.timezone.localize(result_time)
                
                return result_time
        
        # Parse relative times
        result_time = reference_time
        
        # Parse month names with ordinal/numeric days (e.g., "fifteenth March", "March 15", "15th March")
        month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sept': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        ordinal_to_number = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20,
            'twenty-first': 21, 'twenty-second': 22, 'twenty-third': 23, 'twenty-fourth': 24,
            'twenty-fifth': 25, 'twenty-sixth': 26, 'twenty-seventh': 27, 'twenty-eighth': 28,
            'twenty-ninth': 29, 'thirtieth': 30, 'thirty-first': 31,
            # Common transcription errors
            'nighteenth': 19, 'ninteenth': 19  # Common misspellings of "nineteenth"
        }
        
        # Try to extract month and day
        month_day_found = False
        target_month = None
        target_day = None
        
        # Pattern 1: "fifteenth March" or "March fifteenth"
        for ordinal, day_num in ordinal_to_number.items():
            for month_name, month_num in month_names.items():
                if ordinal in callback_lower and month_name in callback_lower:
                    target_month = month_num
                    target_day = day_num
                    month_day_found = True
                    break
            if month_day_found:
                break
        
        # Pattern 2: "March 15" or "15 March" or "15th March"
        if not month_day_found:
            for month_name, month_num in month_names.items():
                # Match "March 15" or "15 March" or "15th March"
                month_day_pattern = rf'\b({month_name})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_name})\b'
                month_day_match = re.search(month_day_pattern, callback_lower)
                if month_day_match:
                    target_month = month_num
                    # Day could be in group 2 or 3
                    target_day = int(month_day_match.group(2) or month_day_match.group(3))
                    month_day_found = True
                    break
        
        # If month and day found, set the date
        if month_day_found and target_month and target_day:
            try:
                current_year = reference_time.year
                # Try current year first
                target_date = result_time.replace(year=current_year, month=target_month, day=target_day, hour=0, minute=0, second=0, microsecond=0)
                
                # If the date is in the past, use next year
                if target_date.date() < reference_time.date():
                    target_date = target_date.replace(year=current_year + 1)
                
                result_time = target_date
                logger.info(f"Parsed month/day from '{callback_string}': {target_month}/{target_day} -> {result_time.date()}")
            except ValueError as e:
                logger.warning(f"Invalid date: month={target_month}, day={target_day}: {e}")
                month_day_found = False
        
        # Extract day
        if not month_day_found and "tomorrow" in callback_lower:
            result_time = result_time + timedelta(days=1)
        elif "monday" in callback_lower:
            days_until_monday = (0 - result_time.weekday()) % 7
            if days_until_monday == 0:  # Today is Monday
                days_until_monday = 7  # Next Monday
            result_time = result_time + timedelta(days=days_until_monday)
        elif "tuesday" in callback_lower:
            days_until_tuesday = (1 - result_time.weekday()) % 7
            if days_until_tuesday == 0:
                days_until_tuesday = 7
            result_time = result_time + timedelta(days=days_until_tuesday)
        elif "wednesday" in callback_lower:
            days_until_wednesday = (2 - result_time.weekday()) % 7
            if days_until_wednesday == 0:
                days_until_wednesday = 7
            result_time = result_time + timedelta(days=days_until_wednesday)
        elif "thursday" in callback_lower:
            days_until_thursday = (3 - result_time.weekday()) % 7
            if days_until_thursday == 0:
                days_until_thursday = 7
            result_time = result_time + timedelta(days=days_until_thursday)
        elif "friday" in callback_lower:
            days_until_friday = (4 - result_time.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            result_time = result_time + timedelta(days=days_until_friday)
        elif "saturday" in callback_lower:
            days_until_saturday = (5 - result_time.weekday()) % 7
            if days_until_saturday == 0:
                days_until_saturday = 7
            result_time = result_time + timedelta(days=days_until_saturday)
        elif "sunday" in callback_lower:
            days_until_sunday = (6 - result_time.weekday()) % 7
            if days_until_sunday == 0:
                days_until_sunday = 7
            result_time = result_time + timedelta(days=days_until_sunday)
        
        # Extract time
        import re
        time_patterns = [
            # Match "2:00 PM" or "2:00PM" first (most specific)
            (r'(\d{1,2}):(\d{2})\s*(?:pm|p\.m\.)', lambda m: (int(m.group(1)) + (12 if int(m.group(1)) < 12 else 0), int(m.group(2)))),
            (r'(\d{1,2}):(\d{2})\s*(?:am|a\.m\.)', lambda m: (int(m.group(1)) if int(m.group(1)) < 12 else 0, int(m.group(2)))),
            # Match "evening 7" or "evening at 7" - evening means PM
            (r'evening\s+(?:at\s+)?(\d{1,2})\b', lambda m: int(m.group(1)) + (12 if int(m.group(1)) < 12 else 0)),
            # Match "morning 7" or "morning at 7" - morning means AM
            (r'morning\s+(?:at\s+)?(\d{1,2})\b', lambda m: int(m.group(1)) if int(m.group(1)) < 12 else 0),
            # Match "2 PM" or "2PM" (without colon) - check for word boundaries to avoid matching day numbers
            (r'\b(\d{1,2})\s*(?:pm|p\.m\.)\b', lambda m: int(m.group(1)) + (12 if int(m.group(1)) < 12 else 0)),
            (r'\b(\d{1,2})\s*(?:am|a\.m\.)\b', lambda m: int(m.group(1)) if int(m.group(1)) < 12 else 0),
            # Match "2:00" (24-hour format)
            (r'(\d{1,2}):(\d{2})', lambda m: (int(m.group(1)), int(m.group(2)))),
            # Match "at 11" or "at 11 AM" - This handles cases like "Sunday at 11" or "book at 11"
            # CRITICAL: This pattern should match "at 11" even without AM/PM
            (r'(?:at|@)\s*(\d{1,2})(?:\s*(?:am|pm|a\.m\.|p\.m\.))?\b', lambda m: int(m.group(1))),
            # Match "2 o'clock" (with o'clock)
            (r'\b(\d{1,2})\s+o\'?clock\b', lambda m: int(m.group(1))),
        ]
        
        hour = None  # No default hour
        minute = None
        time_found = False
        
        for pattern, extractor in time_patterns:
            match = re.search(pattern, callback_lower)
            if match:
                result = extractor(match)
                if isinstance(result, tuple):
                    hour, minute = result
                else:
                    hour = result
                time_found = True
                logger.info(f"Extracted hour {hour} from pattern '{pattern}' in: {callback_string}")
                break
        
        # If no time pattern matched, try to find hour numbers in time context
        if not time_found:
            # Try to find hour number after "at" (most common case: "Sunday at 11")
            at_time_match = re.search(r'(?:at|@)\s*(\d{1,2})\b', callback_lower)
            if at_time_match:
                hour = int(at_time_match.group(1))
                time_found = True
                logger.info(f"Extracted hour {hour} from 'at' pattern in: {callback_string}")
        
        # If still no time found, check if there's a standalone number that could be an hour (1-23)
        # Only in time-related context to avoid false matches
        if not time_found:
            # Look for numbers that are likely hours (1-23) near time-related words
            # Check for context like "book", "schedule", "time", "hour", etc.
            context_match = re.search(r'(?:book|schedule|time|hour|slot|meeting|appointment).*?(\d{1,2})\b', callback_lower)
            if context_match:
                potential_hour = int(context_match.group(1))
                if 1 <= potential_hour <= 23:  # Valid hour range
                    hour = potential_hour
                    time_found = True
                    logger.info(f"Extracted hour {hour} from time context in: {callback_string}")
        # If no time found, do not set a default time. Let the caller handle it (should use first timestamp time)
        if hour is not None:
            # If minute is None, default to 0
            if minute is None:
                minute = 0
            result_time = result_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # else: do not set time, let the caller combine with first timestamp time if needed
        
        # Ensure timezone awareness
        if result_time.tzinfo is None:
            result_time = self.timezone.localize(result_time)
        
        # Ensure time is in the future
        # Allow same-day scheduling if the target time is later than reference time
        if result_time < reference_time:
            result_time = result_time + timedelta(days=1)
        
        return result_time


def calculate_next_call_time(outcome: str, outcome_details: Dict, lead_info: Optional[Dict] = None, timezone_str: str = 'Asia/Dubai') -> Optional[datetime]:
    """
    Main function to calculate next call time
    
    Args:
        outcome: Outcome type
        outcome_details: Outcome details dictionary
        lead_info: Optional lead information
        timezone_str: Timezone string (default: 'Asia/Dubai' for GST - Gulf Standard Time)
    
    Returns:
        datetime object for next call time (within working hours on working day, in GST)
    """
    calculator = ScheduleCalculator(timezone_str)
    return calculator.calculate_next_call(outcome, outcome_details, lead_info)


if __name__ == "__main__":
    # Test the calculator
    from datetime import datetime
    
    calculator = ScheduleCalculator()
    
    # Test Stage 1
    lead_info = {
        "stage": 1,
        "created_at": datetime.now() - timedelta(days=5),
        "last_call_time": datetime.now() - timedelta(days=1),
        "call_count": 2
    }
    
    outcome = "no_answer"
    outcome_details = {}
    
    next_call = calculator.calculate_next_call(outcome, outcome_details, lead_info)
    print(f"Next call (Stage 1): {next_call}")