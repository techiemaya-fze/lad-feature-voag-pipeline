"""
Simplified Schedule Calculator - GLINKS Follow-up Rules
Validates working hours, working days, and grade-based timeline
"""

import logging
from datetime import datetime, timedelta, time
from typing import Optional, List
import pytz

logger = logging.getLogger(__name__)

# GST timezone
GST = pytz.timezone('Asia/Dubai')

# Working hours (GST)
WORKING_HOUR_START = time(10, 0)   # 10:00 AM
WORKING_HOUR_END = time(18, 30)    # 6:30 PM

# UAE Public Holidays (month, day)
UAE_PUBLIC_HOLIDAYS = [
    (1, 1),    # New Year's Day
    (5, 1),    # Labour Day
    (12, 2),   # National Day
    (12, 3),   # National Day (2nd day)
]

# Grade-based timeline (days to add for followup)
GRADE_TIMELINE = {
    12: 1,   # Grade 12: +1 day (most urgent)
    11: 2,   # Grade 11: +2 days
    10: 3,   # Grade 10: +3 days
    9: 4,    # Grade 9:  +4 days
}


class ScheduleCalculator:
    """schedule calculator for GLINKS tenant"""
    
    def __init__(self, public_holidays: Optional[List[tuple]] = None):
        self.timezone = GST
        self.public_holidays = public_holidays or UAE_PUBLIC_HOLIDAYS
    
    def is_working_day(self, date: datetime) -> bool:
        """Check if working day (no Mondays, no holidays)"""
        if date.weekday() == 0:  # Monday
            return False
        if (date.month, date.day) in self.public_holidays:
            return False
        return True
    
    def is_working_hour(self, dt: datetime) -> bool:
        """Check if within working hours"""
        return WORKING_HOUR_START <= dt.time() <= WORKING_HOUR_END
    
    def get_next_working_datetime(self, dt: datetime) -> datetime:
        """Move to next valid working day and hour"""
        current = dt
        original_time = dt.time()
        
        # Skip non-working days (max 7 iterations to avoid infinite loop)
        for _ in range(7):
            if self.is_working_day(current):
                # Always preserve the original time from first timestamp
                current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
                break
            current = current + timedelta(days=1)
            # Always preserve original time even for new days
            current = current.replace(hour=original_time.hour, minute=original_time.minute, second=0, microsecond=0)
        
        return current
    
    def add_working_days(self, start_date: datetime, working_days: int) -> datetime:
        """Add N working days (skipping Mondays and holidays)"""
        current = start_date
        days_added = 0
        
        while days_added < working_days:
            current = current + timedelta(days=1)
            if self.is_working_day(current):
                days_added += 1
        
        return current
    
    def calculate_next_call(
        self,
        current_time: datetime,
        student_grade: int = 12,
        booking_type: str = "auto_followup"
    ) -> datetime:
        """
        Calculate next call time with grade-based timeline + working hours/days validation
        
        Args:
            current_time: Reference time (first transcription timestamp)
            student_grade: Student grade (9-12), default 12
            booking_type: auto_followup or auto_consultation
        
        Returns:
            Next valid call datetime
        """
        # Get days to add based on grade
        days_to_add = GRADE_TIMELINE.get(student_grade, 1)
        
        # Add working days (skips Mondays and holidays)
        next_call = self.add_working_days(current_time, days_to_add)
        
        # Preserve original time from current_time
        next_call = next_call.replace(
            hour=current_time.hour,
            minute=current_time.minute,
            second=0,
            microsecond=0
        )
        
        # Validate working hours and days
        next_call = self.get_next_working_datetime(next_call)
        
        logger.info(f"Grade {student_grade}: +{days_to_add} working days -> {next_call}")
        
        return next_call
