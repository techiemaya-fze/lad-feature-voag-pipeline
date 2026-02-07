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

# Grade-based timeline (frequency-based followup)
GRADE_TIMELINE = {
    12: 2,   # Grade 12: 3 times per week = every ~2 days
    11: 3,   # Grade 11: 2 times per week = every ~3 days  
    10: 3,   # Grade 10: 2 times per week = every ~3 days
    9: 5,    # Grade 9: 1 time per week = every ~5 days
    8: 5,    # Grade 8: 1 time per week = every ~5 days
    7: 5,    # Grade 7: 1 time per week = every ~5 days
    6: 5,    # Grade 6: 1 time per week = every ~5 days
    5: 5,    # Grade 5: 1 time per week = every ~5 days
    4: 5,    # Grade 4: 1 time per week = every ~5 days
    3: 5,    # Grade 3: 1 time per week = every ~5 days
    2: 5,    # Grade 2: 1 time per week = every ~5 days
    1: 5,    # Grade 1: 1 time per week = every ~5 days
}

# Default timeline when no grade mentioned
DEFAULT_TIMELINE = 2  # After 2 days when no grade mentioned


class ScheduleCalculator:
    """schedule calculator for GLINKS tenant"""
    
    def __init__(self, public_holidays: Optional[List[tuple]] = None):
        self.timezone = GST
        self.public_holidays = public_holidays or UAE_PUBLIC_HOLIDAYS
    
    def is_working_day(self, date: datetime) -> bool:
        """Check if working day (no holidays)"""
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
        booking_type: str = "auto_followup",
        confirmed_date_mentioned: bool = False
    ) -> datetime:
        """
        Calculate the next call datetime based on grade and working days.
        
        Args:
            current_time: The base time to calculate from
            student_grade: Student grade (9-12) for timeline calculation
            booking_type: Type of booking (auto_followup, consultation, etc.)
            confirmed_date_mentioned: If True, skip grade timeline and use given date/day
            
        Returns:
            Next valid call datetime
        """
        if confirmed_date_mentioned:
            # Date/day confirmed - use given time, only validate working hours/days
            logger.info(f"Confirmed date/day mentioned - using {current_time} directly")
            next_call = current_time
            grade_display = "confirmed_date"
        else:
            # No date/day confirmed - apply grade timeline
            if student_grade is None:
                days_to_add = DEFAULT_TIMELINE
                grade_display = "None (default)"
            else:
                days_to_add = GRADE_TIMELINE.get(student_grade, DEFAULT_TIMELINE)
                grade_display = str(student_grade)
            
            # Add working days (skips Mondays and holidays)
            next_call = self.add_working_days(current_time, days_to_add)
            
            # Preserve original time from current_time
            next_call = next_call.replace(
                hour=current_time.hour,
                minute=current_time.minute,
                second=0,
                microsecond=0
            )
            
            logger.info(f"Grade {grade_display}: +{days_to_add} working days -> {next_call}")
        
        # Validate working hours and days
        next_call = self.get_next_working_datetime(next_call)
        
        return next_call
