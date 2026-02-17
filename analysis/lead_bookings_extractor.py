"""
Lead Bookings Extractor - Simplified Version
Production-ready with full database integration
Works with both local and production databases
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import pytz

# Add parent directory to path
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import Gemini client
from analysis.gemini_client import generate_with_schema_async, IMPROVED_BOOKING_SCHEMA
from google.genai import types

# Import Gemini configuration
from analysis.gemini_config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_RETRIES
)

# Import database storage
from db.storage.lead_bookings import LeadBookingsStorage, LeadBookingsStorageError

# Import schedule calculator for GLINKS
try:
    from analysis.schedule_calculator import ScheduleCalculator
    SCHEDULE_CALCULATOR_AVAILABLE = True
except ImportError:
    SCHEDULE_CALCULATOR_AVAILABLE = False
    ScheduleCalculator = None

# GLINKS tenant ID
GLINKS_TENANT_ID = "926070b5-189b-4682-9279-ea10ca090b84"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GST timezone
GST = pytz.timezone('Asia/Dubai')


class LeadBookingsExtractor:
    """
    Lead Bookings Extractor
    
    Extracts followup time based on:
    1. User must confirm the time
    2. Relative time (e.g., "after 5 minutes") adds to LAST timestamp
    3. No specific time -> uses FIRST timestamp time
    4. No confirmation -> schedules 2 days from FIRST timestamp time
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in .env file")
        
        # Initialize database storage
        self.storage = LeadBookingsStorage()
        
        # Initialize schedule calculator for GLINKS
        self.schedule_calculator = None
        if SCHEDULE_CALCULATOR_AVAILABLE:
            self.schedule_calculator = ScheduleCalculator()
            logger.info("Schedule calculator initialized for GLINKS support")
        
    
    async def close(self):
        """Close the database connection pool"""
        await self.storage.close()
    
    def extract_first_timestamp(self, transcriptions_data) -> Optional[datetime]:
        """Extract the first timestamp from transcriptions (UTC) and convert to GST"""
        if transcriptions_data is None:
            return None
        
        try:
            UTC = pytz.timezone('UTC')
            messages_list = None
            
            if isinstance(transcriptions_data, dict):
                if 'messages' in transcriptions_data:
                    messages_list = transcriptions_data['messages']
                elif 'segments' in transcriptions_data:
                    messages_list = transcriptions_data['segments']
            elif isinstance(transcriptions_data, list):
                messages_list = transcriptions_data
            
            if not messages_list:
                return None
            
            for entry in messages_list:
                if not isinstance(entry, dict):
                    continue
                
                timestamp = (entry.get('timestamp') or entry.get('created_at') or 
                           entry.get('time') or entry.get('date'))
                
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        elif isinstance(timestamp, datetime):
                            dt = timestamp
                        else:
                            continue
                        
                        if dt.tzinfo is None:
                            dt = UTC.localize(dt)
                        elif dt.tzinfo != UTC:
                            dt = dt.astimezone(UTC)
                        
                        gst_timestamp = dt.astimezone(GST)
                        logger.info(f"First timestamp: {gst_timestamp} (GST)")
                        return gst_timestamp.replace(microsecond=0)
                    except Exception as e:
                        logger.debug(f"Error parsing timestamp: {e}")
                        continue
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting first timestamp: {e}")
            return None
    
    def extract_last_timestamp(self, transcriptions_data) -> Optional[datetime]:
        """Extract the last timestamp from transcriptions (UTC) and convert to GST"""
        if transcriptions_data is None:
            return None
        
        try:
            UTC = pytz.timezone('UTC')
            messages_list = None
            
            if isinstance(transcriptions_data, dict):
                if 'messages' in transcriptions_data:
                    messages_list = transcriptions_data['messages']
                elif 'segments' in transcriptions_data:
                    messages_list = transcriptions_data['segments']
            elif isinstance(transcriptions_data, list):
                messages_list = transcriptions_data
            
            if not messages_list:
                return None
            
            for entry in reversed(messages_list):
                if not isinstance(entry, dict):
                    continue
                
                timestamp = (entry.get('timestamp') or entry.get('created_at') or 
                           entry.get('time') or entry.get('date'))
                
                if timestamp:
                    try:
                        if isinstance(timestamp, str):
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        elif isinstance(timestamp, datetime):
                            dt = timestamp
                        else:
                            continue
                        
                        if dt.tzinfo is None:
                            dt = UTC.localize(dt)
                        elif dt.tzinfo != UTC:
                            dt = dt.astimezone(UTC)
                        
                        gst_timestamp = dt.astimezone(GST)
                        logger.info(f"Last timestamp: {gst_timestamp} (GST)")
                        return gst_timestamp.replace(microsecond=0)
                    except Exception as e:
                        logger.debug(f"Error parsing timestamp: {e}")
                        continue
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting last timestamp: {e}")
            return None
    
    def parse_transcription(self, transcriptions_data) -> str:
        """Parse transcription from various formats and clean up text"""
        if transcriptions_data is None:
            return ""
        
        if isinstance(transcriptions_data, str):
            if not transcriptions_data.strip():
                return ""
            # Try to parse JSON string into dict/list
            try:
                transcriptions_data = json.loads(transcriptions_data)
            except (json.JSONDecodeError, TypeError):
                return ""
        
        if isinstance(transcriptions_data, dict):
            if 'segments' in transcriptions_data and isinstance(transcriptions_data['segments'], list):
                conversation_log = transcriptions_data['segments']
                return "\n".join([
                    f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('text', '')}"
                    for entry in conversation_log
                ])
            elif 'messages' in transcriptions_data and isinstance(transcriptions_data['messages'], list):
                conversation_log = transcriptions_data['messages']
                return "\n".join([
                    f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('message', entry.get('text', ''))}"
                    for entry in conversation_log
                ])
        elif isinstance(transcriptions_data, list):
            return "\n".join([
                f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('message', entry.get('text', ''))}"
                for entry in transcriptions_data
            ])
        
        return str(transcriptions_data)
    
    async def extract_followup_time(
        self,
        conversation_text: str,
        first_timestamp: datetime,
        last_timestamp: datetime,
        tenant_id: str = None
    ) -> Dict:
        """
        Extract followup time using simplified logic
        
        Returns:
            {
                "booking_type": "auto_followup" or "auto_consultation",
                "scheduled_at": datetime or None,
                "time_phrase": str,
                "user_confirmed": bool,
                "calculation_method": str,
                "student_grade": int or None
            }
        """
        
        # Use improved schema from Google client
        schema = IMPROVED_BOOKING_SCHEMA
        
        # Debug: Show conversation content being sent to Gemini
        logger.info(f"CONVERSATION SENT TO GEMINI:\n{conversation_text}\n---END CONVERSATION---")
        
        # Check for meaningful user content to prevent LLM hallucination
        user_lines = [line for line in conversation_text.split('\n') if line.strip().startswith('User:')]
        user_content = ' '.join([line.replace('User:', '').strip() for line in user_lines])
        
        # If no meaningful user content, skip LLM and return defaults
        if len(user_content.strip()) < 1:  # Less than 1 character of user content
            logger.info(f"Insufficient user content ({len(user_content)} chars) - skipping LLM, using defaults")
            result = {
                "booking_type": "auto_followup",
                "time_phrase": None,
                "user_confirmed": False,
                "student_grade": None
            }
        else:
            # Proceed with LLM extraction only if there's meaningful user content
            # Define prompt for logging purposes
            prompt = f"""You are analyzing a phone conversation between an Agent and a User to extract followup scheduling information.

CONVERSATION:
{conversation_text}

INSTRUCTIONS - Read the ENTIRE conversation carefully before answering:

1. **booking_type** - Determine the type based on the FINAL outcome of the conversation:
   - "auto_consultation": User EXPLICITLY confirms a booking/appointment/session/meeting OR agrees to a specific time
     * Clear confirmations: "Yes I'll attend", "Book it", "Confirmed", "I'll be there", "Sure, book the session"
     * Scheduling intent: "Can we schedule", "Let's book", "Schedule for", "Book a consultation"
     * Time agreement: User agrees to ANY specific time/place (even if vague like "tomorrow", "next week")
     * Keywords indicating consultation: "schedule", "book", "appointment", "session", "meeting", "consultation"
     * Examples: "Tomorrow at 3 PM works", "Next Tuesday 10 AM", "Schedule for Friday", "Can we schedule tomorrow?"
   - "auto_followup": Everything else - callback requests, declined, no confirmation, vague
     * Callback: "Call me back", "Call me tomorrow", "Call me later", "I'll call you"
     * Declined: "No thank you", "Not interested", "Maybe later", "I'll think about it", "End the call"
     * Vague: "That one", "Hmm okay", no clear yes/no, "I'm not sure"
     * Agent notes: "without booking", "will follow up"
   - DEFAULT: auto_followup (when unclear)

2. **time_phrase** - Extract the FINAL CONFIRMED time for next contact:
   - CRITICAL: If multiple times are discussed, extract ONLY the LAST one the user AGREED to
   - CRITICAL: If user declines AFTER agreeing to a time, return null
   - CRITICAL: Read the END of conversation - the final decision overrides earlier ones
   - Extract the phrase AS-IS from conversation (e.g., "after 15 mins", "tomorrow 3 PM", "Sunday 11 AM", "evening", "noon", "day after tomorrow")
   - Include day/date when mentioned ("Friday noon" not just "noon")
   - Return null if:
     * User declines at the end ("No", "Not interested", "End the call", "I can't make it", "Let's not schedule")
     * User says "That one" without being specific about which option
     * User is vague and never confirms ("maybe", "I'll think about it", "I'm not sure", "Hmm okay")
     * User says "Maybe later" without specific time
     * Only PAST events are discussed with no future followup
     * User responds with single words like "Okay", "Sure" without time context
   - Edge case examples:
     * Agent: "How about 3 PM?" User: "No, make it 5 PM" User: "Yes" -> "5 PM"
     * Agent: "Sunday 11 AM?" User: "That one" Later User: "End the call" -> null
     * Agent: "Tomorrow?" User: "Yes, around evening" -> "tomorrow evening"
     * Agent: "Call you at 4?" User: "Can you make it 4:30?" Agent: "Sure" User: "Okay" -> "4:30"
     * User: "Call me after 5 minutes" Agent: "I'll call you at 5:35" User: "Okay" -> "5:35"
     * User: "Call me after 2 days" Agent: "I'll call you on Friday at 3 PM" User: "Okay" -> "Friday 3 PM"
     * User: "Call me next week" Agent: "I'll call you next Tuesday at 10 AM" User: "Sure" -> "next Tuesday 10 AM"
     * User: "Call me after 2 weeks" Agent: "I'll call you on March 15th" User: "Okay" -> "March 15th"
     * User: "Call me after a month" Agent: "I'll call you on April 5th at 11 AM" User: "Confirmed" -> "April 5th 11 AM"
     * User: "Call me after lunch" -> "after lunch"
     * User: "Call me back in half an hour" -> "in half an hour"
     * User: "Next week sometime" -> "next week"

3. **user_confirmed** - Did the user EXPLICITLY confirm the time at the END?
   - true ONLY if: User clearly said yes/okay/sure/confirmed to a specific time AND did NOT decline later
   - false if: No confirmation, declined, vague, changed mind, or said no at the end
   - CRITICAL: If user confirmed earlier but declined/ended later, this is FALSE

4. **student_grade**: DEFAULT is null. ONLY return a number (9, 10, 11, or 12) if USER confirms EXACT match:
   
   CRITICAL: The USER must explicitly state their grade. Do NOT extract from agent statements.
   
   REQUIRED USER PHRASES:
   - "I'm in grade 9", "I'm in grade 10", "I'm in grade 11", "I'm in grade 12"
   - "I am in class 9", "I am in class 10", "I am in class 11", "I am in class 12"
   - "I study in 9th grade", "I study in 10th grade", "I study in 11th grade", "I study in 12th grade"
   - "I'm in 9th standard", "I'm in 10th standard", "I'm in 11th standard", "I'm in 12th standard"
   - "I'm in ninth class", "I'm in tenth class", "I'm in eleventh class", "I'm in twelfth class"
   - "My grade is 9", "My grade is 10", "My grade is 11", "My grade is 12"
   
   IF USER EXACTLY CONFIRMS GRADE → return corresponding number
   IF USER DOES NOT CONFIRM GRADE → return null (this is the default)
   
   IMPORTANT: Do NOT extract grade from agent statements like "I see you're in 10th grade". 
   Only extract if the USER themselves confirm their grade.

Return JSON only."""

        try:
            result = await generate_with_schema_async(
                prompt=prompt,
                schema=schema,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS
            )
            
            if not result:
                result = {
                    "booking_type": "auto_followup",
                    "time_phrase": None,
                    "user_confirmed": False,
                    "student_grade": None
                }
            
            result.pop('_usage_metadata', None)
            
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            result = {
                "booking_type": "auto_followup",
                "time_phrase": None,
                "user_confirmed": False,
                "student_grade": None
            }
        
        booking_type = result.get("booking_type", "auto_followup")
        time_phrase = result.get("time_phrase")
        user_confirmed = result.get("user_confirmed", False)
        student_grade = result.get("student_grade")
        
        # Post-process to handle string vs None issues
        if time_phrase == "null" or time_phrase == "":
            time_phrase = None
        
        # Validate student_grade range (9-12)
        if student_grade is not None:
            if not isinstance(student_grade, int) or student_grade < 9 or student_grade > 12:
                logger.warning(f"Invalid student_grade {student_grade}, setting to None")
                student_grade = None
        
        logger.info(f"Extraction: type={booking_type}, phrase='{time_phrase}', confirmed={user_confirmed}")
        
        # Calculate scheduled_at
        scheduled_at = None
        calculation_method = "none"
        
        if not time_phrase:
            # No time phrase -> use grade timeline if available, else default timeline
            if tenant_id == GLINKS_TENANT_ID:
                if student_grade is not None:
                    # GLINKS: Use grade-based timeline
                    from schedule_calculator import GRADE_TIMELINE
                    default_days = GRADE_TIMELINE.get(student_grade, int(os.getenv('GLINKS_DEFAULT_TIMELINE_DAYS', '1')))
                    calculation_method = f"no_confirmation_grade{student_grade}_{default_days}day"
                    logger.info(f"No time phrase: grade {student_grade} -> {default_days} days")
                else:
                    # GLINKS: Use environment variable (no grade mentioned)
                    default_days = int(os.getenv('GLINKS_DEFAULT_TIMELINE_DAYS', '1'))
                    calculation_method = f"no_confirmation_{default_days}day"
                    logger.info(f"No time phrase: no grade -> {default_days} days default")
            else:
                # Non-GLINKS: Use 1 day default (tomorrow)
                default_days = 1
                calculation_method = "no_confirmation_1day"
            
            scheduled_at = first_timestamp + timedelta(days=default_days)
            logger.info(f"No time phrase: {scheduled_at} (using {default_days} days for {'GLINKS grade ' + str(student_grade) if tenant_id == GLINKS_TENANT_ID and student_grade else 'GLINKS default' if tenant_id == GLINKS_TENANT_ID else 'Non-GLINKS'} tenant)")
        else:
            # Parse time phrase (even if not confirmed)
            scheduled_at, calculation_method = await self._parse_time_phrase(
                time_phrase, first_timestamp, last_timestamp
            )
        
        return {
            "booking_type": booking_type,
            "scheduled_at": scheduled_at,
            "time_phrase": time_phrase,
            "user_confirmed": user_confirmed,
            "calculation_method": calculation_method,
            "student_grade": student_grade
        }
    
    def _resolve_time_of_day(self, keyword: str) -> tuple:
        """Resolve time-of-day keywords to (hour, minute)"""
        time_of_day = {
            'morning': (9, 0),
            'noon': (12, 0),
            'afternoon': (14, 0),
            'evening': (18, 0),
            'night': (20, 0),
            'midnight': (0, 0),
            'after lunch': (14, 0),  # Put after lunch last to avoid conflicts
        }
        for keyword_name, (h, m) in time_of_day.items():
            # Use word boundaries to avoid partial matches (e.g., "noon" in "afternoon")
            import re
            if re.search(rf'\b{re.escape(keyword_name)}\b', keyword):
                # Check if there's a specific time after the keyword (e.g., "morning at 10", "afternoon at 3", "evening 5", "midnight at 11")
                import re
                remaining = keyword[keyword.index(keyword_name) + len(keyword_name):].strip()
                
                # Remove "at" if present and then match time
                remaining_clean = remaining.replace('at', '').strip()
                
                # Convert written numbers to digits for any time
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
                    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
                    'nineteen': 19, 'twenty': 20, 'twenty one': 21, 'twenty-two': 22,
                    'twenty three': 23, 'twenty-four': 24
                }
                for word, num in word_to_num.items():
                    remaining_clean = re.sub(rf'\b{word}\b', str(num), remaining_clean)
                
                # Match any time pattern (1-24 hours, optional minutes, optional AM/PM)
                time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?', remaining_clean)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    am_pm = time_match.group(3)
                    
                    # Handle AM/PM
                    if am_pm and am_pm.replace('.', '') == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm and am_pm.replace('.', '') == 'am' and hour == 12:
                        hour = 0
                    # For time-of-day keywords, assume PM for evening/night/afternoon if no AM/PM specified
                    elif not am_pm and keyword_name in ('evening', 'night', 'afternoon') and hour < 12:
                        hour += 12
                    # For morning, assume AM if no AM/PM specified and hour <= 12
                    elif not am_pm and keyword_name == 'morning' and hour <= 12:
                        hour = hour  # Keep as AM
                    # For noon/midnight, handle special cases
                    elif keyword_name == 'noon' and hour > 12:
                        hour = 12  # Force to noon
                    elif keyword_name == 'midnight' and hour != 0:
                        hour = 0  # Force to midnight
                    
                    return hour, minute
                # Only return default if no specific time found
                return h, m
        return None, None
    
    def _add_months(self, dt: datetime, months: int) -> datetime:
        """Add N months to a datetime, handling month overflow"""
        month = dt.month + months
        year = dt.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        import calendar
        max_day = calendar.monthrange(year, month)[1]
        day = min(dt.day, max_day)
        return dt.replace(year=year, month=month, day=day)
    
    def _parse_hour_minute(self, time_phrase_lower: str, call_timestamp: datetime = None) -> tuple:
        """Extract hour and minute from a time phrase. Returns (hour, minute) or (None, None)"""
        import re
        
        # Check time-of-day keywords first
        h, m = self._resolve_time_of_day(time_phrase_lower)
        if h is not None:
            return h, m
        
        # Check "half an hour" / "half hour"
        if 'half an hour' in time_phrase_lower or 'half hour' in time_phrase_lower:
            return None, None  # Handled as relative time
        
        # Convert written numbers to digits for time parsing
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }
        
        # Convert written numbers in the phrase
        normalized = time_phrase_lower
        for word, num in word_to_num.items():
            normalized = re.sub(rf'\b{word}\b', str(num), normalized)
        
        # Match numeric time (e.g., "3 PM", "15:00", "4:30 pm", "11 am", "5 30 pm")
        # Handle commas and extra text in the phrase
        # Look for time patterns with minutes or AM/PM to be more specific
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)(?:\s*gst)?', normalized)
        if not time_match:
            # Fallback: try to find any hour:minute pattern
            time_match = re.search(r'(\d{1,2}):(\d{2})', normalized)
        
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            am_pm = time_match.group(3) if len(time_match.groups()) >= 3 else None
            
            # Check if PM is mentioned anywhere in the phrase (not just in the regex match)
            if not am_pm and 'pm' in normalized.lower():
                am_pm = 'pm'
            elif not am_pm and 'am' in normalized.lower():
                am_pm = 'am'
            
            logger.debug(f"Time match: hour={hour}, minute={minute}, am_pm={am_pm}")
            
            if am_pm and am_pm.replace('.', '') == 'pm' and hour != 12:
                hour += 12
                logger.debug(f"Converted to PM: hour={hour}")
            elif am_pm and am_pm.replace('.', '') == 'am' and hour == 12:
                hour = 0
                logger.debug(f"Converted 12 AM to 0: hour={hour}")
            elif am_pm is None:
                # Universal AM/PM inference for all tenants and all times
                # If no AM/PM specified, use call timestamp to decide
                if call_timestamp:
                    current_hour = call_timestamp.hour
                else:
                    current_hour = datetime.now().hour
                    
                if 1 <= hour <= 11:
                    # Simple universal rule: If call was in afternoon/evening, assume PM
                    # If call was in morning, assume AM (unless it's a common callback time)
                    if current_hour >= 12:
                        # If call was in afternoon/evening, likely means PM
                        hour += 12
                        logger.debug(f"No AM/PM, assumed PM: hour={hour}")
                    elif hour >= 4 and hour <= 8:
                        # Common callback times (4-8) usually mean PM
                        hour += 12
                        logger.debug(f"No AM/PM, callback time assumed PM: hour={hour}")
                    # Otherwise keep as AM (early morning times like 7 AM, 9 AM, 10 AM, 11 AM)
                    logger.debug(f"No AM/PM, kept as AM: hour={hour}")
                # For hour 12 without AM/PM, keep as 12 (noon)
                # For hour > 12, already in 24-hour format
            
            logger.debug(f"Final parsed time: hour={hour}, minute={minute}, am_pm={am_pm}")
            return hour, minute
        
        return None, None
    
    async def _parse_time_phrase(
        self,
        time_phrase: str,
        first_timestamp: datetime,
        last_timestamp: datetime
    ) -> tuple:
        """Parse time phrase and calculate scheduled_at"""
        import re
        
        if not time_phrase:
            scheduled_at = first_timestamp + timedelta(days=1)
            return scheduled_at, "no_time_phrase_1day"
        
        time_phrase_lower = time_phrase.lower().strip()
        
        # Rule 1: Relative time - add to LAST timestamp
        # Matches: "after 5 minutes", "in 30 mins", "within 2 hours", "in half an hour", "after an hour"
        
        # "half an hour" / "half hour"
        if re.search(r'(?:half\s+an?\s+hour|half\s+hour)', time_phrase_lower):
            scheduled_at = last_timestamp + timedelta(minutes=30)
            logger.info(f"Relative time: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "relative_+30m_from_last"
        
        # "an hour" / "1 hour" / "one hour"
        if re.search(r'(?:an\s+hour|one\s+hour)', time_phrase_lower):
            scheduled_at = last_timestamp + timedelta(hours=1)
            logger.info(f"Relative time: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "relative_+1h_from_last"
        
        # "after/in/within X minutes/hours" - handle both numeric and written numbers
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
            'a': 1, 'an': 1
        }
        
        # Convert written numbers to digits for matching
        normalized = time_phrase_lower
        for word, num in word_to_num.items():
            normalized = re.sub(rf'\b{word}\b', str(num), normalized)
        
        relative_match = re.search(r'(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?|hours?|hrs?)', normalized)
        if relative_match:
            amount = int(relative_match.group(1))
            if re.search(r'hours?|hrs?', normalized):
                scheduled_at = last_timestamp + timedelta(hours=amount)
                calculation_method = f"relative_+{amount}h_from_last"
            else:
                scheduled_at = last_timestamp + timedelta(minutes=amount)
                calculation_method = f"relative_+{amount}m_from_last"
            logger.info(f"Relative time: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, calculation_method
        
        # Also handle standalone time phrases (without after/in/within)
        standalone_match = re.search(r'(\d+)\s*(?:mins?|minutes?|hours?|hrs?)', normalized)
        if standalone_match:
            amount = int(standalone_match.group(1))
            if re.search(r'hours?|hrs?', normalized):
                scheduled_at = last_timestamp + timedelta(hours=amount)
                calculation_method = f"standalone_+{amount}h_from_last"
            else:
                scheduled_at = last_timestamp + timedelta(minutes=amount)
                calculation_method = f"standalone_+{amount}m_from_last"
            logger.info(f"Standalone relative time: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, calculation_method
        
        # Rule 2: "day after tomorrow"
        if 'day after tomorrow' in time_phrase_lower:
            target = last_timestamp + timedelta(days=2)
            hour, minute = self._parse_hour_minute(time_phrase_lower.replace('day after tomorrow', ''))
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Day after tomorrow: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "day_after_tomorrow"
        
        # Rule 3: "tomorrow"
        if 'tomorrow' in time_phrase_lower:
            tomorrow = last_timestamp + timedelta(days=1)
            hour, minute = self._parse_hour_minute(time_phrase_lower.replace('tomorrow', ''))
            if hour is not None:
                scheduled_at = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = tomorrow.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Tomorrow: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "tomorrow_parsed"
        
        # Rule 4: "today" + time
        if 'today' in time_phrase_lower:
            time_remaining = time_phrase_lower.replace('today', '')
            logger.info(f"DEBUG: Parsing time from: '{time_remaining}'")
            hour, minute = self._parse_hour_minute(time_remaining, last_timestamp)
            logger.info(f"DEBUG: Parsed hour={hour}, minute={minute}")
            if hour is not None:
                if hour > 23:
                    logger.error(f"ERROR: Invalid hour {hour} - must be 0-23")
                    raise ValueError(f"Invalid hour {hour} - must be 0-23")
                scheduled_at = last_timestamp.replace(hour=hour, minute=minute, second=0, microsecond=0)
                # If user explicitly said "today", keep it today even if time has passed
                # This respects the user's explicit preference
            else:
                scheduled_at = last_timestamp + timedelta(days=1)
            logger.info(f"Today: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "today_parsed"
        
        # Rule 5a: Relative days/weeks/months/years
        # "after 3 days", "in 2 weeks", "after two months", "in 3 months"
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'a': 1, 'an': 1
        }
        
        # Convert written numbers to digits for matching
        normalized = time_phrase_lower
        for word, num in word_to_num.items():
            normalized = re.sub(rf'\b{word}\b', str(num), normalized)
        
        # "after/in X days"
        rel_days = re.search(r'(?:after|in|within)\s+(\d+)\s*days?', normalized)
        if rel_days:
            days = int(rel_days.group(1))
            target = last_timestamp + timedelta(days=days)
            # Check for time in remaining phrase
            remaining = re.sub(r'(?:after|in|within)\s+\d+\s*days?', '', normalized).strip()
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Relative days: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, f"relative_+{days}d"
        
        # "after/in X weeks"
        rel_weeks = re.search(r'(?:after|in|within)\s+(\d+)\s*weeks?', normalized)
        if rel_weeks:
            weeks = int(rel_weeks.group(1))
            target = last_timestamp + timedelta(weeks=weeks)
            remaining = re.sub(r'(?:after|in|within)\s+\d+\s*weeks?', '', normalized).strip()
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Relative weeks: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, f"relative_+{weeks}w"
        
        # "after/in X months" with optional day ("after two months at 17", "in 2 months on 15th")
        rel_months = re.search(r'(?:after|in|within)\s+(\d+)\s*months?', normalized)
        if rel_months:
            months = int(rel_months.group(1))
            target = self._add_months(last_timestamp, months)
            # Check for specific day ("at 17", "on 15th", "17th")
            remaining = re.sub(r'(?:after|in|within)\s+\d+\s*months?', '', normalized).strip()
            day_match = re.search(r'(?:at|on)?\s*(\d{1,2})(?:st|nd|rd|th)?', remaining)
            if day_match:
                day = int(day_match.group(1))
                if 1 <= day <= 31:
                    import calendar
                    max_day = calendar.monthrange(target.year, target.month)[1]
                    day = min(day, max_day)
                    target = target.replace(day=day)
            # Check for time
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Relative months: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, f"relative_+{months}mo"
        
        # "after/in X years"
        rel_years = re.search(r'(?:after|in|within)\s+(\d+)\s*years?', normalized)
        if rel_years:
            years = int(rel_years.group(1))
            target = self._add_months(last_timestamp, years * 12)
            remaining = re.sub(r'(?:after|in|within)\s+\d+\s*years?', '', normalized).strip()
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Relative years: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, f"relative_+{years}y"
        
        # Rule 5b: "next month" with optional day ("next month 17", "next month 15th")
        if 'next month' in time_phrase_lower:
            target = self._add_months(last_timestamp, 1)
            remaining = time_phrase_lower.replace('next month', '').strip()
            day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', remaining)
            if day_match:
                day = int(day_match.group(1))
                if 1 <= day <= 31:
                    import calendar
                    max_day = calendar.monthrange(target.year, target.month)[1]
                    day = min(day, max_day)
                    target = target.replace(day=day)
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Next month: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "next_month"
        
        # Rule 5c: "next year"
        if 'next year' in time_phrase_lower:
            target = self._add_months(last_timestamp, 12)
            remaining = time_phrase_lower.replace('next year', '').strip()
            hour, minute = self._parse_hour_minute(remaining)
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Next year: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "next_year"
        
        # Rule 5d: Specific month name + day ("March 15", "January 20th", "15th March")
        months_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        for month_name, month_num in months_map.items():
            if month_name in time_phrase_lower:
                # Extract day number
                remaining = time_phrase_lower.replace(month_name, '').strip()
                day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?', remaining)
                day = int(day_match.group(1)) if day_match else 1
                if day < 1 or day > 31:
                    day = 1
                
                # Remove day number and weekday from remaining text for time parsing
                remaining = re.sub(r'\d{1,2}(?:st|nd|rd|th)\b', '', remaining).strip()
                # Remove weekday names
                weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                for weekday in weekdays:
                    remaining = remaining.replace(weekday, '').strip()
                # Remove extra words like "at"
                remaining = remaining.replace('at', '').strip()
                
                # Determine year
                target_year = last_timestamp.year
                target_date = last_timestamp.replace(month=month_num, day=min(day, 28))
                # Correct day for month
                import calendar
                max_day = calendar.monthrange(target_year, month_num)[1]
                day = min(day, max_day)
                target_date = last_timestamp.replace(month=month_num, day=day)
                # If date is in the past, use next year
                if target_date.date() <= last_timestamp.date():
                    target_year += 1
                    max_day = calendar.monthrange(target_year, month_num)[1]
                    day = min(day, max_day)
                    target_date = target_date.replace(year=target_year, day=day)
                
                # Extract time
                hour, minute = self._parse_hour_minute(remaining)
                if hour is not None:
                    scheduled_at = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    scheduled_at = target_date.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
                logger.info(f"Month name: '{time_phrase}' -> {scheduled_at}")
                return scheduled_at, f"month_{month_name}"
        
        # Rule 5e: Specific day of current/next month ("17th", "on the 15th", "at 17")
        # Only match if it looks like a date (with ordinal suffix or preceded by "on/at the")
        date_match = re.search(r'(?:on\s+(?:the\s+)?)?(\d{1,2})(?:st|nd|rd|th)', time_phrase_lower)
        if date_match:
            day = int(date_match.group(1))
            if 1 <= day <= 31:
                target = last_timestamp
                if day <= last_timestamp.day:
                    # Day already passed this month, use next month
                    target = self._add_months(last_timestamp, 1)
                import calendar
                max_day = calendar.monthrange(target.year, target.month)[1]
                day = min(day, max_day)
                target = target.replace(day=day)
                
                remaining = re.sub(r'(?:on\s+(?:the\s+)?)?\d{1,2}(?:st|nd|rd|th)', '', time_phrase_lower).strip()
                hour, minute = self._parse_hour_minute(remaining)
                if hour is not None:
                    scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
                logger.info(f"Day of month: '{time_phrase}' -> {scheduled_at}")
                return scheduled_at, "day_of_month"
        
        # Rule 5f: "next week"
        if 'next week' in time_phrase_lower:
            target = last_timestamp + timedelta(days=7)
            hour, minute = self._parse_hour_minute(time_phrase_lower.replace('next week', ''))
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Next week: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "next_week"
        
        # Rule 6: "this weekend" / "weekend"
        if 'weekend' in time_phrase_lower:
            current_weekday = last_timestamp.weekday()
            days_to_saturday = (5 - current_weekday) % 7
            if days_to_saturday == 0:
                days_to_saturday = 7
            target = last_timestamp + timedelta(days=days_to_saturday)
            hour, minute = self._parse_hour_minute(time_phrase_lower.replace('this weekend', '').replace('weekend', ''))
            if hour is not None:
                scheduled_at = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                scheduled_at = target.replace(hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0)
            logger.info(f"Weekend: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "weekend"
        
        # Rule 7: Weekday name (Monday, Tuesday, etc.)
        weekdays = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                   'friday': 4, 'saturday': 5, 'sunday': 6}
        
        for day_name, day_num in weekdays.items():
            if day_name in time_phrase_lower:
                current_weekday = last_timestamp.weekday()
                days_ahead = (day_num - current_weekday) % 7
                if days_ahead == 0:
                    days_ahead = 7
                
                target_date = last_timestamp + timedelta(days=days_ahead)
                hour, minute = self._parse_hour_minute(time_phrase_lower.replace(day_name, ''))
                if hour is not None:
                    scheduled_at = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    scheduled_at = target_date.replace(
                        hour=first_timestamp.hour, minute=first_timestamp.minute, second=0, microsecond=0
                    )
                logger.info(f"Weekday: '{time_phrase}' -> {scheduled_at}")
                return scheduled_at, f"weekday_{day_name}"
        
        # Rule 8: Time-of-day keyword only ("evening", "morning", "noon", "after lunch")
        h, m = self._resolve_time_of_day(time_phrase_lower)
        if h is not None:
            scheduled_at = last_timestamp.replace(hour=h, minute=m, second=0, microsecond=0)
            if scheduled_at <= last_timestamp:
                scheduled_at = scheduled_at + timedelta(days=1)
            logger.info(f"Time of day: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "time_of_day_parsed"
        
        # Rule 9: Just numeric time (e.g., "3 PM", "4:30", "15:00")
        hour, minute = self._parse_hour_minute(time_phrase_lower, last_timestamp)
        if hour is not None:
            scheduled_at = last_timestamp.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if scheduled_at <= last_timestamp:
                scheduled_at = scheduled_at + timedelta(days=1)
            logger.info(f"Time only: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "time_only_parsed"
        
        # Rule 10: Vague phrases - use first timestamp + 1 day
        vague_phrases = ['later', 'sometime', 'some time', 'not sure', 'i\'ll call', 'will call']
        if any(vp in time_phrase_lower for vp in vague_phrases):
            scheduled_at = first_timestamp + timedelta(days=1)
            logger.info(f"Vague phrase: '{time_phrase}' -> {scheduled_at}")
            return scheduled_at, "vague_phrase_1day"
        
        # Fallback: use first timestamp + 1 day
        scheduled_at = first_timestamp + timedelta(days=1)
        logger.info(f"Fallback: '{time_phrase}' -> {scheduled_at}")
        return scheduled_at, "fallback_1day"
    
    async def process_call_log(self, call_id: str) -> Optional[Dict]:
        """Process a single call log and create booking"""
        
        logger.info(f"Processing call_id: {call_id}")
        
        # Get call data from database
        call_data = await self.storage.get_call_log(call_id)
        if not call_data:
            logger.error(f"Call log not found: {call_id}")
            return None
        
        tenant_id = call_data['tenant_id']
        lead_id = call_data['lead_id']
        transcripts = call_data['transcripts']
        initiated_by_user_id = call_data['initiated_by_user_id']
        agent_id = call_data['agent_id']
        started_at = call_data['started_at']
        
        
        # Parse transcription
        conversation_text = self.parse_transcription(transcripts)
        if not conversation_text:
            logger.warning(f"Empty conversation for call {call_id}")
            logger.info(f"Started at: {started_at}")
            
            # Convert started_at to GST like other calls
            if started_at:
                if started_at.tzinfo is None:
                    started_at_gst = GST.localize(started_at)
                elif started_at.tzinfo != GST:
                    started_at_gst = started_at.astimezone(GST)
                else:
                    started_at_gst = started_at
                
                # For GLINKS tenant, use simple calendar days (no working day logic)
                if tenant_id == GLINKS_TENANT_ID:
                    logger.info("Empty conversation for GLINKS - using simple calendar days")
                    try:
                        # Use GLINKS default timeline as simple calendar days
                        default_days = int(os.getenv('GLINKS_DEFAULT_TIMELINE_DAYS', '1'))
                        scheduled_at = started_at_gst + timedelta(days=default_days)
                        calculation_method = f"empty_transcript_{default_days}day"
                        logger.info(f"GLINKS empty conversation: {scheduled_at} (using {default_days} calendar days)")
                    except Exception as e:
                        logger.error(f"GLINKS empty conversation error: {e}")
                        scheduled_at = started_at_gst + timedelta(days=1)
                        calculation_method = "empty_transcript_1day"
                else:
                    # Non-GLINKS tenant: use default next day
                    scheduled_at = started_at_gst + timedelta(days=1)
                    calculation_method = "empty_transcript_1day"
                    logger.info(f"Next day from started_at (GST): {scheduled_at}")
            else:
                scheduled_at = None
                calculation_method = "empty_transcript_no_time"
                logger.warning("No started_at time available")
            
            booking_type = "auto_followup"
            student_grade = None
            # Skip Gemini extraction for empty conversation to prevent hallucination
        else:
            # Extract timestamps
            first_timestamp = self.extract_first_timestamp(transcripts)
            last_timestamp = self.extract_last_timestamp(transcripts)
            
            if not first_timestamp:
                first_timestamp = started_at if started_at else GST.localize(datetime.now())
            if not last_timestamp:
                last_timestamp = first_timestamp
            
            # Extract followup time
            result = await self.extract_followup_time(
                conversation_text=conversation_text,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                tenant_id=tenant_id
            )
            
            booking_type = result['booking_type']
            scheduled_at = result['scheduled_at']
            calculation_method = result['calculation_method']
            student_grade = result['student_grade']
            
            # Debug Gemini extraction
            logger.info(f"Gemini raw result: {result}")
            logger.info(f"Extracted student_grade: {student_grade} (type: {type(student_grade)})")
            
            # Validate student_grade range (9-12)
            if student_grade is not None:
                if not isinstance(student_grade, int) or student_grade < 9 or student_grade > 12:
                    logger.warning(f"Invalid student_grade {student_grade}, setting to None")
                    student_grade = None
        
        # Apply GLINKS schedule calculator if tenant matches and booking_type is auto_consultation
        logger.info(f"GLINKS check: tenant_id={tenant_id}, GLINKS_TENANT_ID={GLINKS_TENANT_ID}")
        logger.info(f"GLINKS check: booking_type={booking_type}, schedule_calculator={self.schedule_calculator is not None}, scheduled_at={scheduled_at is not None}, student_grade={student_grade}")
        
        if (self.schedule_calculator and 
            tenant_id == GLINKS_TENANT_ID and 
            scheduled_at and 
            booking_type == 'auto_consultation'):
            
            logger.info("GLINKS conditions met - applying schedule calculator")
            
            try:
                # Check if any date or day was confirmed (not just default timeline)
                confirmed_date_mentioned = self._has_confirmed_date_or_day(calculation_method)
                
                # Get valid schedule from calculator
                valid_schedule = self.schedule_calculator.calculate_next_call(
                    current_time=scheduled_at,
                    student_grade=student_grade,
                    booking_type=booking_type,
                    confirmed_date_mentioned=confirmed_date_mentioned
                )
                
                if valid_schedule:
                    scheduled_at = valid_schedule
                    calculation_method = f"{calculation_method}_glinks_adjusted"
                    logger.info(f"GLINKS schedule adjusted: {scheduled_at}")
                else:
                    logger.warning(f"GLINKS calculator returned no valid schedule for {scheduled_at}, grade {student_grade}")
                    
            except Exception as e:
                logger.error(f"GLINKS schedule calculator error: {e}")
                # Keep original scheduled_at if calculator fails
        
        # Get voice_id from agent_id
        voice_id = await self.storage.get_voice_id_from_agent_id(agent_id)
        
        # Calculate retry_count and parent_booking_id
        retry_count = 0
        parent_booking_id = None
        
        if lead_id and booking_type:
            existing_count = await self.storage.count_bookings_by_lead_id_and_booking_type(lead_id, booking_type)
            if existing_count > 0:
                max_retry = await self.storage.get_max_retry_count_by_lead_id_and_booking_type(lead_id, booking_type)
                retry_count = max_retry + 1
                
                original_booking = await self.storage.get_original_booking_by_lead_id_and_booking_type(lead_id, booking_type)
                if original_booking:
                    parent_booking_id = original_booking.get('call_id')
        
        # Create booking data
        booking_id = str(uuid.uuid4())
        
        # Calculate buffer_until as 15 minutes after scheduled_at
        buffer_until = None
        if scheduled_at:
            buffer_until = scheduled_at + timedelta(minutes=15)
        
        # Get current timestamp for created_at and updated_at
        now = datetime.now(GST).replace(microsecond=0)
        
        booking_data = {
            "id": booking_id,
            "tenant_id": tenant_id,
            "lead_id": lead_id,
            "assigned_user_id": initiated_by_user_id,
            "booking_type": booking_type,
            "booking_source": "system",
            "scheduled_at": scheduled_at.strftime("%Y-%m-%d %H:%M:%S") if scheduled_at else None,
            "timezone": "GST",
            "status": "scheduled",
            "call_result": None,
            "retry_count": retry_count,
            "parent_booking_id": parent_booking_id,
            "notes": None,
            "metadata": {
                "call_id": call_id
            },
            "created_by": initiated_by_user_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "is_deleted": False,
            "buffer_until": buffer_until.strftime("%Y-%m-%d %H:%M:%S") if buffer_until else None
        }
        
        logger.info(f"   Booking prepared: {booking_id}")
        logger.info(f"   Type: {booking_type}")
        logger.info(f"   Scheduled: {scheduled_at}")
        logger.info(f"   Buffer Until: {buffer_until}")
        logger.info(f"   Method: {calculation_method}")
        logger.info(f"   Retry: {retry_count}")
        return booking_data
    
    def _has_confirmed_date_or_day(self, calculation_method: str) -> bool:
        """Check if any date or day was confirmed (not just default timeline)"""
        # Methods that indicate user confirmed a specific date/day/time
        # These should NOT be overridden by grade timelines
        confirmed_date_methods = {
            # Specific times
            'relative_+1h_from_last', 'relative_+2h_from_last', 'relative_+3h_from_last',
            'relative_+4h_from_last', 'relative_+5h_from_last', 'relative_+6h_from_last',
            'relative_+7h_from_last', 'relative_+8h_from_last', 'relative_+9h_from_last',
            'relative_+10h_from_last', 'relative_+11h_from_last', 'relative_+12h_from_last',
            'relative_+13h_from_last', 'relative_+14h_from_last', 'relative_+15h_from_last',
            'relative_+16h_from_last', 'relative_+17h_from_last', 'relative_+18h_from_last',
            'relative_+19h_from_last', 'relative_+20h_from_last', 'relative_+21h_from_last',
            'relative_+22h_from_last', 'relative_+23h_from_last', 'relative_+24h_from_last',
            # All minutes from 1-60
            'relative_+1m_from_last', 'relative_+2m_from_last', 'relative_+3m_from_last',
            'relative_+4m_from_last', 'relative_+5m_from_last', 'relative_+6m_from_last',
            'relative_+7m_from_last', 'relative_+8m_from_last', 'relative_+9m_from_last',
            'relative_+10m_from_last', 'relative_+11m_from_last', 'relative_+12m_from_last',
            'relative_+13m_from_last', 'relative_+14m_from_last', 'relative_+15m_from_last',
            'relative_+16m_from_last', 'relative_+17m_from_last', 'relative_+18m_from_last',
            'relative_+19m_from_last', 'relative_+20m_from_last', 'relative_+21m_from_last',
            'relative_+22m_from_last', 'relative_+23m_from_last', 'relative_+24m_from_last',
            'relative_+25m_from_last', 'relative_+26m_from_last', 'relative_+27m_from_last',
            'relative_+28m_from_last', 'relative_+29m_from_last', 'relative_+30m_from_last',
            'relative_+31m_from_last', 'relative_+32m_from_last', 'relative_+33m_from_last',
            'relative_+34m_from_last', 'relative_+35m_from_last', 'relative_+36m_from_last',
            'relative_+37m_from_last', 'relative_+38m_from_last', 'relative_+39m_from_last',
            'relative_+40m_from_last', 'relative_+41m_from_last', 'relative_+42m_from_last',
            'relative_+43m_from_last', 'relative_+44m_from_last', 'relative_+45m_from_last',
            'relative_+46m_from_last', 'relative_+47m_from_last', 'relative_+48m_from_last',
            'relative_+49m_from_last', 'relative_+50m_from_last', 'relative_+51m_from_last',
            'relative_+52m_from_last', 'relative_+53m_from_last', 'relative_+54m_from_last',
            'relative_+55m_from_last', 'relative_+56m_from_last', 'relative_+57m_from_last',
            'relative_+58m_from_last', 'relative_+59m_from_last', 'relative_+60m_from_last',
            
            # Specific days/dates
            'tomorrow_parsed', 'today_parsed', 'day_after_tomorrow',
            'weekday_monday', 'weekday_tuesday', 'weekday_wednesday', 
            'weekday_thursday', 'weekday_friday', 'weekday_saturday', 'weekday_sunday',
            
            # Time-only parsing (when user confirms specific time)
            'time_only_parsed', 'time_of_day_parsed',
            
            # Next/This/Last weekday variations
            'next_monday', 'next_tuesday', 'next_wednesday', 'next_thursday', 'next_friday', 'next_saturday', 'next_sunday',
            'this_monday', 'this_tuesday', 'this_wednesday', 'this_thursday', 'this_friday', 'this_saturday', 'this_sunday',
            'last_monday', 'last_tuesday', 'last_wednesday', 'last_thursday', 'last_friday', 'last_saturday', 'last_sunday',
            
            # Month names
            'month_january', 'month_february', 'month_march', 'month_april',
            'month_may', 'month_june', 'month_july', 'month_august',
            'month_september', 'month_october', 'month_november', 'month_december',
            
            # Relative dates (days, weeks, months)
            'relative_+1d', 'relative_+2d', 'relative_+3d', 'relative_+4d', 'relative_+5d',
            'relative_+6d', 'relative_+7d', 'relative_+8d', 'relative_+9d', 'relative_+10d',
            'relative_+11d', 'relative_+12d', 'relative_+13d', 'relative_+14d', 'relative_+15d',
            'relative_+16d', 'relative_+17d', 'relative_+18d', 'relative_+19d', 'relative_+20d',
            'relative_+21d', 'relative_+22d', 'relative_+23d', 'relative_+24d', 'relative_+25d',
            'relative_+26d', 'relative_+27d', 'relative_+28d', 'relative_+29d', 'relative_+30d',
            
            # Weeks (1-52 weeks for complete year coverage)
            'relative_+1w', 'relative_+2w', 'relative_+3w', 'relative_+4w', 'relative_+5w',
            'relative_+6w', 'relative_+7w', 'relative_+8w', 'relative_+9w', 'relative_+10w',
            'relative_+11w', 'relative_+12w', 'relative_+13w', 'relative_+14w', 'relative_+15w',
            'relative_+16w', 'relative_+17w', 'relative_+18w', 'relative_+19w', 'relative_+20w',
            'relative_+21w', 'relative_+22w', 'relative_+23w', 'relative_+24w', 'relative_+25w',
            'relative_+26w', 'relative_+27w', 'relative_+28w', 'relative_+29w', 'relative_+30w',
            'relative_+31w', 'relative_+32w', 'relative_+33w', 'relative_+34w', 'relative_+35w',
            'relative_+36w', 'relative_+37w', 'relative_+38w', 'relative_+39w', 'relative_+40w',
            'relative_+41w', 'relative_+42w', 'relative_+43w', 'relative_+44w', 'relative_+45w',
            'relative_+46w', 'relative_+47w', 'relative_+48w', 'relative_+49w', 'relative_+50w',
            'relative_+51w', 'relative_+52w',
            
            # Months (1-24 months for 2-year coverage)
            'relative_+1month', 'relative_+2month', 'relative_+3month', 'relative_+4month', 'relative_+5month',
            'relative_+6month', 'relative_+7month', 'relative_+8month', 'relative_+9month', 'relative_+10month',
            'relative_+11month', 'relative_+12month', 'relative_+13month', 'relative_+14month', 'relative_+15month',
            'relative_+16month', 'relative_+17month', 'relative_+18month', 'relative_+19month', 'relative_+20month',
            'relative_+21month', 'relative_+22month', 'relative_+23month', 'relative_+24month'
        }
        
        # Methods that indicate NO specific date mentioned - these should use grade timelines
        default_timeline_methods = {
            'no_confirmation_1day', 'empty_transcript_1day', 'no_confirmation'
        }
        
        # If it's a default timeline method, then no date was confirmed
        if calculation_method in default_timeline_methods:
            return False
        
        # If it's a confirmed date method, then date was confirmed
        return calculation_method in confirmed_date_methods
    
    async def save_booking(self, booking_data: Dict) -> Dict:
        """Save booking to database with lead assignment logic"""
        result = {"db": "enabled", "errors": []}
        
        try:
            # Save to database using storage
            booking_id = await self.storage.save_booking(booking_data)
            
            if booking_id:
                result["db"] = "success"
                result["booking_id"] = booking_id
                logger.info(f"Booking saved successfully: {booking_id}")
            else:
                result["db"] = "failed"
                result["booking_id"] = None
                result["errors"].append("Failed to save booking - no ID returned")
                logger.error("Failed to save booking - no ID returned")
                
        except LeadBookingsStorageError as e:
            result["db"] = "error"
            result["booking_id"] = None
            result["errors"].append(str(e))
            logger.error(f"Database storage error: {e}")
        except Exception as e:
            result["db"] = "error"
            result["booking_id"] = None
            result["errors"].append(str(e))
            logger.error(f"Unexpected error saving booking: {e}")
        
        return result

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Lead Bookings Extractor')
    parser.add_argument('--call-id', help='Process specific call ID')
    parser.add_argument('--list', action='store_true', help='List recent calls')
    parser.add_argument('--limit', type=int, default=10, help='Limit for list (default: 10)')
    
    args = parser.parse_args()
    
    extractor = LeadBookingsExtractor()
    
    try:
        if args.list:
            # List recent calls
            calls = await extractor.storage.list_calls(limit=args.limit)
            print(f"\n{'='*100}")
            print(f"Recent Calls (limit: {args.limit})")
            print(f"{'='*100}")
            print(f"{'#':<6} {'Call ID':<40} {'Lead ID':<40} {'Started At':<25} {'Agent ID':<40}")
            print(f"{'-'*100}")
            
            for idx, call in enumerate(calls, 1):
                call_id_short = call['id'][:36] if call['id'] else 'N/A'
                lead_id_short = call['lead_id'][:36] if call['lead_id'] else 'N/A'
                started_at_short = call['started_at'][:25] if call['started_at'] else 'N/A'
                agent_id_short = call['agent_id'][:36] if call['agent_id'] else 'N/A'
                print(f"{idx:<6} {call_id_short:<40} {lead_id_short:<40} {started_at_short:<25} {agent_id_short:<40}")
            
            print(f"{'='*100}\n")
            print(f"To process a call:")
            print(f"  python lead_bookings_extractor.py --call-id <call_id>")
            if calls:
                print(f"\nExample:")
                print(f"  python lead_bookings_extractor.py --call-id {calls[0]['id']}")
        
        elif args.call_id:
            # Process specific call
            booking_data = await extractor.process_call_log(args.call_id)
            if booking_data:
                save_results = await extractor.save_booking(booking_data)
                if save_results.get("db"):
                    print(f"\n Successfully processed and saved call: {args.call_id}")
                    print(f"   Booking ID: {booking_data['id']}")
                    print(f"   Type: {booking_data['booking_type']}")
                    print(f"   Scheduled: {booking_data['scheduled_at']}")
                    print(f"   Metadata: {booking_data['metadata']}")
                elif save_results.get("errors"):
                    print(f"\n Processed but save failed: {save_results['errors']}")
            else:
                print(f"\n Failed to process call: {args.call_id}")
        
        else:
            parser.print_help()
    
    finally:
        await extractor.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())