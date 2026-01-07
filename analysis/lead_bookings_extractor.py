"""
Lead Bookings Extractor
Reads transcriptions from lad_dev.voice_call_logs and creates lead_bookings JSON files
"""

import os
import sys
import json
import asyncio
import re
import httpx
import logging
import argparse
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import pytz

# Add parent directory to path to allow imports when running as script
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import schedule_calculator - handle both module and script execution
try:
    # Try direct import first (when run as script from analysis directory or project root)
    from schedule_calculator import ScheduleCalculator
except ImportError:
    try:
        # Try relative import (when run as module)
        from .schedule_calculator import ScheduleCalculator
    except ImportError:
        # Try absolute import (when run from project root)
        from analysis.schedule_calculator import ScheduleCalculator

from db.storage.lead_bookings import LeadBookingsStorage, LeadBookingsStorageError

# GST timezone
GST = pytz.timezone('Asia/Dubai')

logger = logging.getLogger(__name__)


class LeadBookingsExtractorError(Exception):
    """Exception raised for lead bookings extractor errors."""
    pass


class LeadBookingsExtractor:
    """
    Extract lead bookings from voice_call_logs transcriptions.
    
    Manages:
    - Extracting booking information from call transcriptions using Gemini AI
    - Creating lead_bookings JSON files
    - Processing all transcriptions without limit
    - Calculating scheduled_at using schedule_calculator
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in .env file")
        
        # Initialize database storage
        self.storage = LeadBookingsStorage()
    
    async def close(self):
        """Close the database connection pool"""
        await self.storage.close()
    
    async def _call_gemini_api(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 8192) -> str:
        """Call Gemini API asynchronously with increased token limits to handle long conversations"""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available")
            return None
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.gemini_api_key}"
            
            headers = {"Content-Type": "application/json"}
            
            # Increase timeout for longer conversations
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens
                }
            }
            
            # Use httpx.AsyncClient for async HTTP requests
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                
                return None
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    def extract_last_timestamp(self, transcriptions_data) -> Optional[datetime]:
        """
        Extract the last timestamp from transcriptions (UTC +00) and convert to GST
        Returns None if no timestamp found
        """
        try:
            UTC = pytz.timezone('UTC')
            messages_list = None
            
            # Extract messages/segments list from various formats
            if isinstance(transcriptions_data, dict):
                # Check for 'messages' field
                if 'messages' in transcriptions_data and isinstance(transcriptions_data['messages'], list):
                    messages_list = transcriptions_data['messages']
                    logger.debug(f"Found 'messages' field with {len(messages_list)} entries")
                # Check for 'segments' field (alternative structure)
                elif 'segments' in transcriptions_data and isinstance(transcriptions_data['segments'], list):
                    messages_list = transcriptions_data['segments']
                    logger.debug(f"Found 'segments' field with {len(messages_list)} entries")
            elif isinstance(transcriptions_data, list):
                messages_list = transcriptions_data
                logger.debug(f"Transcriptions is a list with {len(messages_list)} entries")
            
            if not messages_list:
                logger.warning(f"No messages/segments list found in transcriptions. Type: {type(transcriptions_data)}")
                return None
            
            # Find the last message with a timestamp
            last_timestamp = None
            for idx, entry in enumerate(reversed(messages_list)):  # Start from the end
                if not isinstance(entry, dict):
                    continue
                    
                # Try various timestamp field names (timestamp is the primary one)
                timestamp = (entry.get('timestamp') or entry.get('created_at') or 
                           entry.get('time') or entry.get('date') or 
                           entry.get('timestamp_utc') or entry.get('time_utc'))
                
                if timestamp:
                    try:
                        logger.debug(f"Processing timestamp entry #{idx}: {timestamp} (type: {type(timestamp)})")
                        
                        # Handle different timestamp formats
                        if isinstance(timestamp, str):
                            timestamp_str = timestamp.strip()
                            logger.debug(f"Parsing timestamp string: {timestamp_str}")
                            
                            # Try ISO format first - handles: "2026-01-06T15:56:04.594238+00:00"
                            try:
                                # datetime.fromisoformat handles ISO format with microseconds and timezone
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                logger.debug(f"Successfully parsed ISO format: {dt}")
                            except ValueError as e:
                                logger.debug(f"ISO format failed: {e}, trying strptime formats")
                                # Try parsing with strptime for various formats
                                formats = [
                                    '%Y-%m-%dT%H:%M:%S.%f%z',  # "2026-01-06T15:56:04.594238+00:00"
                                    '%Y-%m-%dT%H:%M:%S%z',      # "2026-01-06T15:56:04+00:00"
                                    '%Y-%m-%d %H:%M:%S.%f%z',   # Space separator with microseconds
                                    '%Y-%m-%d %H:%M:%S%z',      # Space separator with timezone
                                    '%Y-%m-%dT%H:%M:%S.%fZ',    # ISO with Z
                                    '%Y-%m-%dT%H:%M:%SZ',       # ISO with Z
                                    '%Y-%m-%d %H:%M:%S',        # No timezone (assume UTC)
                                    '%Y-%m-%dT%H:%M:%S',        # ISO no timezone (assume UTC)
                                ]
                                
                                dt = None
                                for fmt in formats:
                                    try:
                                        dt = datetime.strptime(timestamp_str, fmt)
                                        logger.debug(f"Successfully parsed with format {fmt}: {dt}")
                                        break
                                    except ValueError:
                                        continue
                                
                                if not dt:
                                    logger.warning(f"Could not parse timestamp: {timestamp_str}")
                                    continue
                        elif isinstance(timestamp, datetime):
                            dt = timestamp
                            logger.debug(f"Timestamp is already datetime: {dt}")
                        else:
                            logger.debug(f"Unknown timestamp type: {type(timestamp)}")
                            continue
                        
                        # Ensure UTC timezone (+00)
                        if dt.tzinfo is None:
                            # No timezone info - assume UTC (as per user's requirement "+00")
                            dt = UTC.localize(dt)
                            logger.debug(f"Localized to UTC: {dt}")
                        elif dt.tzinfo != UTC:
                            # Convert to UTC first
                            dt = dt.astimezone(UTC)
                            logger.debug(f"Converted to UTC: {dt}")
                        
                        last_timestamp = dt
                        logger.info(f"Found last timestamp in transcription: {timestamp} -> {dt} (UTC)")
                        break  # Found the last timestamp
                    except Exception as e:
                        logger.warning(f"Error parsing timestamp {timestamp}: {e}", exc_info=True)
                        continue
            
            if last_timestamp:
                # Convert from UTC to GST
                gst_timestamp = last_timestamp.astimezone(GST)
                logger.info(f"Extracted last timestamp from transcriptions: {last_timestamp} (UTC) -> {gst_timestamp} (GST)")
                return self._normalize_datetime(gst_timestamp)
            else:
                logger.warning(f"No valid timestamp found in {len(messages_list)} transcription entries")
                # Log first few entries for debugging
                for i, entry in enumerate(messages_list[:3]):
                    logger.debug(f"Entry {i} keys: {list(entry.keys()) if isinstance(entry, dict) else 'Not a dict'}")
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting last timestamp from transcriptions: {e}", exc_info=True)
            return None
    
    def extract_first_timestamp(self, transcriptions_data) -> Optional[datetime]:
        """
        Extract the first timestamp from transcriptions (UTC +00) and convert to GST
        Returns None if no timestamp found
        """
        try:
            UTC = pytz.timezone('UTC')
            messages_list = None
            
            # Extract messages/segments list from various formats
            if isinstance(transcriptions_data, dict):
                # Check for 'messages' field
                if 'messages' in transcriptions_data and isinstance(transcriptions_data['messages'], list):
                    messages_list = transcriptions_data['messages']
                # Check for 'segments' field (alternative structure)
                elif 'segments' in transcriptions_data and isinstance(transcriptions_data['segments'], list):
                    messages_list = transcriptions_data['segments']
            elif isinstance(transcriptions_data, list):
                messages_list = transcriptions_data
            
            if not messages_list:
                return None
            
            # Find the first message with a timestamp (iterate forward from start)
            first_timestamp = None
            for idx, entry in enumerate(messages_list):
                if not isinstance(entry, dict):
                    continue
                    
                # Try various timestamp field names
                timestamp = (entry.get('timestamp') or entry.get('created_at') or 
                           entry.get('time') or entry.get('date') or 
                           entry.get('timestamp_utc') or entry.get('time_utc'))
                
                if timestamp:
                    try:
                        # Handle different timestamp formats (same logic as extract_last_timestamp)
                        if isinstance(timestamp, str):
                            timestamp_str = timestamp.strip()
                            try:
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            except ValueError:
                                formats = [
                                    '%Y-%m-%dT%H:%M:%S.%f%z',
                                    '%Y-%m-%dT%H:%M:%S%z',
                                    '%Y-%m-%d %H:%M:%S.%f%z',
                                    '%Y-%m-%d %H:%M:%S%z',
                                    '%Y-%m-%dT%H:%M:%S.%fZ',
                                    '%Y-%m-%dT%H:%M:%SZ',
                                    '%Y-%m-%d %H:%M:%S',
                                    '%Y-%m-%dT%H:%M:%S',
                                ]
                                dt = None
                                for fmt in formats:
                                    try:
                                        dt = datetime.strptime(timestamp_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                if not dt:
                                    continue
                        elif isinstance(timestamp, datetime):
                            dt = timestamp
                        else:
                            continue
                        
                        # Ensure UTC timezone
                        if dt.tzinfo is None:
                            dt = UTC.localize(dt)
                        elif dt.tzinfo != UTC:
                            dt = dt.astimezone(UTC)
                        
                        first_timestamp = dt
                        logger.info(f"Found first timestamp in transcription: {timestamp} -> {dt} (UTC)")
                        break  # Found the first timestamp
                    except Exception as e:
                        logger.debug(f"Error parsing first timestamp {timestamp}: {e}")
                        continue
            
            if first_timestamp:
                # Convert from UTC to GST
                gst_timestamp = first_timestamp.astimezone(GST)
                logger.info(f"Extracted first timestamp from transcriptions: {first_timestamp} (UTC) -> {gst_timestamp} (GST)")
                return self._normalize_datetime(gst_timestamp)
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting first timestamp from transcriptions: {e}")
            return None
    
    def parse_transcription(self, transcriptions_data) -> str:
        """Parse transcription from various formats - processes ALL conversations without limit"""
        if isinstance(transcriptions_data, dict):
            if 'messages' in transcriptions_data and isinstance(transcriptions_data['messages'], list):
                conversation_log = transcriptions_data['messages']
                # Process ALL messages without any limit
                conversation_text = "\n".join([
                    f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}"
                    for entry in conversation_log  # No limit - processes all entries
                ])
                logger.debug(f"Parsed {len(conversation_log)} messages from transcripts")
                return conversation_text
            elif any(key in transcriptions_data for key in ['role', 'message', 'text']):
                role = transcriptions_data.get('role', 'Unknown').title()
                message = transcriptions_data.get('message') or transcriptions_data.get('text', '')
                return f"{role}: {message}"
            else:
                # If it's a complex dict structure, convert to JSON to preserve all data
                return json.dumps(transcriptions_data, ensure_ascii=False)
        elif isinstance(transcriptions_data, list):
            # Process ALL entries in the list without any limit
            conversation_text = "\n".join([
                f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}"
                for entry in transcriptions_data  # No limit - processes all entries
            ])
            logger.debug(f"Parsed {len(transcriptions_data)} entries from transcripts list")
            return conversation_text
        else:
            # Convert to string to preserve all data
            return str(transcriptions_data)
    
    async def extract_booking_info(self, conversation_text: str) -> Dict:
        """Extract booking information using Gemini"""
        prompt = f"""Analyze this phone call conversation and extract booking information.

CONVERSATION:
{conversation_text}

Determine:
1. booking_type: 
   - "auto_consultation" if lead explicitly books a counselling session, consultation appointment, or meeting (look for phrases like: "book consultation", "schedule counselling", "book a session", "book meeting", "schedule meeting", "book appointment", "I want to book consultation", "let's book counselling", "book counselling for", "schedule a consultation", "book a counselling session", "meeting booked", "appointment scheduled", "counselling scheduled", "let's schedule", "I'd like to book", "can we book", "book it for", "schedule it for", "set up a meeting", "arrange a consultation", "book a call", "schedule a call for consultation")
   - "auto_followup" for ALL other cases (e.g., "call me after X mins/hours", "call me back in 30 mins", "call me within 1 hour", "call me tomorrow", "call me next week", "follow up with me", "call me later", any callback request, or any conversation where a follow-up is needed but NO meeting/consultation is booked)
   
IMPORTANT: 
- If ANY meeting, consultation, counselling, or appointment is booked/scheduled, use "auto_consultation"
- Only use "auto_followup" if it's just a callback request with NO meeting booking
- booking_type MUST always be either "auto_followup" or "auto_consultation", NEVER null. If unsure, default to "auto_followup".

2. scheduled_at: The exact time mentioned for follow-up or consultation
   - Extract time in format like "2025-12-27 09:00:00" (GST timezone)
   - Handle formats like "within 30 mins", "after 50 mins", "call me after 15 mins", "call me in 30 minutes", "tomorrow 3 PM", "Monday at 11:00", "next Sunday", "next week", "book for next Sunday"
   - CRITICAL: Extract the EXACT time phrase mentioned by the user OR agent, even if it's relative
   - IMPORTANT: Look for phrases like:
     * "call me after X mins/minutes" → extract as "after X mins" (e.g., "after 15 mins")
     * "call me in X mins/minutes" → extract as "in X minutes" or "within X mins" (e.g., "in 30 minutes")
     * "call me within X mins/minutes" → extract as "within X mins"
     * "book for next Sunday" → extract as "next Sunday"
     * "book a meeting for next Sunday" → extract as "next Sunday"
     * "can I book slot at Sunday at 11" and user says "yeah" → extract as "Sunday at 11" or "Sunday at 11 AM"
     * "book slot at Sunday at 11" → extract as "Sunday at 11" or "Sunday at 11 AM"
     * "Sunday at 11" → extract as "Sunday at 11" or "Sunday at 11 AM"
     * "Sunday at 11 AM" → extract as "Sunday at 11 AM"
     * "schedule for tomorrow" → extract as "tomorrow"
   - CRITICAL: When agent asks "can I book slot at Sunday at 11?" or "Sunday at 11 AM or 3 PM?" and user confirms with "yeah", "yes", "okay", etc., extract the COMPLETE time mentioned by agent (e.g., "Sunday at 11" or "Sunday at 11 AM")
   - CRITICAL: Return the time phrase AS-IS (e.g., "after 15 mins", "in 30 minutes", "next Sunday", "tomorrow 3 PM", "Sunday at 11", "Sunday at 11 AM")
   - DO NOT try to convert relative times like "after 15 mins" to datetime format - just return the phrase exactly as mentioned
   - For absolute dates/times, you can return in "YYYY-MM-DD HH:MM:SS" format if you're certain, but prefer returning the phrase
   - If NO time is mentioned at all, return null

3. student_grade: Extract the student's current grade/class if mentioned
   - Look for phrases like "I'm in grade 10", "class 11", "12th standard", "grade 9", "I'm studying in 10th", "currently in grade 12", etc.
   - Return as integer (9, 10, 11, 12, etc.) or null if not mentioned
   - If student mentions they are in college/university/UG/PG/Masters, return 12 (Grade 12+)

4. call_id: Extract any call ID or reference number mentioned, or return null

IMPORTANT: 
- Use "auto_consultation" if ANY meeting, consultation, counselling, or appointment is booked/scheduled/confirmed
- Use "auto_followup" ONLY if it's just a callback request with NO meeting/consultation booking
- Look carefully for booking phrases: "book", "schedule", "appointment", "meeting", "counselling", "consultation"
- booking_type MUST always be either "auto_followup" or "auto_consultation", NEVER null
- Extract student_grade carefully - look for grade numbers (9, 10, 11, 12) or educational level mentions

Respond ONLY in JSON format:
{{
    "booking_type": "auto_followup" or "auto_consultation",
    "scheduled_at": "2025-12-27 09:00:00" or null,
    "student_grade": 10 or null,
    "call_id": "call-id-value" or null
}}"""

        # Increased max_output_tokens to handle longer conversations and detailed responses
        response = await self._call_gemini_api(prompt, temperature=0.1, max_output_tokens=4096)
        if not response:
            # Default to auto_followup if API fails
            return {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                booking_info = json.loads(json_match.group())
                # Ensure booking_type is never null - default to auto_followup
                if not booking_info.get("booking_type") or booking_info.get("booking_type") not in ["auto_followup", "auto_consultation"]:
                    booking_info["booking_type"] = "auto_followup"
                # Ensure student_grade is an integer or null
                if "student_grade" in booking_info and booking_info["student_grade"] is not None:
                    try:
                        booking_info["student_grade"] = int(booking_info["student_grade"])
                    except (ValueError, TypeError):
                        booking_info["student_grade"] = None
                return booking_info
            else:
                logger.warning(f"Could not parse JSON from response: {response}")
                return {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response}")
            return {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
    
    def normalize_time_string(self, time_str: str) -> str:
        """
        Normalize time string by converting written numbers to numeric format.
        This handles cases like "in twenty minutes" -> "in 20 minutes"
        
        Production-safe: Handles None, empty strings, and non-string inputs gracefully.
        """
        if not time_str:
            return "" if time_str is None else str(time_str)
        
        # Ensure time_str is a string (handle any non-string input)
        if not isinstance(time_str, str):
            try:
                time_str = str(time_str)
            except Exception as e:
                logger.warning(f"Could not convert time_str to string: {e}, returning original")
                return str(time_str) if time_str is not None else ""
        
        # Dictionary mapping written numbers to digits (for common time expressions)
        written_numbers = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100'
        }
        
        # Handle compound numbers like "twenty-one", "thirty-five", etc.
        def convert_written_to_number(text: str) -> str:
            text_lower = text.lower()
            
            # Handle compound numbers (twenty-one, thirty-five, etc.)
            compound_pattern = r'(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]?(one|two|three|four|five|six|seven|eight|nine)'
            def replace_compound(match):
                tens = match.group(1)
                ones = match.group(2)
                tens_num = written_numbers.get(tens, '0')
                ones_num = written_numbers.get(ones, '0')
                try:
                    result = str(int(tens_num) + int(ones_num))
                    logger.debug(f"Converted compound number '{match.group(0)}' to '{result}'")
                    return result
                except:
                    return match.group(0)
            
            text_lower = re.sub(compound_pattern, replace_compound, text_lower)
            
            # Replace simple written numbers
            for written, numeric in written_numbers.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(written) + r'\b'
                if re.search(pattern, text_lower):
                    text_lower = re.sub(pattern, numeric, text_lower)
                    logger.debug(f"Converted written number '{written}' to '{numeric}'")
            
            return text_lower
        
        normalized = convert_written_to_number(time_str)
        
        if normalized != time_str.lower():
            logger.info(f"Normalized time string: '{time_str}' -> '{normalized}'")
        
        return normalized
    
    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Remove microseconds from datetime to keep format as YYYY-MM-DD HH:MM:SS"""
        if dt:
            return dt.replace(microsecond=0)
        return dt
    
    async def calculate_scheduled_at(self, booking_type: str, scheduled_at_str: str, reference_time: datetime, conversation_text: Optional[str] = None, transcriptions_data: Optional[Dict] = None, started_at: Optional[datetime] = None) -> Optional[datetime]:
        """
        Calculate scheduled_at using schedule_calculator based on booking_type (async-compatible)
        For simple "X minutes" patterns, uses direct calculation to ensure correct time
        
        Args:
            booking_type: Type of booking (auto_followup or auto_consultation)
            scheduled_at_str: Time string extracted from conversation
            reference_time: Reference time for calculations (usually last transcription timestamp)
            conversation_text: Full conversation text to check for "tomorrow" or date mentions
            transcriptions_data: Raw transcriptions data to extract first timestamp
            started_at: Call start time (fallback if no transcription timestamp found)
        """
        if not booking_type or not scheduled_at_str:
            return None
        
        # Normalize time string to convert written numbers to numeric format
        # e.g., "in twenty minutes" -> "in 20 minutes"
        try:
            normalized_time_str = self.normalize_time_string(scheduled_at_str)
        except Exception as e:
            logger.warning(f"Error normalizing time string '{scheduled_at_str}': {e}, using original string")
            normalized_time_str = scheduled_at_str  # Fallback to original if normalization fails
        
        # Check if "tomorrow" is mentioned IN RELATION TO the specific time phrase
        # Only check if explicitly mentioned with the time, not anywhere in conversation
        is_tomorrow_mentioned = False
        is_today_mentioned = False
        specific_date = None
        parsed_date = None
        
        # PRIORITY 0: Parse explicit dates (e.g., "January 10th", "Saturday, January 10th", "next Monday")
        # Extract date first, then combine with time
        normalized_lower = normalized_time_str.lower()
        
        # Check for specific month+day dates (e.g., "January 10th", "Saturday, January 10th")
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        date_patterns = [
            r'(?:saturday|sunday|monday|tuesday|wednesday|thursday|friday)[,\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?',  # "Saturday, January 10th"
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?',  # "January 10th"
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)',  # "10th January"
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, normalized_lower)
            if date_match:
                try:
                    day = int(date_match.group(1))
                    # Extract month from the matched text
                    matched_text = date_match.group(0).lower()
                    month = None
                    for month_name, month_num in months.items():
                        if month_name in matched_text:
                            month = month_num
                            break
                    
                    if month and 1 <= day <= 31:
                        # Get current year from reference_time
                        current_year = reference_time.year
                        # Try to create the date
                        try:
                            parsed_date = datetime(current_year, month, day).date()
                            # If the date has already passed this year, use next year
                            if parsed_date < reference_time.date():
                                parsed_date = datetime(current_year + 1, month, day).date()
                            logger.info(f"Found specific date in time string: {parsed_date} (from '{normalized_time_str}')")
                            break
                        except ValueError:
                            logger.debug(f"Invalid date: {month}/{day}")
                            continue
                except (ValueError, IndexError, AttributeError) as e:
                    logger.debug(f"Error parsing date from '{normalized_time_str}': {e}")
                    continue
        
        # Check for weekday mentions (e.g., "next Monday", "Monday", "this Saturday")
        if not parsed_date:
            weekday_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            for weekday_name, weekday_num in weekday_map.items():
                if weekday_name in normalized_lower:
                    # Check if it's "next [weekday]" or just "[weekday]"
                    is_next = 'next' in normalized_lower
                    # Calculate days until that weekday
                    current_weekday = reference_time.weekday()
                    days_until = (weekday_num - current_weekday) % 7
                    if days_until == 0:  # Today is that weekday
                        days_until = 7 if is_next else 0  # If "next", use next week, else use today
                    elif is_next and days_until < 7:  # "next" explicitly mentioned
                        days_until = days_until  # Already correct
                    elif not is_next and days_until == 0:  # Today, use today
                        days_until = 0
                    elif not is_next:  # Not today, assume next occurrence
                        days_until = days_until if days_until > 0 else 7
                    
                    parsed_date = (reference_time + timedelta(days=days_until)).date()
                    logger.info(f"Found weekday '{weekday_name}' (next={is_next}), scheduling for: {parsed_date}")
                    break
        
        # FIRST: Check the normalized scheduled_at_str itself for "tomorrow" mentions
        # This is the most reliable indicator - if Gemini extracted "tomorrow at 6:30", it's in the string
        if 'tomorrow' in normalized_lower or 'next day' in normalized_lower:
            is_tomorrow_mentioned = True
            if not parsed_date:
                parsed_date = (reference_time + timedelta(days=1)).date()
            logger.info(f"Found 'tomorrow' directly in scheduled_at_str: '{scheduled_at_str}'")
        elif 'today' in normalized_lower or 'this day' in normalized_lower:
            is_today_mentioned = True
            if not parsed_date:
                parsed_date = reference_time.date()
            logger.info(f"Found 'today' directly in scheduled_at_str: '{scheduled_at_str}'")
        
        # SECOND: If not found in scheduled_at_str, check conversation context ONLY around time mentions
        # Look for patterns like "call me tomorrow at 6:30" or "tomorrow after 10 minutes"
        # BUT: Only check if it's clearly related to the booking time, not just mentioned anywhere
        if not is_tomorrow_mentioned and conversation_text:
            conversation_lower = conversation_text.lower()
            
            # Look for "tomorrow" in context with time phrases
            # Patterns: "tomorrow at [time]", "call [me/you] tomorrow at [time]", "tomorrow [time]"
            tomorrow_context_patterns = [
                r'tomorrow\s+(?:at|in|after|around)?\s*(?:\d{1,2}[:.]\d{2}|\d+\s*(?:mins?|minutes?|hours?))',  # "tomorrow at 6:30" or "tomorrow in 20 minutes"
                r'(?:call|ring|phone|will\s+call)\s+(?:me|you)?\s+tomorrow\s+(?:at|in|after)?',  # "call me tomorrow at" or "will call you tomorrow"
                r'tomorrow\s+(?:we|i|let)\s+(?:will|shall)?\s*(?:call|ring|phone)',  # "tomorrow we will call"
            ]
            
            for pattern in tomorrow_context_patterns:
                if re.search(pattern, conversation_lower):
                    is_tomorrow_mentioned = True
                    logger.info(f"Found 'tomorrow' in context with time phrase: '{pattern}'")
                    break
            
            # Only check for "today" if we didn't find tomorrow
            if not is_tomorrow_mentioned:
                today_context_patterns = [
                    r'today\s+(?:at|in|after|around)?\s*(?:\d{1,2}[:.]\d{2}|\d+\s*(?:mins?|minutes?|hours?))',
                    r'(?:call|ring|phone|will\s+call)\s+(?:me|you)?\s+today\s+(?:at|in|after)?',
                ]
                for pattern in today_context_patterns:
                    if re.search(pattern, conversation_lower):
                        is_today_mentioned = True
                        logger.info(f"Found 'today' in context with time phrase: '{pattern}'")
                        break
        
        # PRIORITY 1: Check for absolute time patterns (e.g., "6:20", "6:20 PM", "6:20 GST", "18:20")
        # These should be used as-is without relative calculation
        absolute_time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "6:20", "6:20 PM", "6:20 GST" - group 1=hour, 2=min, 3=AM/PM
            r'\b(\d{1,2}):(\d{2})\s*(?:GST|IST|UTC)\b',  # "6:20 GST" - group 1=hour, 2=min
            r'\b(\d{1,2})\.(\d{2})\s*(AM|PM|am|pm)?\b',  # "6.20 PM" - group 1=hour, 2=min, 3=AM/PM
            r'(?:^|\s)at\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b',  # "at 6:20 PM" - group 1=hour, 2=min, 3=AM/PM
            r'(?:^|\s)at\s+(\d{1,2}):(\d{2})\s*(?:GST|IST|UTC)\b',  # "at 6:20 GST" - group 1=hour, 2=min
        ]
        
        for pattern in absolute_time_patterns:
            time_match = re.search(pattern, normalized_time_str, re.IGNORECASE)
            if time_match:
                try:
                    # Groups 1 and 2 are always hour and minute
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    
                    # Check for AM/PM in group 3 (if present) or in full match
                    is_pm = False
                    is_am = False
                    
                    # Check if group 3 exists and contains AM/PM
                    if len(time_match.groups()) >= 3 and time_match.group(3):
                        ampm_str = time_match.group(3).upper()
                        if ampm_str == 'PM':
                            is_pm = True
                        elif ampm_str == 'AM':
                            is_am = True
                    
                    # Fallback: check full match text for AM/PM
                    if not is_pm and not is_am:
                        matched_text = time_match.group(0).upper()
                        if 'PM' in matched_text:
                            is_pm = True
                        elif 'AM' in matched_text:
                            is_am = True
                    
                    # Convert to 24-hour format if AM/PM is specified
                    if is_pm and hour != 12:
                        hour = hour + 12
                    elif is_am and hour == 12:
                        hour = 0
                    elif not is_am and not is_pm:
                        # No AM/PM specified - assume 24-hour format if hour > 12, else assume current context
                        if hour > 12:
                            # Already 24-hour format
                            pass
                        else:
                            # Could be 12-hour format without AM/PM - use as-is and let schedule_calculator handle it
                            pass
                    
                    # Validate time
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        # If we have a parsed_date (from explicit date, weekday, or tomorrow/today), use it
                        if parsed_date:
                            # Use the parsed date with the specified time
                            target_time = GST.localize(datetime.combine(parsed_date, datetime.min.time().replace(hour=hour, minute=minute, second=0, microsecond=0)))
                            logger.info(f"Absolute time {hour:02d}:{minute:02d} with parsed date {parsed_date}, scheduling for: {target_time}")
                        elif is_tomorrow_mentioned:
                            # Explicitly mentioned "tomorrow" with the time - schedule for tomorrow
                            target_time_today = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            target_time = target_time_today + timedelta(days=1)
                            logger.info(f"Absolute time {hour:02d}:{minute:02d} - 'tomorrow' explicitly mentioned, scheduling for: {target_time}")
                        elif is_today_mentioned:
                            # Explicitly mentioned "today" - schedule for today even if time has passed
                            target_time_today = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            target_time = target_time_today
                            logger.info(f"Absolute time {hour:02d}:{minute:02d} - 'today' explicitly mentioned, scheduling for: {target_time}")
                        else:
                            # No explicit date mentioned - use the time as-is for TODAY
                            # Even if time has passed, schedule for today (same day as conversation)
                            target_time_today = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            target_time = target_time_today
                            if target_time_today < reference_time:
                                logger.info(f"Absolute time {hour:02d}:{minute:02d} has passed, but no 'tomorrow' mentioned - scheduling for today: {target_time}")
                            else:
                                logger.info(f"Absolute time {hour:02d}:{minute:02d} is later today, scheduling for: {target_time}")
                        
                        logger.info(f"Absolute time detected: '{normalized_time_str}' -> {target_time}")
                        return self._normalize_datetime(target_time)
                except (ValueError, IndexError, AttributeError) as e:
                    logger.debug(f"Error parsing absolute time from '{normalized_time_str}': {e}")
                    continue
        
        # PRIORITY 2: Direct calculation for simple "X minutes" patterns (relative time)
        # This ensures correct time calculation for "in 20 minutes", "after 30 minutes", etc.
        # For relative times, use FIRST TIMESTAMP DATE (or started_at) + the relative time
        direct_minutes_match = re.search(r'(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)', normalized_time_str.lower())
        if direct_minutes_match:
            try:
                minutes = int(direct_minutes_match.group(1))
                # Validate reasonable range (1 minute to 7 days = 10080 minutes)
                if 1 <= minutes <= 10080:
                    # For relative times, use FIRST TIMESTAMP DATE (or started_at) as the base date
                    base_time = None
                    
                    # Get the date from first transcription timestamp (or started_at as fallback)
                    if transcriptions_data:
                        first_timestamp = self.extract_first_timestamp(transcriptions_data)
                        if first_timestamp:
                            base_time = first_timestamp
                            logger.info(f"Using first transcription timestamp as base for relative time: {base_time}")
                    
                    # Fallback to started_at if no transcription timestamp
                    if not base_time and started_at:
                        if started_at.tzinfo is None:
                            base_time = GST.localize(started_at)
                        else:
                            base_time = started_at.astimezone(GST)
                        logger.info(f"Using started_at as base for relative time: {base_time}")
                    
                    # Final fallback to reference_time (last timestamp)
                    if not base_time:
                        base_time = reference_time
                        logger.info(f"Using reference_time (last timestamp) as base for relative time: {base_time}")
                    
                    # Calculate relative time from base_time
                    if is_tomorrow_mentioned:
                        # User said "tomorrow after X minutes" - schedule for tomorrow
                        calculated_time = base_time + timedelta(days=1, minutes=minutes)
                        logger.info(f"Relative time '{normalized_time_str}' with 'tomorrow' mentioned: {base_time} + 1 day + {minutes} minutes = {calculated_time}")
                    else:
                        # Normal relative time - add minutes to base_time (first timestamp or started_at)
                        calculated_time = base_time + timedelta(minutes=minutes)
                        logger.info(f"Relative time (same day): '{normalized_time_str}' = {base_time} + {minutes} minutes = {calculated_time}")
                    return self._normalize_datetime(calculated_time)
                else:
                    logger.warning(f"Minutes value {minutes} out of reasonable range (1-10080), trying schedule_calculator")
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse minutes from '{normalized_time_str}': {e}, trying schedule_calculator")
        
        # For complex patterns or if direct calculation fails, use schedule_calculator
        try:
            # Wrap CPU-bound calculations in asyncio.to_thread to avoid blocking event loop
            def _calculate():
                calculator = ScheduleCalculator()
                
                # Determine outcome based on booking_type
                if booking_type == "auto_followup":
                    # Use callback_requested outcome for auto_followup
                    outcome = "callback_requested"
                    outcome_details = {
                        "callback_time": normalized_time_str  # Use normalized string
                    }
                elif booking_type == "auto_consultation":
                    # Use meeting_booked outcome for auto_consultation
                    outcome = "meeting_booked"
                    outcome_details = {
                        "callback_time": normalized_time_str,  # Use normalized string
                        "followup_time": normalized_time_str  # Use normalized string
                    }
                else:
                    # Fallback: just parse the time string
                    return calculator.parse_callback_time(normalized_time_str, reference_time)  # Use normalized string
                
                # Calculate next call time using schedule calculator
                return calculator.calculate_next_call(outcome, outcome_details, None)
            
            scheduled_at = await asyncio.to_thread(_calculate)
            
            # CRITICAL: Only combine with first timestamp for RELATIVE times (not absolute times)
            # Check if this is a relative time pattern (if absolute time, it would have been caught earlier)
            is_relative_time = direct_minutes_match is not None or any(keyword in normalized_time_str.lower() for keyword in ['after', 'in', 'within', 'minutes', 'mins'])
            is_absolute_time = bool(re.search(r'\b\d{1,2}[:.]\d{2}', normalized_time_str))  # Contains time pattern like "7:45"
            
            if scheduled_at:
                # If absolute time is explicitly mentioned, check if we have a parsed_date
                # If we have a parsed_date, use it instead of schedule_calculator's date
                if is_absolute_time and not is_relative_time:
                    if parsed_date:
                        # Use parsed_date with the time from schedule_calculator
                        schedule_time = scheduled_at.time()
                        combined_datetime = GST.localize(datetime.combine(parsed_date, schedule_time.replace(second=0, microsecond=0)))
                        logger.info(f"Absolute time with parsed date: using parsed date {parsed_date} with schedule_calculator time {schedule_time} = {combined_datetime}")
                        return self._normalize_datetime(combined_datetime)
                    else:
                        logger.info(f"Absolute time detected in '{normalized_time_str}', using schedule_calculator result as-is: {scheduled_at}")
                        return self._normalize_datetime(scheduled_at)
                
                # For relative times, combine with first timestamp date
                try:
                    # Get the date from first transcription timestamp (or started_at as fallback)
                    first_timestamp_date = None
                    if transcriptions_data:
                        first_timestamp = self.extract_first_timestamp(transcriptions_data)
                        if first_timestamp:
                            first_timestamp_date = first_timestamp.date()
                            logger.info(f"Using first transcription timestamp date for relative time: {first_timestamp_date}")
                    
                    # Fallback to started_at if no transcription timestamp
                    if not first_timestamp_date and started_at:
                        if started_at.tzinfo is None:
                            started_at_gst = GST.localize(started_at)
                        else:
                            started_at_gst = started_at.astimezone(GST)
                        first_timestamp_date = started_at_gst.date()
                        logger.info(f"Using started_at date as fallback for relative time: {first_timestamp_date}")
                    
                    # If we have a date from first timestamp, use it with the time from schedule_calculator
                    if first_timestamp_date:
                        # Extract time (hour, minute) from schedule_calculator result
                        calc_hour = scheduled_at.hour
                        calc_minute = scheduled_at.minute
                        
                        # Combine first timestamp date with schedule_calculator time
                        combined_time = GST.localize(datetime.combine(first_timestamp_date, scheduled_at.time().replace(second=0, microsecond=0)))
                        logger.info(f"Combined first timestamp date ({first_timestamp_date}) with schedule_calculator time ({calc_hour:02d}:{calc_minute:02d}) = {combined_time}")
                        return self._normalize_datetime(combined_time)
                    else:
                        # No first timestamp or started_at - use schedule_calculator result as-is
                        logger.warning(f"No first timestamp or started_at found, using schedule_calculator result as-is: {scheduled_at}")
                        return self._normalize_datetime(scheduled_at)
                except Exception as e:
                    logger.warning(f"Error combining first timestamp date with schedule_calculator time: {e}, using schedule_calculator result as-is")
                    return self._normalize_datetime(scheduled_at)
            
            # Validate that schedule_calculator result is reasonable (for relative times)
            if scheduled_at and direct_minutes_match:
                try:
                    minutes = int(direct_minutes_match.group(1))
                    expected_time = reference_time + timedelta(minutes=minutes)
                    time_diff = abs((scheduled_at - expected_time).total_seconds())
                    # If schedule_calculator result differs by more than 1 hour from expected, use direct calculation
                    if time_diff > 3600:  # More than 1 hour difference
                        logger.warning(f"schedule_calculator returned {scheduled_at} but expected {expected_time} for '{normalized_time_str}', using direct calculation")
                        return self._normalize_datetime(expected_time)
                except Exception as e:
                    logger.debug(f"Error validating schedule_calculator result: {e}")
            
            return self._normalize_datetime(scheduled_at) if scheduled_at else None
            
        except Exception as e:
            logger.error(f"Error calculating scheduled_at using schedule_calculator: {e}")
            # Final fallback: try direct calculation if we have minutes
            if direct_minutes_match:
                try:
                    minutes = int(direct_minutes_match.group(1))
                    if 1 <= minutes <= 10080:
                            return self._normalize_datetime(reference_time + timedelta(minutes=minutes))
                except Exception:
                    pass
            # Fallback to simple parsing with normalized string
            try:
                def _parse():
                    calculator = ScheduleCalculator()
                    return calculator.parse_callback_time(normalized_time_str, reference_time)  # Use normalized string
                fallback_result = await asyncio.to_thread(_parse)
                
                # CRITICAL: Only combine with first timestamp for RELATIVE times (not absolute times)
                if fallback_result:
                    # Check if this is a relative time pattern
                    is_relative_time_fallback = direct_minutes_match is not None or any(keyword in normalized_time_str.lower() for keyword in ['after', 'in', 'within', 'minutes', 'mins'])
                    is_absolute_time_fallback = bool(re.search(r'\b\d{1,2}[:.]\d{2}', normalized_time_str))
                    
                    # If absolute time, use result as-is
                    if is_absolute_time_fallback and not is_relative_time_fallback:
                        logger.info(f"Absolute time in fallback, using result as-is: {fallback_result}")
                        return self._normalize_datetime(fallback_result)
                    
                    # For relative times, combine with first timestamp date
                    try:
                        # Get the date from first transcription timestamp (or started_at as fallback)
                        first_timestamp_date = None
                        if transcriptions_data:
                            first_timestamp = self.extract_first_timestamp(transcriptions_data)
                            if first_timestamp:
                                first_timestamp_date = first_timestamp.date()
                        
                        if not first_timestamp_date and started_at:
                            if started_at.tzinfo is None:
                                started_at_gst = GST.localize(started_at)
                            else:
                                started_at_gst = started_at.astimezone(GST)
                            first_timestamp_date = started_at_gst.date()
                        
                        if first_timestamp_date:
                            # Combine first timestamp date with fallback result time
                            combined_time = GST.localize(datetime.combine(first_timestamp_date, fallback_result.time().replace(second=0, microsecond=0)))
                            logger.info(f"Combined first timestamp date ({first_timestamp_date}) with fallback parse time = {combined_time}")
                            return self._normalize_datetime(combined_time)
                    except Exception as e:
                        logger.warning(f"Error combining date with fallback result: {e}")
                
                return self._normalize_datetime(fallback_result) if fallback_result else None
            except Exception as e2:
                logger.error(f"Error in fallback parsing: {e2}")
                return None
    
    
    async def get_retry_count(self, lead_id: Optional[str]) -> int:
        """Get retry count based on lead_id (counts existing bookings with same lead_id)"""
        # Count existing bookings for the same lead_id
        # First call (no existing bookings): retry_count = 0
        # Second call (1 existing booking): retry_count = 1
        # Third call (2 existing bookings): retry_count = 2, etc.
        if not lead_id:
            return 0  # Default to 0 if no lead_id
        
        try:
            # Count existing bookings with the same lead_id
            count = await self.storage.count_bookings_by_lead_id(lead_id)
            retry_count = count  # Start from 0, so first call = 0, second = 1, etc.
            logger.info(f"Found {count} existing booking(s) with lead_id {lead_id}, retry_count = {retry_count}")
            return retry_count
        except Exception as e:
            logger.warning(f"Error getting retry count for lead_id {lead_id}: {e}. Defaulting to 0.")
            return 0
    
    async def process_call_log(self, call_log_id: str) -> Optional[Dict]:
        """Process a single call log and create booking"""
        try:
            # Get call log data from storage
            try:
                call_data = await self.storage.get_call_log(call_log_id)
            except Exception as e:
                logger.error(f"Error fetching call log {call_log_id} from database: {e}")
                raise
            
            if not call_data:
                logger.warning(f"Call log {call_log_id} not found in database")
                logger.info(f"Tip: Use '--list' to see available call IDs")
                return None
            
            call_id = call_data['id']
            tenant_id = call_data['tenant_id']
            lead_id = call_data['lead_id']
            transcripts = call_data['transcripts']
            initiated_by_user_id = call_data['initiated_by_user_id']
            agent_id = call_data['agent_id']
            started_at = call_data['started_at']
            call_status = call_data.get('status')  # Status from voice_call_logs table
            
            # If lead_id is null, try to use call_id as fallback
            # (Some calls might not have lead_id assigned, but we still want to save the booking)
            if not lead_id:
                logger.warning(f"lead_id is null for call {call_id}, using call_id as fallback for lead_id")
                lead_id = call_id
            
            # Parse transcription
            conversation_text = self.parse_transcription(transcripts)
            
            # Extract call_id from transcription JSON (if present)
            transcription_call_id = None
            if isinstance(transcripts, dict):
                # Check for call_id in the transcription JSON structure
                transcription_call_id = transcripts.get('call_id') or transcripts.get('id')
                if not transcription_call_id and 'messages' in transcripts and isinstance(transcripts['messages'], list):
                    # Check first message for call_id
                    if transcripts['messages']:
                        first_msg = transcripts['messages'][0]
                        transcription_call_id = first_msg.get('call_id') or first_msg.get('id')
            
            # Use call_id from transcription if available, otherwise use voice_call_logs.id (call_id)
            call_id_from_transcription = str(transcription_call_id) if transcription_call_id else str(call_id) if call_id else None
            logger.info(f"call_id from transcription: {transcription_call_id}, fallback to voice_call_logs.id: {call_id}, using: {call_id_from_transcription}")
            
            # Check if transcriptions are missing or very short (likely user declined/no answer)
            # If transcriptions are very short (< 50 chars) or empty, treat as Stage 1 (non-responsive)
            is_transcription_missing = not conversation_text or len(conversation_text.strip()) < 50
            
            # Check status column to confirm if user declined
            is_declined = False
            if call_status:
                status_lower = str(call_status).lower()
                is_declined = any(keyword in status_lower for keyword in ['declined', 'rejected', 'no_answer', 'not_interested', 'busy', 'failed'])
            
            if is_transcription_missing:
                logger.warning(f"Transcription missing or very short ({len(conversation_text) if conversation_text else 0} chars) for call {call_log_id}")
                if is_declined:
                    logger.info(f"Status column confirms user declined: {call_status}")
                # Continue processing but will use Stage 1 timeline pattern
                # Set default conversation_text for processing
                if not conversation_text:
                    conversation_text = "No transcription available. User likely declined or did not answer."
            
            # Extract booking info using Gemini first
            logger.info(f"Extracting booking info from conversation (length: {len(conversation_text)} chars)")
            try:
                booking_info = await self.extract_booking_info(conversation_text)
                logger.info(f"Extracted booking info: {booking_info}")
            except Exception as e:
                logger.error(f"Gemini API extraction failed: {e}, will use fallback extraction", exc_info=True)
                # Fallback to empty booking info - will trigger regex extraction
                booking_info = {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
            
            # parent_booking_id: Stores the call_id from metadata of the first booking (where retry_count = 0, parent_booking_id IS NULL)
            # Store call_id in metadata for reference
            # Logic: 
            # - First, check if a booking already exists for this call_id (in metadata)
            # - If no booking exists for this call_id, check if it's a fresh lead or follow-up
            # - Fresh lead (no bookings for this lead_id): parent_booking_id = NULL, retry_count = 0
            # - Follow-up call (bookings exist for this lead_id): parent_booking_id = call_id from metadata of original booking (where retry_count = 0, parent_booking_id IS NULL)
            parent_booking_id = None
            original_call_id = call_id_from_transcription  # Current call's call_id (voice_call_logs.id)
            
            # First, check if a booking already exists for this call_id
            is_fresh_call = True
            if original_call_id:
                try:
                    existing_booking_by_call_id = await self.storage.get_booking_by_call_id_in_metadata(original_call_id)
                    if existing_booking_by_call_id:
                        # A booking already exists for this call_id - this is a duplicate call
                        logger.warning(f"Booking already exists for call_id {original_call_id} - this is a duplicate call")
                        is_fresh_call = False
                except Exception as e:
                    logger.debug(f"Error checking for existing booking by call_id: {e}")
            
            booking_type = booking_info.get('booking_type')
            scheduled_at_str = booking_info.get('scheduled_at')
            student_grade = booking_info.get('student_grade')  # Extract student grade
            
            # Ensure booking_type is never null - default to auto_followup
            if not booking_type or booking_type not in ["auto_followup", "auto_consultation"]:
                booking_type = "auto_followup"
                logger.info(f"booking_type was null or invalid, defaulting to auto_followup")
            
            # Log student grade if found
            if student_grade:
                logger.info(f"Student grade mentioned in conversation: Grade {student_grade}")
            else:
                logger.info(f"Student grade not mentioned in conversation, will use default Grade 12+ timeline")
            
            # Get reference time for calculations
            # IMPORTANT: Use the LAST TIMESTAMP from transcriptions (UTC) converted to GST
            # This ensures scheduling is based on when the conversation actually happened, not when it's processed
            reference_time = None
            last_transcript_timestamp = self.extract_last_timestamp(transcripts)
            
            if last_transcript_timestamp:
                reference_time = last_transcript_timestamp
                logger.info(f"Using last transcription timestamp as reference_time: {reference_time} (GST)")
            else:
                # Fallback to current time if no timestamp found in transcriptions
                reference_time = self._normalize_datetime(datetime.now(GST))
                logger.warning(f"No timestamp found in transcriptions, using current time as reference_time: {reference_time}")
            
            # For logging purposes, we still track the original call start time
            call_start_time = None
            if started_at:
                if started_at.tzinfo is None:
                    call_start_time = GST.localize(started_at)
                else:
                    call_start_time = started_at.astimezone(GST)
                logger.debug(f"Call started at: {call_start_time}, using transcription timestamp {reference_time} as reference for scheduling")
            
            # Process scheduled_at based on booking_type
            # PRODUCTION-READY MULTI-LAYER TIME EXTRACTION STRATEGY:
            # Layer 1: Gemini API extraction (primary method)
            # Layer 2: Regex pattern extraction if Gemini fails or misses time
            # Layer 3: Final extraction attempt before delegating to Glinks scheduler
            # This ensures any time mentioned in conversation is captured, even if Gemini API fails
            #
            # PRIORITY 1: If time is explicitly mentioned in the call, use that time directly (no stages)
            # PRIORITY 2: If no explicit time mentioned, DO NOT auto-generate a scheduled_at here —
            #             send the booking to the Glinks scheduler instead (booking_source='glinks')

            # Initialize scheduled_at and buffer_until to None before conditional logic
            scheduled_at = None
            buffer_until = None
            use_glinks_scheduler = False

            # Determine whether an explicit time phrase was provided by Gemini or fallback extraction
            # Also check if the time is "confirmed" vs "uncertain/unconfirmed" (e.g., questions like "within next ten minutes?")
            explicit_time_mentioned = bool(scheduled_at_str and str(scheduled_at_str).strip())
            time_is_confirmed = False
            
            # Check if time is confirmed (not a question or uncertainty)
            if explicit_time_mentioned:
                time_str_lower = str(scheduled_at_str).strip().lower()
                # Check for uncertainty indicators
                uncertainty_indicators = ['?', 'maybe', 'perhaps', 'possibly', 'might', 'could', 'should', 'may', 'not sure', 'uncertain']
                is_uncertain = any(indicator in time_str_lower for indicator in uncertainty_indicators)
                # Check for question patterns
                is_question = time_str_lower.endswith('?') or '?' in time_str_lower
                
                if is_uncertain or is_question:
                    logger.info(f"Time phrase '{scheduled_at_str}' contains uncertainty/question - treating as UNCONFIRMED time")
                    time_is_confirmed = False
                    explicit_time_mentioned = False  # Treat as if no time was mentioned
                else:
                    time_is_confirmed = True
                    logger.info(f"Time phrase '{scheduled_at_str}' is CONFIRMED")

            # Comprehensive time extraction function - Production-ready with robust error handling
            def extract_time_from_conversation(text: str) -> Optional[str]:
                """Extract time phrases from conversation text with multiple patterns - Production hardened"""
                if not text or not isinstance(text, str):
                    logger.warning("extract_time_from_conversation: Invalid text input")
                    return None
                
                # Local helper function to normalize written numbers (duplicated from class method for nested function access)
                def normalize_written_numbers(time_str: str) -> str:
                    """Normalize written numbers to numeric format - Production-safe with error handling"""
                    if not time_str:
                        return "" if time_str is None else str(time_str) if not isinstance(time_str, str) else time_str
                    
                    try:
                        # Ensure input is string
                        if not isinstance(time_str, str):
                            time_str = str(time_str)
                        
                        written_numbers = {
                            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
                            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
                            'eighty': '80', 'ninety': '90', 'hundred': '100'
                        }
                        
                        text_lower = time_str.lower()
                        
                        # Handle compound numbers (twenty-one, thirty-five, etc.)
                        compound_pattern = r'(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]?(one|two|three|four|five|six|seven|eight|nine)'
                        def replace_compound(match):
                            try:
                                tens = match.group(1)
                                ones = match.group(2)
                                tens_num = written_numbers.get(tens, '0')
                                ones_num = written_numbers.get(ones, '0')
                                return str(int(tens_num) + int(ones_num))
                            except (ValueError, IndexError, AttributeError):
                                return match.group(0)  # Return original if conversion fails
                        
                        text_lower = re.sub(compound_pattern, replace_compound, text_lower)
                        
                        # Replace simple written numbers
                        for written, numeric in written_numbers.items():
                            try:
                                pattern = r'\b' + re.escape(written) + r'\b'
                                text_lower = re.sub(pattern, numeric, text_lower)
                            except Exception:
                                continue  # Skip this replacement if it fails, continue with others
                        
                        return text_lower
                    except Exception as e:
                        logger.debug(f"Error in normalize_written_numbers for '{time_str}': {e}, returning original")
                        return str(time_str) if not isinstance(time_str, str) else time_str
                
                try:
                    # Normalize text: remove extra whitespace, handle quotes, punctuation
                    text_cleaned = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text_lower = text_cleaned.lower()
                    
                    # Pattern 1: "call me after/in/within X mins/minutes" (with or without "me")
                    # Enhanced patterns to handle various formats and edge cases
                    patterns = [
                        # "call me after 5 mins", "call after 5 mins", "just call me after 5 mins"
                        r'(?:just\s+)?(?:call\s+(?:me\s+)?|calling\s+)(?:back\s+)?(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)',
                        # "after 5 mins", "in 5 mins", "within 5 mins" (standalone, with punctuation)
                        r'(?:^|[.\s,\'"])(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)(?:\s|\.|$|,|;|\'|")',
                        # "5 mins later", "5 minutes later"
                        r'(\d+)\s*(?:mins?|minutes?)\s+later',
                        # "call me in 5", "call after 5" (without mins/minutes, context-dependent)
                        r'(?:just\s+)?(?:call\s+(?:me\s+)?|calling\s+)(?:back\s+)?(?:after|in|within)\s+(\d+)(?:\s|\.|$|,|;|\'|")(?!\s*(?:hours?|hrs?|days?|weeks?|months?))',
                        # "after 5", "in 5" (standalone, likely means minutes if context suggests it)
                        r'(?:^|[.\s,\'"])(?:after|in|within)\s+(\d+)(?:\s|\.|$|,|;|\'|")(?!\s*(?:hours?|hrs?|days?|weeks?|months?))',
                        # "call me 5 mins", "call 5 mins" (time before "mins")
                        r'(?:just\s+)?(?:call\s+(?:me\s+)?|calling\s+)(?:back\s+)?(\d+)\s*(?:mins?|minutes?)',
                        # "ring me after 5 mins", "phone me after 5 mins" (alternative verbs)
                        r'(?:ring|phone|reach)\s+(?:me\s+)?(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)',
                        # "get back to me in 5 mins"
                        r'get\s+back\s+(?:to\s+)?(?:me\s+)?(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)',
                        # "connect in 5 mins", "reach out in 5 mins"
                        r'(?:connect|reach\s+out)\s+(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)',
                    ]
                    
                    # Pattern 1b: Written numbers (e.g., "twenty minutes", "thirty minutes")
                    written_number_patterns = [
                        # "call me in twenty minutes", "after twenty minutes"
                        r'(?:just\s+)?(?:call\s+(?:me\s+)?|calling\s+)(?:back\s+)?(?:after|in|within)\s+(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|thirty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|forty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|fifty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))\s*(?:mins?|minutes?)',
                        # "in twenty minutes", "after thirty minutes" (standalone)
                        r'(?:^|[.\s,\'"])(?:after|in|within)\s+(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|thirty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|forty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|fifty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))\s*(?:mins?|minutes?)(?:\s|\.|$|,|;|\'|")',
                    ]
                    
                    for idx, pattern in enumerate(patterns):
                        try:
                            match = re.search(pattern, text_lower)
                            if match:
                                minutes_str = match.group(1)
                                if not minutes_str or not minutes_str.isdigit():
                                    continue
                                
                                minutes = int(minutes_str)
                                
                                # Validation: reasonable time range (1 minute to 7 days = 10080 minutes)
                                # Production safety: cap at reasonable maximum
                                MAX_MINUTES = 10080  # 7 days
                                MIN_MINUTES = 1
                                
                                if minutes < MIN_MINUTES:
                                    logger.warning(f"Extracted invalid minutes value: {minutes} (too small), skipping")
                                    continue
                                if minutes > MAX_MINUTES:
                                    logger.warning(f"Extracted very large minutes value: {minutes}, capping at {MAX_MINUTES}")
                                    minutes = MAX_MINUTES
                                
                                # Use "after X mins" format as it's most common and well-handled
                                result = f"after {minutes} mins"
                                logger.info(f"Extracted time from conversation using pattern #{idx+1} '{pattern[:50]}...': '{result}'")
                                return result
                        except (ValueError, IndexError, AttributeError) as e:
                            logger.debug(f"Error processing pattern #{idx+1}: {e}")
                            continue
                    
                    # Pattern 1b: Process written number patterns (e.g., "twenty minutes")
                    for idx, pattern in enumerate(written_number_patterns):
                        try:
                            match = re.search(pattern, text_lower)
                            if match:
                                # Extract the matched text and normalize it
                                matched_text = match.group(0).strip()
                                # Normalize written numbers to numeric format
                                normalized_text = normalize_written_numbers(matched_text)
                                
                                # Extract numeric minutes from normalized text
                                minutes_match = re.search(r'(\d+)\s*(?:mins?|minutes?)', normalized_text.lower())
                                if minutes_match:
                                    minutes = int(minutes_match.group(1))
                                    
                                    # Validation: reasonable time range
                                    MAX_MINUTES = 10080  # 7 days
                                    MIN_MINUTES = 1
                                    
                                    if minutes < MIN_MINUTES:
                                        logger.warning(f"Extracted invalid minutes value from written number: {minutes} (too small), skipping")
                                        continue
                                    if minutes > MAX_MINUTES:
                                        logger.warning(f"Extracted very large minutes value from written number: {minutes}, capping at {MAX_MINUTES}")
                                        minutes = MAX_MINUTES
                                    
                                    # Use "after X mins" format
                                    result = f"after {minutes} mins"
                                    logger.info(f"Extracted time from conversation using written number pattern #{idx+1} '{pattern[:50]}...': '{matched_text}' -> '{result}'")
                                    return result
                        except (ValueError, IndexError, AttributeError) as e:
                            logger.debug(f"Error processing written number pattern #{idx+1}: {e}")
                            continue
                    
                    # Pattern 2: Absolute times (e.g., "7:45 PM GST", "6:30 PM", "at 7:45")
                    # These should be extracted directly from conversation
                    absolute_time_patterns = [
                        r'(?:call|ring|phone|reach|connect)\s+(?:you|me|him|her|them|us)?\s+(?:at|around|about)?\s*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "call you at 7:45 PM GST"
                        r'(?:at|around|about)\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "at 7:45 PM GST"
                        r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)\b',  # "7:45 PM GST", "6:30 PM GST"
                        r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b',  # "7:45 PM", "6:30 AM"
                        r'\b(\d{1,2}):(\d{2})\s*(?:GST|IST|UTC)\b',  # "7:45 GST"
                        r'\b(\d{1,2})\.(\d{2})\s*(AM|PM|am|pm)?\b',  # "7.45 PM"
                    ]
                    
                    for idx, pattern in enumerate(absolute_time_patterns):
                        try:
                            match = re.search(pattern, text_lower, re.IGNORECASE)
                            if match:
                                hour = match.group(1)
                                minute = match.group(2)
                                ampm = match.group(3) if len(match.groups()) >= 3 and match.group(3) else None
                                
                                # Validate hour and minute
                                try:
                                    hour_int = int(hour)
                                    minute_int = int(minute)
                                    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
                                        logger.debug(f"Invalid time values: hour={hour_int}, minute={minute_int}, skipping")
                                        continue
                                except ValueError:
                                    logger.debug(f"Could not parse hour/minute: hour={hour}, minute={minute}")
                                    continue
                                
                                # Build time string
                                if ampm:
                                    result = f"{hour}:{minute} {ampm.upper()}"
                                else:
                                    result = f"{hour}:{minute}"
                                
                                # Check if GST/IST/UTC is in the match
                                matched_text = match.group(0).upper()
                                if 'GST' in matched_text:
                                    result += " GST"
                                elif 'IST' in matched_text:
                                    result += " IST"
                                elif 'UTC' in matched_text:
                                    result += " UTC"
                                
                                logger.info(f"Extracted absolute time from conversation using pattern #{idx+1} '{pattern[:60]}...': matched '{match.group(0)}' -> '{result}'")
                                return result
                        except (ValueError, IndexError, AttributeError) as e:
                            logger.debug(f"Error processing absolute time pattern #{idx+1}: {e}")
                            continue
                    
                    # Pattern 3: "next Sunday", "tomorrow", etc.
                    day_patterns = [
                        r'(?:book|schedule|meeting|call).*?next\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday)',
                        r'(?:book|schedule|meeting|call).*?tomorrow',
                        r'next\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday)',
                        r'tomorrow(?:\s+at)?\s*(\d{1,2})?\s*(?:AM|PM|am|pm)?',
                    ]
                    
                    for idx, pattern in enumerate(day_patterns):
                        try:
                            match = re.search(pattern, text_lower)
                            if match:
                                result = match.group(0).strip()
                                logger.info(f"Extracted time from conversation using day pattern #{idx+1} '{pattern[:50]}...': '{result}'")
                                return result
                        except (AttributeError, IndexError) as e:
                            logger.debug(f"Error processing day pattern #{idx+1}: {e}")
                            continue
                    
                    logger.debug("No time pattern matched in conversation text")
                    return None
                    
                except Exception as e:
                    logger.error(f"Error in extract_time_from_conversation: {e}", exc_info=True)
                    return None

            # First, try to extract time directly from conversation if Gemini didn't extract it
            if not explicit_time_mentioned:
                extracted_time = extract_time_from_conversation(conversation_text)
                if extracted_time:
                    scheduled_at_str = extracted_time
                    explicit_time_mentioned = True
                    logger.info(f"Extracted time from conversation text (fallback): '{scheduled_at_str}'")
            # If an explicit CONFIRMED time phrase was provided, calculate scheduled_at using schedule_calculator
            if explicit_time_mentioned and time_is_confirmed:
                # Time is explicitly mentioned - use it directly, no stage-based calculation
                logger.info(f"Time explicitly mentioned in conversation: '{scheduled_at_str}' - using this time directly (no stage-based calculation)")
                scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time, conversation_text, transcripts, started_at)
                if not scheduled_at:
                    logger.warning(f"Could not calculate scheduled_at from: '{scheduled_at_str}'")
                    logger.info("Trying to extract time from conversation directly...")
                    # Try to extract time directly from conversation if schedule_calculator didn't work
                    extracted_time = extract_time_from_conversation(conversation_text)
                    if extracted_time:
                        # CRITICAL: Normalize extracted time (convert written numbers to numeric)
                        # This handles cases where extraction finds "in twenty minutes" etc.
                        try:
                            normalized_extracted_time = self.normalize_time_string(extracted_time)
                        except Exception as e:
                            logger.warning(f"Error normalizing extracted time '{extracted_time}': {e}, using as-is")
                            normalized_extracted_time = extracted_time
                        
                        # Try to extract minutes value for direct calculation
                        minutes_match = re.search(r'(\d+)\s*(?:mins?|minutes?)', normalized_extracted_time.lower())
                        if minutes_match:
                            minutes = int(minutes_match.group(1))
                            # Try schedule_calculator first with normalized time
                            try:
                                def _parse_time():
                                    calculator = ScheduleCalculator()
                                    return calculator.parse_callback_time(normalized_extracted_time, reference_time)
                                calc_result = await asyncio.to_thread(_parse_time)
                                
                                # Only combine with first timestamp for RELATIVE times
                                is_relative_check = any(keyword in normalized_extracted_time.lower() for keyword in ['after', 'in', 'within', 'minutes', 'mins'])
                                is_absolute_check = bool(re.search(r'\b\d{1,2}[:.]\d{2}', normalized_extracted_time))
                                
                                # If absolute time, use result as-is
                                if is_absolute_check and not is_relative_check:
                                    scheduled_at = calc_result
                                    logger.info(f"Absolute time detected, using schedule_calculator result as-is: {scheduled_at}")
                                elif calc_result and transcripts:
                                    # For relative times, combine with first timestamp date
                                    try:
                                        first_timestamp = self.extract_first_timestamp(transcripts)
                                        if first_timestamp:
                                            first_timestamp_date = first_timestamp.date()
                                            combined_time = GST.localize(datetime.combine(first_timestamp_date, calc_result.time().replace(second=0, microsecond=0)))
                                            scheduled_at = self._normalize_datetime(combined_time)
                                            logger.info(f"Combined first timestamp date with schedule_calculator time: {scheduled_at}")
                                        elif started_at:
                                            if started_at.tzinfo is None:
                                                started_at_gst = GST.localize(started_at)
                                            else:
                                                started_at_gst = started_at.astimezone(GST)
                                            first_timestamp_date = started_at_gst.date()
                                            combined_time = GST.localize(datetime.combine(first_timestamp_date, calc_result.time().replace(second=0, microsecond=0)))
                                            scheduled_at = self._normalize_datetime(combined_time)
                                            logger.info(f"Combined started_at date with schedule_calculator time: {scheduled_at}")
                                        else:
                                            scheduled_at = self._normalize_datetime(calc_result)
                                    except Exception as e:
                                        logger.warning(f"Error combining date with schedule_calculator result: {e}, using as-is")
                                        scheduled_at = self._normalize_datetime(calc_result)
                                else:
                                    scheduled_at = self._normalize_datetime(calc_result)
                                
                                logger.info(f"Extracted time from conversation using schedule_calculator: {scheduled_at}")
                            except Exception as e:
                                logger.warning(f"schedule_calculator failed: {e}, using direct calculation")
                                # Fallback to direct timedelta calculation
                                scheduled_at = self._normalize_datetime(reference_time + timedelta(minutes=minutes))
                                logger.info(f"Extracted time from conversation (fallback): {scheduled_at}")
                        else:
                            # Try schedule_calculator with the normalized extracted phrase
                            try:
                                def _parse_time():
                                    calculator = ScheduleCalculator()
                                    return calculator.parse_callback_time(normalized_extracted_time, reference_time)
                                calc_result = await asyncio.to_thread(_parse_time)
                                
                                # Only combine with first timestamp for RELATIVE times
                                is_relative_check = any(keyword in normalized_extracted_time.lower() for keyword in ['after', 'in', 'within', 'minutes', 'mins'])
                                is_absolute_check = bool(re.search(r'\b\d{1,2}[:.]\d{2}', normalized_extracted_time))
                                
                                # If absolute time, use result as-is
                                if is_absolute_check and not is_relative_check:
                                    scheduled_at = calc_result
                                    logger.info(f"Absolute time detected, using schedule_calculator result as-is: {scheduled_at}")
                                elif calc_result and transcripts:
                                    # For relative times, combine with first timestamp date
                                    try:
                                        first_timestamp = self.extract_first_timestamp(transcripts)
                                        if first_timestamp:
                                            first_timestamp_date = first_timestamp.date()
                                            combined_time = GST.localize(datetime.combine(first_timestamp_date, calc_result.time().replace(second=0, microsecond=0)))
                                            scheduled_at = self._normalize_datetime(combined_time)
                                            logger.info(f"Combined first timestamp date with schedule_calculator time: {scheduled_at}")
                                        elif started_at:
                                            if started_at.tzinfo is None:
                                                started_at_gst = GST.localize(started_at)
                                            else:
                                                started_at_gst = started_at.astimezone(GST)
                                            first_timestamp_date = started_at_gst.date()
                                            combined_time = GST.localize(datetime.combine(first_timestamp_date, calc_result.time().replace(second=0, microsecond=0)))
                                            scheduled_at = self._normalize_datetime(combined_time)
                                            logger.info(f"Combined started_at date with schedule_calculator time: {scheduled_at}")
                                        else:
                                            scheduled_at = self._normalize_datetime(calc_result)
                                    except Exception as e:
                                        logger.warning(f"Error combining date with schedule_calculator result: {e}, using as-is")
                                        scheduled_at = self._normalize_datetime(calc_result)
                                else:
                                    scheduled_at = self._normalize_datetime(calc_result)
                                
                                logger.info(f"Extracted time from conversation using schedule_calculator: {scheduled_at}")
                            except Exception as e:
                                logger.warning(f"Could not parse extracted time '{normalized_extracted_time}': {e}")
                                scheduled_at = None
                    else:
                        logger.warning("Could not extract time from conversation. Falling back to stage-based timeline.")
                        # Could not compute exact scheduled_at even though time phrase existed; leave scheduled_at None
                        scheduled_at = None
                
                # Calculate buffer_until (scheduled_at + 15 minutes)
                buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15)) if scheduled_at else None
            else:
                # No explicit CONFIRMED time phrase mentioned by Gemini - try one final extraction attempt
                # But skip if time was mentioned but not confirmed (uncertain/question)
                if not time_is_confirmed and scheduled_at_str:
                    logger.info(f"Time mentioned but not confirmed ('{scheduled_at_str}'), skipping final extraction - will use first transcription time with schedule_calculator date")
                    use_glinks_scheduler = True
                    scheduled_at = None
                    buffer_until = None
                else:
                    # This is a production safety net to catch any time mentions that might have been missed
                    logger.info(f"No explicit time from Gemini, attempting final extraction from conversation text...")
                    logger.debug(f"Conversation text sample (first 500 chars): {conversation_text[:500] if conversation_text else 'None'}")
                    final_extracted_time = extract_time_from_conversation(conversation_text)
                    if final_extracted_time:
                        logger.info(f"Final extraction successful! Found time: '{final_extracted_time}'")
                        scheduled_at_str = final_extracted_time
                        # Try to calculate scheduled_at using the extracted time
                        scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time, conversation_text, transcripts, started_at)
                        if scheduled_at:
                            buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                            explicit_time_mentioned = True
                            logger.info(f"Successfully calculated scheduled_at from final extraction: {scheduled_at}")
                        else:
                            # Try direct calculation if schedule_calculator fails
                            # CRITICAL: Normalize extracted time first (convert written numbers to numeric)
                            try:
                                normalized_final_time = self.normalize_time_string(final_extracted_time)
                            except Exception as e:
                                logger.warning(f"Error normalizing final extracted time '{final_extracted_time}': {e}, using as-is")
                                normalized_final_time = final_extracted_time
                            minutes_match = re.search(r'(\d+)\s*(?:mins?|minutes?)', normalized_final_time.lower())
                            if minutes_match:
                                try:
                                    minutes = int(minutes_match.group(1))
                                    # Validate reasonable range
                                    if 1 <= minutes <= 10080:  # 1 minute to 7 days
                                        scheduled_at = self._normalize_datetime(reference_time + timedelta(minutes=minutes))
                                        buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                                        explicit_time_mentioned = True
                                        logger.info(f"Calculated scheduled_at using direct timedelta: {scheduled_at}")
                                    else:
                                        logger.warning(f"Extracted time out of reasonable range: {minutes} minutes")
                                        scheduled_at = None
                                        buffer_until = None
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Could not parse minutes from extracted time: {e}")
                                    scheduled_at = None
                                    buffer_until = None
                            else:
                                scheduled_at = None
                                buffer_until = None
                    
                    # If still no time found after all attempts, or time is not confirmed, delegate to Glinks scheduler
                if not explicit_time_mentioned or not time_is_confirmed or not scheduled_at:
                    if explicit_time_mentioned and not time_is_confirmed:
                        logger.info(f"Time mentioned but not confirmed (uncertain/question), will combine first transcription time with schedule_calculator date for call {call_id}")
                    else:
                        logger.warning(f"No explicit confirmed time found after all extraction attempts for call {call_id}")
                        logger.info(f"Conversation text length: {len(conversation_text) if conversation_text else 0} chars")
                        logger.debug(f"Full conversation text: {conversation_text}")
                    logger.info(f"Delegating scheduling to Glinks scheduler (will combine first transcription time with schedule_calculator date) for call {call_id}")
                    use_glinks_scheduler = True
                    scheduled_at = None
                    buffer_until = None
            
            # Get voice_id from agent_id using storage
            voice_id = await self.storage.get_voice_id_from_agent_id(agent_id)
            
            # Get retry_count and parent_booking_id from database
            # IMPORTANT: auto_followup and auto_consultation are treated separately
            # Each booking_type maintains its own retry_count sequence starting from 0
            # parent_booking_id always points to the call_id from metadata of the original booking 
            # (where parent_booking_id IS NULL) for that lead_id and booking_type combination
            if is_fresh_call:
                # This is a new call_id - check if it's a follow-up call for the same lead_id and booking_type
                if lead_id and booking_type:
                    try:
                        # Check if there are existing bookings for this lead_id and booking_type combination
                        existing_bookings_count = await self.storage.count_bookings_by_lead_id_and_booking_type(lead_id, booking_type)
                        
                        if existing_bookings_count > 0:
                            # This is a follow-up call for the same lead_id and booking_type
                            # Get the maximum retry_count for this lead_id and booking_type combination and add 1
                            max_retry_count = await self.storage.get_max_retry_count_by_lead_id_and_booking_type(lead_id, booking_type)
                            retry_count = max_retry_count + 1
                            
                            # Find the original booking for this lead_id and booking_type (where parent_booking_id IS NULL)
                            # Use the call_id from its metadata as parent_booking_id for all subsequent bookings
                            original_booking = await self.storage.get_original_booking_by_lead_id_and_booking_type(lead_id, booking_type)
                            if original_booking:
                                original_booking_call_id = original_booking.get('call_id')  # Get call_id from metadata
                                if original_booking_call_id:
                                    parent_booking_id = original_booking_call_id
                                    logger.info(f"Follow-up call for lead_id {lead_id} and booking_type {booking_type} - {existing_bookings_count} existing booking(s)")
                                    logger.info(f"Maximum retry_count for this lead_id and booking_type: {max_retry_count}")
                                    logger.info(f"retry_count = {retry_count} (max_retry_count + 1)")
                                    logger.info(f"parent_booking_id = {parent_booking_id} (call_id from metadata of first booking where parent_booking_id IS NULL)")
                                else:
                                    parent_booking_id = None
                                    logger.warning(f"Original booking found but no call_id in metadata, setting parent_booking_id to NULL")
                            else:
                                parent_booking_id = None
                                logger.warning(f"Found {existing_bookings_count} booking(s) but could not find original booking")
                        else:
                            # This is the first booking for this lead_id and booking_type combination
                            retry_count = 0
                            parent_booking_id = None
                            logger.info(f"First booking for lead_id {lead_id} and booking_type {booking_type}")
                            logger.info(f"retry_count = 0, parent_booking_id = NULL")
                    except Exception as e:
                        logger.warning(f"Error checking for existing bookings by lead_id and booking_type: {e}. Treating as fresh booking.")
                        retry_count = 0
                        parent_booking_id = None
                else:
                    # No lead_id or booking_type - treat as fresh booking
                    retry_count = 0
                    parent_booking_id = None
                    logger.info(f"Fresh booking - new call_id {original_call_id}, lead_id={lead_id}, booking_type={booking_type}")
                    logger.info(f"retry_count = 0, parent_booking_id = NULL")
            else:
                # Duplicate call_id detected - a booking already exists for this call_id
                # Even for duplicates, retry_count should be based on lead_id and booking_type, not call_id
                # For duplicate call_ids, we still need to get parent_booking_id from the original booking
                # (where parent_booking_id IS NULL) for the same lead_id and booking_type
                try:
                    # Get the original booking for this call_id to find lead_id
                    original_booking_for_call_id = await self.storage.get_booking_by_call_id_in_metadata(original_call_id)
                    if original_booking_for_call_id:
                        original_lead_id = original_booking_for_call_id.get('lead_id')
                        
                        if original_lead_id and booking_type:
                            # Count existing bookings for this lead_id and booking_type combination (not by call_id)
                            existing_bookings_count = await self.storage.count_bookings_by_lead_id_and_booking_type(original_lead_id, booking_type)
                            
                            if existing_bookings_count > 0:
                                # Get max retry_count for this lead_id and booking_type combination (not by call_id)
                                max_retry_count = await self.storage.get_max_retry_count_by_lead_id_and_booking_type(original_lead_id, booking_type)
                                retry_count = max_retry_count + 1
                                
                                # Find the original booking for this lead_id and booking_type (where parent_booking_id IS NULL)
                                # Use the call_id from its metadata as parent_booking_id
                                original_booking_by_type = await self.storage.get_original_booking_by_lead_id_and_booking_type(original_lead_id, booking_type)
                                if original_booking_by_type:
                                    original_call_id_from_metadata = original_booking_by_type.get('call_id')
                                    if original_call_id_from_metadata:
                                        parent_booking_id = original_call_id_from_metadata
                                        logger.warning(f"Duplicate call_id {original_call_id} detected, but retry_count based on lead_id and booking_type")
                                        logger.info(f"Existing bookings for lead_id {original_lead_id} and booking_type {booking_type}: {existing_bookings_count}")
                                        logger.info(f"Maximum retry_count for this lead_id and booking_type: {max_retry_count}")
                                        logger.info(f"retry_count = {retry_count} (max_retry_count + 1)")
                                        logger.info(f"parent_booking_id = {parent_booking_id} (call_id from metadata of original booking where parent_booking_id IS NULL)")
                                    else:
                                        parent_booking_id = None
                                        logger.warning(f"Original booking found but no call_id in metadata, setting parent_booking_id to NULL")
                                else:
                                    parent_booking_id = None
                                    logger.warning(f"Could not find original booking for lead_id {original_lead_id} and booking_type {booking_type}")
                            else:
                                # No bookings found for this booking_type - treat as first booking for this type
                                retry_count = 0
                                parent_booking_id = None
                                logger.info(f"Duplicate call_id but first booking for booking_type {booking_type}, setting retry_count=0, parent_booking_id=NULL")
                        else:
                            # Fallback: treat as fresh booking
                            retry_count = 0
                            parent_booking_id = None
                            logger.warning(f"Duplicate call_id {original_call_id} detected, but missing lead_id or booking_type, treating as fresh booking")
                    else:
                        # Fallback: if original booking not found, treat as fresh booking
                        retry_count = 0
                        parent_booking_id = None
                        logger.warning(f"Could not find original booking for call_id {original_call_id}, treating as fresh booking")
                except Exception as e:
                    logger.warning(f"Error getting retry_count for duplicate call_id {original_call_id}: {e}. Treating as fresh booking.")
                    retry_count = 0
                    parent_booking_id = None
            
            logger.info(f"Final retry count for call_id {original_call_id}: {retry_count}")

            # If no time was confirmed and use_glinks_scheduler is True, combine first transcription time with schedule_calculator date
            if use_glinks_scheduler and not scheduled_at and booking_type:
                try:
                    logger.info("No time confirmed - combining first transcription time with schedule_calculator date")
                    
                    # Get the time from first transcription timestamp
                    first_transcript_timestamp = self.extract_first_timestamp(transcripts)
                    if first_transcript_timestamp:
                        first_time = first_transcript_timestamp.time()  # Extract just the time
                        logger.info(f"Extracted time from first transcription: {first_time}")
                        
                        # Get the date from schedule_calculator based on booking_type and retry_count
                        def _get_schedule_date():
                            calculator = ScheduleCalculator()
                            
                            # Determine outcome based on booking_type
                            if booking_type == "auto_followup":
                                outcome = "callback_requested"
                                outcome_details = {}  # No specific time - will use default schedule
                            elif booking_type == "auto_consultation":
                                outcome = "meeting_booked"
                                outcome_details = {}  # No specific time - will use default schedule
                            else:
                                outcome = "callback_requested"
                                outcome_details = {}
                            
                            # Calculate next call date using schedule calculator
                            # Pass lead_info with retry_count to influence the schedule
                            lead_info = {
                                "retry_count": retry_count,
                                "stage": 1  # Default stage
                            }
                            schedule_result = calculator.calculate_next_call(outcome, outcome_details, lead_info)
                            return schedule_result
                        
                        schedule_datetime = await asyncio.to_thread(_get_schedule_date)
                        
                        if schedule_datetime:
                            # Extract date from schedule_calculator result
                            schedule_date = schedule_datetime.date()
                            logger.info(f"Got date from schedule_calculator: {schedule_date}")
                            
                            # Combine schedule_calculator date with first transcription time
                            combined_datetime = GST.localize(datetime.combine(schedule_date, first_time.replace(second=0, microsecond=0)))
                            scheduled_at = self._normalize_datetime(combined_datetime)
                            buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                            use_glinks_scheduler = False  # We've scheduled it, so don't delegate to Glinks
                            
                            logger.info(f"Combined schedule_calculator date ({schedule_date}) with first transcription time ({first_time}) = {scheduled_at}")
                        else:
                            logger.warning("schedule_calculator returned None, cannot combine with first transcription time")
                    else:
                        logger.warning("No first transcription timestamp found, cannot extract time for combination")
                except Exception as e:
                    logger.error(f"Error combining first transcription time with schedule_calculator date: {e}", exc_info=True)

            # Ensure buffer_until is always calculated if scheduled_at exists
            # This is a safety check to ensure buffer_until is set correctly even if code paths are missed
            if scheduled_at and not buffer_until:
                buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                logger.info(f"Recalculated buffer_until from scheduled_at: {buffer_until}")
            elif scheduled_at and buffer_until:
                # Verify buffer_until is correct (should be scheduled_at + 15 minutes)
                expected_buffer = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                if abs((buffer_until - expected_buffer).total_seconds()) > 60:  # More than 1 minute difference
                    logger.warning(f"buffer_until ({buffer_until}) doesn't match expected ({expected_buffer}), recalculating")
                    buffer_until = expected_buffer
            
            logger.info(f"scheduled_at: {scheduled_at}, buffer_until: {buffer_until}")
            
            # Create booking data (always create, even if no booking found)
            # Generate new UUID for booking id (auto-generated)
            booking_id = str(uuid.uuid4())
            
            booking_data = {
                "id": booking_id,  # Auto-generated UUID
                "tenant_id": tenant_id,  # Already converted to string above
                "lead_id": lead_id,  # Already converted to string above
                "assigned_user_id": initiated_by_user_id,  # Already converted to string above
                "booking_type": booking_type,  # Can be null if no booking found
                "booking_source": ("system" if scheduled_at else ("glinks" if use_glinks_scheduler else None)),  # "system" only if scheduled_at is filled, otherwise delegate to Glinks
                "scheduled_at": scheduled_at.strftime("%Y-%m-%d %H:%M:%S") if scheduled_at else None,
                "timezone": "GST",
                "status": "scheduled" if booking_type else None,  # Only scheduled if booking exists
                "call_result": None,
                "retry_count": retry_count,  # For fresh lead: 0. For follow-up: count of existing bookings for this lead_id
                "parent_booking_id": parent_booking_id,  # For fresh lead: NULL. For follow-up: call_id from metadata of original booking (where retry_count = 0, parent_booking_id IS NULL)
                "notes": None,
                "metadata": {
                    "call_id": original_call_id  # Store current call's call_id (voice_call_logs.id) in metadata
                } if original_call_id else {},  # Fresh lead: metadata = {"call_id": <current_call_id>}. Follow-up: metadata = {"call_id": <current_call_id>}
                "created_by": str(voice_id) if voice_id else None,
                "created_at": datetime.now(GST).replace(microsecond=0).isoformat(),
                "updated_at": datetime.now(GST).replace(microsecond=0).isoformat(),
                "is_deleted": False,
                "buffer_until": buffer_until.strftime("%Y-%m-%d %H:%M:%S") if buffer_until else None
            }
            
            return booking_data
            
        except Exception as e:
            logger.error(f"Error processing call log {call_log_id}: {e}", exc_info=True)
            return None
    
    def save_booking_json(self, booking_data: Dict, call_log_id: str) -> Optional[str]:
        """Save booking to JSON file - DISABLED in v2 refactor"""
        # JSON file saving disabled - data goes directly to database
        logger.debug(f"JSON file saving disabled for call_log_id={call_log_id}")
        return None
    
    async def list_calls(self, limit: Optional[int] = 100) -> List[Dict]:
        """List all calls from voice_call_logs (async)"""
        return await self.storage.list_calls(limit=limit)
    
    async def save_booking(self, booking_data: Dict) -> Dict:
        """
        Save booking to both JSON file and database
        
        Args:
            booking_data: Booking data dictionary
        
        Returns:
            Dictionary with save results
        """
        call_log_id = booking_data.get('parent_booking_id') or booking_data.get('id', 'unknown')
        results = {"json": None, "db": None, "errors": []}
        
        # Save to JSON file
        try:
            json_filepath = self.save_booking_json(booking_data, call_log_id)
            results["json"] = json_filepath
            logger.info(f"Saved booking to JSON: {json_filepath}")
        except Exception as e:
            error_msg = f"Failed to save JSON: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        # Save to database (only if required fields are present)
        try:
            # Check if required fields are present
            if not booking_data.get('lead_id'):
                error_msg = "Skipping database save: lead_id is required but is null"
                logger.warning(error_msg)
                results["errors"].append(error_msg)
            else:
                # Ensure lead is assigned to user before booking (required by DB trigger)
                lead_id = booking_data.get('lead_id')
                assigned_user_id = booking_data.get('assigned_user_id')
                if lead_id and assigned_user_id:
                    try:
                        from db.storage.leads import LeadStorage
                        lead_storage = LeadStorage()
                        lead_storage.assign_lead_to_user_if_unassigned(lead_id, assigned_user_id)
                    except Exception as e:
                        logger.warning(f"Could not assign lead to user: {e}")
                
                db_booking_id = await self.storage.save_booking(booking_data)
                results["db"] = db_booking_id
                logger.info(f"Saved booking to database: {db_booking_id}")
        except LeadBookingsStorageError as e:
            # This is expected when required fields are missing
            error_msg = f"Skipped database save: {e}"
            logger.warning(error_msg)
            results["errors"].append(error_msg)
        except Exception as e:
            error_msg = f"Failed to save to database: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
        
        return results
    
    async def process_all_calls(self) -> List[Dict]:
        """Process all calls and create bookings (no limit)"""
        # Get all calls without limit
        calls = await self.list_calls(limit=None)
        results = []
        
        for call in calls:
            call_id = call['id']
            logger.info(f"Processing call: {call_id}")
            
            booking_data = await self.process_call_log(call_id)
            if booking_data:
                save_results = await self.save_booking(booking_data)
                if save_results["json"] or save_results["db"]:
                    results.append({
                        "status": "success",
                        "call_id": call_id,
                        "json_file": save_results["json"],
                        "db_id": save_results["db"],
                        "errors": save_results["errors"]
                    })
                else:
                    results.append({
                        "status": "error",
                        "call_id": call_id,
                        "errors": save_results["errors"]
                    })
            else:
                results.append({"status": "skipped", "call_id": call_id, "reason": "No booking found"})
        
        return results


async def main():
    parser = argparse.ArgumentParser(description="Extract lead bookings from voice_call_logs")
    parser.add_argument("--call-id", type=str, help="Process specific call ID (UUID)")
    parser.add_argument("--list", action="store_true", help="List all call IDs from voice_call_logs table")
    parser.add_argument("--all", action="store_true", help="Process all calls")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of calls to list (default: 100)")
    
    args = parser.parse_args()
    
    extractor = LeadBookingsExtractor()
    
    if args.list:
        calls = await extractor.list_calls(limit=args.limit)
        print(f"\nFound {len(calls)} calls:")
        print("=" * 100)
        print(f"{'Row':<6} {'Call ID':<40} {'Lead ID':<40} {'Started At':<25} {'Agent ID':<40}")
        print("=" * 100)
        for idx, call in enumerate(calls, 1):
            call_id_short = call['id'][:36] + "..." if len(call['id']) > 36 else call['id']
            lead_id_short = (call['lead_id'][:36] + "..." if call['lead_id'] and len(call['lead_id']) > 36 else call['lead_id']) or "None"
            started_at_short = call['started_at'][:19] if call['started_at'] else "None"
            agent_id_short = (call['agent_id'][:36] + "..." if call['agent_id'] and len(call['agent_id']) > 36 else call['agent_id']) or "None"
            print(f"{idx:<6} {call_id_short:<40} {lead_id_short:<40} {started_at_short:<25} {agent_id_short:<40}")
        print("=" * 100)
        print(f"\nTo process a specific call, use:")
        print(f"  python lead_bookings_extractor.py --call-id <call_id>")
        print(f"\nExample:")
        if calls:
            print(f"  python lead_bookings_extractor.py --call-id {calls[0]['id']}")
    
    elif args.call_id:
        booking_data = await extractor.process_call_log(args.call_id)
        if booking_data:
            # Save to both JSON and database
            save_results = await extractor.save_booking(booking_data)
            
            print(f"\n[SUCCESS] Booking created:")
            print(f"  JSON file: {save_results['json']}")
            print(f"  Database ID: {save_results['db']}")
            if save_results['errors']:
                print(f"  Errors: {save_results['errors']}")
            print("\nBooking data:")
            print(json.dumps(booking_data, indent=2))
        else:
            print(f"\n[ERROR] No booking found for call {args.call_id}")
            print("This could mean:")
            print("  - No booking request mentioned in the conversation")
            print("  - Conversation doesn't contain 'call me after X mins' or 'book consultation'")
            print("\nCheck the logs for more details.")
    
    elif args.all:
        results = await extractor.process_all_calls()
        print(f"\nProcessed {len(results)} calls:")
        success = sum(1 for r in results if r['status'] == 'success')
        skipped = sum(1 for r in results if r['status'] == 'skipped')
        errors = sum(1 for r in results if r['status'] == 'error')
        print(f"  Success: {success}")
        print(f"  Skipped: {skipped}")
        print(f"  Errors: {errors}")
        
        # Show summary of save results
        json_saved = sum(1 for r in results if r.get('json_file'))
        db_saved = sum(1 for r in results if r.get('db_id'))
        print(f"\nStorage summary:")
        print(f"  JSON files created: {json_saved}")
        print(f"  Database records created: {db_saved}")
    
    else:
        parser.print_help()
    
    # Close connection pool when done
    await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
