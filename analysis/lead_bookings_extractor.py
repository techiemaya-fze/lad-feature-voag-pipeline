"""
Lead Bookings Extractor
Reads transcriptions from lad_dev.voice_call_logs and creates lead_bookings JSON files
"""

import os
import sys
import json
import asyncio
import re
import logging
import argparse
import uuid
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import pytz

# Add parent directory to path to allow imports when running as script
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Structured output client for guaranteed JSON responses - handle both module and script execution
try:
    # Try direct import first (when run as script from analysis directory or project root)
    from gemini_client import generate_with_schema_async, BOOKING_INFO_SCHEMA
except ImportError:
    try:
        # Try relative import (when run as module)
        from .gemini_client import generate_with_schema_async, BOOKING_INFO_SCHEMA
    except ImportError:
        # Try absolute import (when run from project root)
        from analysis.gemini_client import generate_with_schema_async, BOOKING_INFO_SCHEMA

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
    
    async def _call_gemini_structured(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 4096, max_retries: int = 3) -> Optional[Dict]:
        """Call Gemini API with structured output schema - guarantees proper JSON response"""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available")
            return None
        
        logger.debug(f"Calling Gemini API with structured output: temperature={temperature}, maxOutputTokens={max_output_tokens}")
        
        try:
            # Use structured output for guaranteed JSON response
            result = await generate_with_schema_async(
                prompt=prompt,
                schema=BOOKING_INFO_SCHEMA,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
            )
            
            if result:
                # Extract and log usage metadata if present
                if '_usage_metadata' in result:
                    usage = result.pop('_usage_metadata')
                    logger.info(f"Gemini usage for booking extraction: {usage}")
                
                logger.debug(f"Booking extraction structured response received")
                return result
            else:
                logger.warning("No result from structured generation")
                return None
            
        except Exception as e:
            logger.error(f"Gemini structured API exception: {str(e)}", exc_info=True)
            return None
    
    def extract_last_timestamp(self, transcriptions_data) -> Optional[datetime]:
        """
        Extract the last timestamp from transcriptions (UTC +00) and convert to GST
        Returns None if no timestamp found
        """
        # Handle NULL/None transcripts
        if transcriptions_data is None:
            logger.warning("Cannot extract last timestamp - transcriptions_data is NULL")
            return None
        
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
        # Handle NULL/None transcripts
        if transcriptions_data is None:
            logger.warning("Cannot extract first timestamp - transcriptions_data is NULL")
            return None
        
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
        # Handle NULL/None transcripts
        if transcriptions_data is None:
            logger.warning("Transcripts data is NULL/None")
            return ""
        
        # Handle empty string
        if isinstance(transcriptions_data, str) and not transcriptions_data.strip():
            logger.warning("Transcripts data is empty string")
            return ""
        
        if isinstance(transcriptions_data, dict):
            # Check for 'segments' key (test files format)
            if 'segments' in transcriptions_data and isinstance(transcriptions_data['segments'], list):
                conversation_log = transcriptions_data['segments']
                # Process ALL segments - ONLY use 'text' field, ignore 'intended_text'
                conversation_text = "\n".join([
                    f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('text', '')}"
                    for entry in conversation_log
                ])
                logger.debug(f"Parsed {len(conversation_log)} segments from transcripts")
                return conversation_text
            # Check for 'messages' key (production format)
            elif 'messages' in transcriptions_data and isinstance(transcriptions_data['messages'], list):
                conversation_log = transcriptions_data['messages']
                # Process ALL messages without any limit - ONLY use 'text' field, ignore 'intended_text'
                conversation_text = "\n".join([
                    f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('message', entry.get('text', ''))}"
                    for entry in conversation_log  # No limit - processes all entries
                ])
                logger.debug(f"Parsed {len(conversation_log)} messages from transcripts")
                return conversation_text
            elif any(key in transcriptions_data for key in ['role', 'speaker', 'message', 'text']):
                role = transcriptions_data.get('role') or transcriptions_data.get('speaker', 'Unknown')
                # CRITICAL: Only use 'message' or 'text', NEVER 'intended_text'
                message = transcriptions_data.get('message') or transcriptions_data.get('text', '')
                return f"{role.title()}: {message}"
            else:
                # If it's a complex dict structure, convert to JSON to preserve all data
                return json.dumps(transcriptions_data, ensure_ascii=False)
        elif isinstance(transcriptions_data, list):
            # Process ALL entries in the list without any limit - ONLY use 'text' field, ignore 'intended_text'
            conversation_text = "\n".join([
                f"{entry.get('role', entry.get('speaker', 'Unknown')).title()}: {entry.get('message', entry.get('text', ''))}"
                for entry in transcriptions_data  # No limit - processes all entries
            ])
            logger.debug(f"Parsed {len(transcriptions_data)} entries from transcripts list")
            return conversation_text
        else:
            # Convert to string to preserve all data
            return str(transcriptions_data)
    
    async def extract_booking_info(self, conversation_text: str) -> Dict:
        """Extract booking information using Gemini"""
        prompt = f"""Analyze this conversation and extract booking information.

CONVERSATION:
{conversation_text}

Extract:
1. booking_type:
   - "auto_consultation": User EXPLICITLY confirms booking/appointment with clear "yes/okay/sure/book it/confirmed/I'll be there"
   - "auto_followup": Callback request ("call me back", "call me tomorrow", "I'll call you"), declined, no confirmation, agent notes "without booking"
   - CRITICAL: If user asks agent to "call me back" or "call me tomorrow" or schedules a callback = auto_followup (NOT a booking)
   - CRITICAL: User saying "No thank you", "end the call", "not interested", "maybe later" = auto_followup (NOT a booking)
   - Ambiguous "That one" when agent asks "referring to plan?" = auto_followup
   - Default: auto_followup
   - Examples:
     * User: "Call me tomorrow at 4 PM" = auto_followup (callback)
     * User: "Yes, I'll attend the session at 4 PM" = auto_consultation (confirmed appointment)

2. scheduled_at:
   - Extract EXACT time phrase (e.g., "after 15 mins", "tomorrow 3 PM", "Sunday 11 AM")
   - If multiple times, extract LATEST confirmed time
   - ALWAYS include day when mentioned ("Friday noon" not "noon")
   - Return NULL if: User declines ("No thank you", "end the call", "not interested", "maybe later"), agent rejects, agent notes "without booking", no clear acceptance
   - Return NULL if: ONLY discussing PAST consultations/calls with NO mention of NEXT/future follow-up or consultation
   - CRITICAL: Look at ENTIRE conversation, especially the END. If user declines AFTER time mentioned, return NULL
   - Example: Agent asks "Sunday?" + User "That one" + Later user "end the call" = NULL (user declined at end)
   - Return phrase AS-IS, don't convert to datetime

3. student_grade: Extract from "grade 10", "class 11", "12th standard" - Return integer (9-12) or null

4. call_id: Extract if mentioned, else null

5. has_future_discussion: true if conversation mentions NEXT/future followup/consultation/callback, false if only discussing past events

Respond in JSON:
{{{{
    "booking_type": "auto_followup" or "auto_consultation",
    "scheduled_at": "time phrase" or null,
    "student_grade": 10 or null,
    "call_id": "id" or null,
    "has_future_discussion": true or false
}}}}"""

        # Log the extraction request details being sent to Gemini
        logger.info(f"Calling Gemini API for booking extraction with structured output")
        logger.info(f"Conversation length: {len(conversation_text)} chars")
        
        # Using structured output - response is already a parsed dict
        booking_info = await self._call_gemini_structured(prompt, temperature=0.1, max_output_tokens=4096)
        if not booking_info:
            # Default to auto_followup if API fails
            return {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
        
        # Ensure booking_type is never null - default to auto_followup
        if not booking_info.get("booking_type") or booking_info.get("booking_type") not in ["auto_followup", "auto_consultation"]:
            booking_info["booking_type"] = "auto_followup"
        
        # Ensure student_grade is an integer or null
        if "student_grade" in booking_info and booking_info["student_grade"] is not None:
            try:
                booking_info["student_grade"] = int(booking_info["student_grade"])
            except (ValueError, TypeError):
                booking_info["student_grade"] = None
        
        # Sanitize "null" strings from Gemini
        for key in ["scheduled_at", "call_id"]:
            if isinstance(booking_info.get(key), str) and booking_info[key].lower() == "null":
                booking_info[key] = None

        # If no future discussion mentioned and scheduled_at is null, force auto_followup with default timing
        # This handles cases where conversation only discusses past consultations
        has_future = booking_info.get("has_future_discussion", True)
        if not has_future and not booking_info.get("scheduled_at"):
            logger.info("No future discussion detected - forcing auto_followup with grade 12 default")
            booking_info["booking_type"] = "auto_followup"
            booking_info["scheduled_at"] = "use_default_timing"  # Signal to use first timestamp + schedule calculator
            if not booking_info.get("student_grade"):
                booking_info["student_grade"] = 12  # Default to grade 12
        
        # Remove has_future_discussion from final result (internal use only)
        booking_info.pop("has_future_discussion", None)
        
        return booking_info
    
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
    
    async def calculate_scheduled_at(self, booking_type: str, scheduled_at_str: str, reference_time: datetime, conversation_text: Optional[str] = None, transcriptions_data: Optional[Dict] = None, started_at: Optional[datetime] = None, student_grade: Optional[int] = None) -> Optional[datetime]:
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
        # Include common misspellings to handle extraction errors
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            # Common misspellings and variations
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
            'febrauary': 2, 'febuary': 2, 'feburary': 2,  # Common misspellings of February
            'sepetmber': 9, 'septmber': 9, 'sepember': 9,  # Common misspellings of September
            'decmber': 12, 'decembre': 12,  # Common misspellings of December
        }
        
        date_patterns = [
            r'(?:saturday|sunday|monday|tuesday|wednesday|thursday|friday)[,\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|febrauary|febuary|feburary|sepetmber|septmber|sepember|decmber|decembre)\s+(\d{1,2})(?:st|nd|rd|th)?',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|febrauary|febuary|feburary|sepetmber|septmber|sepember|decmber|decembre)\s+(\d{1,2})(?:st|nd|rd|th)?',
            r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|febrauary|febuary|feburary|sepetmber|septmber|sepember|decmber|decembre)',
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
        
        # Check for relative month patterns (e.g., "this month 20", "next month 15")
        if not parsed_date:
            # Pattern: "this month [day]" or "next month [day]"
            this_month_match = re.search(r'this\s+month\s+(\d{1,2})(?:st|nd|rd|th)?', normalized_lower)
            next_month_match = re.search(r'next\s+month\s+(\d{1,2})(?:st|nd|rd|th)?', normalized_lower)
            
            if this_month_match:
                try:
                    day = int(this_month_match.group(1))
                    # Use current month and year
                    current_month = reference_time.month
                    current_year = reference_time.year
                    
                    # Validate day for current month
                    try:
                        parsed_date = datetime(current_year, current_month, day).date()
                        # If the date has already passed this month, use next month
                        if parsed_date < reference_time.date():
                            # Try next month
                            next_month = current_month + 1 if current_month < 12 else 1
                            next_year = current_year if current_month < 12 else current_year + 1
                            parsed_date = datetime(next_year, next_month, day).date()
                        logger.info(f"Found 'this month' date: {parsed_date} (from '{normalized_time_str}')")
                    except ValueError:
                        logger.debug(f"Invalid day {day} for current month {current_month}")
                except (ValueError, IndexError):
                    pass
            
            elif next_month_match:
                try:
                    day = int(next_month_match.group(1))
                    # Use next month
                    current_month = reference_time.month
                    current_year = reference_time.year
                    next_month = current_month + 1 if current_month < 12 else 1
                    next_year = current_year if current_month < 12 else current_year + 1
                    
                    # Validate day for next month
                    try:
                        parsed_date = datetime(next_year, next_month, day).date()
                        logger.info(f"Found 'next month' date: {parsed_date} (from '{normalized_time_str}')")
                    except ValueError:
                        logger.debug(f"Invalid day {day} for next month {next_month}")
                except (ValueError, IndexError):
                    pass
        
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
                    logger.info(f"Weekday calculation: reference_time={reference_time}, reference_time.date()={reference_time.date()}, current_weekday={current_weekday}, target={weekday_num}, days_until_raw={(weekday_num - current_weekday) % 7}")
                    if days_until == 0:  # Today is that weekday
                        days_until = 7 if is_next else 0  # If "next", use next week, else use today
                    elif is_next and days_until < 7:  # "next" explicitly mentioned
                        days_until = days_until  # Already correct
                    elif not is_next and days_until == 0:  # Today, use today
                        days_until = 0
                    elif not is_next:  # Not today, assume next occurrence
                        days_until = days_until if days_until > 0 else 7
                    
                    target_date = reference_time.date() + timedelta(days=days_until)
                    parsed_date = target_date
                    logger.info(f"Found weekday '{weekday_name}' (next={is_next}), ref_date={reference_time.date()}, days_until={days_until}, scheduling for: {parsed_date}")
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
        
        # PRIORITY 1: Check for absolute time patterns (e.g., "6:20", "6:20 PM", "6:20 GST", "18:20", "noon", "midnight")
        # These should be used as-is without relative calculation
        
        # First check for special times: noon and midnight
        # Use word boundary to avoid matching "afternoon" as "noon"
        if re.search(r'\bnoon\b', normalized_lower) or '12 noon' in normalized_lower:
            hour, minute = 12, 0
            if parsed_date:
                target_time = GST.localize(datetime.combine(parsed_date, datetime.min.time().replace(hour=hour, minute=minute, second=0, microsecond=0)))
                logger.info(f"Found 'noon' with parsed date {parsed_date}, scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
            elif is_tomorrow_mentioned:
                target_time_today = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                target_time = target_time_today + timedelta(days=1)
                logger.info(f"Found 'noon' with 'tomorrow', scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
            elif is_today_mentioned:
                target_time = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                logger.info(f"Found 'noon' with 'today', scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
        
        # Use word boundary to avoid matching "midnight" in other contexts
        if re.search(r'\bmidnight\b', normalized_lower) or '12 midnight' in normalized_lower:
            hour, minute = 0, 0
            if parsed_date:
                target_time = GST.localize(datetime.combine(parsed_date, datetime.min.time().replace(hour=hour, minute=minute, second=0, microsecond=0)))
                logger.info(f"Found 'midnight' with parsed date {parsed_date}, scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
            elif is_tomorrow_mentioned:
                target_time_today = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                target_time = target_time_today + timedelta(days=1)
                logger.info(f"Found 'midnight' with 'tomorrow', scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
            elif is_today_mentioned:
                target_time = reference_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                logger.info(f"Found 'midnight' with 'today', scheduling for: {target_time}")
                return self._normalize_datetime(target_time)
        
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
        # For relative times, use LAST TIMESTAMP (reference_time) as the base - when conversation ended
        direct_minutes_match = re.search(r'(?:after|in|within)\s+(\d+)\s*(?:mins?|minutes?)', normalized_time_str.lower())
        if direct_minutes_match:
            try:
                minutes = int(direct_minutes_match.group(1))
                # Validate reasonable range (1 minute to 7 days = 10080 minutes)
                if 1 <= minutes <= 10080:
                    # For relative times, use LAST TIMESTAMP (reference_time) as the base
                    # This ensures "after 10 minutes" means 10 minutes after the conversation ended
                    base_time = reference_time
                    logger.info(f"Using last transcription timestamp (reference_time) as base for relative time: {base_time}")
                    
                    # Calculate relative time from base_time (last timestamp)
                    if is_tomorrow_mentioned:
                        # User said "tomorrow after X minutes" - schedule for tomorrow
                        calculated_time = base_time + timedelta(days=1, minutes=minutes)
                        logger.info(f"Relative time '{normalized_time_str}' with 'tomorrow' mentioned: {base_time} + 1 day + {minutes} minutes = {calculated_time}")
                    else:
                        # Normal relative time - add minutes to base_time (last timestamp)
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
                return calculator.calculate_next_call(outcome, outcome_details, None, reference_time, allow_outside_hours=True)
            
            scheduled_at = await asyncio.to_thread(_calculate)
            
            # CRITICAL: Only combine with first timestamp for RELATIVE times (not absolute times)
            # Check if this is a relative time pattern (if absolute time, it would have been caught earlier)
            is_relative_time = direct_minutes_match is not None or any(keyword in normalized_time_str.lower() for keyword in ['after', 'in', 'within', 'minutes', 'mins'])
            is_absolute_time = bool(re.search(r'\b\d{1,2}[:.]\d{2}', normalized_time_str))  # Contains time pattern like "7:45"
            
            if scheduled_at:
                # CRITICAL FIX: If we have a parsed_date (from specific date like "December 25"), always use it
                # This ensures that explicit dates mentioned in conversation are properly honored
                if parsed_date:
                    # Extract the correct time from the schedule_calculator or from the original string
                    schedule_time = scheduled_at.time()
                    
                    # CRITICAL FIX: If schedule_calculator adjusted the time (e.g., 12 AM -> 10 AM due to working hours),
                    # but we have explicit time in the original string, extract it directly
                    time_patterns = [
                        r'\b(\d{1,2})\s*(?:am|a\.m\.)\b',  # "12 AM", "12 am"  
                        r'\b(\d{1,2})\s*(?:pm|p\.m\.)\b',  # "3 PM", "3 pm"
                        r'\b(\d{1,2}):(\d{2})\s*(?:am|pm|a\.m\.|p\.m\.)\b',  # "3:30 PM"
                        r'\b(\d{1,2})\s*(?:in the |in |o\'?clock )?(morning|am)\b',  # "10 in the morning", "10 morning", "10 o'clock morning"
                        r'\b(\d{1,2})\s*(?:in the |in |o\'?clock )?(afternoon|evening|pm)\b',  # "3 in the afternoon", "6 evening"
                    ]
                    
                    original_hour = None
                    original_minute = 0
                    for pattern in time_patterns:
                        time_match = re.search(pattern, scheduled_at_str, re.IGNORECASE)
                        if time_match:
                            original_hour = int(time_match.group(1))
                            if len(time_match.groups()) > 1 and time_match.group(2) and time_match.group(2).isdigit():  # Has minute component (not text like "morning")
                                original_minute = int(time_match.group(2))
                            
                            # Handle AM/PM conversion
                            matched_text = time_match.group(0).lower()
                            if 'pm' in matched_text or 'afternoon' in matched_text or 'evening' in matched_text:
                                if original_hour != 12:  # 12 PM stays 12
                                    original_hour += 12
                            elif 'am' in matched_text or 'morning' in matched_text:
                                if original_hour == 12:  # 12 AM becomes 0
                                    original_hour = 0
                            
                            logger.info(f"Extracted explicit time from original string '{scheduled_at_str}': {original_hour:02d}:{original_minute:02d}")
                            break
                    
                    # Use the extracted time if we found it, otherwise try schedule_calculator's time, then transcription time
                    if original_hour is not None:
                        final_time = time(original_hour, original_minute, 0)
                        combined_datetime = GST.localize(datetime.combine(parsed_date, final_time))
                        logger.info(f"Using explicit parsed date and time: {parsed_date} at {final_time} = {combined_datetime}")
                    else:
                        # Check if schedule_calculator extracted a meaningful time (not default 10 AM, or has time context words, or hour numbers mentioned)
                        has_time_context = 'morning' in normalized_time_str.lower() or 'afternoon' in normalized_time_str.lower() or 'evening' in normalized_time_str.lower()
                        has_hour_mention = any(str(h) in scheduled_at_str for h in range(1, 24))
                        is_not_default_10am = schedule_time.hour != 10 or schedule_time.minute != 0
                        
                        logger.info(f"Time extraction check: schedule_time={schedule_time}, has_time_context={has_time_context}, has_hour_mention={has_hour_mention}, is_not_default_10am={is_not_default_10am}")
                        
                        if has_time_context or has_hour_mention or is_not_default_10am:
                            # Use schedule_calculator's extracted time
                            combined_datetime = GST.localize(datetime.combine(parsed_date, schedule_time.replace(second=0, microsecond=0)))
                            logger.info(f"Using explicit parsed date with schedule_calculator extracted time: {parsed_date} at {schedule_time} = {combined_datetime}")
                        else:
                            # Use first transcription timestamp time (conversation start time) for future dates
                            if transcriptions_data:
                                first_timestamp = self.extract_first_timestamp(transcriptions_data)
                                if first_timestamp:
                                    transcription_time = first_timestamp.time()
                                    combined_datetime = GST.localize(datetime.combine(parsed_date, transcription_time.replace(second=0, microsecond=0)))
                                    logger.info(f"Using explicit parsed date with first transcription time: {parsed_date} at {transcription_time} = {combined_datetime}")
                                elif reference_time:
                                    transcription_time = reference_time.time()
                                    combined_datetime = GST.localize(datetime.combine(parsed_date, transcription_time.replace(second=0, microsecond=0)))
                                    logger.info(f"Using explicit parsed date with reference time: {parsed_date} at {transcription_time} = {combined_datetime}")
                                else:
                                    # Final fallback to schedule_calculator time if no transcription time
                                    combined_datetime = GST.localize(datetime.combine(parsed_date, schedule_time.replace(second=0, microsecond=0)))
                                    logger.info(f"Using explicit parsed date with schedule_calculator time (no transcription time): {parsed_date} with {schedule_time} = {combined_datetime}")
                            elif reference_time:
                                transcription_time = reference_time.time()
                                combined_datetime = GST.localize(datetime.combine(parsed_date, transcription_time.replace(second=0, microsecond=0)))
                                logger.info(f"Using explicit parsed date with reference time: {parsed_date} at {transcription_time} = {combined_datetime}")
                            else:
                                # Final fallback to schedule_calculator time if no transcription time
                                combined_datetime = GST.localize(datetime.combine(parsed_date, schedule_time.replace(second=0, microsecond=0)))
                                logger.info(f"Using explicit parsed date with schedule_calculator time (no transcription time): {parsed_date} with {schedule_time} = {combined_datetime}")
                    
                    return self._normalize_datetime(combined_datetime)
                
                # CRITICAL FIX: Check if ANY explicit date was mentioned (valid or invalid)
                # If so, NEVER fall back to same-day scheduling - ALWAYS use Grade 12 timeline instead
                
                # Comprehensive detection of explicit date mentions
                contains_month_name = any(month in normalized_time_str.lower() for month in [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december',
                    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 
                    'sep', 'oct', 'nov', 'dec', 'febrauary', 'febuary', 'feburary',
                    'sepetmber', 'septmber', 'sepember', 'decmber', 'decembre'
                ])
                
                # Check for various date patterns
                contains_day_number = bool(re.search(r'\b\d{1,2}(?:st|nd|rd|th)?\b', normalized_time_str))
                contains_date_pattern = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', normalized_time_str))  # "25/12" or "25/12/2026"
                contains_weekday = any(day in normalized_time_str.lower() for day in [
                    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
                ])
                
                # Check for relative date words that indicate future dates
                contains_future_indicators = any(word in normalized_time_str.lower() for word in [
                    'next', 'coming', 'upcoming', 'later this', 'end of', 'this month', 'next month'
                ])
                
                # Check for month-related words (but exclude successfully parsed relative month patterns)
                contains_month_word = any(word in normalized_time_str.lower() for word in [
                    'month', 'week', 'year'
                ])
                
                # Check if this was a relative month pattern that was successfully parsed
                is_parsed_relative_month = (
                    parsed_date and 
                    (re.search(r'this\s+month\s+\d{1,2}', normalized_time_str.lower()) or
                     re.search(r'next\s+month\s+\d{1,2}', normalized_time_str.lower()))
                )
                
                # If ANY explicit date is mentioned but failed to parse properly, use Grade 12 timeline
                # BUT exclude cases where relative month patterns were successfully parsed
                # OR where schedule_calculator successfully parsed the date
                explicit_date_mentioned = (
                    (contains_month_name and contains_day_number) or  # "march 31", "febrauary 31"
                    contains_date_pattern or  # "31/03", "31/3/2026"
                    (contains_weekday and contains_future_indicators) or  # "next monday"
                    (contains_month_word and contains_day_number and not is_parsed_relative_month) or  # "this month 31", but exclude successfully parsed ones
                    contains_month_name  # Just month name might indicate date intent
                )
                
                # Check if schedule_calculator successfully parsed the date
                # If scheduled_at is more than 2 days in the future, consider it successfully parsed
                schedule_calc_parsed_date = False
                if scheduled_at:
                    days_diff = (scheduled_at.date() - reference_time.date()).days
                    if days_diff > 2:  # More than 2 days ahead means a specific date was parsed
                        schedule_calc_parsed_date = True
                        # Extract the parsed date from schedule_calculator result
                        parsed_date = scheduled_at.date()
                        logger.info(f"schedule_calculator successfully parsed date: {parsed_date} ({days_diff} days ahead)")
                
                if explicit_date_mentioned and not parsed_date and not schedule_calc_parsed_date:
                    logger.warning(f"Explicit date mentioned in '{normalized_time_str}' but failed to parse - ALWAYS using Grade 12 timeline (NEVER same-day)")
                    # Extract just the time component and use grade-based timeline for the date
                    schedule_time = scheduled_at.time()
                    calculator = ScheduleCalculator()
                    lead_info = {"student_grade": 12}  # Default grade
                    # Use grade 12 timeline to get a proper future date (minimum 3 working days)
                    timeline_result = calculator.calculate_grade12_timeline(lead_info, reference_time)
                    final_datetime = GST.localize(datetime.combine(timeline_result.date(), schedule_time.replace(second=0, microsecond=0)))
                    logger.info(f"Applied Grade 12 timeline for failed explicit date parsing: {final_datetime} (ensuring future date, never same day)")
                    return self._normalize_datetime(final_datetime)
                
                # If absolute time is explicitly mentioned without specific date, use schedule_calculator result
                if is_absolute_time and not is_relative_time:
                    logger.info(f"Absolute time detected in '{normalized_time_str}', using schedule_calculator result as-is: {scheduled_at}")
                    return self._normalize_datetime(scheduled_at)
                
                # PRIORITY: If parsed_date exists (from weekday/tomorrow/today), use it with schedule_calculator time
                # Otherwise, combine with first timestamp date for relative times
                try:
                    target_date = None
                    
                    # Check if schedule_calculator already provided a valid future date
                    # For absolute time phrases like "on a weekend", "tomorrow", "next Monday", etc.
                    calc_date = scheduled_at.date()
                    today = datetime.now(GST).date()
                    
                    # Check if the phrase contains explicit time (like "3 PM", "10:00", "at 5")
                    time_pattern = r'\b(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2}|at\s+\d{1,2})\b'
                    has_explicit_time = bool(re.search(time_pattern, scheduled_at_str))
                    
                    # Check if a specific month/day was mentioned (like "March 15" or "fifteenth March")
                    # These should be treated as specific dates, not as requests for timeline-based scheduling
                    has_specific_month_day = bool(re.search(r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)', scheduled_at_str.lower()))
                    
                    # If schedule_calculator provided a future date or specific date
                    # Include "after" for phrases like "after 2 days" which are explicit date requests
                    if calc_date > today or any(phrase in scheduled_at_str.lower() for phrase in ['weekend', 'tomorrow', 'next', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'after']):
                        
                        # Check if this is a relative date request like "after X days" which should be treated as explicit
                        is_relative_date_request = any(rel_phrase in scheduled_at_str.lower() for rel_phrase in ['after', 'in'])
                        
                        if has_explicit_time or is_relative_date_request or has_specific_month_day:
                            # Explicit time mentioned OR relative date request like "after 2 days" OR specific month/day - use schedule calculator result as-is
                            # BUT: For specific month/day without time, use first timestamp time
                            if has_specific_month_day and not has_explicit_time:
                                # Specific month/day mentioned but no explicit time - use first timestamp time
                                if transcriptions_data:
                                    first_timestamp = self.extract_first_timestamp(transcriptions_data)
                                    if first_timestamp:
                                        target_time = first_timestamp.time()
                                        final_datetime = GST.localize(datetime.combine(calc_date, target_time.replace(second=0, microsecond=0)))
                                        logger.info(f"Schedule calculator provided specific month/day ({calc_date}) without explicit time - using first timestamp time: {final_datetime}")
                                        return self._normalize_datetime(final_datetime)
                            logger.info(f"Schedule calculator provided date ({calc_date}) with explicit time or relative date request for '{scheduled_at_str}' - using as-is")
                            return self._normalize_datetime(scheduled_at)
                        else:
                            # No explicit time - should follow Grade 12 timeline rules instead of using transcription time
                            logger.info(f"Schedule calculator provided date ({calc_date}) but no explicit time in '{scheduled_at_str}' - applying Grade 12 timeline rules")
                            
                            # NEVER use default 10 AM - use transcription timestamp or reject without explicit time
                            target_time = None
                            if transcriptions_data:
                                first_timestamp = self.extract_first_timestamp(transcriptions_data)
                                if first_timestamp:
                                    target_time = first_timestamp.time().replace(second=0, microsecond=0)
                                    logger.info(f"Using first timestamp time: {target_time} from transcription")
                            
                            # If no timestamp available, use Grade 12 timeline but with transcription time instead of 10 AM default
                            if target_time is None:
                                # Use the first timestamp from transcription for the time (not arbitrary 10 AM)
                                if reference_time:
                                    transcription_time = reference_time.time()
                                    logger.info(f"No explicit time mentioned - using transcription timestamp time: {transcription_time}")
                                    
                                    # Calculate appropriate timeline based on actual student grade (not hardcoded 12)
                                    calculator = ScheduleCalculator()
                                    actual_grade = student_grade if student_grade is not None else 12  # Default to 12 only if not extracted
                                    lead_info = {"student_grade": actual_grade}
                                    
                                    # Use appropriate timeline calculation based on grade
                                    if actual_grade is not None and actual_grade <= 11:
                                        # Use stage2_schedule which handles grades 9, 10, 11 specifically
                                        lead_info["stage2_start"] = reference_time
                                        lead_info["last_call_time"] = reference_time
                                        timeline_result = calculator.calculate_stage2_schedule(lead_info, reference_time)
                                        logger.info(f"Applied Grade {actual_grade} timeline date with transcription time")
                                    else:
                                        # Use Grade 12+ timeline for Grade 12 and above
                                        timeline_result = calculator.calculate_grade12_timeline(lead_info, reference_time)
                                        logger.info(f"Applied Grade {actual_grade}+ timeline date with transcription time")
                                    
                                    # Combine timeline date with transcription time (not 10 AM)
                                    final_date = timeline_result.date()
                                    final_datetime = GST.localize(datetime.combine(final_date, transcription_time))
                                    
                                    logger.info(f"Applied Grade 12 timeline date with transcription time: {final_datetime}")
                                    return self._normalize_datetime(final_datetime)
                                else:
                                    # Fallback if no reference time available
                                    actual_grade = student_grade if student_grade is not None else 12
                                    logger.info(f"No transcription timestamp available - using Grade {actual_grade} timeline default scheduling")
                                    calculator = ScheduleCalculator()
                                    lead_info = {"student_grade": actual_grade}
                                    
                                    if actual_grade is not None and actual_grade <= 11:
                                        lead_info["stage2_start"] = datetime.now(GST)
                                        lead_info["last_call_time"] = datetime.now(GST)
                                        timeline_result = calculator.calculate_stage2_schedule(lead_info, datetime.now(GST))
                                    else:
                                        timeline_result = calculator.calculate_grade12_timeline(lead_info, datetime.now(GST))
                                    return self._normalize_datetime(timeline_result)
                            
                            # Use schedule calculator for grade-based timeline on the target date
                            calculator = ScheduleCalculator()
                            
                            # Create lead info for timeline calculation using actual extracted grade
                            final_grade = student_grade if student_grade is not None else 12  # Default to 12 only if not extracted
                            lead_info = {"student_grade": final_grade}
                            
                            # Set the target date as stage2_start to calculate timeline from that date
                            target_datetime = GST.localize(datetime.combine(calc_date, target_time))
                            lead_info["stage2_start"] = target_datetime
                            lead_info["last_call_time"] = target_datetime
                            
                            # Calculate next call DATE using appropriate timeline based on grade
                            if final_grade is not None and final_grade <= 11:
                                # Use stage2_schedule for grades 9, 10, 11
                                timeline_result = calculator.calculate_stage2_schedule(lead_info, target_datetime)
                            else:
                                # Use Grade 12+ timeline
                                timeline_result = calculator.calculate_grade12_timeline(lead_info, target_datetime)
                            
                            # Use the calculated date but preserve the first timestamp's time
                            final_date = timeline_result.date()
                            final_datetime = GST.localize(datetime.combine(final_date, target_time))
                            
                            logger.info(f"Applied Grade {final_grade} timeline to target date ({calc_date}) using time {target_time} = {final_datetime}")
                            return self._normalize_datetime(final_datetime)
                    
                    # For relative time phrases (like "in 30 minutes"), combine with transcription date
                    # FIRST: Check if we already have a parsed_date from weekday/tomorrow/today
                    if parsed_date:
                        target_date = parsed_date
                        logger.info(f"Using parsed_date from weekday/tomorrow/today: {target_date}")
                    else:
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
                        
                        target_date = first_timestamp_date
                    
                    # If we have a target date, use it with the time from schedule_calculator
                    if target_date:
                        # Extract time (hour, minute) from schedule_calculator result
                        calc_hour = scheduled_at.hour
                        calc_minute = scheduled_at.minute
                        
                        # Combine target date with schedule_calculator time
                        combined_time = GST.localize(datetime.combine(target_date, scheduled_at.time().replace(second=0, microsecond=0)))
                        logger.info(f"Combined target date ({target_date}) with schedule_calculator time ({calc_hour:02d}:{calc_minute:02d}) = {combined_time}")
                        return self._normalize_datetime(combined_time)
                    else:
                        # No date available - use schedule_calculator result as-is
                        logger.warning(f"No date found, using schedule_calculator result as-is: {scheduled_at}")
                        return self._normalize_datetime(scheduled_at)
                except Exception as e:
                    logger.warning(f"Error combining date with schedule_calculator time: {e}, using schedule_calculator result as-is")
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
                            # If fallback_result has a time, use it, else use first timestamp's time
                            if fallback_result.time() != time(0, 0):
                                combined_time = GST.localize(datetime.combine(first_timestamp_date, fallback_result.time().replace(second=0, microsecond=0)))
                            else:
                                # Use first timestamp's time
                                first_timestamp = self.extract_first_timestamp(transcriptions_data)
                                if first_timestamp:
                                    combined_time = GST.localize(datetime.combine(first_timestamp_date, first_timestamp.time().replace(second=0, microsecond=0)))
                                else:
                                    combined_time = GST.localize(datetime.combine(first_timestamp_date, time(10, 0)))  # fallback, should not happen
                            logger.info(f"Combined first timestamp date ({first_timestamp_date}) with time = {combined_time}")
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
                logger.warning(f"Call log {call_log_id} not found in database - creating default booking with 12th grade timeline")
                # When call log doesn't exist, create a default booking using 12th grade timeline and current time
                current_time = self._normalize_datetime(datetime.now(GST))
                
                # ScheduleCalculator already imported at top of file
                calculator = ScheduleCalculator()
                final_grade = 12
                lead_info = {"student_grade": final_grade}
                
                # Calculate schedule date using grade 12 timeline
                schedule_date = calculator.calculate_grade12_timeline(lead_info, current_time)
                # Combine schedule date with current time
                scheduled_at = self._normalize_datetime(GST.localize(datetime.combine(schedule_date.date(), current_time.time().replace(microsecond=0))))
                buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                
                logger.info(f"Default booking scheduled_at: {scheduled_at}, buffer_until: {buffer_until}")
                
                # Create a booking entry for the missing call log
                booking_data = {
                    "id": str(uuid.uuid4()),
                    "tenant_id": "default-tenant",
                    "lead_id": str(uuid.uuid4()),
                    "assigned_user_id": "default-user",
                    "booking_type": "auto_followup",
                    "booking_source": "system",
                    "scheduled_at": scheduled_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "timezone": "GST",
                    "status": "scheduled",
                    "retry_count": 0,
                    "parent_booking_id": None,
                    "metadata": {"call_id": call_log_id, "reason": "missing_call_log"},
                    "buffer_until": buffer_until.strftime("%Y-%m-%d %H:%M:%S"),
                    "created_at": current_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "updated_at": current_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "is_deleted": False
                }
                
                return {
                    'status': 'success',
                    'booking_data': booking_data,
                    'call_id': call_log_id,
                    'message': 'Booking created for missing call log using 12th grade timeline'
                }
            
            call_id = call_data['id']
            tenant_id = call_data['tenant_id']
            lead_id = call_data['lead_id']
            transcripts = call_data['transcripts']
            initiated_by_user_id = call_data['initiated_by_user_id']
            agent_id = call_data['agent_id']
            started_at = call_data['started_at']
            call_status = call_data.get('status')  # Status from voice_call_logs table
            
            # Check for failed calls or NULL transcripts - special handling: auto_followup, grade 12 timeline, use started_at time
            is_failed_call = call_status and str(call_status).lower() in ['failed', 'declined', 'rejected', 'no_answer', 'not_interested', 'busy']
            
            if transcripts is None or is_failed_call:
                reason = "NULL transcripts" if transcripts is None else f"failed call (status: {call_status})"
                logger.warning(f"Special handling for {reason} in call {call_id} - using auto_followup, grade 12 timeline, and started_at time")
                booking_type = "auto_followup"
                # Use grade 12 timeline for date
                # ScheduleCalculator already imported at top of file
                calculator = ScheduleCalculator()
                # Use grade 12 if not present
                final_grade = 12
                lead_info = {"student_grade": final_grade}
                # Use started_at from call_data (should be UTC or aware)
                started_at_val = call_data.get('started_at')
                if not started_at_val:
                    logger.error("started_at is missing in call_data, cannot schedule")
                    return {
                        'status': 'skipped',
                        'reason': 'started_at_missing',
                        'call_id': str(call_id) if call_id else None,
                        'message': 'Cannot process call - started_at is missing in voice_call_logs table'
                    }
                # Parse started_at if string
                if isinstance(started_at_val, str):
                    started_at_dt = datetime.fromisoformat(started_at_val.replace('Z', '+00:00'))
                else:
                    started_at_dt = started_at_val
                # Convert to GST
                if started_at_dt.tzinfo is None:
                    started_at_dt = GST.localize(started_at_dt)
                else:
                    started_at_dt = started_at_dt.astimezone(GST)
                # Remove microseconds
                started_at_dt = started_at_dt.replace(microsecond=0)
                # Get date from grade 12 timeline
                schedule_date = calculator.calculate_grade12_timeline(lead_info, started_at_dt)
                # Combine date from schedule_date with time from started_at_dt
                combined_dt = GST.localize(datetime.combine(schedule_date.date(), started_at_dt.time().replace(microsecond=0)))
                scheduled_at = self._normalize_datetime(combined_dt)
                buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                logger.info(f"Scheduled_at ({reason}): {scheduled_at}, buffer_until: {buffer_until}")
                
                # Set variables for the rest of the processing
                conversation_text = f"No transcription available ({reason}). Auto-scheduled based on started_at time and grade 12 timeline."
                booking_info = {"booking_type": booking_type, "scheduled_at": None, "student_grade": final_grade, "call_id": None}
                explicit_time_mentioned = True
                time_is_confirmed = True
                use_glinks_scheduler = False
                special_handling_mode = True  # Skip normal time processing
                # Continue with normal processing flow instead of returning early
            else:
                # Normal transcription processing - initialize variables
                scheduled_at = None
                buffer_until = None
                booking_type = None
                explicit_time_mentioned = False
                time_is_confirmed = False
                use_glinks_scheduler = False
            
            # If lead_id is null, try to use call_id as fallback
            # (Some calls might not have lead_id assigned, but we still want to save the booking)
            if not lead_id:
                logger.warning(f"lead_id is null for call {call_id}, using call_id as fallback for lead_id")
                lead_id = call_id
            
            # Parse transcription (skip if already set for NULL transcripts)
            if not locals().get('conversation_text'):
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
                
                # Use 12th grade timeline with started_at time for empty/missing transcriptions
                reason = "empty/missing transcription"
                logger.warning(f"Special handling for {reason} in call {call_id} - using auto_followup, grade 12 timeline, and started_at time")
                booking_type = "auto_followup"
                
                # ScheduleCalculator already imported at top of file
                calculator = ScheduleCalculator()
                final_grade = 12
                lead_info = {"student_grade": final_grade}
                
                # Use started_at from call_data
                started_at_val = started_at
                if started_at_val:
                    # Parse started_at if string
                    if isinstance(started_at_val, str):
                        started_at_dt = datetime.fromisoformat(started_at_val.replace('Z', '+00:00'))
                    else:
                        started_at_dt = started_at_val
                    # Convert to GST
                    if started_at_dt.tzinfo is None:
                        started_at_dt = GST.localize(started_at_dt)
                    else:
                        started_at_dt = started_at_dt.astimezone(GST)
                    # Remove microseconds
                    started_at_dt = started_at_dt.replace(microsecond=0)
                    
                    # For missing/empty transcriptions, use Grade 12 timeline (2 days later) at started_at time
                    # Schedule 2 working days later regardless of whether call is today or previous day
                    next_day = started_at_dt + timedelta(days=2)
                    while not calculator.is_working_day(next_day):
                        next_day += timedelta(days=1)
                    scheduled_at = self._normalize_datetime(GST.localize(datetime.combine(next_day.date(), started_at_dt.time())))
                    logger.info(f"Grade 12 timeline: 2-day callback for {reason} at {started_at_dt.time()}: {scheduled_at}")
                    
                    buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                    logger.info(f"Scheduled_at ({reason}): {scheduled_at}, buffer_until: {buffer_until}")
                    
                    # Set variables for the rest of the processing
                    conversation_text = f"No meaningful transcription available ({reason}). Auto-scheduled based on started_at time and grade 12 timeline."
                    booking_info = {"booking_type": booking_type, "scheduled_at": None, "student_grade": final_grade, "call_id": None}
                    explicit_time_mentioned = True
                    time_is_confirmed = True
                    use_glinks_scheduler = False
                    special_handling_mode = True  # Skip normal time processing
                else:
                    # No started_at available, use current time
                    current_time = self._normalize_datetime(datetime.now(GST))
                    schedule_date = calculator.calculate_grade12_timeline(lead_info, current_time)
                    combined_dt = GST.localize(datetime.combine(schedule_date.date(), current_time.time().replace(microsecond=0)))
                    scheduled_at = self._normalize_datetime(combined_dt)
                    buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                    logger.info(f"Scheduled_at ({reason}, no started_at): {scheduled_at}, buffer_until: {buffer_until}")
                    
                    conversation_text = f"No meaningful transcription and no started_at available ({reason}). Auto-scheduled based on current time and grade 12 timeline."
                    booking_info = {"booking_type": booking_type, "scheduled_at": None, "student_grade": final_grade, "call_id": None}
                    explicit_time_mentioned = True
                    time_is_confirmed = True
                    use_glinks_scheduler = False
                    special_handling_mode = True
                # Set default conversation_text for processing
                if not conversation_text:
                    conversation_text = "No transcription available. User likely declined or did not answer."
            
            # Extract booking info using Gemini first (skip if already set for NULL transcripts)
            logger.info(f"Extracting booking info from conversation (length: {len(conversation_text)} chars)")
            if not locals().get('booking_info'):
                try:
                    booking_info = await self.extract_booking_info(conversation_text)
                    logger.info(f"Extracted booking info: {booking_info}")
                except Exception as e:
                    logger.error(f"Gemini API extraction failed: {e}, will use fallback extraction", exc_info=True)
                    # Fallback to empty booking info - will trigger regex extraction
                    booking_info = {"booking_type": "auto_followup", "scheduled_at": None, "student_grade": None, "call_id": None}
            else:
                logger.info(f"Using pre-set booking info for NULL transcripts: {booking_info}")
            
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

            # CRITICAL: If Gemini returned a relative time (like "after around two hours"), 
            # check if there's a specific time mentioned in the conversation that the user accepted.
            # If agent says "after 2 hours" but also mentions "1:59 PM" and user accepts, use the specific time.
            if scheduled_at_str and conversation_text:
                # Check if scheduled_at_str is a relative time
                is_relative_time = bool(re.search(r'(?:after|in|within|around)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*(?:mins?|minutes?|hours?|hrs?)', scheduled_at_str.lower()))
                
                logger.debug(f"Checking for specific time override: scheduled_at_str='{scheduled_at_str}', is_relative_time={is_relative_time}")
                logger.debug(f"Conversation text (first 500 chars): {conversation_text[:500]}")
                
                if is_relative_time:
                    # Try to extract a specific time from conversation text
                    # Look for patterns like "1:59 PM", "1:59", "at 1:59", "1 59", "one fifty-nine", etc.
                    specific_time_patterns = [
                        r'(?:at|around|about|by)\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "at 1:59 PM"
                        r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b',  # "1:59 PM"
                        r'\b(\d{1,2}):(\d{2})\s*(?:GST|IST|UTC)\b',  # "1:59 GST"
                        r'(?:call|ring|phone|reach|connect)\s+(?:you|me|him|her|them|us)?\s+(?:back\s+)?(?:at|around|about)?\s*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "call you at 1:59 PM"
                        # Pattern for times without AM/PM (e.g., "1:59") - infer PM for hours 1-11, AM for 12
                        r'(?:at|around|about|by|call|ring|phone)\s+(\d{1,2}):(\d{2})(?:\s|\.|$|,|;|\'|")(?!\s*(?:AM|PM|am|pm|GST|IST|UTC))',  # "at 1:59" or "call 1:59"
                        r'\b(\d{1,2}):(\d{2})(?:\s|\.|$|,|;|\'|")(?!\s*(?:AM|PM|am|pm|GST|IST|UTC|mins?|minutes?|hours?))',  # "1:59" standalone
                        # Pattern for times with space instead of colon (e.g., "1 59 PM", "at 1 59")
                        r'(?:at|around|about|by)\s+(\d{1,2})\s+(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "at 1 59 PM"
                        r'\b(\d{1,2})\s+(\d{2})\s*(AM|PM|am|pm)\b',  # "1 59 PM"
                        r'(?:call|ring|phone|reach|connect)\s+(?:you|me|him|her|them|us)?\s+(?:back\s+)?(?:at|around|about)?\s*(\d{1,2})\s+(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "call you at 1 59 PM"
                    ]
                    
                    conversation_lower = conversation_text.lower()
                    logger.debug(f"Searching for specific time patterns in conversation (length: {len(conversation_lower)} chars)")
                    
                    for idx, pattern in enumerate(specific_time_patterns):
                        match = re.search(pattern, conversation_lower, re.IGNORECASE)
                        if match:
                            logger.debug(f"Pattern #{idx+1} matched: '{pattern[:60]}...' -> '{match.group(0)}'")
                            hour = match.group(1)
                            minute = match.group(2)
                            ampm = match.group(3) if len(match.groups()) >= 3 and match.group(3) else None
                            
                            # Validate hour and minute
                            try:
                                hour_int = int(hour)
                                minute_int = int(minute)
                                
                                # If no AM/PM specified, infer based on hour
                                # Hours 1-11: likely PM (afternoon/evening callbacks)
                                # Hour 12: could be AM or PM, default to PM for callbacks
                                # Hours 13-23: 24-hour format, keep as-is
                                inferred_ampm = ampm
                                if not ampm:
                                    if 1 <= hour_int <= 11:
                                        inferred_ampm = "PM"  # Afternoon/evening callbacks are more common
                                    elif hour_int == 12:
                                        inferred_ampm = "PM"  # Default to PM for callbacks
                                    elif 13 <= hour_int <= 23:
                                        # 24-hour format, convert to 12-hour
                                        hour_int = hour_int - 12
                                        inferred_ampm = "PM"
                                    else:
                                        continue  # Invalid hour
                                
                                max_hour = 12 if inferred_ampm else 23
                                if 1 <= hour_int <= max_hour and 0 <= minute_int <= 59:
                                    # Build time string - ensure minute is 2 digits
                                    minute_str = f"{minute_int:02d}"
                                    if inferred_ampm:
                                        specific_time = f"{hour_int}:{minute_str} {inferred_ampm.upper()}"
                                    else:
                                        specific_time = f"{hour_int}:{minute_str}"
                                    
                                    # Check if GST/IST/UTC is in the match
                                    matched_text = match.group(0).upper()
                                    if 'GST' in matched_text:
                                        specific_time += " GST"
                                    elif 'IST' in matched_text:
                                        specific_time += " IST"
                                    elif 'UTC' in matched_text:
                                        specific_time += " UTC"
                                    
                                    # Check if this specific time appears AFTER the relative phrase in conversation
                                    # This indicates the agent mentioned it later and user likely accepted it
                                    relative_phrase_pos = conversation_lower.find(scheduled_at_str.lower())
                                    specific_time_pos = conversation_lower.find(match.group(0).lower())
                                    
                                    # If specific time appears after relative phrase, or if we can't determine order,
                                    # prefer the specific time (it's more precise)
                                    logger.debug(f"Found specific time '{specific_time}' at position {specific_time_pos}, relative phrase at position {relative_phrase_pos}")
                                    if specific_time_pos > relative_phrase_pos or relative_phrase_pos == -1:
                                        logger.info(f"Found specific time '{specific_time}' in conversation after relative phrase '{scheduled_at_str}' - using specific time instead")
                                        scheduled_at_str = specific_time
                                        break
                                    else:
                                        logger.debug(f"Found specific time '{specific_time}' but it appears before relative phrase - keeping relative time")
                            except (ValueError, IndexError, AttributeError) as e:
                                logger.debug(f"Error processing matched time pattern: {e}")
                                continue
                    
                    # Log if no specific time was found
                    if is_relative_time and scheduled_at_str and not re.search(r'\d{1,2}:\d{2}', scheduled_at_str):
                        logger.debug(f"No specific time found in conversation, keeping relative time: '{scheduled_at_str}'")

            # --- CUSTOM LOGIC: handle use_default_timing for grade 12 ---
            if scheduled_at_str == "use_default_timing":
                # Use first transcription timestamp as base
                first_transcript_timestamp = self.extract_first_timestamp(transcripts)
                if first_transcript_timestamp:
                    # For grade 12, schedule 2 days later at the same time
                    if student_grade == 12:
                        scheduled_date = first_transcript_timestamp.date() + timedelta(days=2)
                        scheduled_time = first_transcript_timestamp.time().replace(second=0, microsecond=0)
                        scheduled_at = GST.localize(datetime.combine(scheduled_date, scheduled_time))
                        logger.info(f"[use_default_timing] Grade 12 fallback: scheduled_at set to {scheduled_at} (2 days after first transcript)")
                        buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                        # Overwrite scheduled_at_str so downstream logic doesn't re-handle
                        scheduled_at_str = None
                        special_handling_mode = True
                    else:
                        # For other grades, fallback to today + time (or implement other logic as needed)
                        scheduled_date = first_transcript_timestamp.date()
                        scheduled_time = first_transcript_timestamp.time().replace(second=0, microsecond=0)
                        scheduled_at = GST.localize(datetime.combine(scheduled_date, scheduled_time))
                        logger.info(f"[use_default_timing] Non-12th grade fallback: scheduled_at set to {scheduled_at} (same day as first transcript)")
                        buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                        scheduled_at_str = None
                        special_handling_mode = True
                else:
                    # Fallback: use now + 2 days for grade 12
                    now_gst = datetime.now(GST)
                    if student_grade == 12:
                        scheduled_date = now_gst.date() + timedelta(days=2)
                        scheduled_time = now_gst.time().replace(second=0, microsecond=0)
                        scheduled_at = GST.localize(datetime.combine(scheduled_date, scheduled_time))
                        logger.info(f"[use_default_timing] Grade 12 fallback: scheduled_at set to {scheduled_at} (2 days after now)")
                        buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                        scheduled_at_str = None
                        special_handling_mode = True
                    else:
                        scheduled_at = now_gst
                        buffer_until = self._normalize_datetime(scheduled_at + timedelta(minutes=15))
                        logger.info(f"[use_default_timing] Non-12th grade fallback: scheduled_at set to {scheduled_at} (now)")
                        scheduled_at_str = None
                        special_handling_mode = True
            
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
            # IMPORTANT: Use FIRST timestamp for future dates (tomorrow, next week, etc.)
            #           Use LAST timestamp for same-day callbacks (after 10 mins, in 30 mins, etc.)
            # This ensures scheduling is based on when the conversation started for future dates,
            # and when it ended for same-day callbacks
            reference_time = None
            
            # Check if this is a same-day callback or future date
            is_same_day_callback = False
            if scheduled_at_str:
                same_day_patterns = ['after', 'in', 'within', 'minute', 'min', 'hour', 'hr']
                is_same_day_callback = any(pattern in scheduled_at_str.lower() for pattern in same_day_patterns)
            
            if is_same_day_callback:
                # For same-day callbacks, use LAST timestamp (when conversation ended)
                last_transcript_timestamp = self.extract_last_timestamp(transcripts)
                if last_transcript_timestamp:
                    reference_time = last_transcript_timestamp
                    logger.info(f"Same-day callback detected - using last transcription timestamp as reference_time: {reference_time} (GST)")
                else:
                    reference_time = self._normalize_datetime(datetime.now(GST))
                    logger.warning(f"No timestamp found in transcriptions, using current time as reference_time: {reference_time}")
            else:
                # For future dates, use FIRST timestamp (when conversation started)
                first_transcript_timestamp = self.extract_first_timestamp(transcripts)
                if first_transcript_timestamp:
                    reference_time = first_transcript_timestamp
                    logger.info(f"Future date detected - using first transcription timestamp as reference_time: {reference_time} (GST)")
                else:
                    # Fallback to last timestamp if first not available
                    last_transcript_timestamp = self.extract_last_timestamp(transcripts)
                    if last_transcript_timestamp:
                        reference_time = last_transcript_timestamp
                        logger.info(f"Using last transcription timestamp as fallback reference_time: {reference_time} (GST)")
                    else:
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

            # Initialize scheduled_at and buffer_until to None before conditional logic (unless already set by special handling)
            if not locals().get('scheduled_at'):
                scheduled_at = None
            if not locals().get('buffer_until'):
                buffer_until = None
            if not locals().get('use_glinks_scheduler'):
                use_glinks_scheduler = False

            # Determine whether an explicit time phrase was provided by Gemini or fallback extraction
            # Also check if the time is "confirmed" vs "uncertain/unconfirmed" (e.g., questions like "within next ten minutes?")
            # But preserve values already set by special handling (NULL transcripts/failed calls)
            if not locals().get('explicit_time_mentioned'):
                explicit_time_mentioned = bool(scheduled_at_str and str(scheduled_at_str).strip())
            if not locals().get('time_is_confirmed'):
                time_is_confirmed = False
            
            # Check if time is confirmed (not a question, uncertainty, or vague phrase)
            if explicit_time_mentioned:
                time_str_lower = str(scheduled_at_str).strip().lower()
                # Check for uncertainty indicators and vague phrases
                uncertainty_indicators = ['?', 'maybe', 'perhaps', 'possibly', 'might', 'could', 'should', 'may', 'not sure', 'uncertain']
                vague_time_phrases = ['later', 'sometime', 'whenever', 'anytime', 'whenever you can', 'whenever is fine', 'whenever works', 'whenever available', 'whenever suits', 'whenever convenient']
                is_uncertain = any(indicator in time_str_lower for indicator in uncertainty_indicators)
                is_vague = any(phrase in time_str_lower for phrase in vague_time_phrases)
                is_question = time_str_lower.endswith('?') or '?' in time_str_lower
                if is_uncertain or is_question or is_vague:
                    logger.info(f"Time phrase '{scheduled_at_str}' is vague/uncertain/question - treating as UNCONFIRMED time")
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
                    
                    # ==================================================================================
                    # PRIORITY ORDER: Check patterns in order of specificity
                    # 1. Absolute times (HIGHEST PRIORITY) - "11:30 AM GST", "at 7:45 PM"
                    # 2. Relative times - "after 18 mins", "in 30 minutes"
                    # 3. Day references - "tomorrow at 5pm", "next Monday"
                    # ==================================================================================
                    
                    # Pattern 1: Absolute times (e.g., "7:45 PM GST", "6:30 PM", "at 7:45") - CHECK FIRST
                    # These should be extracted directly from conversation
                    # CRITICAL: If agent says "after 2 hours" but also mentions a specific time like "at 3:30 PM",
                    # we should prefer the specific time if user accepts it
                    absolute_time_patterns = [
                        # Pattern 1a: "I'll call you back at 3:30 PM" or "call you at 3:30 PM" (with action verb)
                        r'(?:call|ring|phone|reach|connect|follow\s+up|get\s+back)\s+(?:you|me|him|her|them|us)?\s+(?:back\s+)?(?:at|around|about)?\s*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "call you back at 7:45 PM GST"
                        # Pattern 1b: "at 3:30 PM" or "around 3:30 PM" (standalone)
                        r'(?:at|around|about)\s+(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)?\b',  # "at 7:45 PM GST"
                        # Pattern 1c: "3:30 PM GST" or "7:45 PM" (with timezone or AM/PM)
                        r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\s*(?:GST|IST|UTC)\b',  # "7:45 PM GST", "6:30 PM GST"
                        r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b',  # "7:45 PM", "6:30 AM"
                        # Pattern 1d: "7:45 GST" (without AM/PM but with timezone)
                        r'\b(\d{1,2}):(\d{2})\s*(?:GST|IST|UTC)\b',  # "7:45 GST"
                        # Pattern 1e: "7.45 PM" (dot separator)
                        r'\b(\d{1,2})\.(\d{2})\s*(AM|PM|am|pm)?\b',  # "7.45 PM"
                        # Pattern 1f: "3 PM" or "3:00 PM" (hour only, with AM/PM)
                        r'\b(\d{1,2})(?::00)?\s*(AM|PM|am|pm)\s*(?:GST|IST|UTC)?\b',  # "3 PM", "3:00 PM GST"
                    ]
                    
                    for idx, pattern in enumerate(absolute_time_patterns):
                        try:
                            match = re.search(pattern, text_lower, re.IGNORECASE)
                            if match:
                                hour = match.group(1)
                                # Pattern 1f (hour-only) doesn't have a minute group, others do
                                minute = match.group(2) if len(match.groups()) >= 2 and match.group(2) else "00"
                                # AM/PM is in different group positions depending on pattern
                                ampm = None
                                if len(match.groups()) >= 3:
                                    # Check if group 3 is AM/PM (for patterns with minute)
                                    if match.group(3) and match.group(3).upper() in ['AM', 'PM']:
                                        ampm = match.group(3)
                                elif len(match.groups()) >= 2:
                                    # For pattern 1f, AM/PM is in group 2
                                    if match.group(2) and match.group(2).upper() in ['AM', 'PM']:
                                        ampm = match.group(2)
                                        minute = "00"  # Hour-only pattern, default to :00
                                
                                # Validate hour and minute
                                try:
                                    hour_int = int(hour)
                                    minute_int = int(minute) if minute else 0
                                    # For 12-hour format with AM/PM, hour can be 1-12
                                    # For 24-hour format, hour can be 0-23
                                    max_hour = 12 if ampm else 23
                                    if not (1 <= hour_int <= max_hour):
                                        logger.debug(f"Invalid hour value: {hour_int} (max={max_hour}), skipping")
                                        continue
                                    if not (0 <= minute_int <= 59):
                                        logger.debug(f"Invalid minute value: {minute_int}, skipping")
                                        continue
                                except (ValueError, TypeError):
                                    logger.debug(f"Could not parse hour/minute: hour={hour}, minute={minute}")
                                    continue
                                
                                # Build time string
                                if ampm:
                                    result = f"{hour}:{minute:02d} {ampm.upper()}"
                                else:
                                    result = f"{hour}:{minute:02d}"
                                
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
                    
                    # Pattern 2: "call me after/in/within X mins/minutes" - CHECK AFTER ABSOLUTE TIMES
                    # Only use this if no absolute time was found above
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
                    
                    # Pattern 1b: Written numbers (e.g., "six minutes", "twenty minutes", "thirty minutes")
                    written_number_patterns = [
                        # "call me in twenty minutes", "after twenty minutes"
                        r'(?:just\s+)?(?:call\s+(?:me\s+)?|calling\s+)(?:back\s+)?(?:after|in|within)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|twenty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|thirty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|forty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|fifty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))\s*(?:min(?:ute)?s?)',
                        # "in twenty minutes", "after thirty minutes" (standalone)
                        r'(?:^|[.\s,\'"])(?:after|in|within)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|twenty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|thirty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|forty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine)|fifty[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))\s*(?:min(?:ute)?s?)(?:\s|\.|$|,|;|\'|")',
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
                                
                                # CRITICAL: Reject numbers that look like years (1900-2100 range)
                                # These are likely timestamps or years, not minutes
                                if 1900 <= minutes <= 2100:
                                    logger.warning(f"Extracted value {minutes} looks like a year/timestamp, not minutes - skipping")
                                    continue
                                
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
                    
                    # Pattern 3: "next Sunday", "tomorrow", etc. - CHECK LAST
                    # CRITICAL: Prioritize patterns with TIME first, then fallback to day-only patterns
                    day_patterns = [
                        # Priority 1: Day + specific time ("tomorrow at 5pm", "next Monday at 3pm")
                        r'(?:tomorrow|next\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday))(?:\s+at)?\s+\d{1,2}(?:[:.\s]\d{2})?\s*(?:AM|PM|am|pm)(?:\s*(?:GST|IST|UTC))?',
                        # Priority 2: Day + vague time ("tomorrow morning", "next week evening") 
                        r'(?:tomorrow|next\s+(?:week|month|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday))\s+(?:morning|afternoon|evening|night)',
                        # Priority 3: "call/book tomorrow at 5pm" (capture action + day + time)
                        r'(?:book|schedule|meeting|call).*?(?:tomorrow|next\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday))(?:\s+at)?\s+\d{1,2}(?:[:.\s]\d{2})?\s*(?:AM|PM|am|pm)?',
                        # Priority 4: Day-only fallback (NO specific time) - lowest priority
                        r'(?:book|schedule|meeting|call).*?(?:tomorrow|next\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|sunday|monday|tuesday|wednesday|thursday|friday|saturday))',
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
                    
                    # CRITICAL: Check if extracted time has specific temporal information
                    # Absolute time: "9:40 AM", "at 3pm", "5pm", "7:45 PM GST", "9.40 am"
                    # Must have: hour (with optional minutes) + AM/PM indicator
                    has_absolute_time = bool(re.search(r'\d{1,2}(?:[:.]\d{2})?\s*(?:AM|PM|am|pm)', extracted_time, re.IGNORECASE))
                    
                    # Relative time: "after 18 mins", "in 30 minutes", "within 2 hours" (MUST have number + unit)
                    has_relative_time = bool(re.search(r'(?:after|in|within)\s+\d+\s*(?:min|hour)', extracted_time, re.IGNORECASE))
                    
                    if has_absolute_time or has_relative_time:
                        time_is_confirmed = True
                        time_type = "absolute" if has_absolute_time else "relative"
                        logger.info(f"Fallback extraction found {time_type} time '{extracted_time}' - treating as CONFIRMED")
                    else:
                        logger.warning(f"Fallback extracted '{extracted_time}' but no specific time found (abs={has_absolute_time}, rel={has_relative_time}) - NOT confirming")
                        
            # If an explicit CONFIRMED time phrase was provided, calculate scheduled_at using schedule_calculator
            # Skip this processing if we're in special handling mode (NULL transcripts/failed calls)
            if explicit_time_mentioned and time_is_confirmed and not locals().get('special_handling_mode'):
                # Time is explicitly mentioned - use it directly, no stage-based calculation
                logger.info(f"Time explicitly mentioned in conversation: '{scheduled_at_str}' - using this time directly (no stage-based calculation)")
                scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time, conversation_text, transcripts, started_at, student_grade)
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
                                # Safe to use asyncio.to_thread() - runs in dedicated analysis thread with own executor
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
                                            # If calc_result has a time, use it, else use first timestamp's time
                                            if calc_result.time() != time(0, 0):
                                                combined_time = GST.localize(datetime.combine(first_timestamp_date, calc_result.time().replace(second=0, microsecond=0)))
                                            else:
                                                combined_time = GST.localize(datetime.combine(first_timestamp_date, first_timestamp.time().replace(second=0, microsecond=0)))
                                            scheduled_at = self._normalize_datetime(combined_time)
                                            logger.info(f"Combined first timestamp date with time: {scheduled_at}")
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
                                # Safe to use asyncio.to_thread() - runs in dedicated analysis thread with own executor
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
                # Skip final extraction if special handling mode is active (e.g. use_default_timing)
                if locals().get('special_handling_mode'):
                    logger.info("Skipping final extraction because special_handling_mode is active")
                # No explicit CONFIRMED time phrase mentioned by Gemini - try one final extraction attempt
                # But skip if time was mentioned but not confirmed (uncertain/question)
                elif not time_is_confirmed and scheduled_at_str:
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
                        scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time, conversation_text, transcripts, started_at, student_grade)
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
                        
                        # Get student_grade - default to 12 if not mentioned
                        default_grade = 12
                        final_grade = student_grade if student_grade else default_grade
                        logger.info(f"Using student_grade: {final_grade} (original: {student_grade}, default: {default_grade})")
                        
                        # Get the date from schedule_calculator based on booking_type, retry_count, and grade
                        def _get_schedule_date():
                            # ScheduleCalculator already imported at top of file
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
                            
                            # Calculate next call date using schedule calculator with grade-based pattern
                            # Pass lead_info with retry_count and student_grade to influence the schedule
                            # Use first transcription time as the reference point, not current time
                            first_timestamp_datetime = GST.localize(datetime.combine(first_transcript_timestamp.date(), first_transcript_timestamp.time()))
                            lead_info = {
                                "retry_count": retry_count,
                                "stage": 2,  # Stage 2 uses grade-based follow-up patterns
                                "student_grade": final_grade,  # Use grade for schedule calculation
                                "stage2_start": first_timestamp_datetime,  # Use transcription time as stage2 start reference
                                "last_call_time": first_timestamp_datetime  # Use transcription time as last call reference
                            }
                            schedule_result = calculator.calculate_next_call(outcome, outcome_details, lead_info, first_timestamp_datetime)
                            return schedule_result
                        
                        # Safe to use asyncio.to_thread() - runs in dedicated analysis thread with own executor
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
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging to see Gemini API payload")
    
    args = parser.parse_args()
    
    # Set default logging level to INFO (production standard)
    logging.getLogger().setLevel(logging.DEBUG )
    
    # Set logging level to DEBUG if --debug flag is provided
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled - will show Gemini API payload details")
    
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
