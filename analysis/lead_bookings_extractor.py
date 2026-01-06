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
    
    async def calculate_scheduled_at(self, booking_type: str, scheduled_at_str: str, reference_time: datetime) -> Optional[datetime]:
        """Calculate scheduled_at using schedule_calculator based on booking_type (async-compatible)"""
        if not booking_type or not scheduled_at_str:
            return None
        
        # Normalize time string to convert written numbers to numeric format
        # e.g., "in twenty minutes" -> "in 20 minutes"
        try:
            normalized_time_str = self.normalize_time_string(scheduled_at_str)
        except Exception as e:
            logger.warning(f"Error normalizing time string '{scheduled_at_str}': {e}, using original string")
            normalized_time_str = scheduled_at_str  # Fallback to original if normalization fails
        
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
            return scheduled_at
            
        except Exception as e:
            logger.error(f"Error calculating scheduled_at using schedule_calculator: {e}")
            # Fallback to simple parsing with normalized string
            try:
                def _parse():
                    calculator = ScheduleCalculator()
                    return calculator.parse_callback_time(normalized_time_str, reference_time)  # Use normalized string
                return await asyncio.to_thread(_parse)
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
            reference_time = datetime.now(GST)
            if started_at:
                if started_at.tzinfo is None:
                    reference_time = GST.localize(started_at)
                else:
                    reference_time = started_at.astimezone(GST)
            
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
            explicit_time_mentioned = bool(scheduled_at_str and str(scheduled_at_str).strip())

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
                    
                    # Pattern 2: "next Sunday", "tomorrow", etc.
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
            # If an explicit time phrase was provided, calculate scheduled_at using schedule_calculator
            if explicit_time_mentioned:
                # Time is explicitly mentioned - use it directly, no stage-based calculation
                logger.info(f"Time explicitly mentioned in conversation: '{scheduled_at_str}' - using this time directly (no stage-based calculation)")
                scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time)
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
                                scheduled_at = await asyncio.to_thread(_parse_time)
                                logger.info(f"Extracted time from conversation using schedule_calculator: {scheduled_at}")
                            except Exception as e:
                                logger.warning(f"schedule_calculator failed: {e}, using direct calculation")
                                # Fallback to direct timedelta calculation
                                scheduled_at = reference_time + timedelta(minutes=minutes)
                                logger.info(f"Extracted time from conversation (fallback): {scheduled_at}")
                        else:
                            # Try schedule_calculator with the normalized extracted phrase
                            try:
                                def _parse_time():
                                    calculator = ScheduleCalculator()
                                    return calculator.parse_callback_time(normalized_extracted_time, reference_time)
                                scheduled_at = await asyncio.to_thread(_parse_time)
                                logger.info(f"Extracted time from conversation using schedule_calculator: {scheduled_at}")
                            except Exception as e:
                                logger.warning(f"Could not parse extracted time '{normalized_extracted_time}': {e}")
                                scheduled_at = None
                    else:
                        logger.warning("Could not extract time from conversation. Falling back to stage-based timeline.")
                        # Could not compute exact scheduled_at even though time phrase existed; leave scheduled_at None
                        scheduled_at = None
                
                # Calculate buffer_until (scheduled_at + 15 minutes)
                buffer_until = scheduled_at + timedelta(minutes=15) if scheduled_at else None
            else:
                # No explicit time phrase mentioned by Gemini - try one final extraction attempt
                # This is a production safety net to catch any time mentions that might have been missed
                logger.info(f"No explicit time from Gemini, attempting final extraction from conversation text...")
                final_extracted_time = extract_time_from_conversation(conversation_text)
                if final_extracted_time:
                    logger.info(f"Final extraction successful! Found time: '{final_extracted_time}'")
                    scheduled_at_str = final_extracted_time
                    # Try to calculate scheduled_at using the extracted time
                    scheduled_at = await self.calculate_scheduled_at(booking_type, scheduled_at_str, reference_time)
                    if scheduled_at:
                        buffer_until = scheduled_at + timedelta(minutes=15)
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
                                    scheduled_at = reference_time + timedelta(minutes=minutes)
                                    buffer_until = scheduled_at + timedelta(minutes=15)
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
                
                # If still no time found after all attempts, delegate to Glinks scheduler
                if not explicit_time_mentioned or not scheduled_at:
                    logger.info(f"No explicit time found after all extraction attempts — delegating scheduling to Glinks scheduler for call {call_id}")
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

            # Ensure buffer_until is always calculated if scheduled_at exists
            # This is a safety check to ensure buffer_until is set correctly even if code paths are missed
            if scheduled_at and not buffer_until:
                buffer_until = scheduled_at + timedelta(minutes=15)
                logger.info(f"Recalculated buffer_until from scheduled_at: {buffer_until}")
            elif scheduled_at and buffer_until:
                # Verify buffer_until is correct (should be scheduled_at + 15 minutes)
                expected_buffer = scheduled_at + timedelta(minutes=15)
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
                "created_at": datetime.now(GST).isoformat(),
                "updated_at": datetime.now(GST).isoformat(),
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

