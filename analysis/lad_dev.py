"""
Student Information Extraction for LAD Development
Extracts student/parent information from call transcriptions and stores in lad_dev.education_students table

Usage:
    # List all calls from database
    python lad_dev.py --list-calls
    
    # Extract student info from database call (by row number)
    python lad_dev.py --db-id 123
    
    # Extract student info from database call (by UUID)
    python lad_dev.py --db-id bcc0402b-c290-4242-9873-3cd31052b84a
"""

import os
import json
import asyncio
import re
import requests
import logging
import argparse
import uuid as uuid_lib
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"lad_dev_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.errors
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("psycopg2 not installed. Database features disabled.")
    logger.warning("   Install with: pip install psycopg2-binary")


class StudentExtractor:
    """Student information extractor from call transcriptions"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in .env file")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximately 1 token = 4 characters)"""
        if not text:
            return 0
        return len(text) // 4
    
    def _call_gemini_api(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 500) -> str:
        """Helper function to call Gemini 2.0 Flash API"""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available, skipping API call")
            return None
        
        input_tokens = self._estimate_tokens(prompt)
        logger.debug(f"Calling Gemini API - Input tokens: ~{input_tokens}, Max output: {max_output_tokens}, Temp: {temperature}")
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.gemini_api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens
                }
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                logger.debug("Gemini API call successful")
                
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    if "content" in response_data["candidates"][0]:
                        if "parts" in response_data["candidates"][0]["content"]:
                            output_text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            logger.debug(f"Gemini API response received - Output length: {len(output_text)} chars")
                            return output_text
                
                if "promptFeedback" in response_data:
                    logger.warning(f"Gemini API warning: {response_data.get('promptFeedback', {})}")
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text[:200]}")
            return None
            
        except Exception as e:
            logger.error(f"Gemini API exception: {str(e)}", exc_info=True)
            return None
    
    def _parse_summary_json(self, raw_text: str | None) -> Optional[Dict]:
        """Attempt to extract a JSON object from Gemini output with code fences or trailing text."""
        if not raw_text:
            return None

        text = raw_text.strip()
        if not text:
            return None

        candidates = [text]

        # Extract fenced blocks (handles both ```json and ```)
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)
        for match in fence_pattern.finditer(text):
            snippet = match.group(1).strip()
            if snippet:
                candidates.append(snippet)

        # Handle unterminated fences by slicing after the marker
        for marker in ("```json", "```"):
            marker_index = text.lower().find(marker)
            if marker_index != -1:
                snippet = text[marker_index + len(marker):].strip()
                if snippet:
                    candidates.append(snippet)

        # Always try substring starting from first brace
        brace_index = text.find("{")
        if brace_index != -1:
            json_candidate = text[brace_index:]
            if json_candidate:
                candidates.append(json_candidate)

        decoder = json.JSONDecoder()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            try:
                return decoder.decode(candidate)
            except json.JSONDecodeError:
                pass

            try:
                obj, _ = decoder.raw_decode(candidate)
                return obj
            except json.JSONDecodeError:
                pass

            if "{" in candidate:
                start = candidate.find("{")
                try_candidate = candidate[start:]
                try:
                    obj, _ = decoder.raw_decode(try_candidate)
                    return obj
                except json.JSONDecodeError:
                    continue

        return None
    
    def _extract_user_messages(self, conversation_text: str) -> str:
        """Extract only user messages from conversation (exclude bot/agent messages)"""
        lines = conversation_text.split('\n')
        user_messages = []
        
        for line in lines:
            line = line.strip()
            # Only include lines that start with "User:"
            if line.startswith("User:"):
                user_messages.append(line.replace("User:", "").strip())
            # Exclude agent/bot messages
            elif line.startswith("Bot:") or line.startswith("Agent:"):
                continue
            # Lines without prefix might be user messages (legacy format)
            elif line and not any(line.startswith(prefix) for prefix in ["Bot:", "Agent:", "User:"]):
                user_messages.append(line)
        
        return " ".join(user_messages)
    
    async def extract_student_information(self, conversation_text: str) -> Optional[Dict]:
        """Extract student information from conversation transcription"""
        
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available for student information extraction")
            return None
        
        try:
            user_text = self._extract_user_messages(conversation_text)
            
            if not user_text or len(user_text) < 10:
                logger.debug("Insufficient user text for student information extraction")
                return None
            
            # Log conversation text info for debugging
            logger.info(f"Full conversation text length: {len(conversation_text)} characters")
            logger.info(f"Sample (first 500 chars): {conversation_text[:500]}")
            logger.info(f"Sample (last 500 chars): {conversation_text[-500:]}")
            
            prompt = f"""Extract information from this phone call conversation.

CONVERSATION:
{conversation_text}

INSTRUCTIONS - Follow these steps:

STEP 1 - Find parent name:
Search the conversation for these phrases:
- "My father name is"
- "father name is"  
- "My father is"
If you find any of these, extract the name that comes after. If the name appears in parts (e.g., "Suresh" then later "Harish Kumar"), combine them into one full name.

STEP 2 - Find parent profession:
Search for phrases like:
- "He is doing business"
- "doing business"
- Any mention of parent's job/profession
If found, extract the profession.

STEP 3 - Extract other fields:
- Email addresses (even if spelled phonetically)
- Meeting times when confirmed
- Other information mentioned by the user

REQUIRED FIELDS TO EXTRACT:

1. student_parent_name: 
   - Search for "My father name is" or "father name is" in the conversation
   - Extract the name that follows
   - Combine name parts if mentioned separately
   - Return null ONLY if absolutely no parent name is found

2. parent_designation:
   - Search for parent's profession/job mentions
   - Extract the profession (e.g., "Business", "Doctor")
   - Return null if not mentioned
3. Program Interested In: The educational program, course, or degree the student is interested in (e.g., "Engineering", "MBA", "Medicine", "Computer Science", etc.)
4. Country Interested: The country where the student wants to study (e.g., "USA", "UK", "Canada", "Australia", etc.)
5. Intake Year: The year when the student wants to start the program (e.g., 2025, 2026, etc.)
6. Intake Month: The month when the student wants to start (e.g., "January", "September", "Fall", "Spring", etc.)
7. Any other relevant information about the student, parent, or educational interests (email, phone, percentage, grades, class, school, etc.)

Respond in JSON format. IMPORTANT EXAMPLES:
- If user says "My father name is Suresh" and later "Harish Kumar", extract student_parent_name as "Suresh Harish Kumar"
- If user says "He is doing business", extract parent_designation as "Business"

Full JSON format:
{{
    "student_parent_name": "Extract parent name if you see phrases like 'My father name is X' or 'father name is X' in the conversation. Combine name parts if mentioned separately (e.g., 'Suresh' then 'Harish Kumar' = 'Suresh Harish Kumar'). Return null ONLY if no parent name is mentioned at all.",
    "parent_designation": "Extract parent profession if mentioned (e.g., 'He is doing business' = 'Business', 'My father is a doctor' = 'Doctor'). Return null if not mentioned.",
    "program_interested_in": "Educational program/course/degree student is interested in or null",
    "country_interested": "Country where student wants to study or null",
    "intake_year": "Year when student wants to start (integer like 2025) or null",
    "intake_month": "Month when student wants to start (e.g., 'January', 'September', 'Fall') or null",
    "metadata": {{
        "email": "Email address if mentioned or null",
        "phone": "Phone number if mentioned or null",
        "percentage": "Academic percentage/score if mentioned or null",
        "grades": "Academic grades if mentioned or null",
        "class": "Current class/grade if mentioned or null",
        "school_name": "Current school name if mentioned or null",
        "curriculum": "Curriculum/board type (e.g., 'CBSE', 'ICSE', 'IGCSE', 'IB', 'State Board') if mentioned or null",
        "address": "Address or location if mentioned or null",
        "budget": "Budget or financial information if mentioned or null",
        "preferred_university": "Preferred university or college if mentioned or null",
        "subject_interests": "Subject interests or specialization if mentioned or null",
        "available_time": "Available time/meeting time if mentioned (e.g., 'tomorrow 3pm', 'next week Monday', 'available in evenings', 'Sunday at 11:00 AM' if agent suggests and user agrees) or null",
        "followup_time": "Scheduled meeting/counseling time if a meeting was booked/confirmed (e.g., 'Sunday at 11:00 AM', 'Monday at 3:00 PM') or null",
        "additional_notes": "Any other relevant information provided by the lead"
    }}
}}

IMPORTANT: Use descriptive, specific field names in metadata. For example:
- If they mention "CBSE board" or "ICSE", use "curriculum" field
- If they mention budget or cost, use "budget" field
- If they mention a specific university, use "preferred_university" field
- If they mention when they're available for a meeting, use "available_time" field
- Use clear, meaningful field names instead of generic "any_other_field"

CRITICAL RULES WITH EXAMPLES:
1. Extract information that the STUDENT/PARENT/USER provides - this includes when the student mentions their parent's information
2. Extract EVERYTHING the lead mentioned - be comprehensive and thorough
3. When the student mentions their parent's name or profession, that IS information provided by the user - extract it!

4. PARENT NAME EXTRACTION (student_parent_name) - ABSOLUTELY CRITICAL - DO NOT RETURN NULL IF PARENT NAME IS MENTIONED:
   - MANDATORY: You MUST extract parent name if it is mentioned ANYWHERE in the conversation
   - If user says "My father name is Suresh" and later says "Harish Kumar", combine to "Suresh Harish Kumar"
   - If user says "My father name is Suresh Harish Kumar", extract "Suresh Harish Kumar"
   - Look for phrases: "My father name is...", "My father is...", "father's name is...", "parent name is...", "can you take my parents name?"
   - Read the ENTIRE conversation - parent name may be split across multiple user messages
   - Example: User: "Can you take my parents name?" ΓåÆ Later: "My father name is Suresh" ΓåÆ Later: "Harish Kumar" ΓåÆ Extract: "Suresh Harish Kumar"
   - Example: User: "My father name is Suresh Harish Kumar" ΓåÆ Extract: "Suresh Harish Kumar"
   - CRITICAL: If parent name is mentioned, DO NOT return null - extract the name!

5. PARENT PROFESSION/DESIGNATION (parent_designation) - EXTRACT IF MENTIONED:
   - If user says "He is doing business", extract "Business" or "Businessman"
   - If user says "My father is a doctor", extract "Doctor"
   - Look for phrases about parent's job, profession, or work
   - Example: User: "He is doing business" ΓåÆ Extract: "Business"
   - Example: User: "My father works as an engineer" ΓåÆ Extract: "Engineer"

6. EMAIL EXTRACTION (metadata.email):
   - If user spells email phonetically like "one dot iterate dot one two three at gmail dot com", extract as "one.iterate.123@gmail.com"
   - If user provides email directly, extract exactly as provided
   - Example: User: "one dot iterate dot one two three at gmail dot com" ΓåÆ Extract: "one.iterate.123@gmail.com"

7. SCHEDULED MEETING TIME (metadata.followup_time):
   - If agent asks "Sunday at 11 AM or 3 PM?" and user says "Eleven AM" or "Yes" (referring to 11 AM), extract "Sunday at 11:00 AM"
   - If agent confirms "Sunday at 11:00 AM" and user agrees/confirms, extract "Sunday at 11:00 AM"
   - If a meeting/counseling session is booked or scheduled, extract the confirmed time
   - ALWAYS extract in metadata.followup_time field when a meeting is scheduled/confirmed
   - Example: Agent: "Sunday at 11 AM?" User: "Eleven AM" ΓåÆ Extract in followup_time: "Sunday at 11:00 AM"

8. For intake_year, extract as integer (e.g., 2025, not "2025" as string)
9. For intake_month, use full month names or terms like "Fall", "Spring", "Summer", "Winter"
10. If a field is not mentioned, set it to null (not empty string, not "None")
11. Store ALL additional information in the metadata object using DESCRIPTIVE, SPECIFIC field names
12. Use clear field names in metadata:
   - "curriculum" for CBSE, ICSE, IGCSE, IB, State Board, etc.
   - "budget" for financial information or cost discussions
   - "preferred_university" for specific universities/colleges mentioned
   - "subject_interests" for subjects or specializations
   - "address" for location or address information
   - "available_time" for meeting availability preferences (when user mentions when they're available)
   - "followup_time" for scheduled/confirmed meeting/counseling times (MANDATORY - extract when agent confirms/booked a meeting and user agrees, e.g., "Sunday at 11:00 AM" if meeting is scheduled)
   - "email" for email addresses
13. Do NOT extract agent/bot names (like "Nithya", "Mira Singh", "Pluto Travels representative")
14. Extract information in natural language - preserve exact details when provided
15. If information is provided in multiple parts (e.g., parent name split across messages), combine them into complete values
16. When the user answers agent's questions, extract those answers as user-provided information
17. REMEMBER: If the student mentions their parent's name (e.g., "My father name is Suresh Harish Kumar"), that IS user-provided information - extract it in student_parent_name!
18. REMEMBER: If the student mentions their parent's profession (e.g., "He is doing business"), that IS user-provided information - extract it in parent_designation!
19. Focus on education-related conversations - if this is not education-related, return null for all fields
20. Metadata should contain ANY information the lead provided that doesn't fit in the main columns
"""

            logger.debug("Extracting student information using LLM...")
            result = self._call_gemini_api(prompt, temperature=0.2, max_output_tokens=800)
            
            if not result:
                logger.warning("LLM did not return student information")
                return None
            
            # Parse JSON response
            parsed_data = self._parse_summary_json(result)
            
            if not parsed_data:
                logger.warning("Failed to parse student information JSON from LLM response")
                logger.debug(f"LLM raw response: {result[:500] if result else 'No response'}")
                return None
            
            logger.info(f"Parsed JSON from LLM: {json.dumps(parsed_data, indent=2)}")
            logger.info(f"Parent name from LLM: {parsed_data.get('student_parent_name')}")
            logger.info(f"Parent designation from LLM: {parsed_data.get('parent_designation')}")
            
            # Helper function to clean string values (convert "None" string to None)
            def clean_string(value):
                if value is None:
                    return None
                value_str = str(value).strip()
                if not value_str or value_str.lower() == 'none' or value_str.lower() == 'null':
                    return None
                return value_str
            
            # Clean and validate extracted data
            student_info = {
                'student_parent_name': clean_string(parsed_data.get('student_parent_name')),
                'parent_designation': clean_string(parsed_data.get('parent_designation')),
                'program_interested_in': clean_string(parsed_data.get('program_interested_in')),
                'country_interested': clean_string(parsed_data.get('country_interested')),
                'intake_year': parsed_data.get('intake_year'),
                'intake_month': clean_string(parsed_data.get('intake_month')),
                'metadata': parsed_data.get('metadata', {})
            }
            
            # Validate intake_year - should be integer or convert if possible
            if student_info['intake_year'] is not None:
                try:
                    # Try to convert to int if it's a string
                    if isinstance(student_info['intake_year'], str):
                        student_info['intake_year'] = int(student_info['intake_year'].strip())
                    elif not isinstance(student_info['intake_year'], int):
                        logger.warning(f"Invalid intake_year format: {student_info['intake_year']}")
                        student_info['intake_year'] = None
                except (ValueError, AttributeError):
                    logger.warning(f"Could not convert intake_year to integer: {student_info['intake_year']}")
                    student_info['intake_year'] = None
            
            # Validate metadata is a dict
            if not isinstance(student_info['metadata'], dict):
                student_info['metadata'] = {}
            
            # Clean metadata values (remove "None" strings)
            cleaned_metadata = {}
            if isinstance(student_info['metadata'], dict):
                for key, value in student_info['metadata'].items():
                    if value is not None:
                        value_str = str(value).strip()
                        if value_str and value_str.lower() not in ['none', 'null']:
                            cleaned_metadata[key] = value
            student_info['metadata'] = cleaned_metadata
            
            # Check if we have at least some information (either in main fields or metadata)
            has_info_main = any([
                student_info['student_parent_name'],
                student_info['parent_designation'],
                student_info['program_interested_in'],
                student_info['country_interested'],
                student_info['intake_year'],
                student_info['intake_month']
            ])
            
            has_info_metadata = len(student_info.get('metadata', {})) > 0
            
            if not has_info_main and not has_info_metadata:
                logger.warning("No student information found in conversation")
                logger.info(f"Parsed data was: {parsed_data}")
                logger.info(f"Main fields: student_parent_name={student_info['student_parent_name']}, parent_designation={student_info['parent_designation']}, etc.")
                logger.info(f"Metadata fields: {student_info.get('metadata', {})}")
                return None
            
            logger.debug(f"Student information extracted: {student_info}")
            return student_info
            
        except Exception as e:
            logger.error(f"Error extracting student information: {str(e)}", exc_info=True)
            return None
    
    def save_to_json(self, student_info: Dict, call_log_id, lead_id, tenant_id: Optional[str]) -> Optional[str]:
        """
        Save student information to JSON file locally
        
        Args:
            student_info: The extracted student information dictionary
            call_log_id: ID from call_logs_voiceagent table (for reference in metadata)
            lead_id: Target ID from call_logs_voiceagent.target (to connect with education_students.lead_id)
            tenant_id: Tenant ID from lad_dev.tenants.id (UUID string or None)
        
        Returns:
            Path to saved JSON file or None if failed
        """
        try:
            # Prepare metadata - ONLY store information provided by the lead
            # Do NOT include system-generated fields like call_log_id or extracted_at
            # Store all information that's not in the main columns (email, percentage, grades, etc.)
            metadata_raw = student_info.get('metadata', {})
            
            # Add any additional fields from LLM that aren't in main columns
            # These could include: email, percentage, grades, etc. (ONLY lead-provided information)
            for key, value in student_info.items():
                if key not in ['student_parent_name', 'parent_designation', 'program_interested_in', 
                              'country_interested', 'intake_year', 'intake_month', 'metadata']:
                    if value is not None:
                        metadata_raw[key] = value
            
            # Filter metadata to only include non-null, non-empty values
            metadata = {}
            if isinstance(metadata_raw, dict):
                for key, value in metadata_raw.items():
                    # Only include if value is not None, not empty string, and not "None"/"null" as string
                    if value is not None:
                        value_str = str(value).strip()
                        if value_str and value_str.lower() not in ['none', 'null']:
                            metadata[key] = value
            
            # Convert lead_id (target) to BIGINT to match call_logs_voiceagent.target column type
            lead_id_bigint = None
            if lead_id is not None:
                try:
                    # Convert to integer (BIGINT) for testing - matches call_logs_voiceagent.target
                    if isinstance(lead_id, (int, str)):
                        lead_id_bigint = int(lead_id)
                    else:
                        lead_id_bigint = None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert lead_id to BIGINT: {e}")
                    lead_id_bigint = None
            
            # Prepare output data structure (matches database structure)
            output_data = {
                'tenant_id': tenant_id,
                'lead_id': lead_id_bigint,  # BIGINT from call_logs_voiceagent.target
                'student_parent_name': student_info.get('student_parent_name'),
                'parent_designation': student_info.get('parent_designation'),
                'program_interested_in': student_info.get('program_interested_in'),
                'country_interested': student_info.get('country_interested'),
                'intake_year': student_info.get('intake_year'),
                'intake_month': student_info.get('intake_month'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'is_deleted': False
            }
            
            # Only include metadata if it has actual values (filtered to exclude null/empty)
            if metadata:
                output_data['metadata'] = metadata
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            call_id_short = str(call_log_id)[:8] if call_log_id else 'unknown'
            filename = f"lad_dev_{call_id_short}_{timestamp}.json"
            
            # Create json_exports directory if it doesn't exist
            json_dir = Path(__file__).parent / "json_exports"
            json_dir.mkdir(exist_ok=True)
            
            filepath = json_dir / filename
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Student information saved to JSON file: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save JSON file: {e}", exc_info=True)
            return None
    
    def save_to_database(self, student_info: Dict, call_log_id, lead_id, tenant_id: Optional[str], db_config: Dict) -> bool:
        """
        Save student information to lad_dev.education_students table
        
        Args:
            student_info: The extracted student information dictionary
            call_log_id: ID from call_logs_voiceagent table (for reference in metadata)
            lead_id: Target ID from call_logs_voiceagent.target (to connect with education_students.lead_id)
            tenant_id: Tenant ID from lad_dev.tenants.id (UUID string or None)
            db_config: Dict with db connection parameters
        
        Returns:
            bool: True if saved successfully
        """
        if not DB_AVAILABLE:
            logger.error("Database library not available. Install psycopg2-binary")
            return False
        
        conn = None
        cursor = None
        
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Prepare metadata - ONLY store information provided by the lead
            # Do NOT include system-generated fields like call_log_id or extracted_at
            # Store all information that's not in the main columns (email, percentage, grades, etc.)
            metadata_raw = student_info.get('metadata', {})
            
            # Add any additional fields from LLM that aren't in main columns
            # These could include: email, percentage, grades, etc. (ONLY lead-provided information)
            for key, value in student_info.items():
                if key not in ['student_parent_name', 'parent_designation', 'program_interested_in', 
                              'country_interested', 'intake_year', 'intake_month', 'metadata']:
                    if value is not None:
                        metadata_raw[key] = value
            
            # Filter metadata to only include non-null, non-empty values
            metadata = {}
            if isinstance(metadata_raw, dict):
                for key, value in metadata_raw.items():
                    # Only include if value is not None, not empty string, and not "None"/"null" as string
                    if value is not None:
                        value_str = str(value).strip()
                        if value_str and value_str.lower() not in ['none', 'null']:
                            metadata[key] = value
            
            # Convert lead_id to UUID format for education_students.lead_id column (UUID type)
            # NOTE: All IDs except agent_id should be UUIDs - no int fallback
            lead_id_uuid = None
            if lead_id is not None:
                lead_id_str = str(lead_id).strip()
                # Reject 'None' or 'null' string values
                if lead_id_str.lower() in ('none', 'null', ''):
                    logger.warning(f"lead_id is invalid string value: '{lead_id}'")
                    lead_id_uuid = None
                else:
                    try:
                        # Parse as UUID - this is the only valid format
                        parsed_uuid = uuid_lib.UUID(lead_id_str)
                        lead_id_uuid = str(parsed_uuid)
                        logger.debug(f"lead_id '{lead_id_str}' validated as UUID")
                    except ValueError as e:
                        logger.error(f"lead_id '{lead_id}' is not a valid UUID: {e}")
                        lead_id_uuid = None
            
            # Prepare INSERT query with tenant_id and lead_id
            query = """
                INSERT INTO lad_dev.education_students (
                    tenant_id,
                    lead_id,
                    student_parent_name,
                    parent_designation,
                    program_interested_in,
                    country_interested,
                    intake_year,
                    intake_month,
                    metadata,
                    created_at,
                    updated_at,
                    is_deleted
                ) VALUES (
                    %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, FALSE
                )
            """
            
            values = (
                tenant_id,  # UUID from lad_dev.tenants.id
                lead_id_uuid,  # UUID converted from call_logs_voiceagent.target (BIGINT -> UUID)
                student_info.get('student_parent_name'),
                student_info.get('parent_designation'),
                student_info.get('program_interested_in'),
                student_info.get('country_interested'),
                student_info.get('intake_year'),
                student_info.get('intake_month'),
                Json(metadata)
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            logger.info(f"Student information saved to database (call_log_id: {call_log_id})")
            return True
            
        except Exception as e:
            logger.error(f"Database save failed: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


class StandaloneStudentExtractor:
    """Standalone student extractor - database input only"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize standalone student extractor
        
        Args:
            db_config: Database connection config (optional)
                      {'host': 'localhost', 'database': 'db', 'user': 'user', 'password': 'pass'}
        """
        self.extractor = StudentExtractor()
        self.db_config = db_config
    
    def _get_tenant_id(self, cursor) -> Optional[str]:
        """
        Get tenant_id from lad_dev.tenants table
        
        Args:
            cursor: Database cursor
            
        Returns:
            tenant_id (UUID string) or None if no tenants found
        """
        try:
            cursor.execute("""
                SELECT id FROM lad_dev.tenants 
                ORDER BY created_at ASC NULLS LAST, id ASC
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                tenant_id = result[0]
                logger.debug(f"Found tenant_id: {tenant_id}")
                return str(tenant_id) if tenant_id else None
            else:
                logger.warning("No tenants found in lad_dev.tenants table. tenant_id will be NULL.")
                return None
        except Exception as e:
            logger.warning(f"Error fetching tenant_id: {e}. tenant_id will be NULL.")
            return None
    
    async def list_database_calls(self) -> None:
        """List all calls from database with row numbers matching the query order"""
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        if not self.db_config:
            raise ValueError("Database config not provided. Use --db-host, --db-name, --db-user, --db-pass or .env file")
        
        logger.info("Listing calls from database...")
        logger.info("=" * 80)
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Fetch calls in pgAdmin's default display order (physical storage order using ctid)
            # Select transcriptions as-is and handle conversion in Python to avoid JSONB validation errors
            cursor.execute("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY ctid) as row_num,
                    id,
                    started_at,
                    ended_at,
                    transcriptions
                FROM voice_agent.call_logs_voiceagent
                ORDER BY ctid
                LIMIT 500000
            """)
            
            calls = cursor.fetchall()
            
            if not calls:
                logger.warning("No calls found in database.")
                logger.info("No calls found in database.")
                return
            
            header = f"{'Row':<6} {'UUID':<40} {'Started At':<20} {'Duration':<10} {'Transcript Preview'}"
            logger.info(header)
            logger.info("-" * 80)
            
            for row_num, call_id, started_at, ended_at, transcriptions in calls:
                # Convert transcriptions to preview text safely in Python
                transcript_preview = 'No transcript'
                if transcriptions:
                    try:
                        if isinstance(transcriptions, (dict, list)):
                            transcript_str = json.dumps(transcriptions)
                        else:
                            transcript_str = str(transcriptions)
                        
                        if transcript_str:
                            preview = transcript_str[:50]
                            if len(transcript_str) > 50:
                                preview += '...'
                            transcript_preview = preview
                    except Exception:
                        transcript_preview = 'Invalid transcript format'
                duration = ""
                if started_at and ended_at:
                    duration_seconds = int((ended_at - started_at).total_seconds())
                    duration = f"{duration_seconds}s"
                
                started_str = started_at.strftime('%Y-%m-%d %H:%M:%S') if started_at else 'N/A'
                call_info = f"{row_num:<6} {str(call_id):<40} {started_str:<20} {duration:<10} {transcript_preview}"
                logger.info(call_info)
            
            logger.info("-" * 80)
            logger.info("Row numbers match pgAdmin's default display order (physical storage order)")
            logger.info("To extract student info from a call, use: python lad_dev.py --db-id <row_number>")
            logger.info("Or use UUID directly: python lad_dev.py --db-id <uuid_string>")
            
        finally:
            cursor.close()
            conn.close()
    
    async def extract_from_database(self, call_log_id, save_to_db: bool = False, save_to_json: bool = True) -> Optional[Dict]:
        """
        Extract student information from database call_logs table
        
        Args:
            call_log_id: ID from call_logs_voiceagent table
            save_to_db: If True, save to database (default: False)
            save_to_json: If True, save to JSON file (default: True)
            
        Returns:
            Extracted student information dictionary or None
        """
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        if not self.db_config:
            raise ValueError("Database config not provided. Use --db-host, --db-name, --db-user, --db-pass")
        
        logger.info(f"Fetching call from database (ID: {call_log_id})")
        
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Try to parse as integer first, otherwise treat as UUID string
            try:
                call_log_id_int = int(call_log_id)
                call_log_id = call_log_id_int
            except (ValueError, TypeError):
                # Not an integer, treat as UUID string
                pass
            
            if isinstance(call_log_id, int):
                # Integer ID: Use ROW_NUMBER() to find the Nth record
                cursor.execute("""
                    SELECT id, transcriptions, started_at, ended_at, target
                    FROM (
                        SELECT 
                            id, 
                            transcriptions, 
                            started_at, 
                            ended_at,
                            target,
                            ROW_NUMBER() OVER (ORDER BY ctid) as row_num
                        FROM voice_agent.call_logs_voiceagent
                    ) ranked
                    WHERE row_num = %s
                """, (call_log_id,))
            else:
                # UUID string: Try direct UUID match or text match
                try:
                    cursor.execute("""
                        SELECT id, transcriptions, started_at, ended_at, target
                        FROM voice_agent.call_logs_voiceagent
                        WHERE id = %s::uuid
                    """, (str(call_log_id),))
                except (psycopg2.errors.InvalidTextRepresentation, psycopg2.errors.UndefinedFunction):
                    # Fallback: try text match
                    cursor.execute("""
                        SELECT id, transcriptions, started_at, ended_at, target
                        FROM voice_agent.call_logs_voiceagent
                        WHERE id::text = %s
                    """, (str(call_log_id),))
            
            call_data = cursor.fetchone()
            
            if not call_data:
                raise ValueError(f"Call log {call_log_id} not found in database")
            
            db_call_id, transcriptions, started_at, ended_at, target_id = call_data
            
            # Get transcript from transcriptions column
            # Handle different data types: dict (JSONB), list, or string
            if not transcriptions:
                raise ValueError(f"No transcript found for call {call_log_id}")
            
            # Convert transcriptions to conversation text
            if isinstance(transcriptions, dict):
                # If it's a dict, check if it contains a list of messages
                if 'messages' in transcriptions and isinstance(transcriptions['messages'], list):
                    # Structured format with messages array
                    conversation_log = transcriptions['messages']
                    conversation_text = "\n".join([f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}" for entry in conversation_log])
                elif isinstance(transcriptions, dict) and any(key in transcriptions for key in ['role', 'message', 'text']):
                    # Single message dict - convert to text
                    role = transcriptions.get('role', 'Unknown').title()
                    message = transcriptions.get('message') or transcriptions.get('text', '')
                    conversation_text = f"{role}: {message}"
                else:
                    # Try to extract text from dict
                    if 'text' in transcriptions:
                        conversation_text = str(transcriptions['text'])
                    elif 'transcript' in transcriptions:
                        conversation_text = str(transcriptions['transcript'])
                    elif 'content' in transcriptions:
                        conversation_text = str(transcriptions['content'])
                    else:
                        # Fallback: convert entire dict to JSON string
                        conversation_text = json.dumps(transcriptions)
            elif isinstance(transcriptions, list):
                # List format - convert to text
                conversation_text = "\n".join([f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}" if isinstance(entry, dict) else str(entry) for entry in transcriptions])
            else:
                # String format - use as-is
                conversation_text = str(transcriptions)
            
            logger.info(f"Call ID: {db_call_id}, Started: {started_at}, Ended: {ended_at}, Target ID: {target_id}")
            logger.info(f"Conversation text length: {len(conversation_text)} characters")
            
            # Get tenant_id from lad_dev.tenants table (for JSON structure)
            tenant_id = self._get_tenant_id(cursor)
            
            # Extract student information
            student_info = await self.extractor.extract_student_information(conversation_text)
            
            if student_info:
                logger.info("Student information extracted successfully")
                logger.info(f"Student/Parent Name: {student_info.get('student_parent_name')}")
                logger.info(f"Program Interested: {student_info.get('program_interested_in')}")
                logger.info(f"Country Interested: {student_info.get('country_interested')}")
                
                # Save to JSON file (default behavior)
                if save_to_json:
                    logger.info("Saving student information to JSON file...")
                    json_file = self.extractor.save_to_json(student_info, db_call_id, target_id, tenant_id)
                    
                    if json_file:
                        logger.info(f"Student information saved to JSON file: {json_file}")
                    else:
                        logger.error("Failed to save student information to JSON file")
                
                # Save to database (optional)
                if save_to_db:
                    logger.info("Saving student information to database...")
                    db_saved = self.extractor.save_to_database(
                        student_info, 
                        db_call_id, 
                        target_id, 
                        tenant_id, 
                        self.db_config
                    )
                    if db_saved:
                        logger.info("Student information saved to database successfully")
                    else:
                        logger.error("Failed to save student information to database")
                
                return student_info
            else:
                logger.info("No student information found in this call transcript")
            
            return None
            
        finally:
            cursor.close()
            conn.close()


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Student Information Extraction Tool - Extract from call transcriptions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all calls from database (shows row numbers matching --db-id usage)
    python lad_dev.py --list-calls
    
    # Extract student info and save to JSON file (default behavior)
    python lad_dev.py --db-id 23
    
    # Extract student info, save to JSON file, and also save to database
    python lad_dev.py --db-id 23 --save-db
    
    # Extract student info but skip JSON file (save to database only)
    python lad_dev.py --db-id 23 --save-db --no-save-json
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--db-id', help='Call log ID from database (integer for row number, or UUID string)')
    input_group.add_argument('--list-calls', action='store_true', help='List all calls with row numbers')
    
    parser.add_argument('--db-host', help='Database host (default: from .env or localhost)')
    parser.add_argument('--db-name', help='Database name (default: from .env)')
    parser.add_argument('--db-user', help='Database user (default: from .env or postgres)')
    parser.add_argument('--db-pass', help='Database password (default: from .env)')
    parser.add_argument('--save-db', action='store_true', help='Also save extracted data to database (optional)')
    parser.add_argument('--no-save-json', action='store_true', help='Skip saving to JSON file (JSON is saved by default)')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host or os.getenv('DB_HOST', 'localhost'),
        'database': args.db_name or os.getenv('DB_NAME', 'salesmaya_agent'),
        'user': args.db_user or os.getenv('DB_USER', 'postgres'),
        'password': args.db_pass or os.getenv('DB_PASSWORD')
    }
    
    if not db_config['password']:
        parser.error("Database password required. Provide via --db-pass or DB_PASSWORD in .env file")
    
    extractor = StandaloneStudentExtractor(db_config=db_config)
    result = None
    
    try:
        if args.list_calls:
            await extractor.list_database_calls()
            return
        elif args.db_id:
            result = await extractor.extract_from_database(
                args.db_id, 
                save_to_db=args.save_db,  # Default is False, set to True with --save-db flag
                save_to_json=not args.no_save_json  # Default is True, unless --no-save-json flag is set
            )
        
        if result:
            logger.info("="*60)
            logger.info("EXTRACTION SUMMARY")
            logger.info("="*60)
            logger.info(f"Student/Parent Name: {result.get('student_parent_name', 'N/A')}")
            logger.info(f"Parent Designation: {result.get('parent_designation', 'N/A')}")
            logger.info(f"Program Interested: {result.get('program_interested_in', 'N/A')}")
            logger.info(f"Country Interested: {result.get('country_interested', 'N/A')}")
            logger.info(f"Intake Year: {result.get('intake_year', 'N/A')}")
            logger.info(f"Intake Month: {result.get('intake_month', 'N/A')}")
            logger.info("="*60)
            logger.info("Extraction complete!")
        else:
            logger.info("No student information extracted from this call.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

