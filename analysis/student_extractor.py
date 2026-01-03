"""
Student Information Extraction Module
Extracts student information from call transcriptions and saves to local JSON files

Phase 13: Updated to use lad_dev schema (education_students table)

Usage:
    # List all calls from database
    python student_extraction.py --list-calls
    
    # Extract student info from database call (by row number)
    python student_extraction.py --db-id 123
    
    # Extract student info from database call (by UUID)
    python student_extraction.py --db-id bcc0402b-c290-4242-9873-3cd31052b84a
"""

import os
import json
import asyncio
import re
import requests
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Schema constants for table names
from db.schema_constants import (
    CALL_LOGS_FULL,
    ANALYSIS_FULL,
    LEADS_FULL,
    STUDENTS_FULL,
)

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"student_extraction_{datetime.now().strftime('%Y%m%d')}.log"

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
    """Student information extractor from call transcriptions - matches glinks.students_glinks table"""
    
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
        """
        Extract student information from conversation transcription
        Matches glinks.students_glinks table columns
        """
        
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
- Student name if mentioned
- Email addresses (even if spelled phonetically)
- Meeting times when confirmed
- Other information mentioned by the user

REQUIRED FIELDS TO EXTRACT:

1. student_name: Search for student's name if mentioned, or null
2. parent_name: SEARCH for "My father name is" or "father name is" - extract the name that follows. Combine name parts if mentioned separately. Return null ONLY if no parent name is found.
3. parents_profession: SEARCH for "He is doing business" or similar - extract the profession. Return null if not mentioned.
4. email: Email address if mentioned (even if spelled phonetically), or null
5. parent_contact: Phone number if mentioned, or null
6. parents_workplace: Parent's workplace if mentioned, or null
7. country_of_residence: Country where student lives (default: 'Unknown' if not mentioned)
8. nationality: Student's nationality if mentioned, or null
9. grade_year: Current grade/year if mentioned, or null
10. curriculum: Curriculum type if mentioned, or null
11. school_name: School name if mentioned, or null
12. lead_source: How they found us (default: 'Phone Call')
13. program_country_of_interest: Country of interest if mentioned, or null
14. academic_grades: Academic performance if mentioned, or null
15. counsellor_meeting_link: Always null (do not extract meeting times here)
16. tags: Relevant tags if mentioned, or null
17. stage: Current stage if mentioned, or null
18. status: Status if mentioned, or null
19. counsellor_email: Counsellor email if mentioned, or null

Respond in JSON format:
{{
    "student_name": "Full name of the student if mentioned, or null",
    "parent_name": "Extract parent name if you see phrases like 'My father name is X' or 'father name is X' in the conversation. Combine name parts if mentioned separately. Return null ONLY if no parent name is found.",
    "parent_contact": "Phone number of parent if mentioned, or null",
    "parents_profession": "Extract parent profession if mentioned (e.g., 'He is doing business' = 'Business'). Return null if not mentioned.",
    "parents_workplace": "Parent's workplace/company if mentioned, or null",
    "email": "Email address if mentioned (even if spelled phonetically like 'one dot iterate dot one two three at gmail dot com' = 'one.iterate.123@gmail.com'), or null",
    "country_of_residence": "Country where student lives (if mentioned, default: 'Unknown')",
    "nationality": "Student's nationality if mentioned, or null",
    "grade_year": "Current grade/year (e.g., '10', '11', '12') if mentioned, or null",
    "curriculum": "Curriculum type (e.g., 'CBSE', 'ICSE', 'State Board') if mentioned, or null",
    "school_name": "Name of current school if mentioned, or null",
    "lead_source": "How they found us (default: 'Phone Call')",
    "program_country_of_interest": "Country they're interested in studying if mentioned, or null",
    "academic_grades": "Academic performance/grade if mentioned, or null",
    "counsellor_meeting_link": null,
    "tags": "Relevant tags if mentioned, or null",
    "stage": "Current stage if mentioned, or null",
    "status": "Status if mentioned, or null",
    "counsellor_email": "Counsellor email if mentioned, or null"
}}

CRITICAL RULES:
1. Extract information that the USER/STUDENT/PARENT provides - this includes when the student mentions their parent's information
2. Extract EVERYTHING the lead mentioned - be comprehensive and thorough
3. When the student mentions their parent's name or profession, that IS user-provided information - extract it!
4. REMEMBER: If the student says "My father name is Suresh Harish Kumar", extract "Suresh Harish Kumar" in parent_name!
5. REMEMBER: If the student says "He is doing business", extract "Business" in parents_profession!
6. If a field is not mentioned, set it to null (not empty string, not "None")
7. For country_of_residence: If not mentioned, use 'Unknown' (this field is required)
8. For lead_source: Default to 'Phone Call' if not specified
9. Extract email addresses in standard format even if provided phonetically
10. Extract scheduled meeting times when a meeting is confirmed/booked
11. Do NOT extract agent/bot names (like "Nithya", "Mira Singh", "Pluto Travels representative")
12. Extract information in natural language - preserve exact details when provided
13. If information is provided in multiple parts (e.g., parent name split across messages), combine them into complete values
14. Store ALL information the lead provided - be thorough and comprehensive
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
            logger.info(f"Parent name from LLM: {parsed_data.get('parent_name')}")
            logger.info(f"Parent profession from LLM: {parsed_data.get('parents_profession')}")
            
            # Helper function to clean string values (convert "None" string to None)
            def clean_string(value, default=None):
                if value is None:
                    return default
                value_str = str(value).strip()
                if not value_str or value_str.lower() == 'none' or value_str.lower() == 'null':
                    return default
                return value_str
            
            # Clean and validate extracted data
            student_info = {
                'student_name': clean_string(parsed_data.get('student_name')),
                'parent_name': clean_string(parsed_data.get('parent_name')),
                'parent_contact': clean_string(parsed_data.get('parent_contact')),
                'parents_profession': clean_string(parsed_data.get('parents_profession')),
                'parents_workplace': clean_string(parsed_data.get('parents_workplace')),
                'email': clean_string(parsed_data.get('email')),
                'country_of_residence': clean_string(parsed_data.get('country_of_residence'), 'Unknown'),
                'nationality': clean_string(parsed_data.get('nationality')),
                'grade_year': clean_string(parsed_data.get('grade_year')),
                'curriculum': clean_string(parsed_data.get('curriculum')),
                'school_name': clean_string(parsed_data.get('school_name')),
                'lead_source': clean_string(parsed_data.get('lead_source'), 'Phone Call'),
                'program_country_of_interest': clean_string(parsed_data.get('program_country_of_interest')),
                'academic_grades': clean_string(parsed_data.get('academic_grades')),
                'counsellor_meeting_link': None,  # Always null
                'tags': clean_string(parsed_data.get('tags')),
                'stage': clean_string(parsed_data.get('stage')),
                'status': clean_string(parsed_data.get('status')),
                'counsellor_email': clean_string(parsed_data.get('counsellor_email'))
            }
            
            # Check if we have at least some information
            has_info = any([
                student_info['student_name'],
                student_info['parent_name'],
                student_info['parent_contact'],
                student_info['email'],
                student_info['grade_year'],
                student_info['school_name'],
                student_info['curriculum'],
                student_info['parents_profession'],
                False  # counsellor_meeting_link is always null, so skip this check
            ])
            
            if not has_info:
                logger.warning("No student information found in conversation")
                logger.info(f"Parsed data was: {parsed_data}")
                logger.info(f"Student info extracted: {student_info}")
                return None
            
            logger.debug(f"Student information extracted: {student_info}")
            return student_info
            
        except Exception as e:
            logger.error(f"Error extracting student information: {str(e)}", exc_info=True)
            return None
    
    def save_to_json(self, student_info: Dict, call_log_id, target_id: Optional[int]) -> Optional[str]:
        """
        Save student information to JSON file locally
        
        Args:
            student_info: The extracted student information dictionary (matching glinks.students_glinks columns)
            call_log_id: ID from call_logs_voiceagent table (for reference)
            target_id: Target ID from call_logs_voiceagent.target (BIGINT) - used as id in students_glinks table
        
        Returns:
            Path to saved JSON file or None if failed
        """
        try:
            # Convert target_id (BIGINT) to use as id in students_glinks table
            id_value = None
            if target_id is not None:
                try:
                    if isinstance(target_id, (int, str)):
                        id_value = int(target_id)
                    else:
                        id_value = None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert target_id to integer: {e}")
                    id_value = None
            
            # Prepare output data structure (matches glinks.students_glinks table structure)
            # id column uses the same value as target from call_logs_voiceagent
            output_data = {
                'id': id_value,  # Use target value from call_logs_voiceagent.target
                'student_name': student_info.get('student_name'),
                'parent_name': student_info.get('parent_name'),
                'parent_contact': student_info.get('parent_contact'),
                'parents_profession': student_info.get('parents_profession'),
                'parents_workplace': student_info.get('parents_workplace'),
                'email': student_info.get('email'),
                'country_of_residence': student_info.get('country_of_residence'),
                'nationality': student_info.get('nationality'),
                'grade_year': student_info.get('grade_year'),
                'curriculum': student_info.get('curriculum'),
                'school_name': student_info.get('school_name'),
                'lead_source': student_info.get('lead_source'),
                'program_country_of_interest': student_info.get('program_country_of_interest'),
                'academic_grades': student_info.get('academic_grades'),
                'counsellor_meeting_link': student_info.get('counsellor_meeting_link'),
                'tags': student_info.get('tags'),
                'stage': student_info.get('stage'),
                'status': student_info.get('status'),
                'counsellor_email': student_info.get('counsellor_email'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            call_id_short = str(call_log_id)[:8] if call_log_id else 'unknown'
            filename = f"student_extraction_{call_id_short}_{timestamp}.json"
            
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
    
    def save_to_database(self, student_info: Dict, call_log_id, target_id: Optional[int], db_config: Dict) -> bool:
        """
        Save student information to glinks.students_glinks table
        
        Args:
            student_info: The extracted student information dictionary (matching glinks.students_glinks columns)
            call_log_id: ID from call_logs_voiceagent table (for reference)
            target_id: Target ID from call_logs_voiceagent.target (BIGINT) - used as id in students_glinks table
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
            
            # Convert target_id (BIGINT) to use as id in students_glinks table
            id_value = None
            if target_id is not None:
                try:
                    if isinstance(target_id, (int, str)):
                        id_value = int(target_id)
                    else:
                        id_value = None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert target_id to integer: {e}")
                    id_value = None
            
            if id_value is None:
                logger.error("target_id is required to save to database (connects to students_glinks.id)")
                return False
            
            # Prepare INSERT query - matches glinks.students_glinks table structure
            # Note: id column uses target value from call_logs_voiceagent.target (BIGINT)
            # Database column types are not changed - we're inserting data that matches existing structure
            query = """
                INSERT INTO glinks.students_glinks (
                    id,
                    student_name,
                    parent_name,
                    parent_contact,
                    parents_profession,
                    parents_workplace,
                    email,
                    country_of_residence,
                    nationality,
                    grade_year,
                    curriculum,
                    school_name,
                    lead_source,
                    program_country_of_interest,
                    academic_grades,
                    counsellor_meeting_link,
                    tags,
                    stage,
                    status,
                    counsellor_email,
                    created_at,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
            """
            
            values = (
                id_value,  # id: uses target value from call_logs_voiceagent.target
                student_info.get('student_name'),
                student_info.get('parent_name'),
                student_info.get('parent_contact'),
                student_info.get('parents_profession'),
                student_info.get('parents_workplace'),
                student_info.get('email'),
                student_info.get('country_of_residence', 'Unknown'),  # Default to 'Unknown' if not provided
                student_info.get('nationality'),
                student_info.get('grade_year'),
                student_info.get('curriculum'),
                student_info.get('school_name'),
                student_info.get('lead_source', 'Phone Call'),  # Default to 'Phone Call' if not provided
                student_info.get('program_country_of_interest'),
                student_info.get('academic_grades'),
                None,  # counsellor_meeting_link is always null
                student_info.get('tags'),
                student_info.get('stage'),
                student_info.get('status'),
                student_info.get('counsellor_email')
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            logger.info(f"Student information saved to database (call_log_id: {call_log_id}, id: {id_value})")
            return True
            
        except psycopg2.errors.UniqueViolation:
            logger.warning(f"Student with id {id_value} already exists in database (duplicate entry)")
            if conn:
                conn.rollback()
            return False
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
    
    def _get_lead_category_from_analysis(self, cursor, call_log_id) -> Optional[str]:
        """
        Get lead_category from lad_dev.voice_call_analysis table
        Connected via voice_call_logs.id = voice_call_analysis.call_log_id
        
        Args:
            cursor: Database cursor
            call_log_id: Call log ID (UUID) from voice_call_logs table
            
        Returns:
            lead_category string if found, None otherwise
        """
        if not call_log_id:
            return None
        
        try:
            # Query voice_call_analysis using call_log_id - get from lead_extraction JSONB
            cursor.execute(f"""
                SELECT lead_extraction->>'lead_category' as lead_category
                FROM {ANALYSIS_FULL}
                WHERE call_log_id = %s::uuid
            """, (str(call_log_id),))
            
            row = cursor.fetchone()
            
            if row and row[0] is not None:
                lead_category = str(row[0]).strip()
                if lead_category:
                    logger.debug(f"Found lead_category '{lead_category}' for call_log_id {call_log_id}")
                    return lead_category
            
            logger.debug(f"No lead_category found for call_log_id {call_log_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting lead_category from voice_call_analysis: {e}")
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
            # Fetch calls ordered by created_at
            cursor.execute(f"""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY created_at) as row_num,
                    id,
                    started_at,
                    ended_at,
                    transcripts
                FROM {CALL_LOGS_FULL}
                ORDER BY created_at
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
            
            for row_num, call_id, started_at, ended_at, transcripts in calls:
                # Convert transcripts to preview text safely in Python
                transcript_preview = 'No transcript'
                if transcripts:
                    try:
                        if isinstance(transcripts, (dict, list)):
                            transcript_str = json.dumps(transcripts)
                        else:
                            transcript_str = str(transcripts)
                        
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
            logger.info("To extract student info from a call, use: python student_extraction.py --db-id <row_number>")
            logger.info("Or use UUID directly: python student_extraction.py --db-id <uuid_string>")
            
        finally:
            cursor.close()
            conn.close()
    
    async def extract_from_database(self, call_log_id, save_to_db: bool = False, save_to_json: bool = True) -> Optional[Dict]:
        """
        Extract student information from database call_logs table
        
        Args:
            call_log_id: ID from call_logs_voiceagent table (integer row number or UUID string)
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
                cursor.execute(f"""
                    SELECT id, transcripts, started_at, ended_at, lead_id
                    FROM (
                        SELECT 
                            id, 
                            transcripts, 
                            started_at, 
                            ended_at,
                            lead_id,
                            ROW_NUMBER() OVER (ORDER BY created_at) as row_num
                        FROM {CALL_LOGS_FULL}
                    ) ranked
                    WHERE row_num = %s
                """, (call_log_id,))
            else:
                # UUID string: Try direct UUID match or text match
                try:
                    cursor.execute(f"""
                        SELECT id, transcripts, started_at, ended_at, lead_id
                        FROM {CALL_LOGS_FULL}
                        WHERE id = %s::uuid
                    """, (str(call_log_id),))
                except (psycopg2.errors.InvalidTextRepresentation, psycopg2.errors.UndefinedFunction):
                    # Fallback: try text match
                    cursor.execute(f"""
                        SELECT id, transcripts, started_at, ended_at, lead_id
                        FROM {CALL_LOGS_FULL}
                        WHERE id::text = %s
                    """, (str(call_log_id),))
            
            call_data = cursor.fetchone()
            
            if not call_data:
                raise ValueError(f"Call log {call_log_id} not found in database")
            
            db_call_id, transcripts, started_at, ended_at, lead_id = call_data
            
            # Get transcript from transcripts column
            # Handle different data types: dict (JSONB), list, or string
            if not transcripts:
                raise ValueError(f"No transcript found for call {call_log_id}")
            
            # Convert transcripts to conversation text
            if isinstance(transcripts, dict):
                # If it's a dict, check if it contains a list of messages
                if 'messages' in transcripts and isinstance(transcripts['messages'], list):
                    # Structured format with messages array
                    conversation_log = transcripts['messages']
                    conversation_text = "\n".join([f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}" for entry in conversation_log])
                elif isinstance(transcripts, dict) and any(key in transcripts for key in ['role', 'message', 'text']):
                    # Single message dict - convert to text
                    role = transcripts.get('role', 'Unknown').title()
                    message = transcripts.get('message') or transcripts.get('text', '')
                    conversation_text = f"{role}: {message}"
                else:
                    # Try to extract text from dict
                    if 'text' in transcripts:
                        conversation_text = str(transcripts['text'])
                    elif 'transcript' in transcripts:
                        conversation_text = str(transcripts['transcript'])
                    elif 'content' in transcripts:
                        conversation_text = str(transcripts['content'])
                    else:
                        # Fallback: convert entire dict to JSON string
                        conversation_text = json.dumps(transcripts)
            elif isinstance(transcripts, list):
                # List format - convert to text
                conversation_text = "\n".join([f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}" if isinstance(entry, dict) else str(entry) for entry in transcripts])
            else:
                # String format - use as-is
                conversation_text = str(transcripts)
            
            logger.info(f"Call ID: {db_call_id}, Started: {started_at}, Ended: {ended_at}, Lead ID: {lead_id}")
            logger.info(f"Conversation text length: {len(conversation_text)} characters")
            
            # Get lead_category from post_call_analysis_voiceagent table (connected via call_log_id)
            lead_category = self._get_lead_category_from_analysis(cursor, db_call_id)
            if lead_category:
                logger.info(f"Found lead_category from post_call_analysis: {lead_category}")
            
            # Extract student information
            student_info = await self.extractor.extract_student_information(conversation_text)
            
            if student_info:
                logger.info("Student information extracted successfully")
                logger.info(f"Student Name: {student_info.get('student_name')}")
                logger.info(f"Parent Name: {student_info.get('parent_name')}")
                logger.info(f"Parent Contact: {student_info.get('parent_contact')}")
                logger.info(f"Program Country of Interest: {student_info.get('program_country_of_interest')}")
                
                # Set tags to lead_category if available (from post_call_analysis_voiceagent.lead_category)
                if lead_category:
                    student_info['tags'] = lead_category
                    logger.info(f"Setting tags to lead_category: {lead_category}")
                
                # Save to JSON file (default behavior)
                if save_to_json:
                    logger.info("Saving student information to JSON file...")
                    json_file = self.extractor.save_to_json(student_info, db_call_id, lead_id)
                    
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
                        lead_id, 
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
        description='Student Information Extraction Tool - Extract from call transcriptions (glinks.students_glinks structure)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all calls from database (shows row numbers matching --db-id usage)
    python student_extraction.py --list-calls
    
    # Extract student info and save to JSON file (default behavior)
    python student_extraction.py --db-id 23
    
    # Extract student info, save to JSON file, and also save to database
    python student_extraction.py --db-id 23 --save-db
    
    # Extract student info but skip JSON file (save to database only)
    python student_extraction.py --db-id 23 --save-db --no-save-json
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
            logger.info(f"Student Name: {result.get('student_name', 'N/A')}")
            logger.info(f"Parent Name: {result.get('parent_name', 'N/A')}")
            logger.info(f"Parent Contact: {result.get('parent_contact', 'N/A')}")
            logger.info(f"Email: {result.get('email', 'N/A')}")
            logger.info(f"Grade Year: {result.get('grade_year', 'N/A')}")
            logger.info(f"Curriculum: {result.get('curriculum', 'N/A')}")
            logger.info(f"School Name: {result.get('school_name', 'N/A')}")
            logger.info(f"Program Country of Interest: {result.get('program_country_of_interest', 'N/A')}")
            logger.info(f"Tags: {result.get('tags', 'N/A')}")
            logger.info("="*60)
            logger.info("Extraction complete! Data saved to JSON file in json_exports/ directory")
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
