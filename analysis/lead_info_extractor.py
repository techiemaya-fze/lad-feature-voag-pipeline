"""
Lead Information Extraction Module
Extracts comprehensive lead information from call transcriptions and saves to local JSON files.

This module is designed to be called as part of the post-call analysis pipeline.
It extracts user-provided information such as names, contact details, parent info,
education details, meeting times, etc.

Usage:
    from lead_info_extractor import LeadInfoExtractor
    
    extractor = LeadInfoExtractor()
    lead_info = await extractor.extract_lead_information(conversation_text, summary)
    if lead_info:
        json_path = extractor.save_to_json(lead_info, call_id)
"""

import os
import json
import re
import logging
import httpx
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


load_dotenv()

# Use existing logger from parent module or create new one
logger = logging.getLogger(__name__)


class LeadInfoExtractor:
    """Extract comprehensive lead information from conversation transcriptions"""
    
    def __init__(self, gemini_api_key: Optional[str] = None, cost_tracker: Optional[Dict] = None):
        """
        Initialize lead information extractor
        
        Args:
            gemini_api_key: Gemini API key (default: from GEMINI_API_KEY env var)
            cost_tracker: Optional reference to parent CallAnalytics cost_tracker dict
                         If provided, API call costs will be tracked there
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found - lead info extraction will be skipped")
        
        # Use provided cost_tracker or None (no tracking if standalone)
        self.cost_tracker = cost_tracker
        
        # Ensure json_exports directory exists
        self.exports_dir = Path(__file__).parent / "json_exports"
        self.exports_dir.mkdir(exist_ok=True)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximately 1 token = 4 characters)"""
        if not text:
            return 0
        return len(text) // 4
    
    async def _call_gemini_api(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 500) -> Optional[str]:
        """
        Call Gemini 2.0 Flash API asynchronously using httpx.AsyncClient
        
        Args:
            prompt: The prompt to send to Gemini
            temperature: Temperature for generation (default 0.2 for extraction)
            max_output_tokens: Maximum output tokens
            
        Returns:
            API response text or None if failed
        """
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available, skipping API call")
            return None
        
        input_tokens = self._estimate_tokens(prompt)
        logger.debug(f"Lead extraction API call - Input tokens: ~{input_tokens}, Max output: {max_output_tokens}")
        
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
            
            # Use httpx.AsyncClient for async HTTP requests
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                response_data = response.json()
                logger.debug("Lead extraction API call successful")
                
                # Track costs if cost_tracker is provided (from parent CallAnalytics)
                if self.cost_tracker is not None:
                    self.cost_tracker['api_calls'] += 1
                    
                    # Get actual token counts from response if available
                    if "usageMetadata" in response_data:
                        usage = response_data["usageMetadata"]
                        if "promptTokenCount" in usage:
                            self.cost_tracker['total_input_tokens'] += usage["promptTokenCount"]
                        else:
                            self.cost_tracker['total_input_tokens'] += input_tokens
                        if "candidatesTokenCount" in usage:
                            self.cost_tracker['total_output_tokens'] += usage["candidatesTokenCount"]
                    else:
                        self.cost_tracker['total_input_tokens'] += input_tokens
                
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    if "content" in response_data["candidates"][0]:
                        if "parts" in response_data["candidates"][0]["content"]:
                            output_text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            logger.debug(f"API response length: {len(output_text)} chars")
                            
                            # Track output tokens if no usageMetadata
                            if self.cost_tracker is not None and "usageMetadata" not in response_data:
                                output_tokens = self._estimate_tokens(output_text)
                                self.cost_tracker['total_output_tokens'] += output_tokens
                            
                            return output_text
                
                if "promptFeedback" in response_data:
                    logger.warning(f"Gemini API warning: {response_data.get('promptFeedback', {})}")
            
            return None
            
        except httpx.TimeoutException:
            logger.error("Gemini API timeout after 10 seconds")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Gemini API exception: {str(e)}", exc_info=True)
            return None
    
    def _parse_json_response(self, raw_text: Optional[str]) -> Optional[Dict]:
        """
        Parse JSON from LLM response - handles code fences and trailing text
        
        Args:
            raw_text: Raw text from LLM response
            
        Returns:
            Parsed dictionary or None if parsing failed
        """
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
        
        return " ".join(user_messages)
    
    async def extract_lead_information(self, conversation_text: str, summary: Optional[Dict] = None) -> Optional[Dict]:
        """
        Extract ALL lead information from conversation
        
        Args:
            conversation_text: Full conversation transcript
            summary: Optional summary dict from post-call analysis (for context)
            
        Returns:
            Dict with all lead-provided information or None if extraction failed
        """
        
        if not self.gemini_api_key:
            logger.debug("Skipping lead info extraction - no API key")
            return None
        
        # Extract user messages only
        user_text = self._extract_user_messages(conversation_text)
        
        if not user_text or len(user_text) < 10:
            logger.debug("Insufficient user text for lead info extraction")
            return None
        
        # Get context from summary if available
        contact_person = summary.get('contact_person', '') if summary else ''
        call_summary = summary.get('call_summary', '') if summary else ''
        
        # Log at DEBUG level to avoid PII exposure at INFO level
        logger.debug(f"Conversation length: {len(conversation_text)} chars, User text: {len(user_text)} chars")
        
        prompt = f"""Extract ALL information about the lead from this sales conversation. Capture EVERYTHING the LEAD/USER mentioned (NOT the agent/bot).

CONVERSATION:
{conversation_text}

IMPORTANT - Meeting Time Extraction:
- If agent suggests "Sunday at 11 AM or 3 PM?" and user says "Eleven AM", extract "Sunday at 11:00 AM" (NOT just "Eleven AM")
- Always include the day/date when extracting scheduled meeting times

CALL SUMMARY (for context):
{call_summary[:700] if call_summary else 'Not available'}

TASK: Extract ALL information that the LEAD provided during the conversation. This includes:

1. Personal Information:
   - Lead's name (first name, full name - the person speaking, NOT agent/bot names)
   - Contact details (email, phone, WhatsApp number, etc.)
   - Position/title/role
   - Company/business name

2. Contact Preferences & Meeting Scheduling:
   - Available time (when they prefer to be contacted)
   - Agreed meeting times (if agent suggests and user agrees - capture that time WITH day/date)
   - Callback requests
   - Preferred contact method

3. Educational/Background Information:
   - Education level/grade/class (e.g., "10th standard", "Grade 10")
   - School/college name
   - Curriculum/board type (CBSE, ICSE, IGCSE, IB, etc.)
   - Academic performance/grades/percentage

4. Family/Parent Information:
   - Parent/guardian name
   - Parent phone number or contact
   - Parent designation/profession
   - Parent workplace/company

5. Program/Interest Information:
   - Program interested in
   - Country interested
   - Intake year/month
   - Budget or pricing discussions

Respond in JSON format:

{{
    "first_name": "First name of the lead or null",
    "full_name": "Full name if mentioned or null",
    "email": "Email address if mentioned or null",
    "phone": "Phone number if mentioned or null",
    "whatsapp": "WhatsApp number if mentioned or null",
    "position": "Job title/position if mentioned or null",
    "company": "Company/business name if mentioned or null",
    "available_time": "Scheduled/confirmed meeting time with day (e.g., 'Sunday at 11:00 AM') or null",
    "contact_preference": "Preferred contact method if mentioned or null",
    "location": "Location/city if mentioned or null",
    "education_level": "Education level/grade/class if mentioned or null",
    "school_name": "School/college name if mentioned or null",
    "curriculum": "Curriculum/board type if mentioned or null",
    "academic_performance": "Academic grades/percentage if mentioned or null",
    "parent_name": "Parent/guardian name if mentioned or null",
    "parent_phone": "Parent phone if mentioned or null",
    "parent_designation": "Parent profession if mentioned or null",
    "parent_workplace": "Parent workplace if mentioned or null",
    "program_interested": "Program/course interested in or null",
    "country_interested": "Country of interest or null",
    "intake_year": "Year when student wants to start or null",
    "intake_month": "Month when student wants to start or null",
    "budget": "Budget if mentioned or null",
    "additional_notes": "Any other relevant information or null"
}}

CRITICAL RULES:
1. ONLY extract information PROVIDED BY THE LEAD/USER, NOT the agent/bot
2. EXCEPTION: If agent suggests a meeting time and user AGREES, extract the COMPLETE agreed time
3. Extract EVERYTHING the lead mentioned - be comprehensive
4. Do NOT extract agent/bot names (like "Nithya", "Mira Singh", etc.)
5. If a field is not mentioned, set it to null
"""
        
        logger.debug("Calling Gemini API for lead info extraction...")
        result = await self._call_gemini_api(prompt, temperature=0.2, max_output_tokens=500)
        
        if not result:
            logger.warning("LLM did not return lead information")
            return None
        
        # Parse JSON response
        parsed_data = self._parse_json_response(result)
        
        if not parsed_data:
            logger.warning("Failed to parse lead info JSON from LLM response")
            return None
        
        # Clean the data - remove null/None string values
        lead_info = {}
        for key, value in parsed_data.items():
            if value is not None:
                if isinstance(value, str):
                    cleaned = value.strip()
                    if cleaned and cleaned.lower() not in ['none', 'null', 'n/a']:
                        lead_info[key] = cleaned
                else:
                    lead_info[key] = value
        
        # If first_name not found but contact_person available from summary, use it
        if 'first_name' not in lead_info and contact_person:
            name_parts = contact_person.split()
            if name_parts:
                lead_info['first_name'] = name_parts[0]
                if len(name_parts) > 1 and 'full_name' not in lead_info:
                    lead_info['full_name'] = contact_person
        
        if not lead_info:
            logger.debug("No lead information found in this call")
            return None
        
        logger.info(f"Lead info extracted: {len(lead_info)} fields")
        return lead_info
    
    def save_to_json(self, lead_info: Dict, call_id: str) -> Optional[str]:
        """
        Save extracted lead information to JSON file
        
        Args:
            lead_info: Extracted lead information dictionary
            call_id: Call identifier for filename
            
        Returns:
            Path to saved JSON file or None if failed
        """
        try:
            # Prepare output data
            output_data = {
                "call_id": call_id,
                "extracted_at": datetime.now().isoformat(),
                "lead_info": lead_info
            }
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            call_id_short = str(call_id)[:8] if call_id else 'unknown'
            filename = f"lead_info_{call_id_short}_{timestamp}.json"
            
            filepath = self.exports_dir / filename
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Lead info saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save lead info JSON: {e}", exc_info=True)
            return None


class StandaloneLeadInfoExtractor:
    """
    Standalone lead info extractor for local runs, similar to StandaloneStudentExtractor in lad_dev.py.
    
    - Reads calls from lad_dev.voice_call_logs
    - Extracts lead information with LeadInfoExtractor (Gemini)
    - Saves to JSON and optionally to lad_dev.leads via async LeadInfoStorage
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.extractor = LeadInfoExtractor()
        self.db_config = db_config
    
    def _get_db_connection(self):
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        if not self.db_config:
            raise ValueError("Database config not provided. Use --db-host, --db-name, --db-user, --db-pass or .env file")
        return psycopg2.connect(**self.db_config)
    
    async def list_database_calls(self) -> None:
        """
        List calls from lad_dev.voice_call_logs with row numbers (like lad_dev.StandaloneStudentExtractor).
        """
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        conn = self._get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        logger.info("Listing calls from database (lad_dev.voice_call_logs) for lead info...")
        logger.info("=" * 80)
        try:
            cursor.execute(
                """
                SELECT
                    ROW_NUMBER() OVER (ORDER BY ctid) as row_num,
                    id,
                    started_at,
                    ended_at,
                    transcripts,
                    lead_id,
                    tenant_id
                FROM lad_dev.voice_call_logs
                ORDER BY ctid
                LIMIT 500000
                """
            )
            rows = cursor.fetchall()
            if not rows:
                logger.warning("No calls found in lad_dev.voice_call_logs.")
                return
            
            header = f"{'Row':<6} {'UUID':<40} {'Started At':<20} {'Duration':<10} {'Lead ID':<40} {'Transcript Preview'}"
            logger.info(header)
            logger.info("-" * 80)
            for row in rows:
                call_id = row["id"]
                started_at = row.get("started_at")
                ended_at = row.get("ended_at")
                transcripts = row.get("transcripts")
                lead_id = row.get("lead_id")
                
                transcript_preview = "No transcript"
                if transcripts:
                    try:
                        if isinstance(transcripts, (dict, list)):
                            t_str = json.dumps(transcripts)
                        else:
                            t_str = str(transcripts)
                        if t_str:
                            preview = t_str[:50]
                            if len(t_str) > 50:
                                preview += "..."
                            transcript_preview = preview
                    except Exception:
                        transcript_preview = "Invalid transcript format"
                
                duration = ""
                if started_at and ended_at:
                    try:
                        duration_seconds = int((ended_at - started_at).total_seconds())
                        duration = f"{duration_seconds}s"
                    except Exception:
                        duration = ""
                
                started_str = started_at.strftime("%Y-%m-%d %H:%M:%S") if started_at else "N/A"
                lead_id_str = str(lead_id) if lead_id else "None"
                row_num = row["row_num"]
                info = f"{row_num:<6} {str(call_id):<40} {started_str:<20} {duration:<10} {lead_id_str:<40} {transcript_preview}"
                logger.info(info)
            
            logger.info("-" * 80)
            logger.info("Row numbers match pgAdmin's default display order (physical storage order)")
            logger.info("To extract lead info from a call, use: python lead_info_extractor.py --db-id <row_number>")
            logger.info("Or use UUID directly: python lead_info_extractor.py --db-id <uuid_string>")
        finally:
            cursor.close()
            conn.close()

    async def extract_from_database(
        self,
        call_log_id,
        save_to_db: bool = True,
        save_to_json: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract lead info from lad_dev.voice_call_logs by row number or UUID.
        
        - call_log_id: row number (int) or UUID string from lad_dev.voice_call_logs.id
        - save_to_db: if True, also save to lad_dev.leads using async LeadInfoStorage
        - save_to_json: if True, save JSON export
        """
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        conn = self._get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        logger.info(f"Fetching call from database lad_dev.voice_call_logs (ID: {call_log_id}) for lead info")
        try:
            # Interpret call_log_id as row number or UUID
            try:
                row_num = int(call_log_id)
                use_row = True
            except (ValueError, TypeError):
                use_row = False
            
            if use_row:
                cursor.execute(
                    """
                    SELECT id, transcripts, started_at, ended_at, lead_id, tenant_id
                    FROM (
                        SELECT
                            id,
                            transcripts,
                            started_at,
                            ended_at,
                            lead_id,
                            tenant_id,
                            ROW_NUMBER() OVER (ORDER BY ctid) as row_num
                        FROM lad_dev.voice_call_logs
                    ) ranked
                    WHERE row_num = %s
                    """,
                    (row_num,),
                )
            else:
                try:
                    cursor.execute(
                        """
                        SELECT id, transcripts, started_at, ended_at, lead_id, tenant_id
                        FROM lad_dev.voice_call_logs
                        WHERE id = %s::uuid
                        """,
                        (str(call_log_id),),
                    )
                except Exception:
                    cursor.execute(
                        """
                        SELECT id, transcripts, started_at, ended_at, lead_id, tenant_id
                        FROM lad_dev.voice_call_logs
                        WHERE id::text = %s
                        """,
                        (str(call_log_id),),
                    )
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Call log {call_log_id} not found in lad_dev.voice_call_logs")
            
            db_call_id = row["id"]
            transcripts = row["transcripts"]
            started_at = row.get("started_at")
            ended_at = row.get("ended_at")
            lead_id = row.get("lead_id")
            tenant_id = row.get("tenant_id")
            
            if not transcripts:
                raise ValueError(f"No transcript found for call {call_log_id}")
            
            # Convert transcripts to conversation text (same style as lad_dev.py)
            if isinstance(transcripts, dict):
                if "messages" in transcripts and isinstance(transcripts["messages"], list):
                    conversation_log: List[Dict[str, Any]] = transcripts["messages"]
                    conversation_text = "\n".join(
                        f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}"
                        for entry in conversation_log
                    )
                elif any(k in transcripts for k in ["role", "message", "text"]):
                    role = transcripts.get("role", "Unknown").title()
                    message = transcripts.get("message") or transcripts.get("text", "")
                    conversation_text = f"{role}: {message}"
                else:
                    if "text" in transcripts:
                        conversation_text = str(transcripts["text"])
                    elif "transcript" in transcripts:
                        conversation_text = str(transcripts["transcript"])
                    elif "content" in transcripts:
                        conversation_text = str(transcripts["content"])
                    else:
                        conversation_text = json.dumps(transcripts)
            elif isinstance(transcripts, list):
                conversation_text = "\n".join(
                    f"{entry.get('role', 'Unknown').title()}: {entry.get('message', entry.get('text', ''))}"
                    if isinstance(entry, dict)
                    else str(entry)
                    for entry in transcripts
                )
            else:
                conversation_text = str(transcripts)
            
            logger.info(
                "Call ID: %s, Started: %s, Ended: %s, Lead ID: %s, Tenant ID: %s",
                db_call_id,
                started_at,
                ended_at,
                lead_id,
                tenant_id,
            )
            logger.info("Conversation text length: %s characters", len(conversation_text))
            
            # Extract lead information
            result = await extract_and_save_lead_info(conversation_text, str(db_call_id), summary=None)
            if not result:
                logger.info("No lead information found in this call transcript")
                return None
            
            lead_info = result.get("lead_info")
            json_path = result.get("json_path")
            
            if json_path and save_to_json:
                logger.info("Lead info saved to JSON file: %s", json_path)
            
            # Save to lad_dev.leads using asyncpg storage (LeadInfoStorage) if requested
            if save_to_db and lead_info:
                try:
                    from db.lead_info_storage import LeadInfoStorage

                    storage = LeadInfoStorage()
                    try:
                        tenant_str = str(tenant_id) if tenant_id else None
                        lead_db_id = await storage.save_lead_from_extraction(
                            tenant_id=tenant_str,
                            lead_info=lead_info,
                            call_id=str(db_call_id),
                        )
                        if lead_db_id:
                            logger.info(
                                "Lead info saved to lad_dev.leads with id=%s for call_log_id=%s",
                                lead_db_id,
                                db_call_id,
                            )
                        else:
                            logger.warning(
                                "Lead info not saved to lad_dev.leads (missing phone/tenant) for call_log_id=%s",
                                db_call_id,
                            )
                    finally:
                        await storage.close()
                except ImportError:
                    logger.warning(
                        "LeadInfoStorage not available; skipping lad_dev.leads save for call_log_id=%s",
                        db_call_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to save lead info to lad_dev.leads for call_log_id=%s: %s",
                        db_call_id,
                        exc,
                        exc_info=True,
                    )
            
            return {
                "lead_info": lead_info,
                "json_path": json_path,
                "call_id": str(db_call_id),
                "lead_id": str(lead_id) if lead_id else None,
                "tenant_id": str(tenant_id) if tenant_id else None,
            }
        finally:
            cursor.close()
            conn.close()
