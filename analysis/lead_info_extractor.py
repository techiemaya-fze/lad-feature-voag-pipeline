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
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Structured output client for guaranteed JSON responses
from .gemini_client import generate_with_schema_retry, LEAD_INFO_SCHEMA

load_dotenv()

# Schema configuration
SCHEMA = os.getenv("DB_SCHEMA", "lad_dev")

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
    
    def _call_gemini_structured(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 8192) -> Optional[Dict]:
        """
        Call Gemini API with structured output schema - guarantees proper JSON response
        
        Args:
            prompt: The prompt to send to Gemini
            temperature: Temperature for generation (default 0.2 for extraction)
            max_output_tokens: Maximum output tokens
            
        Returns:
            Parsed JSON dict or None if failed
        """
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available, skipping API call")
            return None
        
        logger.debug(f"Lead extraction API call with structured output")
        
        try:
            # Track API call if cost_tracker is available
            if self.cost_tracker is not None:
                self.cost_tracker['api_calls'] += 1
            
            # Use structured output for guaranteed JSON response
            result = generate_with_schema_retry(
                prompt=prompt,
                schema=LEAD_INFO_SCHEMA,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            
            if result:
                # Extract and track usage metadata
                if self.cost_tracker is not None and '_usage_metadata' in result:
                    usage = result.pop('_usage_metadata')
                    self.cost_tracker['total_input_tokens'] += usage.get('prompt_token_count', 0)
                    self.cost_tracker['total_output_tokens'] += usage.get('candidates_token_count', 0)
                    logger.debug(f"Lead extraction Gemini usage: {usage}")

                logger.debug(f"Lead extraction structured response received")
                return result
            else:
                logger.warning("No result from structured generation")
                return None
            
        except Exception as e:
            logger.error(f"Gemini structured API exception: {str(e)}", exc_info=True)
            return None
    
    # NOTE: _parse_json_response removed - no longer needed with structured output
    # The generate_with_schema_retry function guarantees valid JSON responses
    
    def _extract_user_messages(self, conversation_text: str) -> str:
        """Extract only user messages from conversation (exclude bot/agent messages)"""
        lines = conversation_text.split('\n')
        user_messages = []
        
        for line in lines:
            line = line.strip()
            # Only include lines that start with "User:"
            if line.startswith("User:"):
                user_messages.append(line.replace("User:", "").strip())
            # Exclude agent messages
            elif line.startswith("Agent:"):
                continue
            # Lines without prefix might be user messages (legacy format)
            elif line and not any(line.startswith(prefix) for prefix in ["Agent:", "User:"]):
                user_messages.append(line)
        
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
        
        logger.debug("Calling Gemini API for lead info extraction with structured output...")
        # Using structured output - result is already a parsed dict
        parsed_data = self._call_gemini_structured(prompt, temperature=0.2, max_output_tokens=8192)
        
        if not parsed_data:
            logger.warning("LLM did not return lead information")
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


# Module-level instance for easy import
lead_extractor = LeadInfoExtractor()


async def extract_and_save_lead_info(conversation_text: str, call_id: str, summary: Optional[Dict] = None) -> Optional[str]:
    """
    Convenience function to extract and save lead info in one call
    
    Args:
        conversation_text: Full conversation transcript
        call_id: Call identifier
        summary: Optional summary dict from post-call analysis
        
    Returns:
        Path to saved JSON file or None
    """
    lead_info = await lead_extractor.extract_lead_information(conversation_text, summary)
    if lead_info:
        return lead_extractor.save_to_json(lead_info, call_id)
    return None