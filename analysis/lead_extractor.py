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
import requests
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

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
    
    def _call_gemini_api(self, prompt: str, temperature: float = 0.2, max_output_tokens: int = 500) -> Optional[str]:
        """
        Call Gemini 2.0 Flash API - matches merged_analytics.py pattern
        
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
            
            # 10 second timeout - matches merged_analytics.py
            response = requests.post(url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
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
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text[:200]}")
            return None
            
        except requests.exceptions.Timeout:
            logger.error("Gemini API timeout after 10 seconds")
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
        result = self._call_gemini_api(prompt, temperature=0.2, max_output_tokens=500)
        
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
