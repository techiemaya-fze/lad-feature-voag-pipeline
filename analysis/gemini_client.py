"""

Gemini Client with Structured Output



Shared Gemini client wrapper that uses google-genai library with structured output schemas.

Ensures proper JSON responses every time by using response_schema.

"""



import os

import json

import logging

import asyncio

from typing import Optional, Dict, Any, Type, Tuple, Union

from google import genai

from google.genai import types



from .gemini_config import (

    MODEL_NAME,

    DEFAULT_TEMPERATURE,

    DEFAULT_MAX_OUTPUT_TOKENS,

    MAX_RETRIES,

    RETRY_DELAY_BASE,

)



logger = logging.getLogger(__name__)



# Module-level client instance (lazy initialization)

_client: Optional[genai.Client] = None





def get_client() -> genai.Client:

    """Get or create the Gemini client singleton."""

    global _client

    if _client is None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:

            raise ValueError("GEMINI_API_KEY environment variable not set")

        _client = genai.Client(api_key=api_key)

    return _client





def remove_thought_signatures(text: str) -> str:

    """

    Remove thought signatures from Gemini 3 model responses.

    

    Thought signatures are encrypted representations of internal reasoning

    that can appear in model output, often as base64-like strings or

    special markers like [THINKING], <thought>, etc.

    """

    import re

    

    if not text:

        return text

    

    # Remove common thought signature patterns

    patterns = [

        # Base64-like blocks that appear randomly (long alphanumeric strings)

        r'(?<![a-zA-Z0-9])[A-Za-z0-9+/]{50,}={0,2}(?![a-zA-Z0-9])',

        # [THINKING] or [THOUGHT] markers

        r'\[THINKING\][\s\S]*?\[/THINKING\]',

        r'\[THOUGHT\][\s\S]*?\[/THOUGHT\]',

        # <thought> XML-style markers

        r'<thought>[\s\S]*?</thought>',

        r'<thinking>[\s\S]*?</thinking>',

        # Thought summary markers

        r'<thought_summary>[\s\S]*?</thought_summary>',

        # Internal reasoning markers

        r'\*\*Internal reasoning:\*\*[\s\S]*?(?=\n\n|\Z)',

    ]

    

    cleaned = text

    for pattern in patterns:

        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    

    return cleaned.strip()





def extract_json_from_text(text: str) -> Optional[str]:

    """

    Extract JSON from text that may contain preamble like 'Here is the JSON'

    or thought signatures from Gemini 3 thinking models.

    

    Handles cases like:

    - 'Here is the JSON\n{"key": "value"}'

    - 'Here is the JSON:\n```json\n{"key": "value"}\n```'

    - Direct JSON response

    - Response with thought signatures interspersed

    """

    if not text:

        return None

    

    # First, remove any thought signatures

    text = remove_thought_signatures(text)

    text = text.strip()

    

    # If it already starts with { or [, return as-is

    if text.startswith("{") or text.startswith("["):

        return text

    

    # Try to find JSON in the text

    # Look for code blocks first

    import re

    

    # Check for ```json or ``` code blocks

    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)

    if code_block_match:

        json_content = code_block_match.group(1).strip()

        if json_content.startswith("{") or json_content.startswith("["):

            return json_content

    

    # Find first { or [ character

    brace_idx = text.find("{")

    bracket_idx = text.find("[")

    

    if brace_idx == -1 and bracket_idx == -1:

        return None

    

    # Use the first occurring bracket

    start_idx = brace_idx if brace_idx == -1 else (min(brace_idx, bracket_idx) if bracket_idx != -1 else brace_idx)
    

    # Extract from the first bracket to end

    json_text = text[start_idx:]

    

    # Try to find the complete JSON by matching braces

    if json_text.startswith("{"):

        brace_count = 0

        in_string = False

        escape_next = False

        end_pos = 0

        

        for i, char in enumerate(json_text):

            if escape_next:

                escape_next = False

                continue

            if char == "\\":

                escape_next = True

                continue

            if char == '"' and not escape_next:

                in_string = not in_string

                continue

            if in_string:

                continue

            if char == "{":

                brace_count += 1

            elif char == "}":

                brace_count -= 1

                if brace_count == 0:

                    end_pos = i + 1

                    break

        

        if end_pos > 0:

            return json_text[:end_pos]

    

    return json_text





def generate_with_schema(

    prompt: str,

    schema: types.Schema,

    temperature: float = DEFAULT_TEMPERATURE,

    max_output_tokens: int = 65536,  # Gemini 3 Flash maximum limit

    model: str = MODEL_NAME,

    thinking_level: str = "MINIMAL",  # Use MINIMAL to minimize thinking token usage

) -> Optional[Dict[str, Any]]:

    """

    Generate content with a structured JSON schema.

    

    Args:

        prompt: The input prompt

        schema: A genai.types.Schema defining the expected JSON structure

        temperature: Generation temperature (0.0-1.0)

        max_output_tokens: Maximum tokens in response (default 8192)

        model: Model name to use

        thinking_level: Thinking level for Gemini 3 (MINIMAL, LOW, MEDIUM, HIGH)

    

    Returns:

        Parsed JSON dict or None if generation failed

    """

    try:

        client = get_client()

        

        # Add explicit JSON-only instruction to prompt to avoid preamble text

        json_prompt = f"""{prompt}



CRITICAL: Return ONLY raw JSON. No explanations, no preamble, no "Here is the JSON", no markdown code blocks. Just the JSON object starting with {{ and ending with }}."""

        

        # Use thinking_config with thinking_level for Gemini 3 models

        config = types.GenerateContentConfig(

            thinking_config=types.ThinkingConfig(

                thinking_level=thinking_level,

            ),

            response_mime_type="application/json",

            response_schema=schema,

            temperature=temperature,

            max_output_tokens=max_output_tokens,

        )

        

        response = client.models.generate_content(

            model=model,

            contents=[types.Content(role="user", parts=[types.Part.from_text(text=json_prompt)])],

            config=config,

        )

        

        # Log detailed response info for debugging

        if hasattr(response, 'candidates') and response.candidates:

            candidate = response.candidates[0]

            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')

            logger.debug(f"API finish_reason: {finish_reason}")

            

            # Check for truncation or other issues

            if str(finish_reason) not in ['STOP', 'FinishReason.STOP', 'END_TURN', 'FinishReason.END_TURN']:

                logger.warning(f"API response may be incomplete. finish_reason: {finish_reason}")

        

        if response.text:

            raw_text = response.text

            logger.debug(f"Raw Gemini response (first 500 chars): {raw_text[:500]}")

            logger.debug(f"Response length: {len(raw_text)} chars")

            

            # Extract usage metadata

            usage_metadata = {}

            if hasattr(response, 'usage_metadata'):

                usage_metadata = {

                    'prompt_token_count': response.usage_metadata.prompt_token_count,

                    'candidates_token_count': response.usage_metadata.candidates_token_count,

                    'total_token_count': response.usage_metadata.total_token_count

                }

            

            # First, try to clean up any preamble text

            cleaned_text = extract_json_from_text(raw_text)

            if not cleaned_text:

                logger.warning(f"Could not extract JSON from response: {raw_text[:200]}")

                return None

            

            try:

                result = json.loads(cleaned_text)

                # Inject usage metadata if it's a dictionary

                if isinstance(result, dict) and usage_metadata:

                    result['_usage_metadata'] = usage_metadata

                return result

            except json.JSONDecodeError as json_err:

                # Log the full raw response for debugging

                logger.error(f"JSON decode error: {json_err}")

                logger.error(f"Raw response that failed to parse:\n{raw_text}")

                logger.error(f"Response length: {len(raw_text)} chars - may be truncated")

                

                # Save failed response to file for analysis

                import os

                from datetime import datetime

                debug_dir = os.path.join(os.path.dirname(__file__), "logs", "debug_responses")

                os.makedirs(debug_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                debug_file = os.path.join(debug_dir, f"failed_response_{timestamp}.txt")

                with open(debug_file, "w", encoding="utf-8") as f:

                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")

                    f.write(f"Error: {json_err}\n")

                    f.write(f"Response length: {len(raw_text)} chars\n")

                    if hasattr(response, 'candidates') and response.candidates:

                        f.write(f"Finish reason: {getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')}\n")

                    f.write(f"Prompt (first 500 chars): {prompt[:500]}\n")

                    f.write("="*60 + "\n")

                    f.write(f"Raw Response:\n{raw_text}\n")

                logger.error(f"Failed response saved to: {debug_file}")

                

                 # Last resort: try JSONDecoder.raw_decode

                try:

                    decoder = json.JSONDecoder()

                    if cleaned_text.startswith("{") or cleaned_text.startswith("["):

                        result, _ = decoder.raw_decode(cleaned_text)

                        logger.info("JSON recovered using raw_decode")

                        # Inject usage metadata if it's a dictionary

                        if isinstance(result, dict) and usage_metadata:

                            result['_usage_metadata'] = usage_metadata

                        return result

                except Exception as recovery_err:

                    logger.error(f"JSON recovery also failed: {recovery_err}")

                

                return None

        else:

            logger.warning("Empty response from Gemini API")

            if hasattr(response, 'candidates') and response.candidates:

                logger.warning(f"Finish reason: {getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')}")

        return None

        

    except Exception as e:

        logger.error(f"Gemini structured generation error: {e}")

        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        return None





def generate_with_schema_retry(

    prompt: str,

    schema: types.Schema,

    temperature: float = DEFAULT_TEMPERATURE,

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,

    model: str = MODEL_NAME,

    max_retries: int = MAX_RETRIES,

) -> Optional[Dict[str, Any]]:

    """

    Generate content with structured schema and retry logic.

    

    Includes exponential backoff for rate limit errors.

    """

    import time

    

    for attempt in range(max_retries):

        try:

            result = generate_with_schema(

                prompt=prompt,

                schema=schema,

                temperature=temperature,

                max_output_tokens=max_output_tokens,

                model=model,

            )

            if result is not None:

                return result

                

        except Exception as e:

            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str:

                wait_time = RETRY_DELAY_BASE * (2 ** attempt)

                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")

                time.sleep(wait_time)

                continue

            else:

                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")

                if attempt == max_retries - 1:

                    return None

    

    return None





async def generate_with_schema_async(

    prompt: str,

    schema: types.Schema,

    temperature: float = DEFAULT_TEMPERATURE,

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,

    model: str = MODEL_NAME,

    max_retries: int = MAX_RETRIES,

) -> Optional[Dict[str, Any]]:

    """

    Async version of generate_with_schema_retry.

    

    Uses asyncio.sleep for non-blocking retries.

    """

    for attempt in range(max_retries):

        try:

            # Run the sync client in a thread pool to avoid blocking

            result = await asyncio.to_thread(

                generate_with_schema,

                prompt=prompt,

                schema=schema,

                temperature=temperature,

                max_output_tokens=max_output_tokens,

                model=model,

            )

            if result is not None:

                return result

                

        except Exception as e:

            error_str = str(e).lower()

            if "429" in error_str or "rate limit" in error_str:

                wait_time = RETRY_DELAY_BASE * (2 ** attempt)

                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")

                await asyncio.sleep(wait_time)

                continue

            else:

                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")

                if attempt == max_retries - 1:

                    return None

    

    return None





def generate_text(

    prompt: str,

    temperature: float = DEFAULT_TEMPERATURE,

    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,

    model: str = MODEL_NAME,

    include_usage: bool = False,

) -> Any:

    """

    Generate plain text content (no structured schema).

    

    Args:

        include_usage: If True, returns (text, usage_dict). Default False returns just text.

    """

    try:

        client = get_client()

        

        config = types.GenerateContentConfig(

            temperature=temperature,

            max_output_tokens=max_output_tokens,

        )

        

        response = client.models.generate_content(

            model=model,

            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],

            config=config,

        )

        

        text = response.text if response.text else None

        

        if include_usage:

            usage_metadata = {}

            if hasattr(response, 'usage_metadata'):

                usage_metadata = {

                    'prompt_token_count': response.usage_metadata.prompt_token_count,

                    'candidates_token_count': response.usage_metadata.candidates_token_count,

                    'total_token_count': response.usage_metadata.total_token_count

                }

            return text, usage_metadata

            

        return text

        

    except Exception as e:

        logger.error(f"Gemini text generation error: {e}")

        if include_usage:

            return None, {}

        return None





# ============================================================================

# Pre-defined Schemas for Each Module

# Carefully designed to match EXACT JSON structure expected by each file

# ============================================================================



# -----------------------------------------------------------------------------

# LEAD INFO EXTRACTOR SCHEMA (lead_info_extractor.py)

# All 23 fields expected by the extraction function - all STRING type

# -----------------------------------------------------------------------------

LEAD_INFO_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    properties={

        "first_name": types.Schema(type=types.Type.STRING, description="First name of the lead"),

        "full_name": types.Schema(type=types.Type.STRING, description="Full name if mentioned"),

        "email": types.Schema(type=types.Type.STRING, description="Email address if mentioned"),

        "phone": types.Schema(type=types.Type.STRING, description="Phone number if mentioned"),

        "whatsapp": types.Schema(type=types.Type.STRING, description="WhatsApp number if mentioned"),

        "position": types.Schema(type=types.Type.STRING, description="Job title/position if mentioned"),

        "company": types.Schema(type=types.Type.STRING, description="Company/business name if mentioned"),

        "available_time": types.Schema(type=types.Type.STRING, description="Scheduled/confirmed meeting time with day"),

        "contact_preference": types.Schema(type=types.Type.STRING, description="Preferred contact method if mentioned"),

        "location": types.Schema(type=types.Type.STRING, description="Location/city if mentioned"),

        "education_level": types.Schema(type=types.Type.STRING, description="Education level/grade/class if mentioned"),

        "school_name": types.Schema(type=types.Type.STRING, description="School/college name if mentioned"),

        "curriculum": types.Schema(type=types.Type.STRING, description="Curriculum/board type if mentioned"),

        "academic_performance": types.Schema(type=types.Type.STRING, description="Academic grades/percentage if mentioned"),

        "parent_name": types.Schema(type=types.Type.STRING, description="Parent/guardian name if mentioned"),

        "parent_phone": types.Schema(type=types.Type.STRING, description="Parent phone if mentioned"),

        "parent_designation": types.Schema(type=types.Type.STRING, description="Parent profession if mentioned"),

        "parent_workplace": types.Schema(type=types.Type.STRING, description="Parent workplace if mentioned"),

        "program_interested": types.Schema(type=types.Type.STRING, description="Program/course interested in"),

        "country_interested": types.Schema(type=types.Type.STRING, description="Country of interest"),

        "intake_year": types.Schema(type=types.Type.STRING, description="Year when student wants to start"),

        "intake_month": types.Schema(type=types.Type.STRING, description="Month when student wants to start"),

        "budget": types.Schema(type=types.Type.STRING, description="Budget if mentioned"),

        "additional_notes": types.Schema(type=types.Type.STRING, description="Any other relevant information"),

    },

)



# -----------------------------------------------------------------------------

# SENTIMENT ANALYSIS SCHEMA (merged_analytics.py - _calculate_sentiment_with_llm)

# Required: category, sentiment_score, confidence. All NUMBER types for scores

# -----------------------------------------------------------------------------

SENTIMENT_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    required=["category", "sentiment_score", "confidence"],

    properties={

        "category": types.Schema(

            type=types.Type.STRING, 

            description="Positive, Neutral, Negative, or Very Interested"

        ),

        "sentiment_score": types.Schema(

            type=types.Type.NUMBER, 

            description="Score from -1.0 (very negative) to +1.0 (very positive)"

        ),

        "confidence": types.Schema(

            type=types.Type.NUMBER, 

            description="Confidence score from 0.0 to 1.0"

        ),

        "reasoning": types.Schema(

            type=types.Type.STRING, 

            description="Brief explanation of sentiment analysis"

        ),

    },

)



# -----------------------------------------------------------------------------

# LEAD DISPOSITION SCHEMA (merged_analytics.py - _determine_lead_disposition_llm)

# Required: disposition, action, reasoning, confidence (all STRING)

# -----------------------------------------------------------------------------

DISPOSITION_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    required=["disposition", "action", "reasoning", "confidence"],

    properties={

        "disposition": types.Schema(

            type=types.Type.STRING, 

            description="A, B, C, or D representing disposition category"

        ),

        "action": types.Schema(

            type=types.Type.STRING, 

            description="Recommended action for this lead"

        ),

        "reasoning": types.Schema(

            type=types.Type.STRING, 

            description="One sentence explaining the reasoning"

        ),

        "confidence": types.Schema(

            type=types.Type.STRING, 

            description="High, Medium, or Low"

        ),

    },

)



# -----------------------------------------------------------------------------

# LLM VALIDATION SCHEMA (merged_analytics.py - _validate_sentiment_with_llm)

# Required: sentiment, reason (both STRING)

# -----------------------------------------------------------------------------

LLM_VALIDATION_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    required=["sentiment", "reason"],

    properties={

        "sentiment": types.Schema(

            type=types.Type.STRING, 

            description="Positive, Neutral, or Negative"

        ),

        "reason": types.Schema(

            type=types.Type.STRING, 

            description="Brief explanation for the sentiment validation"

        ),

    },

)



# -----------------------------------------------------------------------------

# LAD DEV / STUDENT INFO SCHEMA (lad_dev.py - extract_student_information)

# Contains nested metadata object with 14 optional fields

# -----------------------------------------------------------------------------

LAD_STUDENT_INFO_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    properties={

        "student_parent_name": types.Schema(

            type=types.Type.STRING, 

            description="Parent name extracted from phrases like 'My father name is'"

        ),

        "parent_designation": types.Schema(

            type=types.Type.STRING, 

            description="Parent profession extracted from phrases like 'He is doing business'"

        ),

        "program_interested_in": types.Schema(

            type=types.Type.STRING, 

            description="Educational program/course/degree student is interested in"

        ),

        "country_interested": types.Schema(

            type=types.Type.STRING, 

            description="Country where student wants to study"

        ),

        "intake_year": types.Schema(

            type=types.Type.INTEGER, 

            description="Year when student wants to start (e.g., 2025, 2026)"

        ),

        "intake_month": types.Schema(

            type=types.Type.STRING, 

            description="Month when student wants to start (e.g., 'January', 'September')"

        ),

        "metadata": types.Schema(

            type=types.Type.OBJECT,

            properties={

                "email": types.Schema(type=types.Type.STRING, description="Email address if mentioned"),

                "phone": types.Schema(type=types.Type.STRING, description="Phone number if mentioned"),

                "percentage": types.Schema(type=types.Type.STRING, description="Academic percentage/score if mentioned"),

                "grades": types.Schema(type=types.Type.STRING, description="Academic grades if mentioned"),

                "class": types.Schema(type=types.Type.STRING, description="Current class/grade if mentioned"),

                "school_name": types.Schema(type=types.Type.STRING, description="Current school name if mentioned"),

                "curriculum": types.Schema(type=types.Type.STRING, description="Curriculum/board type if mentioned"),

                "address": types.Schema(type=types.Type.STRING, description="Address or location if mentioned"),

                "budget": types.Schema(type=types.Type.STRING, description="Budget or financial information if mentioned"),

                "preferred_university": types.Schema(type=types.Type.STRING, description="Preferred university if mentioned"),

                "subject_interests": types.Schema(type=types.Type.STRING, description="Subject interests if mentioned"),

                "followup_time": types.Schema(type=types.Type.STRING, description="For AUTO_FOLLOWUP - callback time like 'in 30 minutes'"),

                "slot_booked_for": types.Schema(type=types.Type.STRING, description="For AUTO_CONSULTATION - full date like 'Tuesday, January 13, 2026 at 1:30 PM'"),

                "available_time": types.Schema(type=types.Type.STRING, description="User availability when no callback confirmed"),

                "summary_last_call": types.Schema(type=types.Type.STRING, description="1-2 sentence summary of the call"),

                "additional_notes": types.Schema(type=types.Type.STRING, description="Any other relevant information"),

            },

        ),

    },

)



# -----------------------------------------------------------------------------

# STUDENT EXTRACTOR SCHEMA (student_extractor.py - glinks.students_glinks table)

# 19 fields for student record extraction - mostly STRING type

# -----------------------------------------------------------------------------

STUDENT_EXTRACTOR_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    properties={

        "student_name": types.Schema(type=types.Type.STRING, description="Full name of the student"),

        "parent_name": types.Schema(type=types.Type.STRING, description="Parent name from phrases like 'My father name is'"),

        "parent_contact": types.Schema(type=types.Type.STRING, description="Phone number of parent"),

        "parents_profession": types.Schema(type=types.Type.STRING, description="Parent profession"),

        "parents_workplace": types.Schema(type=types.Type.STRING, description="Parent's workplace/company"),

        "email": types.Schema(type=types.Type.STRING, description="Email address"),

        "country_of_residence": types.Schema(type=types.Type.STRING, description="Country where student lives"),

        "nationality": types.Schema(type=types.Type.STRING, description="Student's nationality"),

        "grade_year": types.Schema(type=types.Type.STRING, description="Current grade/year (e.g., '10', '11', '12')"),

        "curriculum": types.Schema(type=types.Type.STRING, description="Curriculum type (e.g., 'CBSE', 'ICSE')"),

        "school_name": types.Schema(type=types.Type.STRING, description="Name of current school"),

        "lead_source": types.Schema(type=types.Type.STRING, description="How they found us (default: 'Phone Call')"),

        "program_country_of_interest": types.Schema(type=types.Type.STRING, description="Country interested in studying"),

        "academic_grades": types.Schema(type=types.Type.STRING, description="Academic performance/grade"),

        "counsellor_meeting_link": types.Schema(type=types.Type.STRING, description="Always null"),

        "tags": types.Schema(type=types.Type.STRING, description="Relevant tags"),

        "stage": types.Schema(type=types.Type.STRING, description="Current stage"),

        "status": types.Schema(type=types.Type.STRING, description="Status"),

        "counsellor_email": types.Schema(type=types.Type.STRING, description="Counsellor email"),

    },

)

# -----------------------------------------------------------------------------

# IMPROVED BOOKING INFO SCHEMA (lead_bookings_extractor.py - current implementation)

# Required: booking_type, user_confirmed. Better validation and focused fields

# -----------------------------------------------------------------------------

IMPROVED_BOOKING_SCHEMA = types.Schema(

    type=types.Type.OBJECT,

    required=["booking_type", "user_confirmed"],

    properties={

        "booking_type": types.Schema(

            type=types.Type.STRING,

            description="auto_consultation or auto_followup"

        ),

        "time_phrase": types.Schema(

            type=types.Type.STRING,

            description="Time phrase extracted (e.g., 'after 5 minutes', 'tomorrow 3 PM')"

        ),

        "user_confirmed": types.Schema(

            type=types.Type.BOOLEAN,

            description="true if user explicitly confirmed the followup time"

        ),

        "student_grade": types.Schema(

            type=types.Type.INTEGER,

            nullable=True,

            description="Grade (9-12) ONLY if USER explicitly confirms their grade with phrases like: 'I am in grade 10', 'I am in class 9', 'I study in 11th grade', 'My grade is 12', etc. Do NOT extract from agent statements. Default to null."

        ),

    }

)

