"""
Simple Call Analytics for Pluto Travels
Uses LLM (Gemini) for sentiment analysis and All processing happens AFTER the call to avoid delays
"""

import os
import json
import asyncio
import re
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"analytics_{datetime.now().strftime('%Y%m%d')}.log"

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
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")

import openai

# Lead information extractor
try:
    from .lead_info_extractor import LeadInfoExtractor
    LEAD_EXTRACTOR_AVAILABLE = True
except ImportError:
    LEAD_EXTRACTOR_AVAILABLE = False
    logger.debug("LeadInfoExtractor not available - lead extraction disabled")

# Phase 13: Import storage classes for lad_dev schema
try:
    from db.storage.call_analysis import CallAnalysisStorage
    from db.storage.calls import CallStorage
    STORAGE_CLASSES_AVAILABLE = True
except ImportError:
    STORAGE_CLASSES_AVAILABLE = False
    logger.warning("Storage classes not available - using legacy direct SQL")


class SentimentCategory(Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_INTERESTED = "Very Interested"


@dataclass
class CallAnalytics:
    """Simple call analytics with sentiment and summarization"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in .env file")
        
        # Cost tracking for Gemini API calls
        self.cost_tracker = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'api_calls': 0
        }
        
        # Lead information extractor (optional)
        self.lead_extractor = None
        if LEAD_EXTRACTOR_AVAILABLE:
            # Pass cost_tracker reference so lead extraction costs are tracked together
            self.lead_extractor = LeadInfoExtractor(self.gemini_api_key, self.cost_tracker)

    @staticmethod
    def _ensure_list(value) -> List[str]:
        """Normalize summary list fields to a list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (set, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return [str(value).strip()]

    def _normalize_summary_fields(self, summary_data: Dict | None) -> Dict:
        """Ensure the structured summary uses consistent field types."""
        summary_data = dict(summary_data or {})
        summary_data['call_summary'] = str(summary_data.get('call_summary', '') or '').strip()
        summary_data['key_discussion_points'] = self._ensure_list(summary_data.get('key_discussion_points'))
        summary_data['prospect_questions'] = self._ensure_list(summary_data.get('prospect_questions'))
        summary_data['prospect_concerns'] = self._ensure_list(summary_data.get('prospect_concerns'))
        summary_data['next_steps'] = self._ensure_list(summary_data.get('next_steps'))

        summary_data['business_name'] = str(summary_data.get('business_name') or '').strip() or None
        summary_data['contact_person'] = str(summary_data.get('contact_person') or '').strip() or None
        summary_data['phone_number'] = str(summary_data.get('phone_number') or '').strip() or None
        summary_data['recommendations'] = str(summary_data.get('recommendations', '') or '').strip()

        return summary_data

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

    def _extract_call_summary_from_text(self, text: str) -> str:
        """Extract call summary section from raw LLM text response."""
        if not text:
            return ""
        
        # Look for "call_summary", "Call Summary", "Summary" sections
        patterns = [
            r'(?:call_summary|Call Summary|Summary)[:\-]?\s*(.+?)(?=\n\n|\n(?:Key|Recommendations|Next|Questions|Concerns|Business|Prospect)|$)',
            r'Summary[:\-]?\s*(.+?)(?=\n\n|Recommendations|Key|$)',
            r'The prospect.*?(?=\n\n|Recommendations|Key|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 50:  # Meaningful summary
                    return summary
        
        # If no pattern matches, return first paragraph or first 1000 chars
        paragraphs = text.split('\n\n')
        if paragraphs:
            first_para = paragraphs[0].strip()
            if len(first_para) > 50:
                return first_para[:2000]  # Limit to 2000 chars if no clear section
        
        return text[:1000]  # Fallback: first 1000 chars

    def _extract_recommendations_from_text(self, text: str) -> str:
        """Extract recommendations section from raw LLM text response."""
        if not text:
            return ""
        
        # Look for "recommendations", "Recommendations", "Recommendation" sections
        patterns = [
            r'(?:recommendations|Recommendations|Recommendation)[:\-]?\s*(.+?)(?=\n\n|\n(?:Key|Next|Questions|Concerns|Business|$)|$)',
            r'Recommendations?[:\-]?\s*(.+?)(?=\n\n|$)',
            r'Based on.*?(?=\n\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                recommendations = match.group(1).strip()
                if len(recommendations) > 30:  # Meaningful recommendations
                    return recommendations
        
        # If no pattern matches, look for sentences starting with recommendation keywords
        sentences = re.split(r'[.!?]+', text)
        rec_sentences = []
        rec_keywords = ['recommend', 'suggest', 'should', 'priority', 'next step', 'action', 'follow up']
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in rec_keywords):
                rec_sentences.append(sentence.strip())
        
        if rec_sentences:
            return '. '.join(rec_sentences)  # Return all recommendation sentences
        
        return ""  # No recommendations found

    def _extract_simple_points_from_text(self, text: str) -> List[str]:
        """Extract ALL bullet points or numbered items from raw text - NO LIMITS."""
        points = []
        lines = text.split('\n')
        
        # Look for section headers first
        in_points_section = False
        points_section_keywords = ['key discussion', 'discussion points', 'key points', 'points']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering a points section
            if any(keyword in line_lower for keyword in points_section_keywords):
                in_points_section = True
                continue
            
            # Check if we're leaving the points section
            if in_points_section and (line_lower.startswith('prospect') or line_lower.startswith('recommendations') or line_lower.startswith('next') or line_lower.startswith('business')):
                in_points_section = False
            
            # Extract bullet points
            if in_points_section or re.match(r'^[•\-\*]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
                point = re.sub(r'^[•\-\*\d+[\.\)]\s+', '', line).strip()
                if len(point) > 10:  # Only meaningful points
                    points.append(point)
            # Look for lines starting with keywords
            elif any(keyword in line.lower() for keyword in ['discussed', 'talked about', 'mentioned']):
                if len(line.strip()) > 15:
                    points.append(line.strip())
        
        return points  # NO LIMIT - Return ALL points found

    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extract ALL questions from raw text - NO LIMITS."""
        questions = []
        lines = text.split('\n')
        
        # Look for questions section
        in_questions_section = False
        questions_section_keywords = ['prospect questions', 'questions', 'question']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering a questions section
            if any(keyword in line_lower for keyword in questions_section_keywords):
                in_questions_section = True
                continue
            
            # Check if we're leaving the questions section
            if in_questions_section and (line_lower.startswith('concerns') or line_lower.startswith('recommendations') or line_lower.startswith('next') or line_lower.startswith('business')):
                in_questions_section = False
            
            # Extract questions
            if in_questions_section or '?' in line:
                question = line.strip()
                if '?' in question and len(question) > 10:
                    # Clean up bullet points
                    question = re.sub(r'^[•\-\*\d+[\.\)]\s+', '', question).strip()
                    if len(question) > 10:
                        questions.append(question)
        
        # Also extract questions from sentences
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if '?' in sentence and len(sentence) > 10:
                question = sentence.split('?')[0].strip()
                if len(question) > 10 and question not in questions:
                    questions.append(question + '?')
        
        return questions  # NO LIMIT - Return ALL questions found

    def _extract_concerns_from_text(self, text: str) -> List[str]:
        """Extract ALL concerns using keyword detection - NO LIMITS."""
        concerns = []
        lines = text.split('\n')
        
        # Look for concerns section
        in_concerns_section = False
        concerns_section_keywords = ['prospect concerns', 'concerns', 'concern']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering a concerns section
            if any(keyword in line_lower for keyword in concerns_section_keywords):
                in_concerns_section = True
                continue
            
            # Check if we're leaving the concerns section
            if in_concerns_section and (line_lower.startswith('recommendations') or line_lower.startswith('next') or line_lower.startswith('business')):
                in_concerns_section = False
            
            # Extract concerns
            if in_concerns_section:
                concern = line.strip()
                concern = re.sub(r'^[•\-\*\d+[\.\)]\s+', '', concern).strip()
                if len(concern) > 10:
                    concerns.append(concern)
        
        # Also extract concerns from sentences using keywords
        concern_keywords = ['concern', 'worried', 'issue', 'problem', 'not sure', 'hesitant', 'anxious', 'apprehensive']
        sentences = re.split(r'[.!?]+', text.lower())
        for sentence in sentences:
            if any(keyword in sentence for keyword in concern_keywords):
                concern = sentence.strip().capitalize()
                if len(concern) > 15 and concern not in [c.lower() for c in concerns]:
                    concerns.append(concern)
        
        return concerns  # NO LIMIT - Return ALL concerns found

    def _extract_business_info_from_text(self, text: str) -> Dict:
        """Try to extract business name, contact person, phone from text."""
        business_name = None
        contact_person = None
        phone_number = None
        
        # Look for phone patterns
        phone_patterns = [
            r'\b\d{10}\b',
            r'\b\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b',
            r'\b\d{5}[-.\s]?\d{5}\b'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                phone_number = phone_match.group()
                break
        
        # Look for business name patterns
        business_patterns = [
            r'(?:business name|company|business)[:\-]?\s*([A-Z][A-Za-z\s&]+?)(?=\n|,|\.|$)',
            r'from\s+([A-Z][A-Za-z\s&]+?)(?=\s+expressed|\s+mentioned|\s+said|,|\.|$)',
        ]
        for pattern in business_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                if len(potential_name) > 3 and 'pluto' not in potential_name.lower() and 'travels' not in potential_name.lower():
                    business_name = potential_name
                    break
        
        # Look for contact person patterns
        contact_patterns = [
            r'(?:contact person|contact|person)[:\-]?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+from',
            r'I\'m\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        for pattern in contact_patterns:
            match = re.search(pattern, text)
            if match:
                potential_contact = match.group(1).strip()
                if len(potential_contact.split()) >= 2:
                    contact_person = potential_contact
                    break
        
        return {
            'business_name': business_name,
            'contact_person': contact_person,
            'phone_number': phone_number
        }
    
    def _parse_voiceagent_segments(self, segments: List[Dict]) -> str:
        """
        Parse voiceagent transcription segments format.
        
        CRITICAL RULES:
        - Include BOTH agent and user messages for full context
        - ALWAYS use 'text' field (actual spoken), NEVER use 'intended_text'
        - Filter out phone system messages (voicemail prompts)
        - Analysis will focus on USER behavior with full conversation context
        """
        # Phone system messages to filter out
        SYSTEM_MESSAGES = [
            "call has been forwarded to voice mail",
            "the person you're trying to reach is not available",
            "at the tone",
            "please record your message",
            "when you",
            "to leave a callback number",
            "press",
            "hang up",
            "send a numeric page",
            "to repeat this message",
            "for more options"
        ]
        
        conversation_lines = []
        user_message_count = 0
        voicemail_indicators = 0
        
        for segment in segments:
            speaker = segment.get('speaker', '').lower()
            text = segment.get('text', '').strip()
            
            # ALWAYS use 'text' field only, ignore 'intended_text'
            if not text:
                continue
            
            # Count voicemail indicators (from system/user messages)
            if speaker == 'user' and any(indicator in text.lower() for indicator in ["forwarded", "voice mail", "voicemail", "not available", "at the tone", "record"]):
                voicemail_indicators += 1
            
            # Filter out phone system messages (usually from 'user' speaker in voicemail)
            is_system_message = any(sys_msg in text.lower() for sys_msg in SYSTEM_MESSAGES)
            if is_system_message:
                logger.debug(f"Filtered out system message: {text[:50]}...")
                continue
            
            # Filter out very short likely-system messages from 'user' speaker
            if speaker == 'user' and len(text.split()) <= 2 and text.lower() in ['hello', 'hello?', 'hi', 'when you', 'this', 'that']:
                logger.debug(f"Filtered out short system fragment: {text}")
                continue
            
            # Map speaker names
            if speaker == 'agent':
                role = 'Bot'
            elif speaker == 'user':
                role = 'User'
                user_message_count += 1
            else:
                role = speaker.title()
            
            conversation_lines.append(f"{role}: {text}")
        
        conversation_text = "\n".join(conversation_lines)
        logger.info(f"Parsed {len(conversation_lines)} total messages from voiceagent format ({user_message_count} user, {len(conversation_lines) - user_message_count} agent, filtered from {len(segments)} total segments)")
        
        # Detect voicemail: no user messages OR multiple voicemail indicators with minimal real content
        if user_message_count == 0 or (voicemail_indicators >= 2 and user_message_count <= 1):
            logger.warning(f"VOICEMAIL DETECTED - voicemail indicators: {voicemail_indicators}, user messages: {user_message_count}")
            conversation_text = "[VOICEMAIL - No customer response, call went to voicemail]"
        
        return conversation_text
    
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
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximately 1 token = 4 characters)"""
        if not text:
            return 0
        return len(text) // 4
    
    def _call_gemini_api(self, prompt: str, temperature: float = 0.3, max_output_tokens: int = 800) -> str:
        """Helper function to call Gemini 2.0 Flash API with cost tracking"""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not available, skipping API call")
            return None
        
        input_tokens = self._estimate_tokens(prompt)
        logger.debug(f"Calling Gemini API - Input tokens: ~{input_tokens}, Max output: {max_output_tokens}, Temp: {temperature}")
        
        try:
            import requests
            
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
                
                self.cost_tracker['api_calls'] += 1
                logger.debug(f"Gemini API call successful (Total calls: {self.cost_tracker['api_calls']})")
                
                # Check for usage info in response first (if available, use exact counts)
                if "usageMetadata" in response_data:
                    usage = response_data["usageMetadata"]
                    if "promptTokenCount" in usage:
                        self.cost_tracker['total_input_tokens'] += usage["promptTokenCount"]
                    else:
                        self.cost_tracker['total_input_tokens'] += input_tokens
                    
                    if "candidatesTokenCount" in usage:
                        self.cost_tracker['total_output_tokens'] += usage["candidatesTokenCount"]
                else:
                    # Fallback to estimated tokens
                    self.cost_tracker['total_input_tokens'] += input_tokens
                
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    if "content" in response_data["candidates"][0]:
                        if "parts" in response_data["candidates"][0]["content"]:
                            output_text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                            if "usageMetadata" not in response_data:
                                output_tokens = self._estimate_tokens(output_text)
                                self.cost_tracker['total_output_tokens'] += output_tokens
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
    
    def _calculate_llm_cost(self) -> Dict:
        """Calculate total LLM cost (USD) and provide formatted values
        Note: kept INR conversion for backward compatibility but formatted
        output requested to be in dollars (USD).
        """
        # Gemini 2.0 Flash pricing (as of 2024)
        # Input: $0.075 per 1M tokens
        # Output: $0.30 per 1M tokens
        INPUT_COST_PER_1M_TOKENS = 0.075  # USD
        OUTPUT_COST_PER_1M_TOKENS = 0.30  # USD
        
        # USD to INR conversion rate (approximate) - kept for compatibility
        USD_TO_INR = 83.0
        
        # Calculate costs
        input_cost_usd = (self.cost_tracker['total_input_tokens'] / 1_000_000) * INPUT_COST_PER_1M_TOKENS
        output_cost_usd = (self.cost_tracker['total_output_tokens'] / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS
        total_cost_usd = input_cost_usd + output_cost_usd
        
        # Convert to INR (kept for reference)
        total_cost_inr = total_cost_usd * USD_TO_INR

        return {
            "total_api_calls": self.cost_tracker['api_calls'],
            "input_tokens": self.cost_tracker['total_input_tokens'],
            "output_tokens": self.cost_tracker['total_output_tokens'],
            "total_tokens": self.cost_tracker['total_input_tokens'] + self.cost_tracker['total_output_tokens'],
            "cost_usd": round(total_cost_usd, 6),
            "cost_inr": round(total_cost_inr, 4),
            # User requested dollars instead of INR for the formatted value.
            # Provide explicit USD formatted field and also set the existing
            # "cost_inr_formatted" key to USD for compatibility with callers
            # that expect that key name.
            "cost_usd_formatted": f"${total_cost_usd:.6f}",
            "cost_inr_formatted": f"{total_cost_usd:.6f}",
            "pricing_model": "Gemini 2.0 Flash",
            "input_rate": "$0.075 per 1M tokens",
            "output_rate": "$0.30 per 1M tokens",
            "conversion_rate": f"1 USD = {USD_TO_INR} INR"
        }
    
    async def _calculate_sentiment_with_llm(self, user_text: str, conversation_text: str) -> Dict:
        """Calculate sentiment score and category using LLM (replaces TextBlob+VADER)"""
        
        logger.debug("Calling LLM for sentiment calculation...")
        if not self.gemini_api_key:
            # Fallback to neutral if no API key
            return {
                "sentiment_score": 0.0,
                "category": "Neutral",
                "confidence": 0.5,
                "textblob_polarity": 0.0,
                "vader_compound": 0.0,
                "combined_score": 0.0,
                "llm_reasoning": "LLM API key not available"
            }
        
        try:
            prompt = f"""Analyze the sentiment of this sales conversation and provide a sentiment score.

CONVERSATION:
{conversation_text[:1000]}

TASK: Analyze the prospect's sentiment and provide:
1. Sentiment category: Positive, Neutral, Negative, or Very Interested
2. Sentiment score: -1.0 (very negative) to +1.0 (very positive)
3. Confidence: 0.0 to 1.0

Respond in this EXACT JSON format:
{{
  "category": "Positive|Neutral|Negative|Very Interested",
  "sentiment_score": -1.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation"
}}"""
            
            result = self._call_gemini_api(prompt, temperature=0.1, max_output_tokens=150)
            
            if result:
                # Clean JSON response
                if "```json" in result:
                    start = result.find("```json") + 7
                    end = result.find("```", start)
                    if end != -1:
                        result = result[start:end].strip()
                elif "```" in result:
                    start = result.find("```") + 3
                    end = result.find("```", start)
                    if end != -1:
                        result = result[start:end].strip()
                
                # Extract JSON
                if "{" in result:
                    json_start = result.find("{")
                    result = result[json_start:]
                
                try:
                    sentiment_data = json.loads(result)
                    category = sentiment_data.get("category", "Neutral")
                    sentiment_score = float(sentiment_data.get("sentiment_score", 0.0))
                    confidence = float(sentiment_data.get("confidence", 0.5))
                    reasoning = sentiment_data.get("reasoning", "")
                    
                    # Ensure sentiment_score is in valid range (-1.0 to 1.0)
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                    
                    # Map category to ensure proper score range for consistency
                    if category == "Very Interested":
                        sentiment_score = max(sentiment_score, 0.6)
                    elif category == "Positive":
                        sentiment_score = max(sentiment_score, 0.1)
                    elif category == "Negative":
                        sentiment_score = min(sentiment_score, -0.1)
                    else:
                        sentiment_score = max(-0.1, min(0.1, sentiment_score))
                    
                    # For backward compatibility, set textblob_polarity and vader_compound to sentiment_score
                    return {
                        "sentiment_score": sentiment_score,
                        "category": category,
                        "confidence": confidence,
                        "textblob_polarity": sentiment_score,  # Backward compatibility (LLM-generated)
                        "vader_compound": sentiment_score,  # Backward compatibility (LLM-generated)
                        "combined_score": sentiment_score,  # Float value (-1.0 to 1.0)
                        "llm_reasoning": reasoning
                    }
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            logger.error(f"LLM sentiment calculation error: {e}", exc_info=True)
        
        # Fallback to neutral
        return {
            "sentiment_score": 0.0,
            "category": "Neutral",
            "confidence": 0.5,
            "textblob_polarity": 0.0,
            "vader_compound": 0.0,
            "combined_score": 0.0,
            "llm_reasoning": "LLM sentiment analysis failed"
        }
    
    async def _validate_sentiment_with_llm(self, conversation_text: str, initial_sentiment: str, confidence: float) -> Dict:
        """Use LLM to validate sentiment for low-confidence or ambiguous cases"""
        
        if not self.gemini_api_key or confidence > 0.75:
            logger.debug(f"Skipping LLM validation - Confidence {confidence:.2f} is high enough")
            return {"validated": False, "llm_sentiment": None}
        
        logger.debug(f"Validating sentiment with LLM - Initial: {initial_sentiment}, Confidence: {confidence:.2f}")
        try:
            prompt = f"""
            Analyze the sentiment of this sales conversation. Consider:
            - Sarcasm and irony
            - Mixed emotions (e.g., interested but busy)
            - Context beyond individual words
            
            CONVERSATION:
            {conversation_text[:500]}
            
            Is the prospect's overall sentiment:
            A) Positive (interested, engaged, open)
            B) Neutral (professional, noncommittal)
            C) Negative (uninterested, frustrated, dismissive)
            
            Respond with ONLY one letter (A, B, or C) and a brief reason (max 20 words).
            Format: "Letter: Reason"
            """
            
            result = self._call_gemini_api(prompt, temperature=0.1, max_output_tokens=50)
            
            if result:
                # Parse result
                if result.startswith('A'):
                    llm_sentiment = "Positive"
                elif result.startswith('B'):
                    llm_sentiment = "Neutral"
                elif result.startswith('C'):
                    llm_sentiment = "Negative"
                else:
                    llm_sentiment = None
                
                return {
                    "validated": True,
                    "llm_sentiment": llm_sentiment,
                    "llm_reasoning": result.split(':', 1)[1].strip() if ':' in result else result
                }
            
        except Exception as e:
            # Fail gracefully - sentiment validation is optional
            pass
        
        return {"validated": False, "llm_sentiment": None}
    
    def _adjust_sentiment_with_context(self, base_sentiment: str, conversation_text: str, duration: int, word_count: int) -> Dict:
        """Adjust sentiment based on conversation context (duration, engagement)"""
        
        adjusted_sentiment = base_sentiment
        adjustments = []
        confidence_boost = 0
        
        # Rule 1: Very short calls with neutral sentiment = likely negative
        if duration < 30 and word_count < 20 and base_sentiment == "Neutral":
            adjusted_sentiment = "Negative"
            adjustments.append("Short call duration suggests disinterest")
            confidence_boost += 0.1
        
        # Rule 2: Long engagement with neutral = likely positive
        if duration > 120 and word_count > 100 and base_sentiment == "Neutral":
            adjusted_sentiment = "Positive"
            adjustments.append("Extended engagement suggests interest")
            confidence_boost += 0.15
        
        # Rule 3: Detect sarcasm patterns
        sarcasm_indicators = ["oh great", "oh wonderful", "just what i needed", "perfect timing"]
        text_lower = conversation_text.lower()
        
        if any(indicator in text_lower for indicator in sarcasm_indicators):
            if base_sentiment == "Positive":
                adjusted_sentiment = "Negative"
                adjustments.append("Sarcasm detected - flipping positive to negative")
                confidence_boost += 0.2
        
        # Rule 4: "Busy" is not necessarily negative
        if "busy" in text_lower and base_sentiment == "Negative":
            # Check if there are other negative signals
            negative_words = ["not interested", "don't call", "remove", "annoyed"]
            if not any(word in text_lower for word in negative_words):
                adjusted_sentiment = "Neutral"
                adjustments.append("'Busy' is a timing issue, not rejection")
                confidence_boost += 0.1
        
        # Rule 5: Questions about pricing/features = positive signal
        interest_indicators = ["how much", "pricing", "tell me more", "send information", "whatsapp"]
        if any(indicator in text_lower for indicator in interest_indicators):
            if base_sentiment == "Neutral":
                adjusted_sentiment = "Positive"
                adjustments.append("Prospect asked for information - positive signal")
                confidence_boost += 0.15
        
        return {
            "adjusted_sentiment": adjusted_sentiment,
            "adjustments": adjustments,
            "confidence_boost": min(confidence_boost, 0.25)  # Cap at 25% boost
        }
    
    async def analyze_sentiment(self, conversation_text: str, duration: int = 0, word_count: int = 0) -> Dict:
        """Analyze sentiment using LLM (TextBlob + VADER removed) - only on user messages"""
        
        logger.debug("Starting sentiment analysis...")
        user_text = self._extract_user_messages(conversation_text)
        
        if not user_text or len(user_text) < 10:
            logger.warning("Insufficient user text for sentiment analysis")
            # If no user text, return neutral sentiment
            return {
                "sentiment_description": "Prospect did not provide enough response for sentiment analysis",
                "confidence_score": "0.0%",
                "textblob_polarity": 0.0,
                "vader_compound": 0.0,
                "combined_score": "0.0%",
                "key_phrases": [],
                "reasoning": "Insufficient user input for analysis",
                "advanced_emotions": {},
                "llm_validated": False
            }
        
        logger.debug(f"Analyzing sentiment - User text length: {len(user_text)} chars, Word count: {word_count}")
        llm_sentiment_data = await self._calculate_sentiment_with_llm(user_text, conversation_text)
        
        logger.debug(f"Sentiment analysis complete - Category: {llm_sentiment_data.get('category', 'Unknown')}, Score: {llm_sentiment_data.get('combined_score', 0.0)}")
        combined_score = llm_sentiment_data.get("combined_score", 0.0)
        textblob_polarity = llm_sentiment_data.get("textblob_polarity", 0.0)  # Backward compatibility
        vader_compound = llm_sentiment_data.get("vader_compound", 0.0)  # Backward compatibility
        llm_category = llm_sentiment_data.get("category", "Neutral")
        llm_confidence = llm_sentiment_data.get("confidence", 0.5)
        
        # Map LLM category to SentimentCategory enum
        if llm_category == "Very Interested":
            category = SentimentCategory.VERY_INTERESTED
        elif llm_category == "Positive":
            category = SentimentCategory.POSITIVE
        elif llm_category == "Negative":
            category = SentimentCategory.NEGATIVE
        else:
            category = SentimentCategory.NEUTRAL
        
        # Use LLM confidence, but ensure it's in valid range
        initial_confidence = max(0.3, min(0.95, llm_confidence))
        
        # ENHANCEMENT 1: Adjust sentiment based on context (duration, engagement)
        context_adjustment = self._adjust_sentiment_with_context(
            category.value,
            conversation_text,
            duration,
            word_count if word_count > 0 else len(conversation_text.split())
        )
        
        # Apply context adjustments
        if context_adjustment['adjustments']:
            # Update category based on adjusted sentiment
            adjusted_category_str = context_adjustment['adjusted_sentiment']
            if adjusted_category_str == "Positive":
                category = SentimentCategory.POSITIVE
            elif adjusted_category_str == "Negative":
                category = SentimentCategory.NEGATIVE
            else:
                category = SentimentCategory.NEUTRAL
            
            # Boost confidence if we made adjustments
            confidence = min(0.95, initial_confidence + context_adjustment['confidence_boost'])
        else:
            confidence = initial_confidence
        
        # Extract key phrases from user messages only
        key_phrases = self._extract_natural_phrases(user_text)
        
        # Advanced emotion analysis on user messages only
        advanced_emotions = self.analyze_advanced_emotions(user_text)
        
        # ENHANCEMENT 2: LLM validation for low-confidence or ambiguous cases
        confidence_numeric = confidence
        llm_validation = await self._validate_sentiment_with_llm(
            conversation_text,
            category.value,
            confidence_numeric
        )
        
        # Apply LLM validation if available
        llm_validated = False
        llm_sentiment = None
        llm_reasoning = None
        
        if llm_validation['validated'] and llm_validation['llm_sentiment']:
            llm_validated = True
            llm_sentiment = llm_validation['llm_sentiment']
            llm_reasoning = llm_validation.get('llm_reasoning', '')
            
            # Update category if LLM has different opinion
            if llm_sentiment == "Positive" and category != SentimentCategory.POSITIVE:
                category = SentimentCategory.POSITIVE
            elif llm_sentiment == "Negative" and category != SentimentCategory.NEGATIVE:
                category = SentimentCategory.NEGATIVE
            
            # Boost confidence significantly when LLM validates
            confidence = min(0.95, confidence_numeric + 0.15)
        
        # Generate natural sentiment description using LLM (now with validated sentiment)
        sentiment_description = self.generate_sentiment_description(category.value, advanced_emotions, combined_score, user_text)
        
        # Build reasoning with context adjustments and LLM validation
        base_reasoning = f"LLM sentiment score: {round(combined_score * 100, 1)}% (Category: {category.value})"
        if llm_sentiment_data.get("llm_reasoning"):
            base_reasoning += f" | {llm_sentiment_data.get('llm_reasoning', '')}"
        if context_adjustment['adjustments']:
            base_reasoning += f" | Context: {'; '.join(context_adjustment['adjustments'])}"
        if llm_validated:
            base_reasoning += f" | LLM Validation: {llm_sentiment} ({llm_reasoning})"
        
        result = {
            "sentiment_description": sentiment_description,
            "confidence_score": f"{round(confidence * 100, 1)}%",
            "textblob_polarity": round(textblob_polarity, 2),  # Backward compatibility (LLM-generated)
            "vader_compound": round(vader_compound, 2),  # Backward compatibility (LLM-generated)
            "combined_score": f"{round(combined_score * 100, 1)}%",
            "key_phrases": key_phrases[:5],  # Top 5 phrases
            "reasoning": base_reasoning,
            "advanced_emotions": advanced_emotions,
            "context_adjustments": context_adjustment['adjustments'] if context_adjustment['adjustments'] else None,
            "confidence_boost": f"+{round(context_adjustment['confidence_boost'] * 100, 1)}%" if context_adjustment['confidence_boost'] > 0 else None,
            "llm_validated": llm_validated,
            "llm_powered": True  # Indicate LLM was used for sentiment calculation
        }
        
        # Add LLM validation details if available
        if llm_validated:
            result['llm_sentiment'] = llm_sentiment  # This is the validated sentiment category
            result['llm_reasoning'] = llm_reasoning
        
        return result
    
    def _extract_natural_phrases(self, text: str) -> List[str]:
        """Extract key topics and phrases (not full sentences)"""
        
        # Common words to filter out
        stop_words = {
            'i', 'me', 'my', 'you', 'your', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'this', 'that', 'these', 'those', 'it', 'its', 'be', 'been', 'being',
            'at', 'by', 'for', 'with', 'about', 'to', 'of', 'in', 'on', 'and', 'or', 'but'
        }
        
        # Extract 2-3 word phrases that represent key topics
        words = text.lower().split()
        phrases = []
        
        # Extract important 2-3 word phrases
        for i in range(len(words) - 1):
            # 2-word phrases
            word1, word2 = words[i], words[i+1]
            if word1 not in stop_words and word2 not in stop_words:
                phrase = f"{word1} {word2}"
                if len(phrase) > 5 and phrase not in phrases:
                    phrases.append(phrase)
            
            # 3-word phrases
            if i < len(words) - 2:
                word3 = words[i+2]
                if word1 not in stop_words and word2 not in stop_words:
                    phrase = f"{word1} {word2} {word3}"
                    if len(phrase) > 8 and phrase not in phrases:
                        phrases.append(phrase)
        
        # Return top 5 unique key phrases
        return list(set(phrases))[:5]
    
    def analyze_advanced_emotions(self, conversation_text: str) -> Dict:
        """Analyze emotions - returns neutral emotion (Hugging Face models removed, using LLM-only approach)"""
        
        # Return default neutral emotion structure
        # LLM will infer emotion from context in sentiment_description
        return {
            "status": "llm_only",
            "message": "Emotion analysis using LLM inference from context",
            "emotion": "neutral",
            "confidence": 0.0
        }
    
    def _get_emotion_intensity(self, confidence: float) -> str:
        """Determine emotion intensity based on confidence score"""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.75:
            return "High"
        elif confidence >= 0.60:
            return "Medium"
        elif confidence >= 0.45:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_emotion_description(self, emotion: str, confidence: float, user_text: str) -> str:
        """Generate detailed psychological analysis of prospect's emotional state using LLM"""
        
        # If no Gemini API key, fall back to simple template
        if not self.gemini_api_key:
            return self._generate_emotion_description_fallback(emotion, confidence, user_text)
        
        try:
            # Count engagement indicators
            word_count = len(user_text.split())
            question_count = user_text.count('?')
            exclamation_count = user_text.count('!')
            
            # Create optimized prompt for psychological emotion analysis
            # Note: Emotion is inferred from context (Hugging Face models removed)
            prompt = f"""Analyze this sales prospect's emotional state from their conversation:

WORDS SPOKEN: {word_count} | QUESTIONS: {question_count}

PROSPECT'S WORDS:
"{user_text}"

Write 2 sentences analyzing:
1. Their true emotional state and any shifts during conversation
2. What their language reveals about openness, trust, skepticism, or decision-making style

Be specific and psychological, not generic."""
            
            description = self._call_gemini_api(prompt, temperature=0.3, max_output_tokens=120)
            
            if description:
                # Remove quotes if LLM added them
                description = description.strip('"\'')
                return description
            else:
                # Fallback to template on error
                return self._generate_emotion_description_fallback(emotion, confidence, user_text)
                
        except Exception as e:
            # Fallback to template on error
            return self._generate_emotion_description_fallback(emotion, confidence, user_text)
    
    def _generate_emotion_description_fallback(self, emotion: str, confidence: float, user_text: str) -> str:
        """Fallback template-based emotion description (used when LLM unavailable)"""
        
        # Count engagement indicators
        word_count = len(user_text.split())
        question_count = user_text.count('?')
        
        # Determine intensity modifier
        if confidence >= 0.85:
            intensity = "very"
        elif confidence >= 0.70:
            intensity = "clearly"
        elif confidence >= 0.55:
            intensity = "somewhat"
        else:
            intensity = "slightly"
        
        # Build descriptions based on emotion type with engagement context
        descriptions = {
            "joy": {
                "high_engagement": f"The prospect is {intensity} enthusiastic and excited about the services, showing genuine interest with {word_count} words and {question_count} questions asked.",
                "medium_engagement": f"The prospect seems {intensity} positive and interested, engaging moderately with {word_count} words in the conversation.",
                "low_engagement": f"The prospect appears {intensity} upbeat but reserved, with limited engagement of only {word_count} words."
            },
            "neutral": {
                "high_engagement": f"The prospect remains {intensity} calm and cooperative, listening attentively with {word_count} words but not showing strong emotional reactions.",
                "medium_engagement": f"The prospect is {intensity} neutral and polite, maintaining a balanced tone throughout the {word_count}-word conversation.",
                "low_engagement": f"The prospect sounds {intensity} indifferent and detached, providing only {word_count} words with minimal emotional investment."
            },
            "sadness": {
                "high_engagement": f"The prospect seems {intensity} disappointed or pessimistic, expressing concerns despite engaging with {word_count} words.",
                "medium_engagement": f"The prospect appears {intensity} down or lacking enthusiasm, with moderate engagement of {word_count} words.",
                "low_engagement": f"The prospect sounds {intensity} disheartened and disengaged, offering only {word_count} words with low energy."
            },
            "anger": {
                "high_engagement": f"The prospect is {intensity} frustrated and irritated, expressing annoyance actively with {word_count} words.",
                "medium_engagement": f"The prospect seems {intensity} agitated or impatient, showing frustration in the {word_count}-word conversation.",
                "low_engagement": f"The prospect appears {intensity} annoyed and dismissive, responding curtly with only {word_count} words."
            },
            "fear": {
                "high_engagement": f"The prospect shows {intensity} strong concerns and hesitation, expressing worries throughout {word_count} words.",
                "medium_engagement": f"The prospect appears {intensity} cautious and uncertain, showing some apprehension in {word_count} words.",
                "low_engagement": f"The prospect seems {intensity} anxious and reluctant, providing only {word_count} words with noticeable hesitation."
            },
            "surprise": {
                "high_engagement": f"The prospect is {intensity} caught off-guard and intrigued, responding with unexpected reactions across {word_count} words.",
                "medium_engagement": f"The prospect seems {intensity} surprised and uncertain, showing unexpected responses in {word_count} words.",
                "low_engagement": f"The prospect appears {intensity} startled but reserved, offering only {word_count} words with some confusion."
            },
            "disgust": {
                "high_engagement": f"The prospect is {intensity} dismissive and negative, showing clear rejection despite {word_count} words of engagement.",
                "medium_engagement": f"The prospect seems {intensity} unimpressed and resistant, expressing distaste in {word_count} words.",
                "low_engagement": f"The prospect appears {intensity} repulsed and disinterested, responding with only {word_count} words of clear negativity."
            }
        }
        
        # Determine engagement level
        if word_count > 300:
            engagement_level = "high_engagement"
        elif word_count > 100:
            engagement_level = "medium_engagement"
        else:
            engagement_level = "low_engagement"
        
        # Get description for this emotion and engagement level
        emotion_lower = emotion.lower()
        if emotion_lower in descriptions:
            return descriptions[emotion_lower][engagement_level]
        else:
            # Fallback for unknown emotions
            return f"The prospect shows {intensity} {emotion} emotion with {word_count} words of engagement in the conversation."
    
    def generate_sentiment_description(self, category: str, advanced_emotions: Dict, combined_score: float, user_text: str) -> str:
        """Generate comprehensive behavioral and emotional analysis of the prospect"""
        
        if not self.gemini_api_key:
            return f"Prospect shows {category.lower()} sentiment"
        
        try:
            # Get the main emotion from advanced analysis
            main_emotion = self.get_main_emotion(advanced_emotions)
            emotion_confidence = advanced_emotions.get('confidence', 0)
            
            # Analyze conversation patterns
            word_count = len(user_text.split())
            question_count = user_text.count('?')
            exclamation_count = user_text.count('!')
            sentence_count = len([s for s in user_text.split('.') if s.strip()])
            
            # Create comprehensive prompt for detailed behavioral analysis
            prompt = f"""Analyze this B2B sales prospect's behavior and buying signals:

PROSPECT'S WORDS: "{user_text}"

METRICS: {word_count} words | {question_count} questions | Emotion: {main_emotion} ({emotion_confidence:.0%}) | Sentiment: {category} ({combined_score:.0%})

Write 3 sentences covering:
1. Emotional state and engagement level
2. Communication style and buying signals
3. Likelihood to convert and recommended next steps

Be specific about their mindset, not generic."""
            
            description = self._call_gemini_api(prompt, temperature=0.3, max_output_tokens=150)
            
            if description:
                # Remove quotes if LLM added them
                description = description.strip('"\'')
                return description
            else:
                # Fallback to simple description
                return f"Prospect shows {category.lower()} sentiment with {main_emotion} emotion. Engagement level: {word_count} words spoken with {question_count} questions asked."
            
        except Exception as e:
            # Fallback to simple description
            return f"Prospect shows {category.lower()} sentiment with limited behavioral data available."
    
    def get_main_emotion(self, advanced_emotions: Dict) -> str:
        """Get the main emotion from advanced analysis"""
        
        try:
            # New structure has emotion directly at root level
            if 'emotion' in advanced_emotions:
                return advanced_emotions['emotion']
            
            # Fallback to old structure for backwards compatibility
            if 'basic' in advanced_emotions and isinstance(advanced_emotions['basic'], dict):
                return advanced_emotions['basic'].get('emotion', 'neutral')
            
            return 'neutral'
            
        except Exception:
            return 'neutral'
    
    async def generate_call_summary(self, call_id: str, sentiment: Dict, summary: Dict, duration_seconds: int, conversation_text: str) -> str:
        """Generate a comprehensive call summary"""
        
        try:
            # Basic call info
            duration_minutes = duration_seconds // 60
            duration_seconds_remainder = duration_seconds % 60
            word_count = len(conversation_text.split())
            
            # Sentiment info
            sentiment_description = sentiment.get('sentiment_description', 'No description available')
            confidence = sentiment.get('confidence_score', 0)
            combined_score_str = sentiment.get('combined_score', '0%')
            # Extract numeric value from percentage string (e.g., "85.0%" -> 85.0)
            try:
                combined_score_numeric = float(combined_score_str.rstrip('%'))
            except:
                combined_score_numeric = 0
            
            # Summary info
            business_name = summary.get('business_name', 'Not provided')
            contact_person = summary.get('contact_person', 'Not provided')
            phone_number = summary.get('phone_number', 'Not provided')
            
            # Build comprehensive summary
            summary_parts = []
            
            # Call overview
            summary_parts.append(f"CALL OVERVIEW")
            summary_parts.append(f"Call ID: {call_id}")
            summary_parts.append(f"Duration: {duration_minutes}m {duration_seconds_remainder}s")
            summary_parts.append(f"Conversation: {word_count} words")
            summary_parts.append("")
            
            # Sentiment analysis
            summary_parts.append(f"PROSPECT SENTIMENT")
            summary_parts.append(f"Description: {sentiment_description}")
            summary_parts.append(f"Confidence: {sentiment.get('confidence_score', 'N/A')}")
            summary_parts.append(f"Combined Score: {sentiment.get('combined_score', 'N/A')}")
            summary_parts.append("")
            
            # Business information
            summary_parts.append(f"BUSINESS INFORMATION")
            summary_parts.append(f"Company: {business_name}")
            summary_parts.append(f"Contact: {contact_person}")
            summary_parts.append(f"Phone: {phone_number}")
            summary_parts.append("")
            
            # Call summary from LLM (before key discussion points)
            if 'call_summary' in summary and summary['call_summary']:
                summary_parts.append(f"CALL SUMMARY")
                # Split by sentences for vertical display
                import re
                call_summary_text = summary['call_summary']
                # Split on period followed by space or end of string
                sentences = re.split(r'\.\s+|\.\s*$', call_summary_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:  # Only add non-empty sentences
                        # Add period if it doesn't end with punctuation
                        if not sentence.endswith(('.', '!', '?')):
                            sentence += '.'
                        summary_parts.append(sentence)
                summary_parts.append("")
            
            # Key discussion points
            if 'key_discussion_points' in summary and summary['key_discussion_points']:
                summary_parts.append(f"💬 KEY DISCUSSION POINTS")
                for i, point in enumerate(summary['key_discussion_points'], 1):
                    summary_parts.append(f"{i}. {point}")
                summary_parts.append("")
            
            # Prospect questions
            if 'prospect_questions' in summary and summary['prospect_questions']:
                summary_parts.append(f"PROSPECT QUESTIONS")
                for i, question in enumerate(summary['prospect_questions'], 1):
                    summary_parts.append(f"{i}. {question}")
                summary_parts.append("")
            
            # Prospect concerns
            if 'prospect_concerns' in summary and summary['prospect_concerns']:
                summary_parts.append(f"PROSPECT CONCERNS")
                for i, concern in enumerate(summary['prospect_concerns'], 1):
                    summary_parts.append(f"{i}. {concern}")
                summary_parts.append("")
            
            # Next steps
            if 'next_steps' in summary and summary['next_steps']:
                summary_parts.append(f"AGREED NEXT STEPS")
                for i, step in enumerate(summary['next_steps'], 1):
                    summary_parts.append(f"{i}. {step}")
                summary_parts.append("")
            
            # Recommendations
            if 'recommendations' in summary and summary['recommendations']:
                summary_parts.append(f"RECOMMENDATIONS")
                summary_parts.append(summary['recommendations'])
                summary_parts.append("")
            
            # Follow-up priority based on combined score
            if combined_score_numeric >= 60:
                priority = "HIGH PRIORITY - Immediate follow-up recommended"
            elif combined_score_numeric >= 10:
                priority = "MEDIUM PRIORITY - Schedule follow-up within 24-48 hours"
            elif combined_score_numeric >= -10:
                priority = "LOW PRIORITY - Nurture relationship, follow up in 1-2 weeks"
            else:
                priority = "LOW PRIORITY - Address concerns before re-engaging"
            
            summary_parts.append(f"FOLLOW-UP PRIORITY")
            summary_parts.append(priority)
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating call summary: {str(e)}"
    
    def generate_full_summary(self, sentiment_category: str, sentiment_description: str, business_name: str, contact_person: str, summary: Dict, duration_minutes: int, word_count: int) -> str:
        """Generate a comprehensive full summary of the call"""
        
        try:
            # Extract key information
            key_points = summary.get('key_discussion_points', [])
            questions = summary.get('prospect_questions', [])
            concerns = summary.get('prospect_concerns', [])
            next_steps = summary.get('next_steps', [])
            recommendations = summary.get('recommendations', '')
            
            # Build comprehensive summary
            summary_parts = []
            
            # Opening
            summary_parts.append(f"This {duration_minutes}-minute call with {business_name} ({contact_person}) resulted in a {sentiment_category.lower()} outcome.")
            summary_parts.append(f"The prospect appeared {sentiment_description.lower()} throughout the conversation.")
            summary_parts.append("")
            
            # Main discussion
            if key_points:
                summary_parts.append("Key topics discussed included:")
                for point in key_points[:3]:  # Top 3 points
                    summary_parts.append(f"• {point}")
                summary_parts.append("")
            
            # Questions and concerns
            if questions:
                summary_parts.append("The prospect asked several questions:")
                for question in questions[:2]:  # Top 2 questions
                    summary_parts.append(f"• {question}")
                summary_parts.append("")
            
            if concerns:
                summary_parts.append("Main concerns raised:")
                for concern in concerns[:2]:  # Top 2 concerns
                    summary_parts.append(f"• {concern}")
                summary_parts.append("")
            
            # Next steps and outcome
            if next_steps:
                summary_parts.append("Agreed next steps:")
                for step in next_steps:
                    summary_parts.append(f"• {step}")
                summary_parts.append("")
            
            # Recommendations
            if recommendations:
                summary_parts.append("Recommended follow-up approach:")
                summary_parts.append(recommendations)
                summary_parts.append("")
            
            # Overall assessment
            if sentiment_category == "Very Interested":
                summary_parts.append("Overall Assessment: This is a high-priority prospect showing strong interest and engagement. Immediate follow-up is recommended to capitalize on their enthusiasm.")
            elif sentiment_category == "Positive":
                summary_parts.append("Overall Assessment: This prospect shows positive interest and should be followed up within 24-48 hours to maintain momentum.")
            elif sentiment_category == "Neutral":
                summary_parts.append("Overall Assessment: This prospect maintains a neutral stance and requires nurturing. Follow up in 1-2 weeks with additional value.")
            else:
                summary_parts.append("Overall Assessment: This prospect shows resistance or negative sentiment. Address their concerns before re-engaging.")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error generating full summary: {str(e)}"
    
    async def generate_llm_full_summary(self, call_id: str, sentiment: Dict, summary: Dict, duration_seconds: int, conversation_text: str) -> str:
        """Generate comprehensive full summary using LLM (Gemini)"""
        
        if not self.gemini_api_key:
            return "Gemini API key not configured - using fallback summary"
        
        try:
            # Prepare context for LLM
            sentiment_category = sentiment.get('category', 'Unknown')
            sentiment_description = sentiment.get('sentiment_description', 'No description')
            business_name = summary.get('business_name', 'Not provided')
            contact_person = summary.get('contact_person', 'Not provided')
            duration_minutes = duration_seconds // 60
            
            # Create detailed prompt for LLM
            confidence_score = sentiment.get('confidence_score', 'N/A')
            
            prompt = f"""
            Create a simple, easy-to-understand summary of this sales call in 2-3 sentences that anyone can understand.
            Focus on what happened during the call and the outcome.
            
            CALL DETAILS:
            - Call ID: {call_id}
            - Duration: {duration_minutes} minutes
            - Company: {business_name}
            - Contact: {contact_person}
            - Prospect Sentiment: {sentiment_description}
            - Confidence: {confidence_score}
            
            CONVERSATION:
            {conversation_text}
            
            CALL SUMMARY DATA:
            Key Discussion Points: {summary.get('key_discussion_points', [])}
            Prospect Questions: {summary.get('prospect_questions', [])}
            Prospect Concerns: {summary.get('prospect_concerns', [])}
            Next Steps: {summary.get('next_steps', [])}
            Recommendations: {summary.get('recommendations', '')}
            
            Write a simple, clear summary that explains:
            1. What was discussed in the call
            2. How the prospect responded
            3. What happens next (if anything)
            
            Use simple language that anyone can understand. Keep it brief (2-3 sentences).
            """
            
            full_summary = self._call_gemini_api(prompt, temperature=0.3, max_output_tokens=800)
            
            if full_summary:
                # Clean up the response
                if "```" in full_summary:
                    start = full_summary.find("```") + 3
                    end = full_summary.find("```", start)
                    if end != -1:
                        full_summary = full_summary[start:end].strip()
                return full_summary
            else:
                return "LLM API error - unable to generate summary"
            
        except Exception as e:
            return f"LLM full summary error: {str(e)}"
    
    async def _determine_lead_disposition_llm(self, sentiment: Dict, summary: Dict) -> Dict:
        """Use LLM to determine lead disposition (AI-powered, no hardcoded keywords)"""
        
        logger.debug(f"Determining disposition from sentiment and summary")
        if not self.gemini_api_key:
            # Fallback to rule-based if no API key
            return self._determine_lead_disposition_fallback(sentiment, summary)
        
        try:
            call_summary = str(summary.get('call_summary', ''))
            concerns = summary.get('prospect_concerns', [])
            next_steps = summary.get('next_steps', [])
            sentiment_score = sentiment.get('combined_score', '0%')
            
            prompt = f"""Analyze this sales call and determine the lead disposition.

CALL DATA:
- Sentiment: {sentiment_score}
- Call Summary: {call_summary}
- Prospect Concerns: {', '.join(concerns) if concerns else 'None'}
- Next Steps: {', '.join(next_steps) if next_steps else 'None'}

TASK: Classify this lead as ONE of the following:

A. PROCEED IMMEDIATELY
   - Very interested, asked about pricing/features/timeline
   - Strong buying signals
   - Ready to move forward
   - Follow up within 24 hours

B. FOLLOW UP IN 3 DAYS
   - Moderate interest, needs nurturing
   - Asked questions but not ready to decide
   - Said "busy right now" (timing issue, not rejection)
   - Warm lead, follow up in 2-3 days

C. NURTURE (7 DAYS)
   - Low engagement but not negative
   - Has existing partner but open to alternatives (e.g., "secondary option", "if better price")
   - Gatekeeper (can't share info, will forward internally)
   - Long-term nurture, follow up in 1-2 weeks

D. DON'T PURSUE
   - Explicitly not interested ("don't want", "don't need", "not interested")
   - Very happy with current provider and NOT open to switch
   - Asked to stop calling or remove from list
   - Very low score (<3) with no positive signals

CRITICAL RULES:
1. "Busy right now" = B (timing issue, NOT rejection)
2. "Already have + open to switch" = C (nurture opportunity)
3. "Already have + 100% happy + NOT open" = D (don't pursue)
4. Gatekeeper (can't share details) = C (nurture)
5. "Don't want/need" or "not interested" = D (don't pursue)
6. Asked about pricing/cost = A or B (interested)
7. Strong interest signals = A
8. Negative sentiment with no positive signals = D

Respond in this EXACT format:
DISPOSITION: [A/B/C/D]
ACTION: [One-line recommended action]
REASONING: [One sentence explaining why]
CONFIDENCE: [High/Medium/Low]"""

            llm_response = self._call_gemini_api(prompt, temperature=0.1, max_output_tokens=150)
            
            if not llm_response:
                logger.warning("LLM disposition failed, using fallback")
                return self._determine_lead_disposition_fallback(sentiment, summary)
            
            # Parse LLM response
            disposition_map = {
                'A': 'PROCEED IMMEDIATELY',
                'B': 'FOLLOW UP IN 3 DAYS',
                'C': 'NURTURE (7 DAYS)',
                'D': 'DON\'T PURSUE'
            }
            
            action_map = {
                'A': 'Follow up within 24 hours',
                'B': 'Send info, follow up in 2-3 days',
                'C': 'Add to nurture campaign, follow up in 1-2 weeks',
                'D': 'Archive - Focus on other leads'
            }
            
            # Extract disposition letter (A/B/C/D)
            disposition_letter = None
            for line in llm_response.split('\n'):
                if 'DISPOSITION:' in line.upper():
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter in line:
                            disposition_letter = letter
                            break
                    break
            
            if not disposition_letter:
                logger.warning("Could not parse LLM response, using fallback")
                return self._determine_lead_disposition_fallback(sentiment, summary)
            
            # Extract action and reasoning
            action = action_map.get(disposition_letter, "Review manually")
            reasoning = ""
            confidence = "Medium"
            
            for line in llm_response.split('\n'):
                if 'ACTION:' in line.upper():
                    action = line.split(':', 1)[1].strip()
                elif 'REASONING:' in line.upper():
                    reasoning = line.split(':', 1)[1].strip()
                elif 'CONFIDENCE:' in line.upper():
                    confidence = line.split(':', 1)[1].strip()
            
            if not reasoning:
                reasoning = f"LLM analysis based on call context and sentiment"
            
            return {
                "disposition": disposition_map[disposition_letter],
                "recommended_action": action,
                "reasoning": reasoning,
                "decision_confidence": confidence,
                "llm_powered": True
            }
            
        except Exception as e:
            logger.error(f"LLM disposition error: {str(e)}, using fallback", exc_info=True)
            return self._determine_lead_disposition_fallback(sentiment, summary)
    
    def _determine_lead_disposition_fallback(self, sentiment: Dict, summary: Dict) -> Dict:
        """Determine if lead is worth pursuing - clear YES/NO decision"""
        
        concerns = summary.get('prospect_concerns', [])
        next_steps = summary.get('next_steps', [])
        
        # Decision logic
        disposition = "UNKNOWN"
        action = "Review manually"
        reasoning = []
        
        # RED FLAGS - Don't pursue (check both concerns AND call summary)
        red_flags = [
            "already have",
            "not interested",
            "don't call",
            "remove my number",
            "satisfied with current",
            "100% happy",
            "completely satisfied",
            "very happy with",
            "please stop calling",
            "take me off your list",
            "don't contact me",
            "not looking for",
            "don't need this",
            "don't need",
            "didn't need",
            "don't want",
            "not for me",
            "no thanks",
            "not right now",
            "waste of time",
            "stop bothering me"
        ]
        
        # Check red flags in concerns
        has_red_flag_in_concerns = any(
            any(flag in str(concern).lower() for flag in red_flags)
            for concern in concerns
        )
        
        # Also check red flags in call summary (more comprehensive)
        call_summary = str(summary.get('call_summary', '')).lower()
        has_red_flag_in_summary = any(flag in call_summary for flag in red_flags)
        
        # Red flag if found in EITHER location
        has_red_flag = has_red_flag_in_concerns or has_red_flag_in_summary
        
        # GREEN FLAGS - Strong interest signals only (more strict)
        green_flags = [
            "very interested",
            "definitely interested",
            "tell me more about pricing",
            "how much does it cost",
            "what's the price",
            "send me pricing",
            "want to know more",
            "sounds good",
            "sounds interesting",
            "that's exactly what we need",
            "when can we start",
            "let's schedule a demo",
            "can we meet"
        ]
        
        summary_text = str(summary.get('call_summary', '')).lower()
        has_green_flag = any(flag in summary_text for flag in green_flags)
        
        # GATEKEEPER FLAGS - Not decision maker, can't help
        gatekeeper_flags = [
            "i can't share",
            "i'm not authorized",
            "i don't handle",
            "not the right person",
            "can't disclose",
            "don't have that information",
            "i'll forward this",
            "let me check with",
            "my manager handles",
            "my boss decides",
            "i'm just an assistant",
            "talk to hr",
            "contact our admin",
            "someone else manages"
        ]
        
        has_gatekeeper_flag = any(flag in summary_text for flag in gatekeeper_flags)
        
        # SPECIAL CASE: "Already have" (including variations) but open to switching
        already_have_variations = [
            "already have",
            "i have on business travel",
            "i have a travel",
            "i have someone",
            "we have a partner",
            "working with another",
            "currently using",
            "we use",
            "contracted with",
            "signed with",
            "partnered with"
        ]
        has_already_have = any(variation in call_summary for variation in already_have_variations)
        
        open_to_switch = any(phrase in call_summary for phrase in [
            "explore other options",
            "keeping options open",
            "open to exploring",
            "if they don't meet",
            "decide to leave",
            "if i leave",
            "switch",
            "secondary option",
            "backup option",
            "keep in mind",
            "for future reference",
            "if less discount",
            "if better price",
            "if cheaper",
            "i will prefer you",
            "consider switching",
            "might switch",
            "compare prices",
            "looking for alternatives",
            "contract expires",
            "contract renewal",
            "not locked in",
            "willing to change",
            "open to better deals",
            "if you offer more"
        ])
        
        # DECISION MATRIX
        if has_red_flag and not open_to_switch:
            # Hard red flag - not interested at all
            disposition = "DON'T PURSUE"
            action = "Archive - Focus on other leads"
            reasoning.append("Prospect not qualified or explicitly not interested")
            
        elif has_gatekeeper_flag:
            # Gatekeeper who can't/won't share decision maker info
            disposition = "NURTURE (7 DAYS)"
            action = "Send WhatsApp info, wait for internal referral"
            reasoning.append("Not decision maker, can't share details")
            
        elif has_already_have and not open_to_switch:
            # Has partner and NOT open to switching
            disposition = "DON'T PURSUE"
            action = "Archive - Satisfied with current provider"
            reasoning.append("Has existing partner and not open to alternatives")
            
        elif has_already_have and open_to_switch:
            # They have a partner BUT are open to switching - NURTURE lead!
            disposition = "NURTURE (7 DAYS)"
            action = "Send promised info, follow up in 1-2 weeks"
            reasoning.append("Has existing partner but open to alternatives")
            
        elif has_green_flag:
            # Strong interest signals
            disposition = "PROCEED IMMEDIATELY"
            action = "Follow up within 24 hours"
            reasoning.append("High interest with strong buying signals")
            
        elif next_steps and len(next_steps) > 0 and next_steps[0] != "None":
            # Has next steps - good engagement
            disposition = "FOLLOW UP IN 3 DAYS"
            action = "Send info, follow up in 2-3 days"
            reasoning.append("Good engagement with next steps agreed")
            
        else:
            # Default: moderate engagement, needs nurturing
            disposition = "FOLLOW UP IN 3 DAYS"
            action = "Send info, follow up in 2-3 days"
            reasoning.append("Moderate interest, needs nurturing")
        
        # Additional context
        if "busy" in str(concerns).lower() or "busy" in call_summary:
            reasoning.append("Prospect mentioned being busy - timing issue, not rejection")
            if disposition == "DON'T PURSUE":
                disposition = "FOLLOW UP IN 3 DAYS"
                action = "Try again when less busy"
        
        return {
            "disposition": disposition,
            "recommended_action": action,
            "reasoning": " | ".join(reasoning),
            "decision_confidence": "High" if has_red_flag or has_green_flag else "Medium"
        }
    
    def _calculate_lead_score_from_disposition(self, disposition: Dict) -> Dict:
        """
        Calculate lead_score and lead_category based on disposition.
        
        Mapping:
        - "PROCEED IMMEDIATELY" → lead_score = 10/10, lead_category = "Hot Lead"
        - "FOLLOW UP IN 3 DAYS" → lead_score = 8/10, lead_category = "warm Lead"
        - "FOLLOW UP IN 7 DAYS" or "NURTURE (7 DAYS)" → lead_score = 6/10, lead_category = "Warm Lead"
        - "DON'T PURSUE" → lead_score = 4/10, lead_category = "Cold Lead"
        """
        disposition_str = disposition.get('disposition', '').upper()
        
        if disposition_str == 'PROCEED IMMEDIATELY':
            lead_score = 10.0
            lead_category = "Hot Lead"
            priority = "High"
        elif disposition_str == 'FOLLOW UP IN 3 DAYS':
            lead_score = 8.0
            lead_category = "warm Lead"
            priority = "High"
        elif disposition_str in ['FOLLOW UP IN 7 DAYS', 'NURTURE (7 DAYS)', 'NURTURE']:
            lead_score = 6.0
            lead_category = "Warm Lead"
            priority = "Medium"
        elif disposition_str == "DON'T PURSUE" or disposition_str == "DONT PURSUE":
            lead_score = 4.0
            lead_category = "Cold Lead"
            priority = "Low"
        else:
            # Default/Unknown disposition
            lead_score = 5.0
            lead_category = "Warm Lead"
            priority = "Medium"
        
        return {
            "lead_score": lead_score,
            "max_score": 10.0,
            "lead_category": lead_category,
            "priority": priority,
            "scoring_breakdown": {
                "disposition_based": f"Score calculated from disposition: {disposition_str}"
            }
        }
    
    def _enhance_recommendations(self, base_rec: str, lead_score: Dict, sentiment: Dict, quality: Dict, stage: Dict, duration: int) -> str:
        """Enhance recommendations with complete analytics context"""
        
        # Extract key metrics
        score = lead_score.get('lead_score', 0)
        category = lead_score.get('lead_category', 'Unknown')
        engagement = quality.get('engagement_level', 'Unknown')
        quality_rating = quality.get('quality_rating', 'Unknown')
        emotion = sentiment.get('advanced_emotions', {}).get('emotion', 'neutral')
        completion = stage.get('stage_completion_percentage', '0%')
        
        # Build enhanced recommendation
        enhanced = f"{base_rec}\n\n"
        enhanced += f"Analytics Context:\n"
        enhanced += f"- Lead Score: {score}/10 ({category})\n"
        enhanced += f"- Engagement: {engagement} ({quality_rating} quality)\n"
        enhanced += f"- Emotion: {emotion.title()}\n"
        enhanced += f"- Stage Completion: {completion}\n"
        enhanced += f"- Duration: {duration}s\n\n"
        
        # Add priority-based action items
        if score >= 7:
            enhanced += "Priority: HIGH - Immediate follow-up required within 24 hours.\n"
            enhanced += "Action: Call back or send personalized WhatsApp with pricing details."
        elif score >= 5:
            enhanced += "Priority: MEDIUM - Follow up within 2-3 days.\n"
            enhanced += "Action: Send WhatsApp brochure, email summary of value proposition."
        elif score >= 3:
            enhanced += "Priority: LOW - Nurture campaign, follow up in 1 week.\n"
            enhanced += "Action: Add to email list, send monthly newsletter."
        else:
            enhanced += "Priority: VERY LOW - Not qualified, minimal follow-up.\n"
            enhanced += "Action: Archive, focus on higher-scoring leads."
        
        return enhanced
    
    async def generate_summary(self, conversation_text: str, sentiment_data: Dict = None, duration: int = 0) -> Dict:
        """Generate call summary with enhanced recommendations using all analytics data"""
        
        logger.info("Starting summary generation...")
        if not self.gemini_api_key:
            logger.error("Gemini API key not configured for summary generation")
            return {"error": "Gemini API key not configured"}
        
        try:
            # Build analytics context for smarter recommendations
            analytics_context = ""
            if sentiment_data:
                analytics_context = f"""
            
            ANALYTICS DATA (use this for smarter recommendations):
            - Sentiment: {sentiment_data.get('sentiment_description', 'N/A')[:100]}...
            - Confidence: {sentiment_data.get('confidence_score', 'N/A')}
            - Combined Score: {sentiment_data.get('combined_score', 'N/A')}
            - Emotion: {sentiment_data.get('advanced_emotions', {}).get('emotion', 'N/A')}
            - Call Duration: {duration} seconds
            """
            
            prompt = f"""
            Analyze this sales conversation and provide a structured summary:

            CONVERSATION:
            {conversation_text}
            {analytics_context}

            Please provide a JSON response with the following structure:
            {{
                "call_summary": "Complete detailed summary from prospect's perspective (3-5 sentences)",
                "key_discussion_points": ["point1", "point2", "point3"],
                "prospect_questions": ["question1", "question2"],
                "prospect_concerns": ["concern1", "concern2"],
                "next_steps": ["step1", "step2"],
                "business_name": "extracted PROSPECT's business name (NOT Pluto Travels) or null",
                "contact_person": "extracted PROSPECT's contact name (NOT the agent/caller from Pluto Travels) or null",
                "phone_number": "extracted PROSPECT's phone number or null",
                "recommendations": "Detailed, actionable recommendation based on sentiment, emotion, duration, and conversation quality. Include specific next steps, timing, and priority level."
            }}

            CRITICAL INSTRUCTIONS FOR "call_summary":
            - Write a COMPLETE, DETAILED summary (3-5 sentences minimum)
            - DO NOT mention "Nithya", "Pluto Travels", "agent", "caller", or "received a call from"
            - Focus ONLY on what the PROSPECT said, expressed, and their business needs
            - Start directly with the prospect's information: "The prospect from [company] expressed..."
            - Include: their company name (if mentioned), their specific needs, what they asked about, their interest level
            - Write as if describing what the prospect told us about their business situation
            
            Example GOOD call_summary:
            "The prospect, Pavan Kumar from Techie Maya, expressed a need for transportation services for his employees, specifically drivers to travel 15 kilometers from their host to the company. The prospect inquired about pricing for this service. The conversation was an introductory call to explore how we can support Techie Maya's travel needs."
            
            Example BAD call_summary (DO NOT DO THIS):
            "The prospect received a call from Nithya, a corporate travel specialist from Pluto Travels..."
            
            OTHER FIELD INSTRUCTIONS:
            - For "business_name": Extract the PROSPECT's company name, NOT "Pluto Travels"
            - For "contact_person": Extract the PROSPECT's name, NOT the agent's name
            - Focus on what the PROSPECT said, their needs, and their response
            
            Focus on:
            - Business travel/transport needs mentioned by the prospect
            - Pricing discussions
            - Timeline requirements
            - Decision-making process
            - Any commitments or next steps agreed upon
            """
            
            logger.debug(f"Generating summary - Conversation length: {len(conversation_text)} chars, Duration: {duration}s")
            summary_text = self._call_gemini_api(prompt, temperature=0.3, max_output_tokens=800)
            
            if not summary_text:
                logger.error("Gemini API error - unable to generate summary")
                return {"error": f"Gemini API error - unable to generate summary"}
            
            logger.debug(f"Summary API response received - Length: {len(summary_text)} chars")
            parsed_summary = self._parse_summary_json(summary_text)

            if parsed_summary is None:
                # HYBRID FALLBACK: Extract structured data from raw LLM text response
                full_text = summary_text.strip()
                
                logger.warning("JSON parsing failed - using hybrid fallback to extract data from text")
                
                # Extract specific sections from LLM text response
                extracted_call_summary = self._extract_call_summary_from_text(full_text)
                extracted_recommendations = self._extract_recommendations_from_text(full_text)
                
                # Extract structured data (NO LIMITS - all items extracted)
                extracted_points = self._extract_simple_points_from_text(full_text)
                extracted_questions = self._extract_questions_from_text(full_text)
                extracted_concerns = self._extract_concerns_from_text(full_text)
                extracted_info = self._extract_business_info_from_text(full_text)
                
                # Build fallback response with extracted data
                fallback_summary = {
                    "call_summary": extracted_call_summary if extracted_call_summary else "Summary extraction failed. See recommendations for full LLM response.",
                    "key_discussion_points": extracted_points if extracted_points else ["See call_summary and recommendations for full details"],
                    "prospect_questions": extracted_questions if extracted_questions else ["See call_summary and recommendations for questions"],
                    "prospect_concerns": extracted_concerns if extracted_concerns else [],
                    "next_steps": [],
                    "business_name": extracted_info.get('business_name'),
                    "contact_person": extracted_info.get('contact_person'),
                    "phone_number": extracted_info.get('phone_number'),
                    "recommendations": extracted_recommendations if extracted_recommendations else "Recommendations not found in LLM response. Full text available in call_summary field."
                }
                
                logger.info(f"Hybrid fallback extracted - Summary: {len(extracted_call_summary)} chars, Points: {len(extracted_points)}, Questions: {len(extracted_questions)}, Concerns: {len(extracted_concerns)}")
                
                return fallback_summary

            normalized_summary = self._normalize_summary_fields(parsed_summary)
            logger.info(f"Summary generation complete - Key points: {len(normalized_summary.get('key_discussion_points', []))}, Questions: {len(normalized_summary.get('prospect_questions', []))}")
            return normalized_summary
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}", exc_info=True)
            return {"error": f"Gemini API error: {str(e)}"}
    
    def calculate_lead_score(self, conversation_text: str, sentiment: Dict, summary: Dict, duration_seconds: int) -> Dict:
        """
        Calculate lead score from 0-10 based on multiple factors
        Runs AFTER call - no performance impact
        """
        
        logger.debug("Calculating lead score...")
        score = 0.0
        max_score = 10.0
        scoring_breakdown = {}
        
        # 1. Sentiment Score (0-3 points)
        try:
            combined_score_str = sentiment.get('combined_score', '0.0%')
            combined_score = float(combined_score_str.replace('%', '')) / 100.0
            
            if combined_score >= 0.6:
                sentiment_points = 3.0
            elif combined_score >= 0.3:
                sentiment_points = 2.0
            elif combined_score >= 0:
                sentiment_points = 1.0
            else:
                sentiment_points = 0.0
            
            score += sentiment_points
            scoring_breakdown['sentiment'] = f"{sentiment_points}/3.0"
        except:
            scoring_breakdown['sentiment'] = "0/3.0"
        
        # 2. Engagement Level (0-2 points) - based on conversation length
        # BUT penalized if sentiment is negative (filters nuisance calls)
        word_count = len(conversation_text.split())
        
        if word_count >= 200:
            base_engagement = 2.0
        elif word_count >= 100:
            base_engagement = 1.5
        elif word_count >= 50:
            base_engagement = 1.0
        else:
            base_engagement = 0.5
        
        # Apply sentiment penalty for negative calls
        # If sentiment is negative, reduce engagement points by 50%
        if combined_score < 0:
            engagement_points = base_engagement * 0.5  # 50% penalty
        elif combined_score < 0.2:
            engagement_points = base_engagement * 0.75  # 25% penalty
        else:
            engagement_points = base_engagement
        
        score += engagement_points
        scoring_breakdown['engagement'] = f"{engagement_points}/2.0 ({word_count} words)"
        
        # 3. Duration Quality (0-1 point)
        # BUT penalized if sentiment is negative (filters long nuisance calls)
        if duration_seconds >= 180:  # 3+ minutes
            base_duration = 1.0
        elif duration_seconds >= 120:  # 2+ minutes
            base_duration = 0.7
        elif duration_seconds >= 60:  # 1+ minute
            base_duration = 0.4
        else:
            base_duration = 0.1
        
        # Apply sentiment penalty for negative calls
        # If sentiment is negative, reduce duration points by 50%
        if combined_score < 0:
            duration_points = base_duration * 0.5  # 50% penalty
        elif combined_score < 0.2:
            duration_points = base_duration * 0.75  # 25% penalty
        else:
            duration_points = base_duration
        
        score += duration_points
        scoring_breakdown['duration'] = f"{duration_points}/1.0 ({duration_seconds}s)"
        
        # 4. Information Provided (0-2 points)
        info_points = 0.0
        if summary.get('business_name'):
            info_points += 0.7
        if summary.get('contact_person'):
            info_points += 0.7
        if summary.get('phone_number'):
            info_points += 0.6
        
        score += info_points
        scoring_breakdown['information_provided'] = f"{info_points}/2.0"
        
        # 5. Next Steps Agreed (0-2 points)
        next_steps = summary.get('next_steps', [])
        if len(next_steps) >= 2:
            next_steps_points = 2.0
        elif len(next_steps) == 1:
            next_steps_points = 1.0
        else:
            next_steps_points = 0.0
        
        score += next_steps_points
        scoring_breakdown['next_steps'] = f"{next_steps_points}/2.0 ({len(next_steps)} steps)"
        
        final_score = round(min(score, max_score), 1)
        logger.debug(f"Lead score calculated: {final_score}/10 - Breakdown: {scoring_breakdown}")
        
        # Categorize lead quality
        if final_score >= 8.0:
            lead_category = "Hot Lead"
            priority = "High"
        elif final_score >= 6.0:
            lead_category = "Warm Lead"
            priority = "Medium"
        elif final_score >= 4.0:
            lead_category = "Cold Lead"
            priority = "Low"
        else:
            lead_category = "Not Qualified"
            priority = "Very Low"
        
        return {
            "lead_score": final_score,
            "max_score": max_score,
            "lead_category": lead_category,
            "priority": priority,
            "scoring_breakdown": scoring_breakdown
        }
    
    def calculate_conversation_quality(self, conversation_text: str, sentiment: Dict, duration_seconds: int) -> Dict:
        """
        Calculate conversation quality metrics
        Runs AFTER call - no performance impact
        """
        
        logger.debug("Calculating conversation quality metrics...")
        user_text = self._extract_user_messages(conversation_text)
        
        # Count turns (user responses)
        user_turns = len([line for line in conversation_text.split('\n') if line.strip().startswith('User:')])
        bot_turns = len([line for line in conversation_text.split('\n') if line.strip().startswith('Bot:')])
        
        # Calculate response rate
        if bot_turns > 0:
            response_rate = round((user_turns / bot_turns) * 100, 1)
        else:
            response_rate = 0.0
        
        # Average response length
        if user_turns > 0:
            avg_user_response_length = len(user_text.split()) / user_turns
        else:
            avg_user_response_length = 0
        
        # Engagement indicators
        questions_asked = user_text.count('?')
        
        # Quality assessment
        if user_turns >= 5 and avg_user_response_length >= 10:
            quality_rating = "Excellent"
        elif user_turns >= 3 and avg_user_response_length >= 5:
            quality_rating = "Good"
        elif user_turns >= 2:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        return {
            "quality_rating": quality_rating,
            "conversation_turns": {
                "user_turns": user_turns,
                "bot_turns": bot_turns,
                "total_turns": user_turns + bot_turns
            },
            "response_rate": f"{response_rate}%",
            "avg_user_response_length": round(avg_user_response_length, 1),
            "questions_asked_by_user": questions_asked,
            "engagement_level": "High" if user_turns >= 4 else "Medium" if user_turns >= 2 else "Low"
        }
    
    def parse_conversation_transcript(self, conversation_log: List, call_start_time) -> List[Dict]:
        """
        Parse conversation log into structured transcript with REAL timestamps
        
        Args:
            conversation_log: List of dicts with {role, message, timestamp}
            call_start_time: datetime when call started
        
        Returns:
            List of dicts with {timestamp, role, message} using real timestamps
        """
        transcript = []
        
        for entry in conversation_log:
            # Calculate elapsed time from call start
            message_time = entry.get('timestamp')
            if message_time and call_start_time:
                elapsed_seconds = (message_time - call_start_time).total_seconds()
                
                # Format as MM:SS
                minutes = int(elapsed_seconds // 60)
                seconds = int(elapsed_seconds % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
            else:
                timestamp = "00:00"
            
            transcript.append({
                "timestamp": timestamp,
                "role": entry.get('role', 'unknown'),
                "message": entry.get('message', '')
            })
        
        return transcript
    
    def parse_conversation_transcript_legacy(self, conversation_text: str, duration_seconds: int) -> List[Dict]:
        """
        LEGACY: Parse conversation text (old format) into transcript with estimated timestamps
        Used for backwards compatibility with old string-based conversation logs
        """
        transcript = []
        lines = conversation_text.strip().split('\n')
        
        # Calculate time per message (distribute evenly)
        num_messages = len([l for l in lines if l.strip()])
        time_per_message = duration_seconds / num_messages if num_messages > 0 else 0
        
        current_time = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse role and message
            if line.startswith("User:"):
                role = "user"
                message = line.replace("User:", "").strip()
            elif line.startswith("Bot:"):
                role = "agent"
                message = line.replace("Bot:", "").strip()
            else:
                # Skip lines that don't match expected format
                continue
            
            # Format timestamp as MM:SS
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            transcript.append({
                "timestamp": timestamp,
                "role": role,
                "message": message
            })
            
            current_time += time_per_message
        
        return transcript
    
    def extract_call_stages(self, conversation_text: str, summary: Dict) -> Dict:
        """
        Extract call stage information
        Runs AFTER call - no performance impact
        """
        
        logger.debug("Extracting call stages...")
        stages_reached = []
        
        # Detect stages based on conversation content
        text_lower = conversation_text.lower()
        
        # Stage 1: Greeting
        if any(keyword in text_lower for keyword in ['hello', 'hi', 'good morning', 'good afternoon', 'pluto travels']):
            stages_reached.append("1_greeting")
        
        # Stage 2: Decision-maker identification
        if any(keyword in text_lower for keyword in ['manage', 'handle', 'decision', 'in charge', 'responsible']):
            stages_reached.append("2_decision_maker_identification")
        
        # Stage 3: Value proposition
        if any(keyword in text_lower for keyword in ['save', 'cost', 'dashboard', 'partnership', 'convenience']):
            stages_reached.append("3_value_proposition")
        
        # Stage 4: Qualification
        if any(keyword in text_lower for keyword in ['how frequently', 'how often', 'travel', 'agency', 'pain point']):
            stages_reached.append("4_qualification")
        
        # Stage 5: Objection handling
        if any(keyword in text_lower for keyword in ['already have', 'not interested', 'busy', 'security', 'pricing']):
            stages_reached.append("5_objection_handling")
        
        # Stage 6: WhatsApp handoff
        if any(keyword in text_lower for keyword in ['whatsapp', 'send', 'brochure', 'message', 'follow up']):
            stages_reached.append("6_whatsapp_handoff")
        
        # Determine final stage
        if stages_reached:
            final_stage = stages_reached[-1]
        else:
            final_stage = "0_no_engagement"
        
        # Stage completion percentage
        stage_completion = round((len(stages_reached) / 6) * 100, 1)
        
        return {
            "stages_reached": stages_reached,
            "final_stage": final_stage,
            "stage_completion_percentage": f"{stage_completion}%",
            "total_stages_reached": len(stages_reached)
        }
    
    async def analyze_call(self, call_id: str, conversation_log, duration_seconds: int, call_start_time=None) -> Dict:
        """Complete call analysis - sentiment + summarization + scoring
        
        Args:
            call_id: Unique call identifier
            conversation_log: List of dicts with {role, message, timestamp} OR string (legacy)
            duration_seconds: Call duration
            call_start_time: datetime when call started (for real timestamps)
        """
        
        logger.info(f"Analyzing call {call_id}...")
        
        # Handle both new (structured) and old (string) formats
        if isinstance(conversation_log, dict):
            # Dict format - check for voiceagent 'segments' first
            if 'segments' in conversation_log:
                conversation_text = self._parse_voiceagent_segments(conversation_log['segments'])
            elif 'conversation' in conversation_log:
                conversation_text = conversation_log['conversation']
            elif 'messages' in conversation_log:
                # List of messages in dict
                messages = conversation_log['messages']
                if isinstance(messages, list):
                    conversation_text = "\n".join([f"{msg.get('role', 'Unknown').title()}: {msg.get('message', '')}" for msg in messages])
                else:
                    conversation_text = str(messages)
            else:
                # Try to convert entire dict to readable format
                conversation_text = str(conversation_log)
        elif isinstance(conversation_log, list) and len(conversation_log) > 0 and isinstance(conversation_log[0], dict):
            # Check if it's voiceagent segments format (has 'speaker' and 'text' fields)
            if 'speaker' in conversation_log[0] and 'text' in conversation_log[0]:
                conversation_text = self._parse_voiceagent_segments(conversation_log)
            else:
                # New format: List of dicts with timestamps
                # Convert to text for analytics
                conversation_text = "\n".join([f"{entry['role'].title()}: {entry['message']}" for entry in conversation_log])
        else:
            # Legacy format: String with "User:" and "Bot:" lines
            if isinstance(conversation_log, list):
                conversation_text = "\n".join(conversation_log)
            else:
                conversation_text = str(conversation_log)
        
        word_count = len(conversation_text.split())
        logger.info(f"Processing call - Word count: {word_count}, Duration: {duration_seconds}s, Text length: {len(conversation_text)} chars")
        
        logger.info("Step 1/7: Analyzing sentiment...")
        sentiment = await self.analyze_sentiment(conversation_text, duration=duration_seconds, word_count=word_count)
        
        logger.info("Step 2/7: Generating summary...")
        summary = await self.generate_summary(conversation_text, sentiment_data=sentiment, duration=duration_seconds)
        
        logger.info("Step 3/7: Determining lead disposition...")
        lead_disposition = await self._determine_lead_disposition_llm(sentiment, summary)
        logger.info(f"Lead disposition: {lead_disposition.get('disposition', 'Unknown')} - Action: {lead_disposition.get('recommended_action', 'N/A')}")
        
        logger.info("Step 4/7: Calculating lead score from disposition...")
        lead_score = self._calculate_lead_score_from_disposition(lead_disposition)
        logger.info(f"Lead score: {lead_score.get('lead_score', 0)}/10 - Category: {lead_score.get('lead_category', 'Unknown')}")
        
        logger.info("Step 5/7: Calculating quality metrics...")
        quality_metrics = self.calculate_conversation_quality(conversation_text, sentiment, duration_seconds)
        
        logger.info("Step 6/7: Extracting call stages...")
        stage_info = self.extract_call_stages(conversation_text, summary)
        
        # Step 7/7: Extract lead information and save to JSON
        lead_info = None
        lead_info_path = None
        if self.lead_extractor:
            logger.info("Step 7/7: Extracting lead information...")
            try:
                lead_info = await self.lead_extractor.extract_lead_information(conversation_text, summary)
                if lead_info:
                    lead_info_path = self.lead_extractor.save_to_json(lead_info, call_id)
                    logger.info(f"Lead info extracted: {len(lead_info)} fields, saved to: {lead_info_path}")
                else:
                    logger.debug("No lead information found in this call")
            except Exception as e:
                logger.error(f"Lead info extraction failed: {e}", exc_info=True)
        else:
            logger.debug("Lead extractor not available - skipping lead info extraction")
        
        # Enhance recommendations with complete analytics data
        if summary and 'recommendations' in summary and lead_score:
            enhanced_rec = self._enhance_recommendations(
                summary.get('recommendations', ''),
                lead_score,
                sentiment,
                quality_metrics,
                stage_info,
                duration_seconds
            )
            summary['recommendations'] = enhanced_rec
        
        cost_info = self._calculate_llm_cost()
        logger.info(f"LLM cost for this call: ${cost_info.get('cost_usd', 0):.6f} ({cost_info.get('total_tokens', 0)} tokens)")
        
        # Combine results (transcript removed - keeping only analytics)
        result = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "lead_disposition": lead_disposition,  # Clear YES/NO decision for quick filtering
            "sentiment": sentiment,
            "summary": summary,
            "lead_score": lead_score,
            "quality_metrics": quality_metrics,
            "stage_info": stage_info,
            "lead_info": lead_info,  # Extracted lead information (may be None)
            "lead_info_path": lead_info_path,  # Path to saved JSON file (may be None)
            "conversation_length": len(conversation_text),
            "word_count": len(conversation_text.split()),
            "cost": cost_info
        }
        
        logger.info(f"Call analysis complete - Call ID: {call_id}, Lead Score: {lead_score.get('lead_score', 0)}/10, Disposition: {lead_disposition.get('disposition', 'Unknown')}")
        
        self.cost_tracker = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'api_calls': 0
        }
        
        return result
    
    def save_transcript_txt(self, analysis: Dict) -> str:
        """Save conversation transcript as a formatted TXT file"""
        
        call_id = analysis.get('call_id', 'UNKNOWN')
        timestamp = analysis.get('timestamp', datetime.now().isoformat())
        duration_seconds = analysis.get('duration_seconds', 0)
        transcript = analysis.get('conversation_transcript', [])
        sentiment = analysis.get('sentiment', {})
        advanced_emotions = sentiment.get('advanced_emotions', {})
        
        # Format duration
        duration_minutes = duration_seconds // 60
        duration_seconds_remainder = duration_seconds % 60
        
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%B %d, %Y")
            time_start_str = dt.strftime("%H:%M:%S")
        except:
            date_str = "Unknown Date"
            time_start_str = "Unknown Time"
        
        # Build TXT content
        lines = []
        lines.append("╔" + "═" * 62 + "╗")
        lines.append("║" + " " * 20 + "CALL TRANSCRIPT" + " " * 27 + "║")
        lines.append("╚" + "═" * 62 + "╝")
        lines.append("")
        lines.append(f"Call ID: {call_id}")
        lines.append(f"Date: {date_str}")
        lines.append(f"Time: {time_start_str}")
        lines.append(f"Duration: {duration_minutes} minutes {duration_seconds_remainder} seconds")
        lines.append("")
        lines.append("─" * 64)
        lines.append("")
        
        # Add conversation
        for entry in transcript:
            timestamp_str = entry.get('timestamp', '00:00')
            role = entry.get('role', 'unknown')
            message = entry.get('message', '')
            
            # Format role label
            if role == "agent":
                role_label = "Agent (Nithya):"
            elif role == "user":
                role_label = "Lead:"
            else:
                role_label = f"{role.title()}:"
            
            # Write timestamp and role
            lines.append(f"[{timestamp_str}] {role_label}")
            
            # Word wrap message to 60 characters
            words = message.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 60:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())
            
            lines.append("")  # Blank line between messages
        
        lines.append("─" * 64)
        lines.append("")
        lines.append("CALL SUMMARY")
        lines.append("")
        
        # Add sentiment info
        sentiment_desc = sentiment.get('sentiment_description', 'N/A')
        combined_score = sentiment.get('combined_score', 'N/A')
        
        lines.append(f"Sentiment: {combined_score}")
        
        # Add emotion info
        if advanced_emotions:
            emotion = advanced_emotions.get('emotion', 'N/A')
            emotion_conf = advanced_emotions.get('confidence', 0)
            lines.append(f"Main Emotion: {emotion.title()} ({emotion_conf:.0%} confidence)")
        
        lines.append("")
        
        # Add lead score
        lead_score = analysis.get('lead_score', {})
        if lead_score:
            score = lead_score.get('score', 'N/A')
            category = lead_score.get('category', 'N/A')
            lines.append(f"Lead Score: {score}/10 ({category})")
            lines.append("")
        
        # Add summary info
        summary = analysis.get('summary', {})
        if summary:
            next_steps = summary.get('next_steps', [])
            if next_steps:
                lines.append("Key Outcome:")
                for step in next_steps:
                    lines.append(f"• {step}")
                lines.append("")
        
        lines.append("─" * 64)
        lines.append("End of Transcript")
        lines.append("─" * 64)
        
        # Save to file
        filename = f"transcript_{call_id}.txt"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def save_analysis(self, analysis: Dict, filename: Optional[str] = None) -> str:
        """Save analysis to JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"call_analysis_{analysis['call_id']}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        return filename
    
    
    def _determine_sentiment_category(self, combined_score: str) -> str:
        """
        Determine sentiment category from combined score
        
        Args:
            combined_score: Score as string with % (e.g., "61.0%")
            
        Returns:
            Sentiment category: "Positive", "Neutral", or "Negative"
        """
        try:
            score = float(combined_score.replace("%", ""))
            if score >= 60:
                return "Positive"
            elif score >= 40:
                return "Neutral"
            else:
                return "Negative"
        except:
            return "Neutral"
    
    def _calculate_quality_score(self, quality_metrics: Dict) -> float:
        """
        Calculate numeric quality score from quality_rating
        
        Args:
            quality_metrics: Quality metrics dictionary
            
        Returns:
            Quality score (0-10)
        """
        rating = quality_metrics.get("quality_rating", "Average")
        rating_map = {
            "Excellent": 9.0,
            "Good": 7.5,
            "Average": 5.0,
            "Poor": 3.0
        }
        return rating_map.get(rating, 5.0)
    
    def _extract_call_log_id(self, call_id: str) -> Optional[int]:
        """
        Extract call_log_id from call_id string
        
        Args:
            call_id: String like "DB_1_20251029_123045" or "MANUAL_20251029_123045"
            
        Returns:
            Extracted ID as integer, or None if cannot extract
        """
        try:
            # For DB calls: "DB_1_20251029_123045" -> extract "1"
            if call_id.startswith("DB_"):
                parts = call_id.split("_")
                if len(parts) >= 2:
                    return int(parts[1])
            
            # For other formats, return None (no call_log_id available)
            return None
        except:
            return None
    
    def save_to_database(self, analysis: Dict, call_log_id, db_config: Dict) -> bool:
        """
        DEPRECATED: This method uses the old voice_agent schema.
        Use save_to_lad_dev() instead for the new lad_dev schema.
        
        This method is kept for backwards compatibility but will be removed in a future version.
        
        Args:
            analysis: The complete analytics dictionary
            call_log_id: ID from call_logs_voiceagent table (can be UUID, int, or None)
                       If None, will extract from call_id string
            db_config: Dict with db connection parameters
        
        Returns:
            bool: True if saved successfully
        """
        import warnings
        warnings.warn(
            "save_to_database() is deprecated. Use save_to_lad_dev() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not DB_AVAILABLE:
            logger.error("Database library not available. Install psycopg2-binary")
            return False
        
        conn = None
        cursor = None
        
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()
            
            # Extract call_log_id from JSON call_id if not provided
            if call_log_id is None:
                call_id_str = analysis.get("call_id", "")
                call_log_id = self._extract_call_log_id(call_id_str)
                
                if call_log_id is None:
                    logger.warning(f"Cannot extract call_log_id from call_id: {call_id_str} - Skipping database save (only for DB calls)")
                    return False
            
            # Ensure call_log_id is converted to string for UUID column
            # If it's already a UUID object or UUID string, use it as-is
            # If it's an integer, we need to get the actual UUID from the database
            if isinstance(call_log_id, int):
                # Integer provided - need to fetch actual UUID from database using ROW_NUMBER()
                # Ordered by ctid (physical storage order, matches pgAdmin's default display)
                cursor.execute("""
                    SELECT id FROM (
                        SELECT 
                            id,
                            ROW_NUMBER() OVER (ORDER BY ctid) as row_num
                        FROM lad_dev.voice_call_logs
                    ) ranked
                    WHERE row_num = %s
                """, (call_log_id,))
                uuid_result = cursor.fetchone()
                if uuid_result:
                    call_log_id = uuid_result[0]
                else:
                    logger.error(f"Cannot find call with position {call_log_id}")
                    return False
            elif isinstance(call_log_id, str):
                # String - check if it's a valid UUID format
                import uuid as uuid_lib
                try:
                    # Validate UUID format and convert to UUID type
                    uuid_lib.UUID(call_log_id)
                    # Keep as string (PostgreSQL will handle the cast)
                except ValueError:
                    # Not a valid UUID string, try to extract from call_id
                    call_log_id = self._extract_call_log_id(str(call_log_id))
                    if call_log_id is None:
                        logger.error(f"Invalid call_log_id format: {call_log_id}")
                        return False
            
            # Extract data from analysis dictionary
            sentiment = analysis.get("sentiment", {})
            summary = analysis.get("summary", {})
            lead = analysis.get("lead_score", {})
            quality = analysis.get("quality_metrics", {})
            stage = analysis.get("stage_info", {})
            disposition = analysis.get("lead_disposition", {})
            cost_data = analysis.get("cost", {})
            
            # Prepare INSERT query - matches FULL lad_dev.voice_call_analysis schema
            # Includes ALL columns from old post_call_analysis_voiceagent + new columns
            query = """
                INSERT INTO lad_dev.voice_call_analysis (
                    call_log_id,
                    tenant_id,
                    summary,
                    sentiment,
                    key_phrases,
                    key_discussion_points,
                    prospect_questions,
                    prospect_concerns,
                    recommendations,
                    lead_category,
                    engagement_level,
                    stages_reached,
                    disposition,
                    recommended_action,
                    cost,
                    key_points,
                    lead_extraction,
                    raw_analysis,
                    analysis_cost
                ) VALUES (
                    %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            # Prepare call_log_id value
            if call_log_id is None:
                call_log_id_value = None
            elif hasattr(call_log_id, '__str__'):
                call_log_id_value = str(call_log_id)
            else:
                call_log_id_value = call_log_id
            
            # Extract tenant_id from analysis if available
            tenant_id_value = analysis.get("tenant_id")
            if tenant_id_value and hasattr(tenant_id_value, '__str__'):
                tenant_id_value = str(tenant_id_value)
            
            # Build key_points as JSONB from summary data
            key_points_data = summary.get("key_discussion_points", [])
            if isinstance(key_points_data, str):
                key_points_data = [key_points_data] if key_points_data else []
            
            # Build lead_extraction from lead info (special column for extracted lead details)
            lead_extraction_data = analysis.get("lead_info", {})
            
            # Build raw_analysis with all detailed data for backup
            raw_analysis_data = {
                "sentiment_full": sentiment,
                "lead_score_full": lead,
                "quality_metrics_full": quality,
                "stage_info_full": stage,
                "disposition_full": disposition,
                "cost_full": cost_data,
            }
            
            # Extract cost values
            cost_numeric = None
            analysis_cost_value = None
            if cost_data:
                # Try to extract numeric cost
                cost_inr = cost_data.get("cost_inr_formatted", "")
                if cost_inr:
                    try:
                        cost_numeric = float(str(cost_inr).replace("₹", "").replace(",", "").strip())
                        analysis_cost_value = cost_numeric
                    except (ValueError, TypeError):
                        pass
            
            # Prepare text fields for old columns (convert lists/dicts to text)
            def to_text(val):
                if val is None:
                    return ""
                if isinstance(val, (list, dict)):
                    import json
                    return json.dumps(val)
                return str(val)
            
            # Build values tuple (19 columns)
            values = (
                call_log_id_value,                                          # call_log_id
                tenant_id_value,                                            # tenant_id (NEW)
                summary.get("call_summary", ""),                            # summary
                sentiment.get("sentiment_description", ""),                 # sentiment
                to_text(sentiment.get("key_phrases", [])),                  # key_phrases (TEXT)
                to_text(summary.get("key_discussion_points", [])),          # key_discussion_points (TEXT)
                to_text(summary.get("prospect_questions", [])),             # prospect_questions (TEXT)
                to_text(summary.get("prospect_concerns", [])),              # prospect_concerns (TEXT)
                summary.get("recommendations", ""),                         # recommendations
                lead.get("lead_category", ""),                              # lead_category (TEXT)
                quality.get("engagement_level", ""),                        # engagement_level (TEXT)
                to_text(stage.get("stages_reached", [])),                   # stages_reached (TEXT)
                disposition.get("disposition", ""),                         # disposition (TEXT)
                disposition.get("recommended_action", ""),                  # recommended_action
                cost_numeric,                                               # cost (numeric)
                Json(key_points_data),                                      # key_points (JSONB, NEW)
                Json(lead_extraction_data),                                 # lead_extraction (JSONB, NEW)
                Json(raw_analysis_data),                                    # raw_analysis (JSONB, NEW)
                analysis_cost_value                                         # analysis_cost (numeric, NEW)
            )

            if call_log_id_value:
                cursor.execute(
                    "DELETE FROM lad_dev.voice_call_analysis WHERE call_log_id = %s::uuid",
                    (call_log_id_value,),
                )
            
            cursor.execute(query, values)
            conn.commit()
            
            # Update tags column in leads table with lead_category value
            lead_category_value = lead.get("lead_category", "")
            if lead_category_value and call_log_id_value:
                try:
                    import json
                    # Get lead_id from voice_call_logs using call_log_id
                    cursor.execute(
                        "SELECT lead_id FROM lad_dev.voice_call_logs WHERE id = %s::uuid",
                        (call_log_id_value,)
                    )
                    lead_result = cursor.fetchone()
                    
                    if lead_result and lead_result[0]:
                        lead_id_from_call = lead_result[0]
                        # Update tags column in leads table as JSON array
                        from datetime import timezone
                        cursor.execute(
                            "UPDATE lad_dev.leads SET tags = %s, updated_at = %s WHERE id = %s::uuid",
                            (json.dumps([lead_category_value]), datetime.now(timezone.utc), lead_id_from_call)
                        )
                        conn.commit()
                        logger.info(f"Updated tags column in leads table (lead_id: {lead_id_from_call}, tags: {[lead_category_value]})")
                except Exception as tag_update_error:
                    logger.warning(f"Failed to update tags in leads table: {tag_update_error}")
                    # Don't fail the whole operation if tag update fails
            
            logger.info(f"Analytics saved to database (call_log_id: {call_log_id})")
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
    
    async def save_to_lad_dev(
        self,
        analysis: Dict,
        call_log_id: str,
        tenant_id: str,
    ) -> bool:
        """
        Save analytics to lad_dev.voice_call_analysis table using CallAnalysisStorage.
        
        Phase 13: This is the new storage method using the lad_dev schema.
        
        Args:
            analysis: The complete analytics dictionary
            call_log_id: UUID of the call log (from lad_dev.voice_call_logs)
            tenant_id: UUID of the tenant for multi-tenancy
            
        Returns:
            bool: True if saved successfully
        """
        if not STORAGE_CLASSES_AVAILABLE:
            logger.error("CallAnalysisStorage not available - cannot save to lad_dev")
            return False
        
        try:
            # Extract data from analysis dictionary
            sentiment = analysis.get("sentiment", {})
            summary = analysis.get("summary", {})
            lead_score = analysis.get("lead_score", {})
            disposition = analysis.get("lead_disposition", {})
            cost_data = analysis.get("cost", {})
            lead_info = analysis.get("lead_info")
            
            # Build summary text
            summary_text = summary.get("call_summary", "")
            if not summary_text and summary.get("key_discussion_points"):
                summary_text = "; ".join(summary.get("key_discussion_points", []))
            
            # Build sentiment category
            sentiment_category = sentiment.get("sentiment_category", "neutral").lower()
            if sentiment_category not in ["positive", "negative", "neutral"]:
                sentiment_category = "neutral"
            
            # Build key_points list
            key_points = []
            if summary.get("key_discussion_points"):
                key_points.extend(summary.get("key_discussion_points", []))
            if summary.get("next_steps"):
                key_points.extend([f"Next: {s}" for s in summary.get("next_steps", [])])
            
            # Build lead_extraction dict
            lead_extraction = {}
            if lead_info:
                lead_extraction = lead_info
            elif lead_score:
                lead_extraction = {
                    "lead_category": lead_score.get("lead_category"),
                    "lead_score": lead_score.get("lead_score"),
                    "priority": lead_score.get("priority"),
                }
            if disposition:
                lead_extraction["disposition"] = disposition.get("disposition")
                lead_extraction["recommended_action"] = disposition.get("recommended_action")
            
            # Build raw_analysis (full response for debugging)
            raw_analysis = {
                "sentiment": sentiment,
                "summary": summary,
                "lead_score": lead_score,
                "lead_disposition": disposition,
                "quality_metrics": analysis.get("quality_metrics"),
                "stage_info": analysis.get("stage_info"),
            }
            
            # Get analysis cost
            analysis_cost = cost_data.get("cost_usd", 0.0) if cost_data else None
            
            # Use storage class
            storage = CallAnalysisStorage()
            analysis_id = await storage.upsert_analysis(
                call_log_id=call_log_id,
                tenant_id=tenant_id,
                summary=summary_text,
                sentiment=sentiment_category,
                key_points=key_points,
                lead_extraction=lead_extraction,
                raw_analysis=raw_analysis,
                analysis_cost=analysis_cost,
            )
            
            if analysis_id:
                logger.info(f"Analytics saved to lad_dev.voice_call_analysis (id={analysis_id})")
                
                # Update tags column in leads table with lead_category value
                lead_category_value = lead_score.get("lead_category", "")
                if lead_category_value:
                    try:
                        # Import LeadStorage to update tags
                        from db.storage.leads import LeadStorage
                        from db.storage.calls import CallStorage
                        
                        # Get lead_id from voice_call_logs
                        call_storage = CallStorage()
                        call_log = await call_storage.get_call_by_id(call_log_id)
                        
                        if call_log and call_log.get('lead_id'):
                            lead_id_from_call = call_log.get('lead_id')
                            
                            # Update tags in leads table using direct SQL
                            # (LeadStorage doesn't have a method to update just tags)
                            import psycopg2
                            import json
                            from datetime import timezone
                            from db.connection_pool import get_db_connection
                            from db.db_config import get_db_config
                            
                            db_config = get_db_config()
                            with get_db_connection(db_config) as conn:
                                with conn.cursor() as cursor:
                                    cursor.execute(
                                        "UPDATE lad_dev.leads SET tags = %s, updated_at = %s WHERE id = %s::uuid",
                                        (json.dumps([lead_category_value]), datetime.now(timezone.utc), lead_id_from_call)
                                    )
                                    conn.commit()
                                    logger.info(f"Updated tags column in leads table (lead_id: {lead_id_from_call}, tags: {[lead_category_value]})")
                    except Exception as tag_update_error:
                        logger.warning(f"Failed to update tags in leads table: {tag_update_error}")
                        # Don't fail the whole operation if tag update fails
                
                return True
            else:
                logger.error("Failed to save analytics to lad_dev")
                return False
                
        except Exception as e:
            logger.error(f"lad_dev save failed: {e}", exc_info=True)
            return False
    
    
    def print_analysis_report(self, analysis: Dict):
        """Print a formatted analysis report"""
        
        print("\n" + "="*60)
        print("CALL ANALYTICS REPORT")
        print("="*60)
        
        print(f"Call ID: {analysis['call_id']}")
        print(f"Duration: {analysis['duration_seconds']} seconds")
        print(f"Conversation: {analysis['word_count']} words")
        
        # Lead Disposition - NEW! Clear decision
        if 'lead_disposition' in analysis:
            disposition = analysis['lead_disposition']
            print(f"\n{'='*60}")
            print(f"LEAD DISPOSITION: {disposition['disposition']}")
            print(f"   Recommended Action: {disposition['recommended_action']}")
            print(f"   Reasoning: {disposition['reasoning']}")
            print(f"   Confidence: {disposition['decision_confidence']}")
            print(f"{'='*60}")
        
        # Lead Score
        if 'lead_score' in analysis:
            lead_score = analysis['lead_score']
            print(f"\nLEAD SCORE: {lead_score['lead_score']}/{lead_score['max_score']} - {lead_score['lead_category']}")
            print(f"   Priority: {lead_score['priority']}")
            print(f"   Scoring Breakdown:")
            for metric, value in lead_score['scoring_breakdown'].items():
                print(f"      • {metric.replace('_', ' ').title()}: {value}")
        
        # Quality Metrics
        if 'quality_metrics' in analysis:
            quality = analysis['quality_metrics']
            print(f"\n📈 CONVERSATION QUALITY: {quality['quality_rating']}")
            print(f"   Engagement Level: {quality['engagement_level']}")
            print(f"   Turns: User={quality['conversation_turns']['user_turns']}, Bot={quality['conversation_turns']['bot_turns']}")
            print(f"   Response Rate: {quality['response_rate']}")
            print(f"   Avg User Response: {quality['avg_user_response_length']} words")
            print(f"   Questions Asked: {quality['questions_asked_by_user']}")
        
        # Stage Info
        if 'stage_info' in analysis:
            stage = analysis['stage_info']
            print(f"\nCALL STAGES:")
            print(f"   Final Stage: {stage['final_stage']}")
            print(f"   Completion: {stage['stage_completion_percentage']}")
            print(f"   Stages Reached: {', '.join(stage['stages_reached'])}")
        
        print(f"\nSENTIMENT ANALYSIS:")
        print(f"   Description: {analysis['sentiment']['sentiment_description']}")
        print(f"   Confidence Score: {analysis['sentiment']['confidence_score']}")
        print(f"   Combined Score: {analysis['sentiment']['combined_score']}")
        print(f"   Reasoning: {analysis['sentiment']['reasoning']}")
        
        if analysis['sentiment']['key_phrases']:
            print(f"   Key Phrases: {', '.join(analysis['sentiment']['key_phrases'])}")
        
        if 'error' not in analysis['summary']:
            summary = analysis['summary']
            
            print(f"\nCALL SUMMARY:")
            
            if summary.get('business_name'):
                print(f"   Business: {summary['business_name']}")
            if summary.get('contact_person'):
                print(f"   Contact: {summary['contact_person']}")
            if summary.get('phone_number'):
                print(f"   Phone: {summary['phone_number']}")
            
            if summary.get('key_discussion_points'):
                print(f"\n   Key Discussion Points:")
                for point in summary['key_discussion_points']:
                    print(f"   • {point}")
            
            if summary.get('prospect_questions'):
                print(f"\n   Prospect Questions:")
                for question in summary['prospect_questions']:
                    print(f"   • {question}")
            
            if summary.get('prospect_concerns'):
                print(f"\n   Prospect Concerns:")
                for concern in summary['prospect_concerns']:
                    print(f"   • {concern}")
            
            if summary.get('next_steps'):
                print(f"\n   Next Steps:")
                for step in summary['next_steps']:
                    print(f"   • {step}")
            
            if summary.get('recommendations'):
                print(f"\n   Recommendations: {summary['recommendations']}")
        
        else:
            print(f"\nSummary Error: {analysis['summary']['error']}")
        
        print("="*60)

analytics = CallAnalytics()

async def analyze_call_complete(call_id: str, conversation_log, duration_seconds: int, call_start_time=None):
    """Complete call analysis - call this AFTER the call ends
    
    Args:
        call_id: Unique call identifier
        conversation_log: List of dicts with {role, message, timestamp} OR string (legacy)
        duration_seconds: Call duration
        call_start_time: datetime when call started (optional, for real timestamps)
    """
    
    print(f"\nStarting post-call analysis for {call_id}...")
    
    analysis = await analytics.analyze_call(call_id, conversation_log, duration_seconds, call_start_time)
    analytics.print_analysis_report(analysis)
    json_filename = analytics.save_analysis(analysis)
    print(f"JSON Analysis saved to: {json_filename}")
    
    return analysis

"""
Standalone Call Analytics Tool
Database input only for generating call analytics JSON

Usage:
    # List all calls from database
    python merged_analytics.py --list-calls
    
    # From database (by row number)
    python merged_analytics.py --db-id 123
    
    # From database (by UUID)
    python merged_analytics.py --db-id bcc0402b-c290-4242-9873-3cd31052b84a
"""

import argparse
import sys

try:
    import psycopg2
    import psycopg2.errors
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("psycopg2 not installed. Database features disabled.")
    print("   Install with: pip install psycopg2-binary")


class StandaloneAnalytics:
    """Standalone analytics processor - database input only"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize standalone analytics
        
        Args:
            db_config: Database connection config (optional)
                      {'host': 'localhost', 'database': 'db', 'user': 'user', 'password': 'pass'}
        """
        self.analytics = CallAnalytics()
        self.db_config = db_config
    
    async def list_database_calls(self) -> None:
        """List all calls from database with row numbers matching the query order"""
        if not DB_AVAILABLE:
            raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
        
        if not self.db_config:
            raise ValueError("Database config not provided. Use --db-host, --db-name, --db-user, --db-pass or .env file")
        
        logger.info("Listing calls from database...")
        print("=" * 80)
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        try:
            # Fetch calls in pgAdmin's default display order (physical storage order using ctid)
            # This matches exactly how pgAdmin shows rows when no ORDER BY is specified
            cursor.execute("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY ctid) as row_num,
                    id,
                    started_at,
                    ended_at,
                    CASE 
                        WHEN transcripts IS NULL OR transcripts::text = '' THEN 'No transcript'
                        ELSE SUBSTRING(transcripts::text, 1, 50) || '...'
                    END as transcript_preview
                FROM lad_dev.voice_call_logs
                ORDER BY ctid
                LIMIT 500000
            """)
            
            calls = cursor.fetchall()
            
            if not calls:
                logger.warning("No calls found in database.")
                print("No calls found in database.")
                return
            
            print(f"{'Row':<6} {'UUID':<40} {'Started At':<20} {'Duration':<10} {'Transcript Preview'}")
            print("-" * 80)
            
            for row_num, call_id, started_at, ended_at, transcript_preview in calls:
                duration = ""
                if started_at and ended_at:
                    duration_seconds = int((ended_at - started_at).total_seconds())
                    duration = f"{duration_seconds}s"
                
                started_str = started_at.strftime('%Y-%m-%d %H:%M:%S') if started_at else 'N/A'
                print(f"{row_num:<6} {str(call_id):<40} {started_str:<20} {duration:<10} {transcript_preview}")
            
            print("-" * 80)
            print(f"\nRow numbers match pgAdmin's default display order (physical storage order)")
            print(f"To analyze a call, use: python merged_analytics.py --db-id <row_number>")
            print(f"Or use UUID directly: python merged_analytics.py --db-id <uuid_string>")
            
        finally:
            cursor.close()
            conn.close()
    
    async def analyze_from_database(self, call_log_id) -> Dict:
        """
        Analyze call from database call_logs table
        
        Args:
            call_log_id: ID from call_logs_voiceagent table
            
        Returns:
            Complete analytics JSON
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
            # Fetch call data from call_logs_voiceagent table (in voice_agent schema)
            # Handle both UUID and integer IDs
            # If call_log_id is integer and id column is UUID, use ROW_NUMBER() for sequential access
            # If call_log_id is UUID string, match directly
            
            # Try to parse as integer first, otherwise treat as UUID string
            try:
                call_log_id_int = int(call_log_id)
                call_log_id = call_log_id_int
            except (ValueError, TypeError):
                # Not an integer, treat as UUID string
                pass
            
            if isinstance(call_log_id, int):
                # Integer ID: Use ROW_NUMBER() to find the Nth record (ordered by ctid)
                cursor.execute("""
                    SELECT id, transcripts, started_at, ended_at, recording_url
                    FROM (
                        SELECT 
                            id, 
                            transcripts, 
                            started_at, 
                            ended_at, 
                            recording_url,
                            ROW_NUMBER() OVER (ORDER BY ctid) as row_num
                        FROM lad_dev.voice_call_logs
                    ) ranked
                    WHERE row_num = %s
                """, (call_log_id,))
            else:
                # UUID string: Try direct UUID match or text match
                try:
                    cursor.execute("""
                        SELECT id, transcripts, started_at, ended_at, recording_url
                        FROM lad_dev.voice_call_logs
                        WHERE id = %s::uuid
                    """, (str(call_log_id),))
                except (psycopg2.errors.InvalidTextRepresentation, psycopg2.errors.UndefinedFunction):
                    # Fallback: try text match
                    cursor.execute("""
                        SELECT id, transcripts, started_at, ended_at, recording_url
                        FROM lad_dev.voice_call_logs
                        WHERE id::text = %s
                    """, (str(call_log_id),))
            
            call_data = cursor.fetchone()
            
            if not call_data:
                raise ValueError(f"Call log {call_log_id} not found in database")
            
            db_call_id, transcripts, started_at, ended_at = call_data[:4]
            recording_url = call_data[4] if len(call_data) > 4 else None
            
            # Get tenant_id for save_to_lad_dev
            cursor.execute(
                "SELECT tenant_id FROM lad_dev.voice_call_logs WHERE id = %s::uuid",
                (str(db_call_id),)
            )
            tenant_result = cursor.fetchone()
            tenant_id = tenant_result[0] if tenant_result else None
            
            # Calculate duration from timestamps
            if started_at and ended_at:
                duration = int((ended_at - started_at).total_seconds())
            else:
                duration = 0
            
            # Get transcript from transcripts column
            # Handle both JSONB (dict/list) and text formats
            if transcripts:
                if isinstance(transcripts, dict):
                    # JSONB dict - check for voiceagent segments or other formats
                    if 'segments' in transcripts:
                        conversation_text = transcripts  # Pass dict to analyze_call for segment parsing
                    else:
                        # Other dict format
                        conversation_text = transcripts
                elif isinstance(transcripts, list):
                    # JSONB list - check if it's voiceagent segments
                    if transcripts and isinstance(transcripts[0], dict) and 'speaker' in transcripts[0]:
                        conversation_text = transcripts  # Pass list to analyze_call for segment parsing
                    else:
                        # Other list format
                        conversation_text = transcripts
                else:
                    # Plain text
                    conversation_text = str(transcripts)
            else:
                conversation_text = ""
            
            if not conversation_text:
                raise ValueError(f"No transcript found for call {call_log_id}")
            
            call_id = f"DB_{call_log_id}_{started_at.strftime('%Y%m%d_%H%M%S') if started_at else 'unknown'}"
            
            logger.info(f"Call ID: {call_id}, Duration: {duration}s, Started: {started_at}, Ended: {ended_at}")
            
            # Run analytics
            result = await self.analytics.analyze_call(
                call_id=call_id,
                conversation_log=conversation_text,
                duration_seconds=duration,
                call_start_time=started_at
            )
            
            # Add tenant_id to result for tracking
            if tenant_id:
                result['tenant_id'] = str(tenant_id)
            
            # Save to database - use new method if tenant_id is available
            if tenant_id and STORAGE_CLASSES_AVAILABLE:
                logger.info("Saving analytics to lad_dev.voice_call_analysis table...")
                success = await self.analytics.save_to_lad_dev(result, str(db_call_id), str(tenant_id))
            else:
                logger.info("Saving analytics to post_call_analysis_voiceagent table (legacy)...")
                success = self.analytics.save_to_database(result, db_call_id, self.db_config)
            
            if success:
                logger.info("Analytics saved to database successfully!")
            else:
                logger.error("Failed to save analytics to database")
            
            return result
            
        finally:
            cursor.close()
            conn.close()
    
    def save_json(self, result: Dict, output_file: Optional[str] = None) -> str:
        """
        Save analytics result to JSON file
        
        Args:
            result: Analytics result dictionary
            output_file: Optional custom output filename
            
        Returns:
            Path to saved file
        """
        if not output_file:
            call_id = result.get('call_id', 'UNKNOWN')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"call_analysis_{call_id}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis saved to: {output_file}")
        return output_file
    

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Standalone Call Analytics Tool - Database input only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all calls from database (shows row numbers matching --db-id usage)
    python merged_analytics.py --list-calls
    
    # From database (by row number - matches pgAdmin's default display order)
    python merged_analytics.py --db-id 23
    
    # From database (by UUID directly)
    python merged_analytics.py --db-id bcc0402b-c290-4242-9873-3cd31052b84a
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--db-id', help='Call log ID from database (integer for row number, or UUID string)')
    input_group.add_argument('--list-calls', action='store_true', help='List all calls with row numbers')
    
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--db-host', help='Database host (default: from .env or localhost)')
    parser.add_argument('--db-name', help='Database name (default: from .env)')
    parser.add_argument('--db-user', help='Database user (default: from .env or postgres)')
    parser.add_argument('--db-pass', help='Database password (default: from .env)')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host or os.getenv('DB_HOST', 'localhost'),
        'database': args.db_name or os.getenv('DB_NAME', 'salesmaya_agent'),
        'user': args.db_user or os.getenv('DB_USER', 'postgres'),
        'password': args.db_pass or os.getenv('DB_PASSWORD')
    }
    
    if not db_config['password']:
        parser.error("Database password required. Provide via --db-pass or DB_PASSWORD in .env file")
    
    analytics = StandaloneAnalytics(db_config=db_config)
    result = None
    
    try:
        if args.list_calls:
            await analytics.list_database_calls()
            return
        elif args.db_id:
            result = await analytics.analyze_from_database(args.db_id)
        
        if result:
            analytics.save_json(result, args.output)
            
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            print(f"Call ID: {result['call_id']}")
            print(f"Disposition: {result['lead_disposition']['disposition']}")
            print(f"Action: {result['lead_disposition']['recommended_action']}")
            print(f"Lead Score: {result['lead_score']['lead_score']}/10 ({result['lead_score']['lead_category']})")
            print(f"Sentiment: {result['sentiment']['combined_score']} (confidence: {result['sentiment']['confidence_score']})")
            print(f"Engagement: {result['quality_metrics']['engagement_level']}")
            print(f"Duration: {result['duration_seconds']}s")
            print("="*60)
            print("Analysis complete!")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
