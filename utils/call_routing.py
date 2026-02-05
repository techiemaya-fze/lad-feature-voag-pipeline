"""
Call Routing Module
===================

Handles phone number validation, formatting, and routing based on 
carrier-specific rules stored in voice_agent_numbers.rules JSONB column.

Rules Schema:
{
    "inbound": bool,
    "outbound": bool,
    "allowed_outbound": "india" | "global",
    "required_lead_format": ["0", "base_number"] | ["country_code", "base_number"]
}

Example Usage:
    from utils.call_routing import validate_and_format_call
    
    result = validate_and_format_call(
        from_number="+919876543210",
        to_number="09876543210",
        db_config=get_db_config()
    )
    
    if result.success:
        dial_number = result.formatted_to_number
    else:
        raise HTTPException(400, result.error_message)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from db.connection_pool import get_db_connection

logger = logging.getLogger(__name__)

# Country code patterns for detection
# IMPORTANT: Longer prefixes must come BEFORE shorter ones (971 before 91)
COUNTRY_CODE_PATTERNS = [
    ('+971', 'uae', 9),     # UAE with + (must be before +91)
    ('971', 'uae', 9),      # UAE without + (must be before 91)
    ('+91', 'india', 10),   # India: +91 + 10 digits
    ('91', 'india', 10),    # India without +
    ('+1', 'usa', 10),      # USA/Canada
    ('+44', 'uk', 10),      # UK
    ('+61', 'australia', 9), # Australia
]

# India detection patterns
INDIA_PREFIXES = ['+91', '91', '0']
INDIA_COUNTRY_CODE = '+91'


@dataclass
class CallRoutingResult:
    """Result of call routing validation and formatting."""
    success: bool
    formatted_to_number: Optional[str] = None
    error_message: Optional[str] = None
    carrier_name: Optional[str] = None
    detected_country: Optional[str] = None
    outbound_trunk_id: Optional[str] = None  # SIP trunk ID from rules
    

@dataclass
class ParsedNumber:
    """Parsed phone number components."""
    country_code: Optional[str]  # e.g., "+91", "+1"
    base_number: str             # e.g., "9876543210"
    country: Optional[str]       # e.g., "india", "usa"
    original: str                # Original input


def parse_phone_number(phone: str) -> ParsedNumber:
    """
    Parse phone number to extract country code and base number.
    
    Args:
        phone: Raw phone number string
        
    Returns:
        ParsedNumber with country_code, base_number, country
    """
    if not phone:
        return ParsedNumber(None, "", None, phone or "")
    
    # Clean the input
    cleaned = str(phone).strip()
    original = cleaned
    
    # Remove non-digit characters except leading +
    if cleaned.startswith('+'):
        cleaned = '+' + re.sub(r'[^\d]', '', cleaned[1:])
    else:
        cleaned = re.sub(r'[^\d]', '', cleaned)
    
    if not cleaned:
        return ParsedNumber(None, "", None, original)
    
    # Try to match known country codes
    for prefix, country, expected_length in COUNTRY_CODE_PATTERNS:
        if cleaned.startswith(prefix):
            base = cleaned[len(prefix):]
            # Validate base number length
            if len(base) >= expected_length:
                return ParsedNumber(
                    country_code=prefix if prefix.startswith('+') else f"+{prefix}",
                    base_number=base,
                    country=country,
                    original=original
                )
    
    # Check for 0-prefix (India local format: 0 + 10 digits)
    if cleaned.startswith('0') and len(cleaned) == 11:
        # 0 + 10 digits = India local format
        return ParsedNumber(
            country_code='+91',
            base_number=cleaned[1:],  # Remove leading 0
            country='india',
            original=original
        )
    
    # Check for 0-prefix (UAE local format: 0 + 9 digits)
    if cleaned.startswith('0') and len(cleaned) == 10:
        # 0 + 9 digits = UAE local format
        return ParsedNumber(
            country_code='+971',
            base_number=cleaned[1:],  # Remove leading 0
            country='uae',
            original=original
        )
    
    # SMART HEURISTICS for bare numbers (matching expert logic)
    # These numbers have no country code or trunk prefix
    if len(cleaned) == 9:
        # 9 digits → UAE national number
        return ParsedNumber(
            country_code='+971',
            base_number=cleaned,
            country='uae',
            original=original
        )
    elif len(cleaned) == 10:
        # 10 digits → ambiguous between India and US
        # Heuristic: India mobile numbers typically start with 6-9
        if cleaned[0] in '6789':
            return ParsedNumber(
                country_code='+91',
                base_number=cleaned,
                country='india',
                original=original
            )
        else:
            # US/Canada
            return ParsedNumber(
                country_code='+1',
                base_number=cleaned,
                country='usa',
                original=original
            )
    
    # Bare number without recognizable pattern - no assumption
    # Return as unknown format - caller must validate
    return ParsedNumber(
        country_code=None,
        base_number=cleaned,
        country=None,
        original=original
    )


def normalize_phone_to_e164(phone: str) -> str:
    """
    Normalize a phone number to E.164 format using smart heuristics.
    
    This handles:
    - Cleaning formatting characters
    - Normalizing international exit codes (00, 011 -> +)
    - Formatting bare numbers based on length and prefix
    
    Args:
        phone: Raw phone number string
        
    Returns:
        E.164 formatted string (e.g., +971501234567)
        
    Raises:
        ValueError: If number cannot be normalized
    """
    if not phone:
        raise ValueError("Phone number cannot be empty")
        
    cleaned = phone.strip()
    
    # Handle + prefix (keep it, strip formatting)
    if cleaned.startswith('+'):
        cleaned = '+' + re.sub(r'[^\d]', '', cleaned[1:])
    else:
        cleaned = re.sub(r'[^\d]', '', cleaned)
    
    # Normalize international exit codes (00, 011 -> +)
    cleaned = re.sub(r'^(00|011)', '+', cleaned)
    
    if not cleaned or cleaned == '+':
        raise ValueError("Phone number cannot be empty")
    
    # If already has + prefix and looks valid, return it
    if cleaned.startswith('+') and len(cleaned) >= 10:
        return cleaned
    
    # Use parse_phone_number to apply smart heuristics
    parsed = parse_phone_number(cleaned)
    
    if parsed.country_code and parsed.base_number:
        return f"{parsed.country_code}{parsed.base_number}"
        
    raise ValueError(f"Could not normalize '{phone}' to E.164 format")


def format_for_carrier(parsed: ParsedNumber, rules: Dict[str, Any]) -> str:
    """
    Format phone number according to carrier's required_lead_format.
    
    Args:
        parsed: Parsed phone number
        rules: Carrier rules from voice_agent_numbers.rules
        
    Returns:
        Formatted phone number string
    """
    required_format = rules.get('required_lead_format', ['country_code', 'base_number'])
    
    if not required_format:
        required_format = ['country_code', 'base_number']
    
    parts = []
    for part in required_format:
        if part == '0':
            parts.append('0')
        elif part == 'country_code':
            # Use detected country code or default to +91 for India
            cc = parsed.country_code or '+91'
            parts.append(cc)
        elif part == 'base_number':
            parts.append(parsed.base_number)
    
    return ''.join(parts)


def get_number_rules(from_number: str, db_config: dict, tenant_id: str | None = None) -> Optional[Dict[str, Any]]:
    """
    Fetch carrier rules for a from_number from database.
    
    Args:
        from_number: The from/caller number
        db_config: Database configuration
        tenant_id: Optional tenant ID for multi-tenant filtering
        
    Returns:
        Dict with 'carrier_name' and 'rules', or None if not found
    """
    if not from_number:
        return None
    
    # Parse from_number to get components
    parsed = parse_phone_number(from_number)
    
    try:
        with get_db_connection(db_config) as conn:
            with conn.cursor() as cur:
                # Build query with optional tenant_id filter
                if parsed.country_code and tenant_id:
                    cur.execute("""
                        SELECT provider, rules
                        FROM lad_dev.voice_agent_numbers
                        WHERE country_code = %s AND base_number = %s AND tenant_id = %s
                        LIMIT 1
                    """, (parsed.country_code, int(parsed.base_number) if parsed.base_number.isdigit() else 0, tenant_id))
                elif parsed.country_code:
                    # Fallback without tenant_id (legacy calls)
                    cur.execute("""
                        SELECT provider, rules
                        FROM lad_dev.voice_agent_numbers
                        WHERE country_code = %s AND base_number = %s
                        LIMIT 1
                    """, (parsed.country_code, int(parsed.base_number) if parsed.base_number.isdigit() else 0))
                elif tenant_id:
                    cur.execute("""
                        SELECT provider, rules
                        FROM lad_dev.voice_agent_numbers
                        WHERE base_number = %s AND tenant_id = %s
                        LIMIT 1
                    """, (int(parsed.base_number) if parsed.base_number.isdigit() else 0, tenant_id))
                else:
                    # Fallback - just base_number
                    cur.execute("""
                        SELECT provider, rules
                        FROM lad_dev.voice_agent_numbers
                        WHERE base_number = %s
                        LIMIT 1
                    """, (int(parsed.base_number) if parsed.base_number.isdigit() else 0,))
                
                row = cur.fetchone()
                if row:
                    return {
                        'carrier_name': row[0],
                        'rules': row[1] or {}
                    }
                
                # If tenant_id was provided but no match, log a warning
                if tenant_id:
                    logger.warning(f"No carrier rules found for from_number: {from_number} (tenant_id: {tenant_id[:8]}...)")
                else:
                    logger.warning(f"No carrier rules found for from_number: {from_number}")
                return None
                
    except Exception as e:
        logger.error(f"Error fetching carrier rules for {from_number}: {e}")
        return None


def validate_and_format_call(
    from_number: str,
    to_number: str,
    db_config: dict,
    tenant_id: str | None = None,
) -> CallRoutingResult:
    """
    Validate and format phone number for outbound call based on carrier rules.
    
    Args:
        from_number: The from/caller number (determines carrier rules)
        to_number: The destination number to validate and format
        db_config: Database configuration for rule lookup
        tenant_id: Optional tenant ID for multi-tenant filtering
        
    Returns:
        CallRoutingResult with success status and formatted number or error
    """
    if not from_number:
        # No from_number - allow call with original to_number
        logger.warning("No from_number provided, skipping routing validation")
        return CallRoutingResult(
            success=True,
            formatted_to_number=to_number,
            carrier_name="unknown"
        )
    
    if not to_number:
        return CallRoutingResult(
            success=False,
            error_message="to_number is required"
        )
    
    # Parse the destination number
    parsed_to = parse_phone_number(to_number)
    
    # Get carrier rules (with tenant_id for multi-tenant isolation)
    carrier_info = get_number_rules(from_number, db_config, tenant_id)
    
    if not carrier_info:
        # No rules found - still validate country detection
        logger.warning(f"No carrier rules for {from_number}, using default validation")
        if not parsed_to.country_code and not parsed_to.country:
            return CallRoutingResult(
                success=False,
                error_message=f"No country code detected in to_number '{to_number}'. Please provide country code (e.g., +91) or use 0-prefix for India.",
                carrier_name="unknown",
                detected_country=None
            )
        # Default: use E.164 format
        formatted = f"{parsed_to.country_code}{parsed_to.base_number}"
        return CallRoutingResult(
            success=True,
            formatted_to_number=formatted,
            carrier_name="unknown",
            detected_country=parsed_to.country
        )
    
    carrier_name = carrier_info['carrier_name']
    rules = carrier_info['rules'] or {}
    outbound_trunk_id = rules.get('outbound_trunk_id')  # Get trunk ID from rules
    
    # Check if outbound is allowed
    if not rules.get('outbound', True):
        return CallRoutingResult(
            success=False,
            error_message=f"Outbound calls not allowed for carrier: {carrier_name}",
            carrier_name=carrier_name
        )
    
    # Check allowed_outbound restriction
    allowed_outbound = rules.get('allowed_outbound', 'global')
    detected_country = parsed_to.country
    
    if allowed_outbound == 'india':
        if detected_country and detected_country.lower() != 'india':
            return CallRoutingResult(
                success=False,
                error_message=f"Only Indian calls allowed with {carrier_name}. Detected country: {detected_country}",
                carrier_name=carrier_name,
                detected_country=detected_country
            )
        
        # For India-only carriers, if no country detected, assume India
        if not detected_country:
            parsed_to = ParsedNumber(
                country_code='+91',
                base_number=parsed_to.base_number,
                country='india',
                original=parsed_to.original
            )
            detected_country = 'india'
    
    elif allowed_outbound.lower() == 'uae':
        if detected_country and detected_country.lower() != 'uae':
            return CallRoutingResult(
                success=False,
                error_message=f"Only UAE calls allowed with {carrier_name}. Detected country: {detected_country}",
                carrier_name=carrier_name,
                detected_country=detected_country
            )
        
        # For UAE-only carriers, if no country detected, assume UAE
        if not detected_country:
            parsed_to = ParsedNumber(
                country_code='+971',
                base_number=parsed_to.base_number,
                country='uae',
                original=parsed_to.original
            )
            detected_country = 'uae'
    
    elif allowed_outbound == 'global':
        # Global carrier - country code or 0-prefix REQUIRED
        if not parsed_to.country_code and not parsed_to.country:
            # No country detected - error, don't assume
            return CallRoutingResult(
                success=False,
                error_message=f"No country code detected in to_number '{to_number}'. Please provide country code (e.g., +91) or use 0-prefix for India.",
                carrier_name=carrier_name,
                detected_country=None
            )
    
    # Format the number according to carrier rules
    formatted_number = format_for_carrier(parsed_to, rules)
    
    logger.info(
        f"Call routing: carrier={carrier_name}, "
        f"original={to_number}, formatted={formatted_number}, "
        f"country={detected_country}"
    )
    
    return CallRoutingResult(
        success=True,
        formatted_to_number=formatted_number,
        carrier_name=carrier_name,
        detected_country=detected_country,
        outbound_trunk_id=outbound_trunk_id
    )


# Convenience function for quick validation
def is_call_allowed(from_number: str, to_number: str, db_config: dict) -> tuple[bool, str]:
    """
    Quick check if a call is allowed.
    
    Returns:
        Tuple of (is_allowed, error_message_or_formatted_number)
    """
    result = validate_and_format_call(from_number, to_number, db_config)
    if result.success:
        return True, result.formatted_to_number or to_number
    return False, result.error_message or "Unknown error"
