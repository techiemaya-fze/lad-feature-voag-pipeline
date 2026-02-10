"""
Number Storage Module
Handles phone number lookup and management
"""

import os
import logging
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Import connection pool manager (context manager pattern)
from db.connection_pool import get_db_connection
from db.schema_constants import NUMBERS_FULL

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class NumberStorage:
    """Manages phone number data in the database"""
    
    def __init__(self):
        """Initialize database connection"""
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD")
        }
    
    # Note: Uses get_db_connection() context manager for automatic cleanup
    
    async def find_number_by_phone(self, phone_number: str, tenant_id: str | None = None) -> Optional[str]:
        """
        Find number ID by phone number string
        
        Args:
            phone_number: Phone number to search for (e.g., "+919876543210")
            tenant_id: Optional tenant ID for multi-tenant filtering
            
        Returns:
            number_id (UUID string) if found, None otherwise
        """
        if not phone_number:
            return None
        
        try:
            # Split phone number into country code and base number
            country_code, base_number = self._split_phone_number(phone_number)
            
            if base_number is None:
                logger.debug(f"Could not parse phone number: {phone_number}")
                return None
            
            with get_db_connection(self.db_config) as conn:
                with conn.cursor() as cursor:
                    # Search by country_code and base_number with optional tenant_id filter
                    if country_code and tenant_id:
                        cursor.execute(
                            f"""
                            SELECT id
                            FROM {NUMBERS_FULL}
                            WHERE country_code = %s AND base_number = %s AND tenant_id = %s
                            LIMIT 1
                            """,
                            (country_code, base_number, tenant_id)
                        )
                    elif country_code:
                        # Fallback without tenant_id (legacy)
                        cursor.execute(
                            f"""
                            SELECT id
                            FROM {NUMBERS_FULL}
                            WHERE country_code = %s AND base_number = %s
                            LIMIT 1
                            """,
                            (country_code, base_number)
                        )
                    elif tenant_id:
                        cursor.execute(
                            f"""
                            SELECT id
                            FROM {NUMBERS_FULL}
                            WHERE base_number = %s AND tenant_id = %s
                            LIMIT 1
                            """,
                            (base_number, tenant_id)
                        )
                    else:
                        # Only base_number known (legacy fallback)
                        cursor.execute(
                            f"""
                            SELECT id
                            FROM {NUMBERS_FULL}
                            WHERE base_number = %s
                            LIMIT 1
                            """,
                            (base_number,)
                        )
                    
                    result = cursor.fetchone()
                    
                    if result:
                        number_id = str(result[0])
                        logger.info(f"Found number: id={number_id}, phone={phone_number}")
                        return number_id
                    else:
                        logger.debug(f"No number found for phone: {phone_number}")
                        return None
                
        except Exception as e:
            logger.error(f"Error finding number by phone {phone_number}: {e}", exc_info=True)
            return None
    
    def _split_phone_number(self, phone: str) -> tuple[str | None, int | None]:
        """Split phone into country_code and base_number."""
        import re
        
        if not phone:
            return None, None
        
        cleaned = str(phone).strip()
        if cleaned.startswith('+'):
            cleaned = '+' + re.sub(r'[^\d]', '', cleaned[1:])
        else:
            cleaned = re.sub(r'[^\d]', '', cleaned)
        
        if not cleaned or cleaned == '+':
            return None, None
        
        # Known country code patterns
        country_codes = [
            ('+91', 2),   # India
            ('+1', 1),    # USA/Canada
            ('+44', 2),   # UK
            ('+971', 3),  # UAE
        ]
        
        if cleaned.startswith('+'):
            for prefix, length in country_codes:
                if cleaned.startswith(prefix):
                    base = cleaned[len(prefix):]
                    if base.isdigit() and len(base) >= 6:
                        return prefix, int(base)
        
        # No + prefix - detect by known patterns
        for prefix, length in country_codes:
            digits_prefix = prefix[1:]
            if cleaned.startswith(digits_prefix) and len(cleaned) >= length + 8:
                base = cleaned[length:]
                return prefix, int(base)
        
        # Fallback - no country code
        if cleaned.isdigit():
            return None, int(cleaned)
        
        return None, None
    
    def get_provider_by_phone(self, phone_number: str, tenant_id: str | None = None) -> Optional[str]:
        """
        Get telephony provider for a phone number from voice_agent_numbers.
        
        Args:
            phone_number: Phone number to look up (e.g., "+971545335200")
            tenant_id: Optional tenant ID for multi-tenant filtering
            
        Returns:
            Provider string (e.g., 'vonage', 'twilio') if found, None otherwise
        """
        if not phone_number:
            return None
        
        try:
            country_code, base_number = self._split_phone_number(phone_number)
            
            if base_number is None:
                logger.debug(f"Could not parse phone number for provider lookup: {phone_number}")
                return None
            
            with get_db_connection(self.db_config) as conn:
                with conn.cursor() as cursor:
                    if country_code and tenant_id:
                        cursor.execute(
                            f"""
                            SELECT provider
                            FROM {NUMBERS_FULL}
                            WHERE country_code = %s AND base_number = %s AND tenant_id = %s
                            LIMIT 1
                            """,
                            (country_code, base_number, tenant_id)
                        )
                    elif country_code:
                        cursor.execute(
                            f"""
                            SELECT provider
                            FROM {NUMBERS_FULL}
                            WHERE country_code = %s AND base_number = %s
                            LIMIT 1
                            """,
                            (country_code, base_number)
                        )
                    elif tenant_id:
                        cursor.execute(
                            f"""
                            SELECT provider
                            FROM {NUMBERS_FULL}
                            WHERE base_number = %s AND tenant_id = %s
                            LIMIT 1
                            """,
                            (base_number, tenant_id)
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT provider
                            FROM {NUMBERS_FULL}
                            WHERE base_number = %s
                            LIMIT 1
                            """,
                            (base_number,)
                        )
                    
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        provider = str(result[0])
                        logger.info(f"Resolved provider '{provider}' for phone {phone_number[:4]}***")
                        return provider
                    else:
                        logger.debug(f"No provider found for phone: {phone_number[:4]}***")
                        return None
                
        except Exception as e:
            logger.error(f"Error getting provider for phone {phone_number[:4]}***: {e}", exc_info=True)
            return None
    
    async def get_default_outbound_number(self) -> Optional[int]:
        """
        Get the default outbound number for calls
        
        Returns:
            number_id (int) if configured, None otherwise
        """
        try:
            with get_db_connection(self.db_config) as conn:
                with conn.cursor() as cursor:
                    # Try to find an active outbound number
                    cursor.execute(
                        f"""
                        SELECT id
                        FROM {NUMBERS_FULL}
                        WHERE type = 'outbound' AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    )
                    
                    result = cursor.fetchone()
                    
                    # If no outbound number, try any active number
                    if not result:
                        logger.warning("No active outbound number found, trying any active number")
                        cursor.execute(
                            f"""
                            SELECT id
                            FROM {NUMBERS_FULL}
                            WHERE status = 'active'
                            ORDER BY created_at DESC
                            LIMIT 1
                            """
                        )
                        result = cursor.fetchone()
                    
                    if result:
                        number_id = result[0]
                        logger.info(f"Using default outbound number: id={number_id}")
                        return number_id
                    else:
                        logger.warning("No default outbound number configured")
                        return None
                
        except Exception as e:
            logger.error(f"Error getting default outbound number: {e}", exc_info=True)
            return None

    async def get_default_agent_for_number(self, phone_number: str) -> Optional[int]:
        """Resolve the default agent mapped to a specific phone number."""
        if not phone_number:
            return None

        try:
            # Split phone number into country code and base number
            country_code, base_number = self._split_phone_number(phone_number)
            
            if base_number is None:
                return None
            
            with get_db_connection(self.db_config) as conn:
                with conn.cursor() as cursor:
                    if country_code:
                        cursor.execute(
                            f"""
                            SELECT default_agent
                            FROM {NUMBERS_FULL}
                            WHERE country_code = %s AND base_number = %s
                            LIMIT 1
                            """,
                            (country_code, base_number),
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT default_agent
                            FROM {NUMBERS_FULL}
                            WHERE base_number = %s
                            LIMIT 1
                            """,
                            (base_number,),
                        )
                    result = cursor.fetchone()

                    if result and result[0] is not None:
                        resolved = int(result[0])
                        logger.info(
                            "Resolved default agent %s for phone number %s",
                            resolved,
                            phone_number,
                        )
                        return resolved

            logger.debug("No default agent mapping for phone number %s", phone_number)
            return None

        except Exception as e:
            logger.error(
                "Error resolving default agent for number %s: %s",
                phone_number,
                e,
                exc_info=True,
            )
            return None
