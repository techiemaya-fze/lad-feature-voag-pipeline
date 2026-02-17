"""
Call Stage Detection Module

This module contains all the logic for detecting call stages from conversation transcripts.
It handles the extraction of stage information including:
- Stage 1: Contacted (user response detected)
- Stage 2: Follow-up scheduled (follow-up call arranged)
- Stage 3: Email sent (email confirmation)
- Stage 4: Counseling booked (booking confirmed)

This module is designed to be imported and used by the main analytics module.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class CallStageDetector:
    """
    Handles detection of call stages from conversation transcripts.
    """
    
    def __init__(self):
        """Initialize the stage detector with default patterns."""
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize all regex patterns for stage detection."""
        
        # Stage 1: User response patterns (any user message means contacted)
        self.user_patterns = [
            r'User:.*',  # Any user response (full conversation format)
            r'Customer:.*',  # Alternative user marker (full conversation format)
            r'Prospect:.*',  # Alternative user marker (full conversation format)
            r'.*',  # Any non-empty user response (user-only format)
        ]
        
        # Complete rejection patterns - only block ALL stages if user is truly not interested
        self.cancellation_patterns = [
            # User completely rejects service (more specific patterns)
            r'User:.*(not interested.*at all|not interested.*anything)',
            r'User:.*(don\'t call.*me.*ever|never call.*me.*again)',
            r'User:.*(remove.*me.*from.*list|take.*me.*off.*list)',
            r'User:.*(go.*away|leave.*me.*alone|stop.*calling.*me)',
            r'User:.*(don\'t.*contact.*me|no.*contact.*ever)',
            # Customer/Prospect alternatives
            r'Customer:.*(not interested.*at all|not interested.*anything)',
            r'Customer:.*(don\'t call.*me.*ever|never call.*me.*again)',
            r'Customer:.*(remove.*me.*from.*list|take.*me.*off.*list)',
            r'Customer:.*(go.*away|leave.*me.*alone|stop.*calling.*me)',
            r'Customer:.*(don\'t.*contact.*me|no.*contact.*ever)',
            r'Prospect:.*(not interested.*at all|not interested.*anything)',
            r'Prospect:.*(don\'t call.*me.*ever|never call.*me.*again)',
            r'Prospect:.*(remove.*me.*from.*list|take.*me.*off.*list)',
            r'Prospect:.*(go.*away|leave.*me.*alone|stop.*calling.*me)',
            r'Prospect:.*(don\'t.*contact.*me|no.*contact.*ever)',
        ]
        
        # Booking cancellation patterns - only block booking/appointment stages
        self.booking_cancellation_patterns = [
            # User cancels/rejects appointments but may accept callbacks
            r'User:.*(currently not|not now|busy now|busy right now|can\'t talk|cannot talk|no time)',
            r'User:.*(call.*after|call me after|call later|callback later|some other time)',
            r'User:.*(cancel|cancelled|reject|rejected|decline|declined)',
            r'User:.*(doesn\'t work|not work|can\'t make it|unable to make)',
            r'User:.*(schedule.*doesn\'t work|time doesn\'t work|not good time)',
            # Customer/Prospect alternatives
            r'Customer:.*(currently not|not now|busy now|busy right now|can\'t talk|cannot talk|no time)',
            r'Customer:.*(call.*after|call me after|call later|callback later|some other time)',
            r'Customer:.*(cancel|cancelled|reject|rejected|decline|declined)',
            r'Customer:.*(doesn\'t work|not work|can\'t make it|unable to make)',
            r'Customer:.*(schedule.*doesn\'t work|time doesn\'t work|not good time)',
            r'Prospect:.*(currently not|not now|busy now|busy right now|can\'t talk|cannot talk|no time)',
            r'Prospect:.*(call.*after|call me after|call later|callback later|some other time)',
            r'Prospect:.*(cancel|cancelled|reject|rejected|decline|declined)',
            r'Prospect:.*(doesn\'t work|not work|can\'t make it|unable to make)',
            r'Prospect:.*(schedule.*doesn\'t work|time doesn\'t work|not good time)',
        ]
        
        # Follow-up cancellation patterns - only block followup stage
        self.followup_cancellation_patterns = [
            # User rejects follow-up calls specifically
            r'User:.*(no call.*back|don\'t call.*back|not call.*back)',
            r'User:.*(no follow.*up|don\'t follow.*up|not follow.*up)',
            r'User:.*(no callback|don\'t callback|not callback)',
            r'User:.*(call me never|never call|no more calls)',
            r'User:.*(stop calling|don\'t call again|no more calls)',
            # Customer/Prospect alternatives
            r'Customer:.*(no call.*back|don\'t call.*back|not call.*back)',
            r'Customer:.*(no follow.*up|don\'t follow.*up|not follow.*up)',
            r'Customer:.*(no callback|don\'t callback|not callback)',
            r'Customer:.*(call me never|never call|no more calls)',
            r'Customer:.*(stop calling|don\'t call again|no more calls)',
            r'Prospect:.*(no call.*back|don\'t call.*back|not call.*back)',
            r'Prospect:.*(no follow.*up|don\'t follow.*up|not follow.*up)',
            r'Prospect:.*(no callback|don\'t callback|not callback)',
            r'Prospect:.*(call me never|never call|no more calls)',
            r'Prospect:.*(stop calling|don\'t call again|no more calls)',
        ]
        
        # Email cancellation patterns - only block email stage
        self.email_cancellation_patterns = [
            # User rejects email communication specifically
            r'User:.*(no email|don\'t email|not email|no mail|don\'t mail)',
            r'User:.*(not interested.*email|don\'t send.*email|no email.*please)',
            r'User:.*(email.*not.*work|mail.*not.*work|email.*wrong)',
            r'User:.*(no email address|don\'t have email|no email available)',
            r'User:.*(prefer call|only call|call only|no email needed)',
            # Customer/Prospect alternatives
            r'Customer:.*(no email|don\'t email|not email|no mail|don\'t mail)',
            r'Customer:.*(not interested.*email|don\'t send.*email|no email.*please)',
            r'Customer:.*(email.*not.*work|mail.*not.*work|email.*wrong)',
            r'Customer:.*(no email address|don\'t have email|no email available)',
            r'Customer:.*(prefer call|only call|call only|no email needed)',
            r'Prospect:.*(no email|don\'t email|not email|no mail|don\'t mail)',
            r'Prospect:.*(not interested.*email|don\'t send.*email|no email.*please)',
            r'Prospect:.*(email.*not.*work|mail.*not.*work|email.*wrong)',
            r'Prospect:.*(no email address|don\'t have email|no email available)',
            r'Prospect:.*(prefer call|only call|call only|no email needed)',
        ]
        
        # Stage 2: Follow-up discussion patterns
        self.followup_discussion_patterns = [
            # Agent patterns (universal - any day/time)
            r'Agent:.*call.*back|callback|follow.*up.*call',
            r'Agent:.*schedule.*call|call.*schedule',
            r'Agent:.*call.*you.*back|back.*call',
            r'Agent:.*next.*call|call.*again',
            r'Agent:.*follow.*up.*later|later.*follow',
            r'Agent:.*call.*later',
            r'Agent:.*call.*me.*later',
            r'Agent:.*later.*this.*afternoon',
            r'Agent:.*would.*later.*work',
            r'Agent:.*call.*tomorrow',
            r'Agent:.*call.*today',
            r'Agent:.*call.*evening',
            r'Agent:.*call.*afternoon',
            r'Agent:.*call.*morning',
            # User patterns (asking for follow-up)
            r'User:.*call.*later',
            r'User:.*call.*me.*later',
            r'User:.*can.*you.*call.*later',
            r'User:.*busy.*now.*call',
            r'User:.*call.*again',
            r'User:.*follow.*up.*later',
            r'User:.*schedule.*call|call.*schedule',
            r'User:.*call.*you.*back|back.*call',
            r'User:.*next.*call|call.*again',
            r'User:.*follow.*up.*later|later.*follow'
        ]
        
        # Stage 2: User accepts follow-up patterns
        self.user_accepts_followup_patterns = [
            r'User:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed)',
            r'Customer:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed)',
            r'Prospect:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed)',
            # Universal follow-up acceptance patterns (any day/time)
            r'User:.*call.*next',
            r'User:.*can.*you.*call',
            r'User:.*busy.*now.*call',
            r'User:.*call.*later',
            r'User:.*call.*me.*later',
            r'User:.*call.*again',
            r'User:.*yeah.*fine',
            r'User:.*that\'s.*fine',
            r'User:.*follow.*up',
            # Specific patterns from the test conversation
            r'User:.*can.*you.*call.*later',
            r'User:.*call.*me.*later.*busy.*now',
            r'User:.*busy.*now.*call.*later',
            # User-only patterns (for user-only conversation format)
            r'(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed)',
            r'call.*next',
            r'can.*you.*call',
            r'busy.*now.*call',
            r'call.*later',
            r'call.*me.*later',
            r'call.*again',
            r'yeah.*fine',
            r'that\'s.*fine',
            r'follow.*up',
            r'can.*you.*call.*later',
            r'call.*me.*later.*busy.*now',
            r'busy.*now.*call.*later'
        ]
        
        # Stage 2: Agent schedules follow-up patterns
        self.agent_schedules_followup_patterns = [
            # Agent patterns (universal - any day/time)
            r'Agent:.*(call|will call|calling).*at',
            r'Agent:.*(schedule|scheduled|booking).*call',
            r'Agent:.*(i\'ll|i will).*call.*you',
            r'Agent:.*call.*back.*at',
            r'Agent:.*follow.*up.*call',
            r'Agent:.*call.*next',
            r'Agent:.*call.*you.*next',
            r'Agent:.*make.*note.*call',
            r'Agent:.*call.*later',
            r'Agent:.*call.*again',
            # More flexible patterns
            r'Agent:.*make.*note',
            r'Agent:.*call.*tomorrow',
            r'Agent:.*call.*today',
            r'Agent:.*call.*evening',
            r'Agent:.*call.*afternoon',
            r'Agent:.*call.*morning',
            r'Agent:.*next.*week',
            r'Agent:.*next.*day',
            # Specific patterns from the conversation
            r'Agent:.*would.*later.*this.*afternoon.*work',
            r'Agent:.*later.*this.*afternoon',
            r'Agent:.*would.*later.*work',
            r'Agent:.*say.*around.*pm',
            r'Agent:.*around.*pm.*work',
            # More specific patterns for incomplete messages
            r'Agent:.*would.*later.*this.*afternoon',
            r'Agent:.*later.*afternoon.*work',
            r'Agent:.*around.*pm.*gst',
            r'Agent:.*pm.*gst.*work'
        ]
        
        # Stage 3: Email discussion patterns
        self.email_discussion_patterns = [
            r'email.*send|send.*email',
            r'mail.*send|send.*mail',
            r'brochure.*email|email.*brochure',
            r'send.*details|details.*email',
            r'share.*email|email.*share',
            r'email.*address|address.*email',
            r'email.*list|list.*email'
        ]
        
        # Stage 3: User provides email patterns
        self.user_provides_email_patterns = [
            r'User:.*[\w\.-]+@[\w\.-]+\.\w+',  # Standard email format
            r'Customer:.*[\w\.-]+@[\w\.-]+\.\w+',  # Standard email format
            r'Prospect:.*[\w\.-]+@[\w\.-]+\.\w+',  # Standard email format
            r'User:.*my.*email.*is',
            r'Customer:.*my.*email.*is',
            r'Prospect:.*my.*email.*is',
            r'User:.*email.*me.*at',
            r'Customer:.*email.*me.*at',
            r'Prospect:.*email.*me.*at',
            # Spelled-out email patterns
            r'User:.*at.*gmail.*dot.*com',
            r'Customer:.*at.*gmail.*dot.*com',
            r'Prospect:.*at.*gmail.*dot.*com',
            r'User:.*at.*yahoo.*dot.*com',
            r'Customer:.*at.*yahoo.*dot.*com',
            r'Prospect:.*at.*yahoo.*dot.*com',
            r'User:.*at.*outlook.*dot.*com',
            r'Customer:.*at.*outlook.*dot.*com',
            r'Prospect:.*at.*outlook.*dot.*com',
            r'User:.*dot.*com',
            r'Customer:.*dot.*com',
            r'Prospect:.*dot.*com',
            # Email confirmation patterns
            r'User:.*email.*address.*is',
            r'Customer:.*email.*address.*is',
            r'Prospect:.*email.*address.*is',
            r'User:.*correct.*email',
            r'Customer:.*correct.*email',
            r'Prospect:.*correct.*email',
            # User-only patterns (for user-only conversation format)
            r'[\w\.-]+@[\w\.-]+\.\w+',  # Standard email format
            r'my.*email.*is',
            r'email.*me.*at',
            r'at.*gmail.*dot.*com',
            r'at.*yahoo.*dot.*com',
            r'at.*outlook.*dot.*com',
            r'dot.*com',
            r'email.*address.*is',
            r'correct.*email'
        ]
        
        # Stage 3: Agent confirms sending patterns
        self.agent_confirms_sending_patterns = [
            r'Agent:.*(send|sent|sending|will send).*email',
            r'Agent:.*(send|sent|sending|will send).*mail',
            r'Agent:.*(i\'ll|i will).*send.*email',
            r'Agent:.*(i\'ve|i have).*sent.*email',
            r'Agent:.*confirmation.*email',
            r'Agent:.*get.*confirmation.*email',
            r'Agent:.*email.*shortly',
            r'Agent:.*email.*details',
            r'Agent:.*booked.*email',
            r'Agent:.*confirmation.*shortly',
            r'Agent:.*confirmation.*email.*shortly',
            r'Agent:.*get.*confirmation.*email.*shortly',
            # Additional confirmation patterns
            r'Agent:.*you.*get.*email',
            r'Agent:.*send.*confirmation',
            r'Agent:.*email.*confirmation',
            r'Agent:.*confirmation.*details',
            r'Agent:.*session.*details.*email',
            r'Agent:.*counselor.*contact.*email',
            r'Agent:.*use.*that.*for.*confirmation',
            r'Agent:.*confirmation.*goes.*right.*place'
        ]
        
        # Stage 4: Booking discussion patterns
        self.booking_patterns = [
            r'book.*counseling|counseling.*book',
            r'schedule.*counseling|counseling.*schedule',
            r'appointment.*book|book.*appointment',
            r'book.*session|session.*book',
            r'counseling.*appointment|appointment.*counseling',
            r'book.*call|call.*book',
            r'schedule.*meeting|meeting.*schedule',
            r'book.*meeting|meeting.*book',
            r'book.*slot|slot.*book',
            r'set.*meeting|meeting.*set',
            r'arrange.*meeting|meeting.*arrange',
            r'fix.*meeting|meeting.*fix',
            r'organize.*meeting|meeting.*organize',
            r'plan.*meeting|meeting.*plan',
            r'confirm.*meeting|meeting.*confirm'
        ]
        
        # Stage 4: Booking confirmation patterns
        self.booking_confirmation_patterns = [
            r'User:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed|booked)',
            r'Customer:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed|booked)',
            r'Prospect:.*(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed|booked)',
            # User-only patterns (for user-only conversation format)
            r'(yes|okay|ok|sure|perfect|great|sounds good|that works|works for me|confirmed|booked)'
        ]
        
        # Stage mapping for leads table
        self.stage_mapping = {
            '1_contacted': 'contacted',
            '2_followup': 'followup',
            '3_email_sent': 'email_sent',
            '4_counseling_booked': 'meeting_booked'  # Default for non-Glinks tenants
        }
        
        # Tenant-specific stage mappings
        self.tenant_stage_mapping = {
            # Glinks tenant uses counseling_booked for stage 4
            '926070b5-189b-4682-9279-ea10ca090b84': {
                '1_contacted': 'contacted',
                '2_followup': 'followup',
                '3_email_sent': 'email_sent',
                '4_counseling_booked': 'counseling_booked'
            }
        }
    
    def has_meaningful_user_transcription(self, conversation_text: str) -> bool:
        """
        Check if there's meaningful user transcription (at least 15 letters).
        
        Args:
            conversation_text: The formatted conversation transcript
            
        Returns:
            bool: True if meaningful user transcription exists, False otherwise
        """
        lines = conversation_text.split('\n')
        total_user_text = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                user_message = line.replace("User:", "").strip()
                total_user_text += user_message + " "
        
        # Remove spaces and check total length
        clean_user_text = total_user_text.replace(" ", "")
        
        # Consider meaningful if at least 1 character (any user response counts as contacted)
        is_meaningful = len(clean_user_text) >= 1
        
        logger.info(f"User transcription check: {len(clean_user_text)} characters -> {'Meaningful' if is_meaningful else 'Not meaningful'}")
        
        return is_meaningful
    
    def extract_call_stages(self, conversation_text: str, summary: Dict = None, tenant_id: str = None) -> Dict:
        """
        Extract call stage information from conversation text.
        
        Args:
            conversation_text: The formatted conversation transcript
            summary: Optional summary data (not used in current implementation)
            tenant_id: Optional tenant ID for tenant-specific stage mapping
            
        Returns:
            Dict containing stage information:
            - stages_reached: List of stages detected in order
            - final_stage: The highest stage reached
            - stage_completion_percentage: Percentage of stages completed
            - total_stages_reached: Number of stages reached
            - stage_mapping: Tenant-specific stage mapping
        """
        logger.debug("Extracting call stages...")
        
        # Check if there's meaningful user transcription (at least 15 letters)
        if not self.has_meaningful_user_transcription(conversation_text):
            logger.info("No meaningful user transcription detected - returning minimal stage info")
            return self.get_default_stage_info()
        
        stages_reached = []
        
        # Convert to lowercase for pattern matching
        text_lower = conversation_text.lower()
        
        # Check for complete rejection patterns (not interested, don't call, etc.)
        complete_rejection = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.cancellation_patterns)
        if complete_rejection:
            logger.info("Complete rejection detected - not awarding any stages")
            return {
                "stages_reached": [],
                "final_stage": None,
                "stage_completion_percentage": "0%",
                "total_stages_reached": 0,
                "stage_mapping": self._get_tenant_stage_mapping(tenant_id)
            }
        
        # Check for booking cancellation specifically
        booking_cancelled = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.booking_cancellation_patterns)
        
        # Check for follow-up cancellation specifically
        followup_cancelled = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.followup_cancellation_patterns)
        
        # Check for email cancellation specifically
        email_cancelled = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.email_cancellation_patterns)
        
        # Stage 1: Contacted
        # Check if there's any user transcription (response from user)
        has_user_transcription = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.user_patterns)
        if has_user_transcription:
            stages_reached.append("1_contacted")
        
        # Stage 2: Follow-up Call Scheduled
        # Check if follow-up call is scheduled and user accepts - but only if not cancelled
        followup_discussed = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.followup_discussion_patterns)
        user_accepts_followup = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.user_accepts_followup_patterns)
        agent_schedules_followup = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.agent_schedules_followup_patterns)
        
        # Stage 2 reached if follow-up call scheduling occurs and not cancelled
        if followup_discussed and user_accepts_followup and agent_schedules_followup and not followup_cancelled:
            stages_reached.append("2_followup")
        elif followup_discussed and followup_cancelled:
            logger.info(f"Follow-up discussed but cancelled - not awarding Stage 2")
        
        # Stage 3: Email Sent
        # Check if agent discusses sending email, user provides email, and agent confirms sending - but only if not cancelled
        email_discussed = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.email_discussion_patterns)
        user_provides_email = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.user_provides_email_patterns)
        agent_confirms_sending = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.agent_confirms_sending_patterns)
        
        if email_discussed and user_provides_email and agent_confirms_sending and not email_cancelled:
            stages_reached.append("3_email_sent")
        elif email_discussed and email_cancelled:
            logger.info(f"Email discussed but cancelled - not awarding Stage 3")
        
        # Stage 4: Counseling Booked
        # Check if booking is discussed and confirmed - but only if not cancelled
        booking_discussed = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.booking_patterns)
        booking_confirmed = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in self.booking_confirmation_patterns)
        
        if booking_discussed and booking_confirmed and not booking_cancelled:
            stages_reached.append("4_counseling_booked")
            logger.info(f"Stage 4 DETECTED: Booking confirmed - will apply tenant-specific mapping")
        elif booking_discussed and booking_cancelled:
            logger.info(f"Booking discussed but cancelled - not awarding Stage 4")
        
        # Determine final stage - use the highest stage reached (no fallback)
        if stages_reached:
            final_stage = stages_reached[-1]
        else:
            # No stages detected - return empty
            final_stage = None
        
        # Stage completion percentage (now out of 4 stages)
        stage_completion = round((len(stages_reached) / 4) * 100, 1) if stages_reached else 0
        
        # Get tenant-specific stage mapping
        stage_mapping = self._get_tenant_stage_mapping(tenant_id)
        
        return {
            "stages_reached": stages_reached,
            "final_stage": final_stage,
            "stage_completion_percentage": f"{stage_completion}%",
            "total_stages_reached": len(stages_reached),
            "stage_mapping": stage_mapping
        }
    
    def _get_tenant_stage_mapping(self, tenant_id: str) -> Dict:
        """
        Get tenant-specific stage mapping.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Dict containing tenant-specific stage mapping
        """
        logger.info(f"Stage 4 Tenant Check - Comparing tenant_id: {tenant_id}")
        logger.info(f"Stage 4 Tenant Check - Available tenants: {list(self.tenant_stage_mapping.keys())}")
        
        if tenant_id in self.tenant_stage_mapping:
            logger.info(f"Stage 4 Tenant Check - MATCH FOUND: Using Glinks mapping (counseling_booked)")
            return self.tenant_stage_mapping[tenant_id]
        else:
            logger.info(f"Stage 4 Tenant Check - NO MATCH: Using default mapping (meeting_booked)")
            return self.stage_mapping
    
    def get_highest_stage_for_leads(self, conversation_text: str, tenant_id: str = None) -> str:
        """
        Get the highest stage reached for leads table update.
        
        Args:
            conversation_text: The formatted conversation transcript
            tenant_id: Optional tenant ID for tenant-specific stage mapping
            
        Returns:
            String representing the highest stage reached (readable format)
        """
        try:
            stage_info = self.extract_call_stages(conversation_text, {}, tenant_id)
            stages_reached = stage_info.get('stages_reached', [])
            
            # Get the highest stage actually reached (last in the list)
            if stages_reached:
                highest_stage = stages_reached[-1]
            else:
                # No stages detected - use empty string
                highest_stage = ''
                
        except Exception as stage_error:
            logger.warning(f"Error extracting stages: {stage_error}")
            highest_stage = ''
        
        # Get tenant-specific stage mapping
        stage_mapping = self._get_tenant_stage_mapping(tenant_id)
        
        # Map stage codes to readable names
        readable_stage = stage_mapping.get(highest_stage, highest_stage)
        
        # Ensure readable_stage is a string
        if not isinstance(readable_stage, str):
            readable_stage = str(readable_stage)
        
        return readable_stage
    
    def get_default_stage_info(self) -> Dict:
        """
        Get default stage info for calls with no user transcription.
        
        Returns:
            Dict containing default stage information
        """
        return {
            "stages_reached": [],
            "final_stage": None,
            "stage_completion_percentage": "0.0%",
            "total_stages_reached": 0,
        }


# Global instance for easy import
stage_detector = CallStageDetector()
