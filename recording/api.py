"""
API endpoints for accessing call recordings

Provides secure access to recordings via signed URLs

V2 Integration Audit: Uses lad_dev schema column names:
- recording_url (not call_recording_url)
- direction (not call_type)
- transcripts (not transcriptions)
- duration_seconds (not call_duration)
"""

import logging
from typing import Optional

from db.storage import CallStorage
from storage.gcs import GCSStorageManager

logger = logging.getLogger(__name__)


class RecordingAPI:
    """Handles API operations for call recordings"""

    def __init__(self):
        self.storage = CallStorage()
        self.gcs = GCSStorageManager()

    async def get_recording_url(
        self,
        call_log_id: str,
        expiration_hours: int = 24
    ) -> Optional[dict]:
        """
        Get a signed URL for accessing a call recording
        
        Args:
            call_log_id: UUID of the call log entry
            expiration_hours: URL expiration time in hours (default: 24)
        
        Returns:
            Dict with signed_url and metadata if successful, None otherwise
        """
        try:
            # Get call details from database
            call = await self.storage.get_call_by_id(call_log_id)
            
            if not call:
                logger.warning("Call not found: call_log_id=%s", call_log_id)
                return None
            
            # Use new column name: recording_url
            gs_url = call.get("recording_url")
            
            if not gs_url:
                logger.warning("No recording URL for call: call_log_id=%s", call_log_id)
                return None
            
            # Generate signed URL
            signed_url = self.gcs.generate_signed_url_from_gs(
                gs_url,
                expiration_hours=expiration_hours
            )
            
            if not signed_url:
                logger.error("Failed to generate signed URL for: %s", gs_url)
                return None
            
            return {
                "call_log_id": call_log_id,
                "signed_url": signed_url,
                "gs_url": gs_url,
                "expires_in_hours": expiration_hours,
                "status": call.get("status"),
                "started_at": call.get("started_at"),
                "ended_at": call.get("ended_at"),
                "direction": call.get("direction")  # New column name
            }
            
        except Exception as exc:
            logger.error(
                "Failed to get recording URL for call_log_id=%s: %s",
                call_log_id,
                exc,
                exc_info=True
            )
            return None

    async def get_transcription(self, call_log_id: str) -> Optional[dict]:
        """
        Get transcription for a call
        
        Args:
            call_log_id: UUID of the call log entry
        
        Returns:
            Dict with transcription data if successful, None otherwise
        """
        try:
            transcription_json = await self.storage.get_call_transcription(call_log_id)
            
            if not transcription_json:
                logger.warning("No transcription for call: call_log_id=%s", call_log_id)
                return None
            
            return {
                "call_log_id": call_log_id,
                "transcription": transcription_json
            }
            
        except Exception as exc:
            logger.error(
                "Failed to get transcription for call_log_id=%s: %s",
                call_log_id,
                exc,
                exc_info=True
            )
            return None

    async def get_call_details(self, call_log_id: str) -> Optional[dict]:
        """
        Get complete call details including recording URL and transcription
        
        Args:
            call_log_id: UUID of the call log entry
        
        Returns:
            Dict with all call details if successful, None otherwise
        """
        try:
            call = await self.storage.get_call_by_id(call_log_id)
            
            if not call:
                logger.warning("Call not found: call_log_id=%s", call_log_id)
                return None
            
            # Generate signed URL if recording exists (using new column name)
            signed_url = None
            recording_url = call.get("recording_url")  # New column name
            if recording_url:
                signed_url = self.gcs.generate_signed_url_from_gs(
                    recording_url,
                    expiration_hours=24
                )
            
            return {
                "call_log_id": call["id"],
                "tenant_id": call.get("tenant_id"),
                "status": call.get("status"),
                "direction": call.get("direction"),  # New: was call_type
                "started_at": call.get("started_at"),
                "ended_at": call.get("ended_at"),
                "duration_seconds": call.get("duration_seconds"),  # New: was call_duration
                "cost": call.get("cost"),
                "recording": {
                    "gs_url": recording_url,
                    "signed_url": signed_url,
                    "expires_in_hours": 24 if signed_url else None
                } if recording_url else None,
                "transcripts": call.get("transcripts"),  # New: was transcriptions
                "lead_id": call.get("lead_id"),
                "agent_id": call.get("agent_id"),
                "voice_id": call.get("voice_id"),
                "initiated_by_user_id": call.get("initiated_by_user_id")
            }
            
        except Exception as exc:
            logger.error(
                "Failed to get call details for call_log_id=%s: %s",
                call_log_id,
                exc,
                exc_info=True
            )
            return None
