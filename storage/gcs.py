"""
Google Cloud Storage utilities for call recordings

Handles:
1. Uploading recordings to GCS
2. Generating signed URLs for secure access
3. Managing recording metadata

Copied from: gcs_storage.py
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class GCSStorageManager:
    """Manages Google Cloud Storage operations for call recordings"""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize GCS Storage Manager
        
        Args:
            bucket_name: GCS bucket name (defaults to env var GCS_BUCKET)
            credentials_path: Path to service account JSON
            project_id: GCP project ID
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET")
        self.credentials_path = credentials_path or os.getenv("GCS_CREDENTIALS_JSON")
        self.project_id = project_id or os.getenv("GCS_PROJECT_ID")
        
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET not configured")
        
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise ValueError(f"GCS credentials not found at: {self.credentials_path}")
        
        self.credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path
        )
        self.client = storage.Client(
            credentials=self.credentials,
            project=self.project_id
        )
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(
            "GCS Storage Manager initialized: bucket=%s, project=%s",
            self.bucket_name,
            self.project_id
        )

    def get_blob_path(self, call_id: str, room_name: Optional[str] = None) -> str:
        """Generate blob path for a recording"""
        if room_name:
            return f"recordings/{room_name}/{call_id}.ogg"
        return f"recordings/{call_id}.ogg"

    def get_gs_url(self, blob_path: str) -> str:
        """Get gs:// URL for a blob"""
        return f"gs://{self.bucket_name}/{blob_path}"

    def upload_file(
        self,
        local_path: Path,
        call_id: str,
        room_name: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[str]:
        """Upload a recording file to GCS"""
        try:
            if not local_path.exists():
                logger.error("Local file not found: %s", local_path)
                return None
            
            blob_path = self.get_blob_path(call_id, room_name)
            blob = self.bucket.blob(blob_path)
            
            if metadata:
                blob.metadata = metadata
            
            blob.upload_from_filename(str(local_path))
            
            gs_url = self.get_gs_url(blob_path)
            logger.info(
                "Uploaded recording to GCS: %s (size: %d bytes)",
                gs_url,
                local_path.stat().st_size
            )
            
            return gs_url
            
        except Exception as exc:
            logger.error("Failed to upload to GCS: %s", exc, exc_info=True)
            return None

    def generate_signed_url(
        self,
        blob_path: str,
        expiration_hours: int = 24
    ) -> Optional[str]:
        """Generate a signed URL for secure access to a recording"""
        try:
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning("Blob not found: %s", blob_path)
                return None
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiration_hours),
                method="GET",
                credentials=self.credentials
            )
            
            logger.info(
                "Generated signed URL for %s (expires in %d hours)",
                blob_path,
                expiration_hours
            )
            
            return url
            
        except Exception as exc:
            logger.error("Failed to generate signed URL: %s", exc, exc_info=True)
            return None

    def generate_signed_url_from_gs(
        self,
        gs_url: str,
        expiration_hours: int = 24
    ) -> Optional[str]:
        """Generate a signed URL from a gs:// URL"""
        try:
            # Strip any whitespace/newlines that might be in the URL
            gs_url = gs_url.strip()
            
            if not gs_url.startswith("gs://"):
                logger.error("Invalid GCS URL: %s", gs_url)
                return None
            
            parts = gs_url[5:].split("/", 1)
            if len(parts) != 2:
                logger.error("Invalid GCS URL format: %s", gs_url)
                return None
            
            bucket_name, blob_path = parts
            
            if bucket_name != self.bucket_name:
                logger.warning(
                    "Bucket mismatch: expected %s, got %s",
                    self.bucket_name,
                    bucket_name
                )
            
            return self.generate_signed_url(blob_path, expiration_hours)
            
        except Exception as exc:
            logger.error("Failed to generate signed URL from gs URL: %s", exc, exc_info=True)
            return None

    def delete_recording(self, blob_path: str) -> bool:
        """Delete a recording from GCS"""
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            logger.info("Deleted recording from GCS: %s", blob_path)
            return True
            
        except Exception as exc:
            logger.error("Failed to delete from GCS: %s", exc, exc_info=True)
            return False

    def get_blob_metadata(self, blob_path: str) -> Optional[dict]:
        """Get metadata for a blob"""
        try:
            blob = self.bucket.blob(blob_path)
            blob.reload()
            
            return {
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created,
                "updated": blob.updated,
                "metadata": blob.metadata or {},
                "gs_url": self.get_gs_url(blob_path)
            }
            
        except Exception as exc:
            logger.error("Failed to get blob metadata: %s", exc, exc_info=True)
            return None

    def list_recordings(
        self,
        prefix: str = "recordings/",
        max_results: int = 100
    ) -> list[dict]:
        """List recordings in GCS"""
        try:
            blobs = self.client.list_blobs(
                self.bucket_name,
                prefix=prefix,
                max_results=max_results
            )
            
            recordings = []
            for blob in blobs:
                recordings.append({
                    "name": blob.name,
                    "size": blob.size,
                    "created": blob.time_created,
                    "gs_url": self.get_gs_url(blob.name)
                })
            
            logger.info("Listed %d recordings with prefix: %s", len(recordings), prefix)
            return recordings
            
        except Exception as exc:
            logger.error("Failed to list recordings: %s", exc, exc_info=True)
            return []


__all__ = ["GCSStorageManager"]
