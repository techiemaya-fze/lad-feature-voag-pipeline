"""
Google OAuth and credential management.

Merged from:
- utils/google_oauth.py
- utils/google_credentials.py

Contains:
- GoogleOAuthSettings: OAuth configuration from environment
- OAuthStateManager: Signs/validates OAuth state tokens
- TokenEncryptor: Encrypts/decrypts credential payloads
- GoogleCredentialResolver: Loads encrypted tokens from DB
- GoogleCredentialError: Custom exception for credential errors
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Tuple
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken
from google.oauth2.credentials import Credentials
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# Lazy import to avoid circular dependency
# from db.storage.user_token_storage import UserTokenStorage

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
)
STATE_SALT = "voice-agent-google-oauth"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_scopes(raw_scopes: str | None) -> tuple[str, ...]:
    if not raw_scopes:
        return DEFAULT_SCOPES
    scopes = [segment.strip() for segment in raw_scopes.split(",") if segment.strip()]
    return tuple(scopes) if scopes else DEFAULT_SCOPES


def _load_client_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Google OAuth client file not found: {candidate}")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    if "web" in data:
        return data
    if "installed" in data:
        return data
    # Assume already shaped correctly
    return {"web": data}


def _coerce_db_blob(blob: Any) -> bytes | None:
    """Coerce various DB blob types to bytes."""
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        return blob.tobytes()
    if isinstance(blob, bytearray):
        return bytes(blob)
    if isinstance(blob, bytes):
        return blob
    return None


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass(frozen=True)
class GoogleOAuthSettings:
    """Google OAuth configuration loaded from environment variables."""
    
    client_config: Mapping[str, Any]
    redirect_uri: str
    scopes: tuple[str, ...]
    state_secret: str
    state_ttl_seconds: int
    success_fallback: str
    error_fallback: str
    frontend_redirect_map: Mapping[str, str]
    encryption_key: str

    @property
    def client_id(self) -> str:
        block = self._client_block
        value = block.get("client_id")
        if not value:
            raise RuntimeError("google oauth client_id missing from secrets file")
        return value

    @property
    def client_secret(self) -> str:
        block = self._client_block
        value = block.get("client_secret")
        if not value:
            raise RuntimeError("google oauth client_secret missing from secrets file")
        return value

    @property
    def token_uri(self) -> str:
        block = self._client_block
        return block.get("token_uri", "https://oauth2.googleapis.com/token")

    @property
    def auth_uri(self) -> str:
        block = self._client_block
        return block.get("auth_uri", "https://accounts.google.com/o/oauth2/auth")

    @property
    def _client_block(self) -> Mapping[str, Any]:
        return self.client_config.get("web") or self.client_config.get("installed") or {}


@lru_cache(maxsize=1)
def get_google_oauth_settings() -> GoogleOAuthSettings:
    """Load Google OAuth settings from environment variables."""
    client_file = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS")
    redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    state_secret = os.getenv("GOOGLE_OAUTH_STATE_SECRET")
    encryption_key = os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY")
    success_redirect = os.getenv("GOOGLE_OAUTH_SUCCESS_FALLBACK") or "/"
    error_redirect = os.getenv("GOOGLE_OAUTH_ERROR_FALLBACK") or success_redirect
    frontend_map_raw = os.getenv("FRONTEND_REDIRECT_MAP", "{}")
    scopes = _parse_scopes(os.getenv("GOOGLE_OAUTH_SCOPES"))
    ttl_seconds = int(os.getenv("GOOGLE_OAUTH_STATE_TTL_SECONDS", "600"))

    if not client_file or not redirect_uri or not state_secret or not encryption_key:
        missing = [
            name
            for name, value in (
                ("GOOGLE_OAUTH_CLIENT_SECRETS", client_file),
                ("GOOGLE_OAUTH_REDIRECT_URI", redirect_uri),
                ("GOOGLE_OAUTH_STATE_SECRET", state_secret),
                ("GOOGLE_TOKEN_ENCRYPTION_KEY", encryption_key),
            )
            if not value
        ]
        raise RuntimeError(f"Missing Google OAuth env vars: {', '.join(missing)}")

    try:
        frontend_map = json.loads(frontend_map_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FRONTEND_REDIRECT_MAP must be valid JSON") from exc

    return GoogleOAuthSettings(
        client_config=_load_client_config(client_file),
        redirect_uri=redirect_uri,
        scopes=scopes,
        state_secret=state_secret,
        state_ttl_seconds=ttl_seconds,
        success_fallback=success_redirect,
        error_fallback=error_redirect,
        frontend_redirect_map=frontend_map,
        encryption_key=encryption_key,
    )


# =============================================================================
# STATE MANAGER
# =============================================================================

class OAuthStateManager:
    """Signs and validates short-lived OAuth state payloads."""

    def __init__(self, settings: GoogleOAuthSettings) -> None:
        self._serializer = URLSafeTimedSerializer(settings.state_secret, salt=STATE_SALT)
        self._ttl_seconds = settings.state_ttl_seconds

    def issue(self, *, user_id: str, frontend_id: str) -> str:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._ttl_seconds)
        payload = {
            "user_id": user_id,
            "frontend_id": frontend_id,
            "nonce": uuid4().hex,
            "exp": expires_at.isoformat(),
        }
        return self._serializer.dumps(payload)

    def verify(self, token: str) -> dict[str, Any]:
        try:
            data = self._serializer.loads(token, max_age=self._ttl_seconds)
            return data
        except SignatureExpired as exc:
            raise ValueError("OAuth state expired") from exc
        except BadSignature as exc:
            raise ValueError("OAuth state invalid") from exc


# =============================================================================
# TOKEN ENCRYPTION
# =============================================================================

class TokenEncryptor:
    """Encrypts and decrypts serialized credential payloads."""

    def __init__(self, key: str) -> None:
        try:
            key_bytes = key.encode("utf-8")
            self._fernet = Fernet(key_bytes)
        except Exception as exc:
            raise RuntimeError("GOOGLE_TOKEN_ENCRYPTION_KEY must be a valid Fernet key") from exc

    def encrypt_json(self, payload: Mapping[str, Any]) -> bytes:
        serialized = json.dumps(payload).encode("utf-8")
        return self._fernet.encrypt(serialized)

    def decrypt_json(self, blob: bytes | None) -> dict[str, Any] | None:
        if not blob:
            return None
        try:
            decrypted = self._fernet.decrypt(blob)
        except InvalidToken as exc:
            raise ValueError("Stored Google token blob could not be decrypted") from exc
        return json.loads(decrypted.decode("utf-8"))


# =============================================================================
# CREDENTIAL CONVERSION
# =============================================================================

def credentials_to_dict(credentials: Credentials) -> dict[str, Any]:
    """Convert Google Credentials object to dictionary for storage."""
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or []),
        "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
    }


def credentials_from_dict(payload: Mapping[str, Any]) -> Credentials:
    """Reconstruct Google Credentials object from dictionary."""
    expiry_raw = payload.get("expiry")
    expiry = None
    if expiry_raw:
        expiry = datetime.fromisoformat(str(expiry_raw))
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
    return Credentials(
        token=payload.get("token"),
        refresh_token=payload.get("refresh_token"),
        token_uri=payload.get("token_uri") or "https://oauth2.googleapis.com/token",
        client_id=payload.get("client_id"),
        client_secret=payload.get("client_secret"),
        scopes=payload.get("scopes"),
        expiry=expiry,
    )


# =============================================================================
# CREDENTIAL RESOLVER (from google_credentials.py)
# =============================================================================

class GoogleCredentialError(RuntimeError):
    """Raised when Google OAuth credentials cannot be located or decrypted."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class GoogleCredentialResolver:
    """Loads encrypted Google tokens from user_identities table."""

    def __init__(self, storage: Any = None) -> None:
        # Lazy import to avoid circular dependency
        if storage is None:
            from db.storage.tokens import UserTokenStorage
            storage = UserTokenStorage()
        self._storage = storage
        self._encryptor: TokenEncryptor | None = None

    def _canonicalize_identifier(self, identifier: str | int | None) -> str:
        if identifier is None:
            return ""
        if isinstance(identifier, int):
            return str(identifier)
        return identifier.strip()

    def _ensure_encryptor(self) -> TokenEncryptor:
        if self._encryptor is None:
            try:
                settings = get_google_oauth_settings()
            except Exception as exc:
                raise GoogleCredentialError(
                    "Google OAuth configuration is invalid",
                    status_code=500,
                ) from exc
            self._encryptor = TokenEncryptor(settings.encryption_key)
        return self._encryptor

    async def resolve_user(self, identifier: str | int | None) -> Tuple[str, dict[str, Any]]:
        """Resolve user identifier to canonical ID and user record."""
        clean = self._canonicalize_identifier(identifier)
        if not clean:
            raise GoogleCredentialError("user_id is required", status_code=400)
        
        # lad_dev.users uses UUID 'id' as primary key
        record = await self._storage.get_user_by_user_id(clean)
        
        if not record:
            raise GoogleCredentialError(f"User {clean} not found", status_code=404)
        
        # lad_dev.users uses 'id' as primary key, not 'user_id'
        canonical = str(record.get("id") or "").strip()
        if not canonical:
            raise GoogleCredentialError(
                "User record missing canonical identifier",
                status_code=500,
            )
        return canonical, record

    async def load_credentials(self, identifier: str | int | None) -> Credentials:
        """Load and decrypt Google credentials for a user."""
        canonical_id, _ = await self.resolve_user(identifier)
        
        # Get token blob from user_identities table
        blob = await self._storage.get_google_token_blob(canonical_id)
        if not blob:
            raise GoogleCredentialError(
                "User has not authorized Google access",
                status_code=404,
            )
        encryptor = self._ensure_encryptor()
        try:
            payload = encryptor.decrypt_json(blob)
        except ValueError as exc:
            raise GoogleCredentialError(
                "Stored Google OAuth tokens are invalid",
                status_code=500,
            ) from exc
        if not payload:
            raise GoogleCredentialError(
                "Stored Google OAuth tokens are unavailable",
                status_code=404,
            )
        credentials = credentials_from_dict(payload)
        # Prevent timezone-aware/naive comparison issues inside google-auth
        credentials.expiry = None
        # Ensure the credentials retain canonical user reference for auditing
        credentials._canonical_user_id = canonical_id  # type: ignore[attr-defined]
        return credentials


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Settings
    "GoogleOAuthSettings",
    "get_google_oauth_settings",
    # State management
    "OAuthStateManager",
    # Token encryption
    "TokenEncryptor",
    # Credential conversion
    "credentials_to_dict",
    "credentials_from_dict",
    # Credential resolver
    "GoogleCredentialResolver",
    "GoogleCredentialError",
]
