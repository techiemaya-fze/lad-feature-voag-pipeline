"""
Microsoft OAuth and credential management.

Copied from: utils/microsoft_oauth.py

Contains:
- MicrosoftOAuthSettings: OAuth configuration from environment
- MicrosoftAuthService: Handles OAuth flow using MSAL
- token_response_to_storage_format: Converts MSAL response for storage
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import msal

logger = logging.getLogger(__name__)


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass(frozen=True)
class MicrosoftOAuthSettings:
    """Configuration for Microsoft OAuth."""
    
    client_id: str
    client_secret: str
    tenant_id: str
    redirect_uri: str
    scopes: list[str]

    @property
    def authority(self) -> str:
        """Microsoft login authority URL."""
        return f"https://login.microsoftonline.com/{self.tenant_id}"


@lru_cache(maxsize=1)
def get_microsoft_oauth_settings() -> MicrosoftOAuthSettings:
    """Load Microsoft OAuth settings from environment variables."""
    client_id = os.getenv("MICROSOFT_CLIENT_ID")
    client_secret = os.getenv("MICROSOFT_CLIENT_SECRET")
    tenant_id = os.getenv("MICROSOFT_TENANT_ID", "common")
    redirect_uri = os.getenv("MICROSOFT_REDIRECT_URI")
    scopes_raw = os.getenv("MICROSOFT_SCOPES", "")

    if not client_id or not client_secret or not redirect_uri:
        missing = [
            name
            for name, value in (
                ("MICROSOFT_CLIENT_ID", client_id),
                ("MICROSOFT_CLIENT_SECRET", client_secret),
                ("MICROSOFT_REDIRECT_URI", redirect_uri),
            )
            if not value
        ]
        raise RuntimeError(f"Missing Microsoft OAuth env vars: {', '.join(missing)}")

    # Parse space-separated scopes
    scopes = [s.strip() for s in scopes_raw.split() if s.strip()]
    if not scopes:
        scopes = ["openid", "profile", "email", "offline_access", "User.Read"]

    return MicrosoftOAuthSettings(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
    )


# =============================================================================
# AUTH SERVICE
# =============================================================================

class MicrosoftAuthService:
    """Handles Microsoft OAuth flow using MSAL."""

    def __init__(self, settings: MicrosoftOAuthSettings | None = None) -> None:
        self._settings = settings or get_microsoft_oauth_settings()
        self._app = msal.ConfidentialClientApplication(
            self._settings.client_id,
            authority=self._settings.authority,
            client_credential=self._settings.client_secret,
        )

    @property
    def settings(self) -> MicrosoftOAuthSettings:
        return self._settings

    def _filter_reserved_scopes(self, scopes: list[str]) -> list[str]:
        """
        Filter out reserved scopes that MSAL adds automatically.
        
        MSAL throws an error if you include 'openid', 'profile', or 'offline_access'
        in the scopes list - it adds these automatically.
        """
        reserved = {"openid", "profile", "offline_access", "email"}
        return [s for s in scopes if s.lower() not in reserved]

    def get_auth_url(self, state: str) -> str:
        """
        Generate Microsoft authorization URL.

        Args:
            state: CSRF state token (from OAuthStateManager)

        Returns:
            Authorization URL to redirect user to
        """
        filtered_scopes = self._filter_reserved_scopes(self._settings.scopes)
        
        auth_url = self._app.get_authorization_request_url(
            filtered_scopes,
            state=state,
            redirect_uri=self._settings.redirect_uri,
            prompt="select_account",
        )
        return auth_url

    def exchange_code_for_token(self, code: str) -> dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback

        Returns:
            Token response containing access_token, refresh_token, etc.

        Raises:
            ValueError: If token exchange fails
        """
        filtered_scopes = self._filter_reserved_scopes(self._settings.scopes)
        
        result = self._app.acquire_token_by_authorization_code(
            code,
            scopes=filtered_scopes,
            redirect_uri=self._settings.redirect_uri,
        )

        if "error" in result:
            error_desc = result.get("error_description", result.get("error", "Unknown error"))
            logger.error("Microsoft token exchange failed: %s", error_desc)
            raise ValueError(f"Microsoft Auth Error: {error_desc}")

        return result

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            New token response

        Raises:
            ValueError: If refresh fails
        """
        filtered_scopes = self._filter_reserved_scopes(self._settings.scopes)
        
        result = self._app.acquire_token_by_refresh_token(
            refresh_token,
            scopes=filtered_scopes,
        )

        if "error" in result:
            error_desc = result.get("error_description", result.get("error", "Unknown error"))
            logger.error("Microsoft token refresh failed: %s", error_desc)
            raise ValueError(f"Microsoft Token Refresh Error: {error_desc}")

        return result


# =============================================================================
# TOKEN CONVERSION
# =============================================================================

def token_response_to_storage_format(token_result: dict[str, Any]) -> dict[str, Any]:
    """
    Convert MSAL token response to storage format.

    Args:
        token_result: Raw MSAL token response

    Returns:
        Standardized token payload for encryption and storage
    """
    return {
        "access_token": token_result.get("access_token"),
        "refresh_token": token_result.get("refresh_token"),
        "expires_in": token_result.get("expires_in"),
        "scope": token_result.get("scope"),
        "token_type": token_result.get("token_type", "Bearer"),
        "id_token": token_result.get("id_token"),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MicrosoftOAuthSettings",
    "MicrosoftAuthService",
    "get_microsoft_oauth_settings",
    "token_response_to_storage_format",
]
