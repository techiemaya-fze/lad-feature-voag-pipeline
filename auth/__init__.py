"""
Authentication module.

Contains:
- Google OAuth (google.py)
- Microsoft OAuth (microsoft.py)
"""

from auth.google import (
    GoogleOAuthSettings,
    OAuthStateManager,
    TokenEncryptor,
    GoogleCredentialResolver,
    GoogleCredentialError,
    credentials_to_dict,
    credentials_from_dict,
    get_google_oauth_settings,
)

from auth.microsoft import (
    MicrosoftOAuthSettings,
    MicrosoftAuthService,
    get_microsoft_oauth_settings,
    token_response_to_storage_format,
)

__all__ = [
    # Google
    "GoogleOAuthSettings",
    "OAuthStateManager",
    "TokenEncryptor",
    "GoogleCredentialResolver",
    "GoogleCredentialError",
    "credentials_to_dict",
    "credentials_from_dict",
    "get_google_oauth_settings",
    # Microsoft
    "MicrosoftOAuthSettings",
    "MicrosoftAuthService",
    "get_microsoft_oauth_settings",
    "token_response_to_storage_format",
]
