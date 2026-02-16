"""
LiveKit Credential Resolver

Resolves LiveKit credentials from database or environment variables based on
the USE_SELFHOST_ROUTING_TABLE feature flag.

Usage:
    from utils.livekit_resolver import resolve_livekit_credentials
    
    creds = await resolve_livekit_credentials(
        from_number="+971545335200",
        tenant_id="tenant-uuid",
        routing_result=routing_result
    )
    
    # Use credentials
    url = creds.url
    api_key = creds.api_key
    api_secret = creds.api_secret  # Decrypted
    trunk_id = creds.trunk_id
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

from utils.call_routing import CallRoutingResult
from utils.en_de_crypt import encrypt_decrypt
from db.storage.livekit import LiveKitConfigStorage

logger = logging.getLogger(__name__)


@dataclass
class LiveKitCredentials:
    """Resolved LiveKit credentials."""
    url: str
    api_key: str
    api_secret: str  # Decrypted
    trunk_id: Optional[str]
    worker_name: Optional[str]  # Worker/agent name for dispatch
    source: str  # "database" or "environment" for logging


def _get_credentials_from_env(
    routing_result: Optional[CallRoutingResult] = None
) -> LiveKitCredentials:
    """
    Get credentials from environment variables (existing behavior).
    
    Args:
        routing_result: Optional routing result with outbound_trunk_id
        
    Returns:
        LiveKitCredentials from environment
        
    Raises:
        RuntimeError: If required environment variables are missing
    """
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([url, api_key, api_secret]):
        raise RuntimeError(
            "Missing required LiveKit environment variables: "
            "LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"
        )
    
    # Get trunk_id from routing_result or environment
    trunk_id = None
    if routing_result and routing_result.outbound_trunk_id:
        trunk_id = routing_result.outbound_trunk_id
    else:
        trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
    
    logger.debug("Using LiveKit credentials from environment variables")
    
    # Get worker_name from environment (fallback)
    worker_name = os.getenv("VOICE_AGENT_NAME", "inbound-agent")
    
    return LiveKitCredentials(
        url=url,
        api_key=api_key,
        api_secret=api_secret,
        trunk_id=trunk_id,
        worker_name=worker_name,
        source="environment"
    )


async def _get_credentials_from_database(
    from_number: str,
    tenant_id: Optional[str],
    routing_result: Optional[CallRoutingResult] = None,
) -> Optional[LiveKitCredentials]:
    """
    Get credentials from database via livekit_config UUID.
    
    Args:
        from_number: Caller number (used to lookup config)
        tenant_id: Tenant ID for multi-tenant isolation
        routing_result: Optional routing result with livekit_config_id
        
    Returns:
        LiveKitCredentials from database, or None if not found or on error
    """
    try:
        # Get livekit_config UUID from routing_result
        if not routing_result or not routing_result.livekit_config_id:
            logger.debug(f"No livekit_config UUID in routing result for {from_number[:4] if from_number else 'unknown'}***")
            return None
        
        config_id = routing_result.livekit_config_id
        logger.debug(f"Looking up LiveKit config: {config_id[:8]}...")
        
        # Query database for config
        storage = LiveKitConfigStorage()
        config = await storage.get_livekit_config(config_id)
        
        if not config:
            logger.warning(f"LiveKit config not found in database: {config_id}")
            return None
        
        # Decrypt the API secret
        try:
            decrypted_secret = encrypt_decrypt(config['livekit_api_secret'])
        except Exception as decrypt_error:
            logger.error(
                f"Failed to decrypt livekit_api_secret for config {config['name']}: {decrypt_error}",
                exc_info=True
            )
            return None
        
        # Get trunk_id (priority: config > routing_result > env)
        trunk_id = config.get('trunk_id')
        if not trunk_id and routing_result:
            trunk_id = routing_result.outbound_trunk_id
        if not trunk_id:
            trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
        
        # Get worker_name (priority: config > env)
        worker_name = config.get('worker_name') or os.getenv("VOICE_AGENT_NAME", "inbound-agent")
        
        logger.info(
            f"Using LiveKit credentials from database (config: {config['name']}, "
            f"worker: {worker_name}, number: {from_number[:4] if from_number else 'unknown'}***)"
        )
        
        return LiveKitCredentials(
            url=config['livekit_url'],
            api_key=config['livekit_api_key'],
            api_secret=decrypted_secret,
            trunk_id=trunk_id,
            worker_name=worker_name,
            source="database"
        )
        
    except Exception as e:
        logger.error(
            f"Error getting LiveKit credentials from database for {from_number[:4] if from_number else 'unknown'}***: {e}",
            exc_info=True
        )
        return None


async def resolve_livekit_credentials(
    from_number: Optional[str],
    tenant_id: Optional[str],
    routing_result: Optional[CallRoutingResult] = None,
) -> LiveKitCredentials:
    """
    Resolve LiveKit credentials from database or environment variables.
    
    This function implements the credential resolution logic with feature flag support:
    1. Check USE_SELFHOST_ROUTING_TABLE flag
    2. If false: Return environment variables
    3. If true: Try database lookup
    4. On any error: Fallback to environment variables
    
    Args:
        from_number: Caller number (used to lookup config)
        tenant_id: Tenant ID for multi-tenant isolation
        routing_result: Optional routing result with livekit_config_id and outbound_trunk_id
        
    Returns:
        LiveKitCredentials with resolved values
        
    Raises:
        RuntimeError: If no credentials found in database or environment
        
    Examples:
        >>> # With feature flag enabled and database config
        >>> creds = await resolve_livekit_credentials(
        ...     from_number="+971545335200",
        ...     tenant_id="tenant-uuid",
        ...     routing_result=routing_result
        ... )
        >>> creds.source
        'database'
        
        >>> # With feature flag disabled
        >>> creds = await resolve_livekit_credentials(
        ...     from_number="+971545335200",
        ...     tenant_id="tenant-uuid"
        ... )
        >>> creds.source
        'environment'
    """
    # Check feature flag
    use_selfhost = os.getenv("USE_SELFHOST_ROUTING_TABLE", "true").lower() == "true"
    
    if not use_selfhost:
        # Feature disabled - use environment variables only
        logger.debug("USE_SELFHOST_ROUTING_TABLE=false, using environment variables")
        return _get_credentials_from_env(routing_result)
    
    # Feature enabled - try database first
    logger.debug("USE_SELFHOST_ROUTING_TABLE=true, attempting database lookup")
    
    # Try to get credentials from database
    db_creds = await _get_credentials_from_database(
        from_number=from_number,
        tenant_id=tenant_id,
        routing_result=routing_result
    )
    
    if db_creds:
        # Successfully got credentials from database
        return db_creds
    
    # Fallback to environment variables
    logger.info(
        f"Falling back to environment variables for {from_number[:4] if from_number else 'unknown'}*** "
        "(no database config or error occurred)"
    )
    return _get_credentials_from_env(routing_result)


# Convenience function for backward compatibility
def validate_livekit_credentials() -> tuple[str, str, str]:
    """
    Validate and return LiveKit credentials from environment (legacy function).
    
    This function is kept for backward compatibility with existing code.
    New code should use resolve_livekit_credentials() instead.
    
    Returns:
        Tuple of (url, api_key, api_secret)
        
    Raises:
        RuntimeError: If required environment variables are missing
    """
    creds = _get_credentials_from_env()
    return creds.url, creds.api_key, creds.api_secret
