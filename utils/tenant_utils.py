"""
Tenant and Vertical Resolution Utilities.

Legacy Glinks support:
- Previously hardcoded GLINKS_ORG_ID
- Now resolves vertical from "organization slug" (e.g. education-glinks -> education)
"""

# Legacy Glinks organization ID
LEGACY_GLINKS_ORG_ID = "f6de7991-df4f-43de-9f40-298fcda5f723"

def get_vertical_from_org_id(org_id: str | None) -> str:
    """
    Resolve vertical from legacy Organization ID.
    
    Args:
        org_id: UUID string of the organization
        
    Returns:
        Vertical string (default: "general")
    """
    if org_id == LEGACY_GLINKS_ORG_ID:
        return "education"
    return "general"


def get_vertical_from_slug(slug: str | None) -> str:
    """
    Determine vertical from tenant slug.
    
    Format: "{vertical}-{client_name}"
    Example: 
      - "education-glinks" -> "education"
      - "realestate-agency" -> "realestate"
      - "general-company" -> "general"
    
    Args:
        slug: Tenant slug string
        
    Returns:
        Vertical string (default: "general")
    """
    if not slug:
        return "general"
        
    normalized = slug.lower().strip()
    
    if normalized == "g_links" or normalized.startswith("education-"):
        return "education"
    elif normalized.startswith("realestate-"):
        return "realestate"
        
    return "general"

def is_education_vertical(slug: str | None) -> bool:
    """Check if tenant belongs to education vertical."""
    return get_vertical_from_slug(slug) == "education"


async def get_tenant_slug(tenant_id: str) -> str | None:
    """
    Get tenant slug from database by tenant_id.
    
    Phase 18: Used for vertical resolution from tenant_id.
    
    Args:
        tenant_id: Tenant UUID
        
    Returns:
        Tenant slug string or None if not found
    """
    try:
        from db.storage import CallStorage
        
        storage = CallStorage()
        with storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT slug FROM lad_dev.tenants WHERE id = %s",
                    (tenant_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception:
        return None


async def is_education_tenant(tenant_id: str | None) -> bool:
    """
    Check if tenant belongs to education vertical by tenant_id.
    
    Phase 18: Queries lad_dev.tenants for slug and determines vertical.
    
    Args:
        tenant_id: Tenant UUID
        
    Returns:
        True if tenant's vertical is education
    """
    if not tenant_id:
        return False
    
    slug = await get_tenant_slug(tenant_id)
    return is_education_vertical(slug)


def get_vertical_from_tenant_id_sync(tenant_id: str | None) -> str:
    """
    Synchronous version - get vertical from tenant_id.
    
    WARNING: Uses sync DB call, prefer async version in async contexts.
    """
    if not tenant_id:
        return "general"
    
    try:
        from db.storage import CallStorage
        
        storage = CallStorage()
        with storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT slug FROM lad_dev.tenants WHERE id = %s",
                    (tenant_id,)
                )
                row = cur.fetchone()
                slug = row[0] if row else None
                return get_vertical_from_slug(slug)
    except Exception:
        return "general"

