"""
Knowledge Base Routes Module (V2 API).

Handles knowledge base (RAG) endpoints:
- POST /knowledge-base/stores: Create store
- GET /knowledge-base/stores: List stores
- GET /knowledge-base/stores/{store_id}: Get store
- DELETE /knowledge-base/stores/{store_id}: Delete store
- POST /knowledge-base/stores/{store_id}/documents: Upload document
- GET /knowledge-base/stores/{store_id}/documents: List documents
- Agent-store linking endpoints
- Lead-store linking endpoints
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from db.storage import KnowledgeBaseStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

# Lazy initialization
_kb_storage: KnowledgeBaseStorage | None = None


def _get_kb_storage() -> KnowledgeBaseStorage:
    global _kb_storage
    if _kb_storage is None:
        _kb_storage = KnowledgeBaseStorage()
    return _kb_storage


def _check_file_search_enabled() -> None:
    """Check if file search feature is enabled."""
    if not os.getenv("GOOGLE_GENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Knowledge base feature not available - GOOGLE_GENAI_API_KEY not configured"
        )


# =============================================================================
# STATUS ENDPOINT
# =============================================================================

@router.get("/status", response_model=dict)
async def knowledge_base_status() -> dict:
    """
    Check knowledge base (file search) feature status.
    
    Returns:
        Status info including whether feature is enabled
    """
    enabled = bool(os.getenv("GOOGLE_GENAI_API_KEY"))
    
    return {
        "enabled": enabled,
        "feature": "knowledge_base",
        "requires": "GOOGLE_GENAI_API_KEY environment variable",
        "message": "Knowledge base is ready" if enabled else "Knowledge base is disabled",
    }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CreateStoreRequest(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)


class KnowledgeBaseStoreResponse(BaseModel):
    id: str
    gemini_store_name: str
    display_name: str
    description: str | None = None
    document_count: int = 0
    is_active: bool = True
    created_at: Any = None
    updated_at: Any = None


class KnowledgeBaseDocumentResponse(BaseModel):
    name: str
    display_name: str
    state: str | None = None


class KnowledgeBaseUploadResponse(BaseModel):
    document: KnowledgeBaseDocumentResponse
    message: str


class AgentStoreLink(BaseModel):
    store_id: str
    is_default: bool = False
    priority: int = 0


class LeadStoreLink(BaseModel):
    store_id: str
    priority: int = 0


# =============================================================================
# STORE CRUD ROUTES
# =============================================================================

@router.post("/stores", response_model=KnowledgeBaseStoreResponse)
async def create_knowledge_base_store(
    payload: CreateStoreRequest,
    request: Request,
) -> KnowledgeBaseStoreResponse:
    """Create a new knowledge base store."""
    _check_file_search_enabled()
    
    
    user_id: str | None = None  # UUID
    try:
        user_id_header = request.headers.get("X-User-ID")
        if user_id_header:
            user_id = user_id_header.strip()  # UUID string
    except (ValueError, TypeError):
        pass
    
    kb_storage = _get_kb_storage()
    
    try:
        # For now create store record without Gemini store
        # Full Gemini integration requires FileSearchTool
        gemini_store_name = f"stores/{payload.display_name.lower().replace(' ', '-')}"
        
        store_id = await kb_storage.create_store(
            gemini_store_name=gemini_store_name,
            display_name=payload.display_name,
            description=payload.description,
            created_by_user_id=user_id,
        )
        
        store_record = await kb_storage.get_store_by_id(store_id)
        if not store_record:
            raise HTTPException(status_code=500, detail="Failed to retrieve created store")
        
        return KnowledgeBaseStoreResponse(
            id=str(store_record["id"]),
            gemini_store_name=store_record["gemini_store_name"],
            display_name=store_record["display_name"],
            description=store_record.get("description"),
            document_count=store_record.get("document_count", 0),
            is_active=store_record.get("is_active", True),
            created_at=store_record.get("created_at"),
            updated_at=store_record.get("updated_at"),
        )
    except Exception as e:
        logger.error(f"Failed to create store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores", response_model=list[KnowledgeBaseStoreResponse])
async def list_knowledge_base_stores(
    active_only: bool = True,
    user_id: str | None = None,  # UUID
) -> list[KnowledgeBaseStoreResponse]:
    """List all knowledge base stores."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    stores = await kb_storage.list_stores(
        active_only=active_only,
        created_by_user_id=user_id,
    )
    
    return [
        KnowledgeBaseStoreResponse(
            id=str(s["id"]),
            gemini_store_name=s["gemini_store_name"],
            display_name=s["display_name"],
            description=s.get("description"),
            document_count=s.get("document_count", 0),
            is_active=s.get("is_active", True),
            created_at=s.get("created_at"),
            updated_at=s.get("updated_at"),
        )
        for s in stores
    ]


@router.get("/stores/{store_id}", response_model=KnowledgeBaseStoreResponse)
async def get_knowledge_base_store(store_id: str) -> KnowledgeBaseStoreResponse:
    """Get a specific knowledge base store by ID."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    store = await kb_storage.get_store_by_id(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    return KnowledgeBaseStoreResponse(
        id=str(store["id"]),
        gemini_store_name=store["gemini_store_name"],
        display_name=store["display_name"],
        description=store.get("description"),
        document_count=store.get("document_count", 0),
        is_active=store.get("is_active", True),
        created_at=store.get("created_at"),
        updated_at=store.get("updated_at"),
    )


@router.delete("/stores/{store_id}", response_model=dict)
async def delete_knowledge_base_store(
    store_id: str,
    delete_gemini_store: bool = True,
) -> dict:
    """Delete a knowledge base store."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    store = await kb_storage.get_store_by_id(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    deleted = await kb_storage.delete_store(store_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete store record")
    
    return {"message": "Store deleted successfully", "store_id": store_id}


# =============================================================================
# AGENT-STORE LINKING ROUTES
# =============================================================================

@router.post("/agents/{agent_id}/stores", response_model=dict)
async def link_store_to_agent(agent_id: int, payload: AgentStoreLink) -> dict:
    """Link a knowledge base store to an agent."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    try:
        link_id = await kb_storage.link_store_to_agent(
            agent_id=agent_id,
            store_id=payload.store_id,
            is_default=payload.is_default,
            priority=payload.priority,
        )
        return {"message": "Store linked to agent", "link_id": link_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}/stores/{store_id}", response_model=dict)
async def unlink_store_from_agent(agent_id: int, store_id: str) -> dict:
    """Remove a store link from an agent."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    unlinked = await kb_storage.unlink_store_from_agent(agent_id, store_id)
    if not unlinked:
        raise HTTPException(status_code=404, detail="Link not found")
    
    return {"message": "Store unlinked from agent"}


@router.get("/agents/{agent_id}/stores", response_model=list[KnowledgeBaseStoreResponse])
async def get_agent_stores(agent_id: int) -> list[KnowledgeBaseStoreResponse]:
    """Get all knowledge base stores linked to an agent."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    stores = await kb_storage.get_stores_for_agent(agent_id)
    return [
        KnowledgeBaseStoreResponse(
            id=str(s["id"]),
            gemini_store_name=s["gemini_store_name"],
            display_name=s["display_name"],
            description=s.get("description"),
            document_count=s.get("document_count", 0),
            is_active=s.get("is_active", True),
        )
        for s in stores
    ]


# =============================================================================
# LEAD-STORE LINKING ROUTES
# =============================================================================

@router.post("/leads/{lead_id}/stores", response_model=dict)
async def link_store_to_lead(lead_id: str, payload: LeadStoreLink) -> dict:  # lead_id is UUID
    """Link a knowledge base store to a lead for per-call customization."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    try:
        link_id = await kb_storage.link_store_to_lead(
            lead_id=lead_id,
            store_id=payload.store_id,
            priority=payload.priority,
        )
        return {"message": "Store linked to lead", "link_id": link_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/leads/{lead_id}/stores/{store_id}", response_model=dict)
async def unlink_store_from_lead(lead_id: str, store_id: str) -> dict:  # lead_id is UUID
    """Remove a store link from a lead."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    unlinked = await kb_storage.unlink_store_from_lead(lead_id, store_id)
    if not unlinked:
        raise HTTPException(status_code=404, detail="Link not found")
    
    return {"message": "Store unlinked from lead"}


@router.get("/leads/{lead_id}/stores", response_model=list[KnowledgeBaseStoreResponse])
async def get_lead_stores(lead_id: str) -> list[KnowledgeBaseStoreResponse]:  # lead_id is UUID
    """Get all knowledge base stores linked to a lead."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    stores = await kb_storage.get_stores_for_lead(lead_id)
    return [
        KnowledgeBaseStoreResponse(
            id=str(s["id"]),
            gemini_store_name=s["gemini_store_name"],
            display_name=s["display_name"],
            description=s.get("description"),
            document_count=s.get("document_count", 0),
            is_active=s.get("is_active", True),
        )
        for s in stores
    ]


__all__ = ["router"]
