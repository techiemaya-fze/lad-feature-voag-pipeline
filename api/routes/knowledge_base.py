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
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base feature not available - GOOGLE_API_KEY or GEMINI_API_KEY not configured"
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
    enabled = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    
    return {
        "enabled": enabled,
        "feature": "knowledge_base",
        "requires": "GOOGLE_API_KEY or GEMINI_API_KEY environment variable",
        "message": "Knowledge base is ready" if enabled else "Knowledge base is disabled",
    }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CreateStoreRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID that owns this store")
    display_name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)
    is_default: bool = Field(False, description="Auto-attach to tenant's calls")
    priority: int = Field(0, description="Higher = preferred when multiple stores")


class KnowledgeBaseStoreResponse(BaseModel):
    id: str
    tenant_id: str | None = None
    gemini_store_name: str
    display_name: str
    description: str | None = None
    is_default: bool = False
    is_active: bool = True
    priority: int = 0
    document_count: int = 0
    created_at: Any = None
    updated_at: Any = None


class KnowledgeBaseDocumentResponse(BaseModel):
    name: str
    display_name: str
    state: str | None = None


class KnowledgeBaseUploadResponse(BaseModel):
    document: KnowledgeBaseDocumentResponse
    message: str


# Note: AgentStoreLink and LeadStoreLink are deprecated - use tenant_id instead


# =============================================================================
# STORE CRUD ROUTES
# =============================================================================
@router.post("/stores", response_model=KnowledgeBaseStoreResponse)
async def create_knowledge_base_store(
    payload: CreateStoreRequest,
    request: Request,
) -> KnowledgeBaseStoreResponse:
    """Create a new knowledge base store linked to a tenant."""
    _check_file_search_enabled()
    
    # Get optional user_id for created_by
    created_by: str | None = None
    try:
        user_id_header = request.headers.get("X-User-ID")
        if user_id_header:
            created_by = user_id_header.strip()  # UUID string
    except (ValueError, TypeError):
        pass
    
    kb_storage = _get_kb_storage()
    
    try:
        # Create Gemini FileSearchStore first
        from tools.file_search_tool import FileSearchTool
        file_search = FileSearchTool()
        gemini_store = await file_search.create_store(payload.display_name)
        gemini_store_name = gemini_store.name
        
        # Create DB record
        store_id = await kb_storage.create_store(
            tenant_id=payload.tenant_id,
            gemini_store_name=gemini_store_name,
            display_name=payload.display_name,
            description=payload.description,
            is_default=payload.is_default,
            priority=payload.priority,
            created_by=created_by,
        )
        
        store_record = await kb_storage.get_store_by_id(store_id)
        if not store_record:
            raise HTTPException(status_code=500, detail="Failed to retrieve created store")
        
        logger.info(f"Created KB store '{payload.display_name}' for tenant {payload.tenant_id}")
        
        return KnowledgeBaseStoreResponse(
            id=str(store_record["id"]),
            tenant_id=store_record.get("tenant_id"),
            gemini_store_name=store_record["gemini_store_name"],
            display_name=store_record["display_name"],
            description=store_record.get("description"),
            is_default=store_record.get("is_default", False),
            is_active=store_record.get("is_active", True),
            priority=store_record.get("priority", 0),
            document_count=store_record.get("document_count", 0),
            created_at=store_record.get("created_at"),
            updated_at=store_record.get("updated_at"),
        )
    except Exception as e:
        logger.error(f"Failed to create store: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores", response_model=list[KnowledgeBaseStoreResponse])
async def list_knowledge_base_stores(
    tenant_id: str | None = None,
    active_only: bool = True,
) -> list[KnowledgeBaseStoreResponse]:
    """List knowledge base stores, optionally filtered by tenant."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    stores = await kb_storage.list_stores(
        tenant_id=tenant_id,
        active_only=active_only,
    )
    
    return [
        KnowledgeBaseStoreResponse(
            id=str(s["id"]),
            tenant_id=s.get("tenant_id"),
            gemini_store_name=s["gemini_store_name"],
            display_name=s["display_name"],
            description=s.get("description"),
            is_default=s.get("is_default", False),
            is_active=s.get("is_active", True),
            priority=s.get("priority", 0),
            document_count=s.get("document_count", 0),
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
        tenant_id=store.get("tenant_id"),
        gemini_store_name=store["gemini_store_name"],
        display_name=store["display_name"],
        description=store.get("description"),
        is_default=store.get("is_default", False),
        is_active=store.get("is_active", True),
        priority=store.get("priority", 0),
        document_count=store.get("document_count", 0),
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
# DOCUMENT UPLOAD ROUTES
# =============================================================================

@router.post("/stores/{store_id}/documents", response_model=KnowledgeBaseUploadResponse)
async def upload_document_to_store(
    store_id: str,
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
) -> KnowledgeBaseUploadResponse:
    """
    Upload a document to a knowledge base store.
    
    Supported formats: PDF, TXT, HTML, MD, DOC, DOCX, PPT, PPTX, XLS, XLSX, 
    ODT, ODS, ODP, CSV, JSON, XML
    """
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    
    # Verify store exists
    store = await kb_storage.get_store_by_id(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    gemini_store_name = store["gemini_store_name"]
    
    try:
        from tools.file_search_tool import FileSearchTool
        
        file_search = FileSearchTool()
        
        # Read file content
        file_bytes = await file.read()
        
        # Upload to Gemini
        doc_info = await file_search.upload_document_from_bytes(
            store_name=gemini_store_name,
            file_bytes=file_bytes,
            filename=file.filename or "document",
            display_name=display_name or file.filename,
        )
        
        # Update document count in DB
        current_count = store.get("document_count", 0)
        await kb_storage.update_store(store_id, document_count=current_count + 1)
        
        logger.info(f"Uploaded document '{doc_info.display_name}' to store {store_id}")
        
        return KnowledgeBaseUploadResponse(
            document=KnowledgeBaseDocumentResponse(
                name=doc_info.name,
                display_name=doc_info.display_name,
                state=doc_info.state,
            ),
            message="Document uploaded successfully",
        )
    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores/{store_id}/documents", response_model=list[KnowledgeBaseDocumentResponse])
async def list_store_documents(store_id: str) -> list[KnowledgeBaseDocumentResponse]:
    """List all documents in a knowledge base store."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    
    store = await kb_storage.get_store_by_id(store_id)
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    gemini_store_name = store["gemini_store_name"]
    
    try:
        from tools.file_search_tool import FileSearchTool
        
        file_search = FileSearchTool()
        docs = await file_search.list_documents(gemini_store_name)
        
        return [
            KnowledgeBaseDocumentResponse(
                name=d.name,
                display_name=d.display_name,
                state=d.state,
            )
            for d in docs
        ]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TENANT-BASED STORE RETRIEVAL
# =============================================================================

@router.get("/tenants/{tenant_id}/stores", response_model=list[KnowledgeBaseStoreResponse])
async def get_tenant_stores(
    tenant_id: str,
    default_only: bool = True,
) -> list[KnowledgeBaseStoreResponse]:
    """Get all knowledge base stores for a tenant."""
    _check_file_search_enabled()
    
    kb_storage = _get_kb_storage()
    stores = await kb_storage.get_stores_for_tenant(tenant_id, default_only=default_only)
    
    return [
        KnowledgeBaseStoreResponse(
            id=str(s["id"]),
            tenant_id=s.get("tenant_id"),
            gemini_store_name=s["gemini_store_name"],
            display_name=s["display_name"],
            description=s.get("description"),
            is_default=s.get("is_default", False),
            is_active=True,  # Only active stores are returned
            priority=s.get("priority", 0),
            document_count=s.get("document_count", 0),
        )
        for s in stores
    ]


# =============================================================================
# QUERY ENDPOINT (Testing)
# =============================================================================

class KBQueryRequest(BaseModel):
    """Request body for KB query."""
    tenant_id: str = Field(..., description="Tenant UUID to query KB stores for")
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    model: str = Field("gemini-2.5-flash", description="Gemini model to use")


class KBQueryResponse(BaseModel):
    """Response from KB query."""
    answer: str
    sources: list[str] = []
    store_names: list[str] = []


@router.post("/query", response_model=KBQueryResponse)
async def query_knowledge_base(request: KBQueryRequest) -> KBQueryResponse:
    """
    Query the knowledge base for a tenant.
    
    This endpoint retrieves default KB stores for the tenant and queries
    Gemini with file search grounding to get an answer.
    
    Args:
        request: Contains tenant_id, question, and optional model
        
    Returns:
        Answer text and sources cited
    """
    _check_file_search_enabled()
    
    # Get KB stores for tenant
    kb_storage = _get_kb_storage()
    stores = await kb_storage.get_stores_for_tenant(request.tenant_id, default_only=False)
    
    if not stores:
        raise HTTPException(
            status_code=404,
            detail=f"No knowledge base stores found for tenant {request.tenant_id}"
        )
    
    # Get Gemini store names
    gemini_store_names = [s["gemini_store_name"] for s in stores if s.get("gemini_store_name")]
    
    if not gemini_store_names:
        raise HTTPException(
            status_code=404,
            detail="No Gemini stores linked to tenant's KB stores"
        )
    
    logger.info(f"Querying KB with {len(gemini_store_names)} store(s) for tenant {request.tenant_id}")
    
    # Query Gemini with file search
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=request.model,
            contents=request.question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=gemini_store_names
                    )
                )]
            )
        )
        
        # Extract answer
        answer = response.text if response.text else "(No response generated)"
        
        # Extract sources from grounding metadata
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding = response.candidates[0].grounding_metadata
            if grounding.grounding_chunks:
                for chunk in grounding.grounding_chunks:
                    if hasattr(chunk, 'retrieved_context') and chunk.retrieved_context:
                        title = getattr(chunk.retrieved_context, 'title', None)
                        if title and title not in sources:
                            sources.append(title)
        
        return KBQueryResponse(
            answer=answer,
            sources=sources,
            store_names=gemini_store_names,
        )
        
    except Exception as e:
        logger.error(f"KB query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


# =============================================================================
# BULK UPLOAD ENDPOINT
# =============================================================================

class BulkUploadRequest(BaseModel):
    """Request body for bulk upload."""
    folder_path: str = Field(..., description="Absolute path to folder containing files to upload")


class BulkUploadResult(BaseModel):
    """Result of a single file upload."""
    filename: str
    success: bool
    display_name: str | None = None
    error: str | None = None


class BulkUploadResponse(BaseModel):
    """Response from bulk upload."""
    store_id: str
    total_files: int
    successful: int
    failed: int
    results: list[BulkUploadResult]


# Supported file extensions for KB upload
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".html", ".htm", ".json", ".xml",
    ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt", ".csv"
}


@router.post("/stores/{store_id}/bulk-upload", response_model=BulkUploadResponse)
async def bulk_upload_documents(
    store_id: str,
    request: BulkUploadRequest,
) -> BulkUploadResponse:
    """
    Bulk upload all supported files from a folder to a KB store.
    
    Supported formats: txt, md, pdf, html, json, xml, xlsx, xls, docx, doc, pptx, ppt, csv
    
    Args:
        store_id: The KB store UUID
        request: Contains folder_path to upload from
        
    Returns:
        Summary of upload results
    """
    _check_file_search_enabled()
    
    folder_path = request.folder_path
    
    # Validate folder exists
    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {folder_path}"
        )
    
    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {folder_path}"
        )
    
    # Get store info
    kb_storage = _get_kb_storage()
    store = await kb_storage.get_store(store_id)
    
    if not store:
        raise HTTPException(
            status_code=404,
            detail=f"Store not found: {store_id}"
        )
    
    gemini_store_name = store.get("gemini_store_name")
    if not gemini_store_name:
        raise HTTPException(
            status_code=400,
            detail="Store has no Gemini store linked"
        )
    
    # Get all supported files
    files_to_upload = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                files_to_upload.append((filename, file_path))
    
    if not files_to_upload:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files found in {folder_path}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    logger.info(f"Bulk uploading {len(files_to_upload)} files to store {store_id}")
    
    # Upload each file
    from tools.file_search_tool import FileSearchTool
    file_search = FileSearchTool()
    
    results = []
    successful = 0
    failed = 0
    
    for filename, file_path in files_to_upload:
        try:
            # Use filename without extension as display name
            display_name = os.path.splitext(filename)[0]
            
            doc_info = await file_search.upload_document(
                store_name=gemini_store_name,
                file_path=file_path,
                display_name=display_name,
                wait_for_completion=False,  # Don't wait for each file
            )
            
            results.append(BulkUploadResult(
                filename=filename,
                success=True,
                display_name=display_name,
            ))
            successful += 1
            logger.info(f"Uploaded: {filename}")
            
        except Exception as e:
            results.append(BulkUploadResult(
                filename=filename,
                success=False,
                error=str(e)[:200],
            ))
            failed += 1
            logger.error(f"Failed to upload {filename}: {e}")
    
    logger.info(f"Bulk upload complete: {successful} succeeded, {failed} failed")
    
    return BulkUploadResponse(
        store_id=store_id,
        total_files=len(files_to_upload),
        successful=successful,
        failed=failed,
        results=results,
    )


__all__ = ["router"]

