"""
API Routes Module (V2).

Contains route handlers with kebab-case naming:
- calls: Call initiation (/start-call, /call-logs)
- batch: Batch operations (/trigger-batch-call)
- agents: Agent CRUD (/voice-agents)
- knowledge_base: KB operations (/knowledge-base)
- oauth: Google OAuth (/auth/google/*)
- oauth_microsoft: Microsoft OAuth (/auth/microsoft/*)
- recordings: Recording signed URLs (/recordings/*)
"""

from .calls import router as calls_router
from .batch import router as batch_router
from .agents import router as agents_router
from .knowledge_base import router as knowledge_base_router
from .oauth import router as oauth_router
from .recordings import router as recordings_router

# Microsoft OAuth is fully implemented
from .oauth_microsoft import microsoft_router as oauth_microsoft_router

__all__ = [
    "calls_router",
    "batch_router", 
    "agents_router",
    "knowledge_base_router",
    "oauth_router",
    "oauth_microsoft_router",
    "recordings_router",
]

