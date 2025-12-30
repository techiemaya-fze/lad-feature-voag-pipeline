"""
Database storage submodule.

Contains all runtime storage classes for database operations.
"""

from db.storage.agents import AgentStorage
from db.storage.batches import BatchStorage
from db.storage.calls import CallStorage
from db.storage.call_analysis import CallAnalysisStorage
from db.storage.leads import LeadStorage
from db.storage.students import StudentStorage
from db.storage.tokens import UserTokenStorage
from db.storage.voices import VoiceStorage
from db.storage.numbers import NumberStorage
from db.storage.knowledge_base import KnowledgeBaseStorage
from db.storage.email_templates import EmailTemplateStorage

__all__ = [
    "AgentStorage",
    "BatchStorage",
    "CallStorage",
    "CallAnalysisStorage",
    "LeadStorage",
    "StudentStorage",
    "UserTokenStorage",
    "VoiceStorage",
    "NumberStorage",
    "KnowledgeBaseStorage",
    "EmailTemplateStorage",
]

