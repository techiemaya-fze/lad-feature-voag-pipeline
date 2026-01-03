"""Post-call analysis package exposing the legacy merged analytics module."""

from .merged_analytics import CallAnalytics, analyze_call_complete, analytics
from .runner import run_post_call_analysis

# Batch report (optional - may have heavy dependencies)
try:
    from .batch_report import generate_batch_report
    _BATCH_REPORT_AVAILABLE = True
except ImportError:
    _BATCH_REPORT_AVAILABLE = False
    generate_batch_report = None


# Lead information extractor (optional - may not be available)
try:
    from .lead_info_extractor import LeadInfoExtractor, lead_extractor, extract_and_save_lead_info
    _LEAD_EXTRACTOR_AVAILABLE = True
except ImportError:
    _LEAD_EXTRACTOR_AVAILABLE = False
    LeadInfoExtractor = None
    lead_extractor = None
    extract_and_save_lead_info = None

__all__ = [
    "CallAnalytics",
    "analyze_call_complete",
    "analytics",
    "run_post_call_analysis",
    "generate_batch_report",
    "LeadInfoExtractor",
    "lead_extractor",
    "extract_and_save_lead_info",
]
