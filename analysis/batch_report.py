"""
Batch Report Service - Excel Report Generation for Completed Batches

This service generates Excel reports for completed batch calls and can optionally
send them via email. It integrates with the existing database infrastructure
and is called from main.py when a batch completes.

Supports two email methods:
- SMTP (default): Traditional SMTP email sending
- OAuth: Uses Google Gmail API with stored OAuth tokens (bypasses SMTP blocks)

Usage from main.py:
    from analysis.batch_report_service import generate_batch_report
    
    # When batch completes (completed, stopped, or cancelled)
    result = await generate_batch_report(batch_id, send_email=True)

CLI Usage:
    # List all batches
    uv run python -m post_call_analysis.batch_report_service --list-batches
    
    # Generate report for a batch
    uv run python -m post_call_analysis.batch_report_service --batch-id <UUID>
    
    # Generate and email report
    uv run python -m post_call_analysis.batch_report_service --batch-id <UUID> --send-email

Environment Variables Required:
    - DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD (database)
    - BATCH_REPORT_EMAIL or REPORT_EMAIL (optional, for email reports)
      Special value 'self' can be included in the list to automatically include
      the connected Gmail address of the OAuth user (BATCH_EMAIL_OAUTH_USER_ID)
      Example: REPORT_EMAIL=self,user1@example.com,user2@example.com
    
    For SMTP (default):
    - SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    
    For OAuth (set BATCH_EMAIL_METHOD=oauth):
    - BATCH_EMAIL_METHOD=oauth
    - BATCH_EMAIL_OAUTH_USER_ID=10 (user ID with Google OAuth tokens)
"""

import os
import json
import asyncio
import logging
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv

# Gulf Standard Time (GST = UTC+4)
GST = timezone(timedelta(hours=4))

load_dotenv()

# Schema constants for table names
from db.schema_constants import (
    CALL_LOGS_FULL,
    ANALYSIS_FULL,
    LEADS_FULL,
    BATCHES_FULL,
    BATCH_ENTRIES_FULL,
)

# Configure logging
logger = logging.getLogger("post_call_analysis.batch_report_service")

# Optional imports for Excel and email
try:
    import pandas as pd
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("pandas/openpyxl not installed. Excel export disabled. Install with: pip install pandas openpyxl")

try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

# OAuth email support (Gmail API)
try:
    from utils.google_credentials import GoogleCredentialResolver, GoogleCredentialError
    from tools.gmail_email_tool import GmailEmailTool, EmailPayload, GmailToolError
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    OAUTH_EMAIL_AVAILABLE = True
except ImportError:
    OAUTH_EMAIL_AVAILABLE = False
    logger.warning("OAuth email not available (missing google_credentials or gmail_email_tool)")

# Use existing connection pool infrastructure
try:
    from db.connection_pool import get_raw_connection, return_connection, USE_CONNECTION_POOLING
    DB_POOL_AVAILABLE = True
except ImportError:
    DB_POOL_AVAILABLE = False
    logger.warning("Connection pool not available, using direct connections")
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed")

# Create exports directory
EXPORTS_DIR = Path(__file__).parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)


def _convert_to_gst(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert datetime to Gulf Standard Time (UTC+4)
    
    Args:
        dt: datetime object (aware or naive, assumes UTC if naive)
    
    Returns:
        datetime in GST timezone
    """
    if dt is None:
        return None
    
    # If datetime is naive (no timezone), assume it's UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to GST (UTC+4)
    return dt.astimezone(GST)


def _get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "salesmaya_agent"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
        "connect_timeout": 30,
    }


def _get_connection():
    """Get database connection using pool if available (raw connection, must be returned manually)"""
    db_config = _get_db_config()
    if DB_POOL_AVAILABLE:
        return get_raw_connection(db_config)
    else:
        import psycopg2
        return psycopg2.connect(**db_config)


def _return_conn(conn):
    """Return connection to pool if pooling is enabled"""
    if DB_POOL_AVAILABLE and USE_CONNECTION_POOLING:
        return_connection(conn, _get_db_config())
    else:
        conn.close()


def get_email_recipients(env_key: str = "BATCH_REPORT_EMAIL") -> List[str]:
    """
    Get list of email recipients from environment variable.
    
    Supports a special 'self' keyword which will be resolved to the connected
    Gmail address of the OAuth user (BATCH_EMAIL_OAUTH_USER_ID) when 
    get_email_recipients_async() is called.
    
    Args:
        env_key: Environment variable name (default: BATCH_REPORT_EMAIL)
    
    Returns:
        List of email addresses (may include 'self' placeholder)
    """
    emails_str = os.getenv(env_key, "").strip()
    if not emails_str:
        # Fall back to REPORT_EMAIL for backwards compatibility
        emails_str = os.getenv("REPORT_EMAIL", "").strip()
    
    if not emails_str:
        return []
    
    # Remove quotes if present
    emails_str = emails_str.strip('"\'')
    
    # Split by comma, semicolon, or newline
    for sep in [',', ';', '\n']:
        if sep in emails_str:
            emails = [e.strip() for e in emails_str.split(sep)]
            break
    else:
        emails = [emails_str.strip()]
    
    # Filter: keep valid emails OR the special 'self' keyword
    return [e for e in emails if e and ('@' in e or e.lower() == 'self')]


async def _get_oauth_user_connected_gmail(user_id: Optional[int] = None) -> Optional[str]:
    """
    Get the connected Gmail address for the specified user or the default OAuth user.
    
    Args:
        user_id: Optional user ID to look up. Falls back to BATCH_EMAIL_OAUTH_USER_ID env var.
    
    Returns:
        Connected Gmail address or None if not found/configured
    """
    # Use provided user_id or fall back to env var
    if user_id is not None:
        oauth_user_id = user_id
        logger.info(f"Resolving 'self' email using provided user_id={oauth_user_id} (batch initiator)")
    else:
        oauth_user_id_str = os.getenv("BATCH_EMAIL_OAUTH_USER_ID", "")
        if not oauth_user_id_str:
            logger.warning("No user_id provided and BATCH_EMAIL_OAUTH_USER_ID not set, cannot resolve 'self' email recipient")
            return None
        
        try:
            oauth_user_id = int(oauth_user_id_str)
            logger.info(f"Resolving 'self' email using fallback BATCH_EMAIL_OAUTH_USER_ID={oauth_user_id}")
        except ValueError:
            logger.error(f"Invalid BATCH_EMAIL_OAUTH_USER_ID: {oauth_user_id_str}")
            return None
    
    # Query the database for the connected_gmail
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email FROM lad_dev.users WHERE id = %s",
                (oauth_user_id,)
            )
            row = cur.fetchone()
            if row and row[0]:
                connected_gmail = row[0]
                logger.info(f"Resolved 'self' to connected_gmail: {connected_gmail} (user_id={oauth_user_id})")
                return connected_gmail
            else:
                logger.warning(f"No connected_gmail found for user_id={oauth_user_id}")
                return None
    except Exception as e:
        logger.error(f"Failed to fetch connected_gmail for user {oauth_user_id}: {e}")
        return None
    finally:
        _return_conn(conn)


async def get_email_recipients_async(
    env_key: str = "BATCH_REPORT_EMAIL",
    initiated_by: Optional[int] = None,
) -> List[str]:
    """
    Get list of email recipients from environment variable, resolving 'self' to 
    the connected Gmail of the batch initiator or OAuth user.
    
    If 'self' is present in the recipient list, it will be replaced with the 
    connected_gmail address of:
    1. The initiated_by user (if provided) - the person who started the batch
    2. The user specified by BATCH_EMAIL_OAUTH_USER_ID (fallback)
    
    Args:
        env_key: Environment variable name (default: BATCH_REPORT_EMAIL)
        initiated_by: User ID who initiated the batch (used to resolve 'self')
    
    Returns:
        List of resolved email addresses
    """
    raw_recipients = get_email_recipients(env_key)
    
    # Check if 'self' is in the list
    has_self = any(e.lower() == 'self' for e in raw_recipients)
    
    if not has_self:
        return raw_recipients
    
    # Resolve 'self' to the initiator's connected Gmail (or fallback to env var user)
    connected_gmail = await _get_oauth_user_connected_gmail(user_id=initiated_by)
    
    resolved_recipients = []
    for email in raw_recipients:
        if email.lower() == 'self':
            if connected_gmail:
                resolved_recipients.append(connected_gmail)
            else:
                logger.warning("'self' in REPORT_EMAIL but could not resolve connected_gmail - skipping")
        else:
            resolved_recipients.append(email)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recipients = []
    for email in resolved_recipients:
        email_lower = email.lower()
        if email_lower not in seen:
            seen.add(email_lower)
            unique_recipients.append(email)
    
    return unique_recipients


async def get_batch_info(batch_id: str) -> Optional[Dict]:
    """
    Get batch information from database
    
    Args:
        batch_id: Batch UUID
    
    Returns:
        Batch info dict or None if not found
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # v2 schema: lad_dev.voice_call_batches uses finished_at, no cancelled_calls
            cur.execute(f"""
                SELECT 
                    id, metadata->>'job_id' as job_id, status, total_calls, completed_calls,
                    failed_calls, created_at, finished_at,
                    agent_id, initiated_by_user_id
                FROM {BATCHES_FULL}
                WHERE id = %s::uuid
            """, (str(batch_id),))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return {
                "id": str(row[0]),
                "job_id": row[1],
                "status": row[2],
                "total_calls": row[3],
                "completed_calls": row[4],
                "failed_calls": row[5],
                "cancelled_calls": 0,  # v2 schema doesn't have this, default to 0
                "created_at": row[6],
                "completed_at": row[7],  # Using finished_at column
                "agent_id": row[8],
                "initiated_by": row[9],
            }
    finally:
        _return_conn(conn)


async def fetch_batch_call_data(batch_id: str) -> List[Dict]:
    """
    Fetch all call analysis data for a batch
    
    Args:
        batch_id: Batch UUID
    
    Returns:
        List of call data dictionaries
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    cl.id as call_log_id,
                    cl.started_at,
                    cl.ended_at,
                    cl.status as call_status,
                    cl.duration_seconds as call_duration,
                    COALESCE(l.first_name || ' ' || l.last_name, '') as lead_name,
                    COALESCE('+' || l.country_code || l.base_number::text, '') as lead_number,
                    a.summary,
                    a.sentiment,
                    a.key_points,
                    a.lead_extraction,
                    a.raw_analysis
                FROM {BATCH_ENTRIES_FULL} be
                INNER JOIN {CALL_LOGS_FULL} cl ON be.call_log_id = cl.id
                LEFT JOIN {ANALYSIS_FULL} a ON a.call_log_id = cl.id
                LEFT JOIN {LEADS_FULL} l ON cl.lead_id = l.id
                WHERE be.batch_id = %s::uuid
                ORDER BY cl.started_at ASC
            """, (str(batch_id),))

            
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            
            data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                
                # Extract fields from lead_extraction JSONB for backwards compatibility
                lead_extraction = row_dict.get('lead_extraction') or {}
                if isinstance(lead_extraction, str):
                    try:
                        lead_extraction = json.loads(lead_extraction)
                    except json.JSONDecodeError:
                        lead_extraction = {}
                
                # Map new schema fields to old report format
                row_dict['lead_category'] = lead_extraction.get('lead_category', '')
                row_dict['disposition'] = lead_extraction.get('disposition', '')
                row_dict['recommended_action'] = lead_extraction.get('recommended_action', '')
                row_dict['engagement_level'] = lead_extraction.get('engagement_level', '')
                row_dict['recommendations'] = lead_extraction.get('recommendations', '')
                
                # Extract key_points from JSONB
                key_points = row_dict.get('key_points') or []
                if isinstance(key_points, str):
                    try:
                        key_points = json.loads(key_points)
                    except json.JSONDecodeError:
                        key_points = []
                row_dict['key_discussion_points'] = key_points
                row_dict['key_phrases'] = []
                row_dict['prospect_questions'] = []
                row_dict['prospect_concerns'] = []
                
                # Handle sentiment
                sentiment = row_dict.get('sentiment', '')
                row_dict['sentiment_description'] = sentiment or ''
                
                data.append(row_dict)
            
            return data
    finally:
        _return_conn(conn)


async def get_batch_call_count(batch_id: str) -> Dict[str, int]:
    """
    Get count of total calls and analyzed calls for a batch
    
    Args:
        batch_id: Batch UUID
    
    Returns:
        Dict with 'total_calls' and 'analyzed_calls' counts
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    COUNT(be.id) as total_calls,
                    COUNT(a.id) as analyzed_calls
                FROM {BATCH_ENTRIES_FULL} be
                LEFT JOIN {ANALYSIS_FULL} a ON a.call_log_id = be.call_log_id
                WHERE be.batch_id = %s::uuid
            """, (str(batch_id),))
            
            row = cur.fetchone()
            return {
                "total_calls": row[0] if row else 0,
                "analyzed_calls": row[1] if row else 0,
            }
    finally:
        _return_conn(conn)


async def wait_for_analysis_completion(
    batch_id: str,
    max_wait_seconds: int = 120,
    poll_interval_seconds: int = 10
) -> Dict[str, int]:
    """
    Wait for post-call analysis to complete for all calls in a batch
    
    Args:
        batch_id: Batch UUID
        max_wait_seconds: Maximum time to wait (default 2 minutes)
        poll_interval_seconds: How often to check (default 10 seconds)
    
    Returns:
        Dict with 'total_calls' and 'analyzed_calls' counts
    """
    logger.info(f"Waiting for analysis completion for batch {batch_id} (max {max_wait_seconds}s)")
    
    start_time = datetime.now(timezone.utc)
    last_counts = None
    stable_count = 0  # Track how many times counts have been stable
    
    while True:
        counts = await get_batch_call_count(batch_id)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(f"Analysis status: {counts['analyzed_calls']}/{counts['total_calls']} analyzed ({elapsed:.0f}s elapsed)")
        
        # All calls analyzed
        if counts['total_calls'] > 0 and counts['analyzed_calls'] >= counts['total_calls']:
            logger.info(f"All {counts['total_calls']} calls have been analyzed")
            return counts
        
        # Check for stable counts (analysis might be stuck or some calls don't get analyzed)
        if last_counts and counts['analyzed_calls'] == last_counts['analyzed_calls']:
            stable_count += 1
            # If counts have been stable for 3 polls, assume analysis is complete
            if stable_count >= 3:
                logger.warning(f"Analysis counts stable for {stable_count * poll_interval_seconds}s - proceeding with {counts['analyzed_calls']}/{counts['total_calls']} analyzed")
                return counts
        else:
            stable_count = 0
        
        last_counts = counts
        
        # Timeout
        if elapsed >= max_wait_seconds:
            logger.warning(f"Timeout waiting for analysis: {counts['analyzed_calls']}/{counts['total_calls']} analyzed after {elapsed:.0f}s")
            return counts
        
        await asyncio.sleep(poll_interval_seconds)


async def list_batches(limit: int = 50) -> List[Dict]:
    """
    List recent batches with summary info
    
    Args:
        limit: Maximum number of batches to return
    
    Returns:
        List of batch summary dictionaries
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT 
                    id, metadata->>'job_id' as job_id, status, total_calls, completed_calls,
                    failed_calls, cancelled_calls, created_at, updated_at
                FROM {BATCHES_FULL}
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            
            batches = []
            for row in cur.fetchall():
                batches.append({
                    "id": str(row[0]),
                    "job_id": row[1],
                    "status": row[2],
                    "total_calls": row[3],
                    "completed_calls": row[4],
                    "failed_calls": row[5],
                    "cancelled_calls": row[6],
                    "created_at": row[7],
                    "completed_at": row[8],
                })
            
            return batches
    finally:
        _return_conn(conn)


def _get_time_period(started_at: datetime) -> str:
    """Categorize call into time period based on start time (in GST)"""
    if not started_at:
        return "Unknown"
    
    # Convert to GST for time period categorization
    gst_time = _convert_to_gst(started_at)
    if not gst_time:
        return "Unknown"
    
    hour = gst_time.hour
    
    if 9 <= hour < 12:
        return "09:00-12:00 (Morning)"
    elif 12 <= hour < 14:
        return "12:00-14:00 (Midday)"
    elif 14 <= hour < 17:
        return "14:00-17:00 (Afternoon)"
    elif 17 <= hour < 20:
        return "17:00-20:00 (Evening)"
    elif 20 <= hour < 24:
        return "20:00-24:00 (Night)"
    else:
        return "00:00-09:00 (Early Morning)"


def _format_list_field(value) -> str:
    """Format list/JSON field for Excel display"""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else ""
    elif isinstance(value, dict):
        return json.dumps(value, indent=2)
    elif value is None:
        return ""
    return str(value)


def _format_duration(started_at, ended_at) -> str:
    """Format duration from timestamps"""
    if not started_at or not ended_at:
        return "N/A"
    
    try:
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace('Z', '+00:00'))
        
        # Ensure both are timezone-aware for accurate calculation
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if ended_at.tzinfo is None:
            ended_at = ended_at.replace(tzinfo=timezone.utc)
        
        duration_seconds = int((ended_at - started_at).total_seconds())
        if duration_seconds < 0:
            return "N/A"
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        return f"{minutes}m {seconds}s"
    except Exception:
        return "N/A"


def _apply_excel_formatting(worksheet, df):
    """Apply professional formatting to Excel worksheet"""
    if not EXCEL_AVAILABLE:
        return
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    wrap_alignment = Alignment(wrap_text=True, vertical='top', horizontal='left')
    center_alignment = Alignment(vertical='center', horizontal='center')
    
    # Format header row
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_alignment
        cell.border = border
    
    worksheet.row_dimensions[1].height = 25
    
    # Format data rows
    for row_idx, row in enumerate(worksheet.iter_rows(min_row=2, max_row=worksheet.max_row), start=2):
        for cell in row:
            cell.border = border
            cell.alignment = wrap_alignment
        worksheet.row_dimensions[row_idx].height = 20
    
    # Set column widths
    column_widths = {
        'Call Log ID': 40,
        'Lead Name': 20,
        'Lead Number': 15,
        'Date': 12,
        'Start Time (GST)': 14,
        'End Time (GST)': 14,
        'Duration': 10,
        'Call Status': 12,
        'Time Period (GST)': 22,
        'Disposition': 20,
        'Recommended Action': 40,
        'Lead Category': 15,
        'Engagement Level': 15,
        'Sentiment': 50,
        'Summary': 60,
        'Key Discussion Points': 50,
        'Prospect Questions': 50,
        'Prospect Concerns': 50,
        'Recommendations': 50,
    }
    
    for idx, col in enumerate(df.columns, start=1):
        col_letter = get_column_letter(idx)
        if col in column_widths:
            worksheet.column_dimensions[col_letter].width = column_widths[col]
        else:
            max_length = max(
                df[col].astype(str).map(len).max() if len(df) > 0 else 0,
                len(str(col))
            )
            worksheet.column_dimensions[col_letter].width = min(max_length + 2, 60)
    
    # Freeze header row
    worksheet.freeze_panes = 'A2'
    
    # Enable filter
    worksheet.auto_filter.ref = worksheet.dimensions


def export_to_excel(
    data: List[Dict],
    batch_id: str,
    batch_info: Optional[Dict] = None,
    output_file: Optional[str] = None
) -> str:
    """
    Export batch call data to Excel file
    
    Args:
        data: List of call data dictionaries
        batch_id: Batch UUID for filename
        batch_info: Optional batch metadata
        output_file: Optional custom output path
    
    Returns:
        Path to exported Excel file
    """
    if not EXCEL_AVAILABLE:
        raise ImportError("pandas/openpyxl not installed. Install with: pip install pandas openpyxl")
    
    if not data:
        raise ValueError("No data to export")
    
    # Generate output filename
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = str(batch_id)[:8]
        output_file = f"batch_report_{short_id}_{timestamp}.xlsx"
    
    output_path = EXPORTS_DIR / output_file
    
    # Prepare data for Excel
    excel_data = []
    for record in data:
        started_at = record.get('started_at')
        ended_at = record.get('ended_at')
        
        # Convert timestamps to Gulf Standard Time (UTC+4)
        gst_started_at = _convert_to_gst(started_at) if isinstance(started_at, datetime) else None
        gst_ended_at = _convert_to_gst(ended_at) if isinstance(ended_at, datetime) else None
        
        # Format timestamps in GST
        date_str = gst_started_at.strftime('%Y-%m-%d') if gst_started_at else 'N/A'
        start_time_str = gst_started_at.strftime('%H:%M:%S') if gst_started_at else 'N/A'
        end_time_str = gst_ended_at.strftime('%H:%M:%S') if gst_ended_at else 'N/A'
        
        row = {
            'Call Log ID': str(record.get('call_log_id', '')) if record.get('call_log_id') else '',
            'Lead Name': record.get('lead_name', '') or '',
            'Lead Number': record.get('lead_number', '') or '',
            'Date': date_str,
            'Start Time (GST)': start_time_str,
            'End Time (GST)': end_time_str,
            'Duration': _format_duration(started_at, ended_at),
            'Call Status': record.get('call_status', '') or '',
            'Time Period (GST)': _get_time_period(started_at),
            'Disposition': record.get('disposition', '') or '',
            'Recommended Action': record.get('recommended_action', '') or '',
            'Lead Category': record.get('lead_category', '') or '',
            'Engagement Level': record.get('engagement_level', '') or '',
            'Sentiment': record.get('sentiment_description', '') or '',
            'Summary': record.get('summary', '') or '',
            'Key Discussion Points': _format_list_field(record.get('key_discussion_points')),
            'Prospect Questions': _format_list_field(record.get('prospect_questions')),
            'Prospect Concerns': _format_list_field(record.get('prospect_concerns')),
            'Recommendations': record.get('recommendations', '') or '',
        }
        excel_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Column order
    column_order = [
        'Call Log ID', 'Lead Name', 'Lead Number', 'Date', 'Start Time (GST)',
        'End Time (GST)', 'Duration', 'Call Status', 'Time Period (GST)', 'Disposition',
        'Recommended Action', 'Lead Category', 'Engagement Level', 'Sentiment',
        'Summary', 'Key Discussion Points', 'Prospect Questions',
        'Prospect Concerns', 'Recommendations'
    ]
    
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Write to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        if batch_info:
            summary_data = {
                'Metric': ['Batch ID', 'Status', 'Total Calls', 'Completed', 'Failed', 
                          'Cancelled', 'Created At', 'Completed At'],
                'Value': [
                    batch_info.get('id', 'N/A'),
                    batch_info.get('status', 'N/A'),
                    batch_info.get('total_calls', 0),
                    batch_info.get('completed_calls', 0),
                    batch_info.get('failed_calls', 0),
                    batch_info.get('cancelled_calls', 0),
                    str(batch_info.get('created_at', 'N/A')),
                    str(batch_info.get('completed_at', 'N/A')),
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Group by time period
        if len(data) > 1 and 'Time Period (GST)' in df.columns:
            time_periods = df['Time Period (GST)'].unique()
            for period in sorted(time_periods):
                period_df = df[df['Time Period (GST)'] == period].copy()
                period_df = period_df.drop(columns=['Time Period (GST)'], errors='ignore')
                
                sheet_name = period[:31].replace(':', '-').replace('/', '-')
                period_df.to_excel(writer, sheet_name=sheet_name, index=False)
                _apply_excel_formatting(writer.sheets[sheet_name], period_df)
            
            # Also add an 'All Calls' sheet
            df_all = df.drop(columns=['Time Period (GST)'], errors='ignore')
            df_all.to_excel(writer, sheet_name='All Calls', index=False)
            _apply_excel_formatting(writer.sheets['All Calls'], df_all)
        else:
            df.to_excel(writer, sheet_name='Call Analysis', index=False)
            _apply_excel_formatting(writer.sheets['Call Analysis'], df)
    
    logger.info(f"Excel report created: {output_path} ({len(data)} records)")
    return str(output_path)


def _get_email_method() -> str:
    """
    Get the configured email method.
    
    Returns:
        "smtp" (default) or "oauth"
    """
    method = os.getenv("BATCH_EMAIL_METHOD", "smtp").lower().strip()
    if method not in ("smtp", "oauth"):
        logger.warning(f"Invalid BATCH_EMAIL_METHOD '{method}', defaulting to 'smtp'")
        return "smtp"
    return method


async def _send_email_via_oauth(
    to_emails: List[str],
    subject: str,
    html_body: str,
    text_body: str,
    attachment_path: Optional[str] = None,
    oauth_user_id: Optional[int] = None,
) -> bool:
    """
    Send email using Google OAuth tokens via Gmail API.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject
        html_body: HTML email body
        text_body: Plain text email body (fallback)
        attachment_path: Optional path to file attachment
        oauth_user_id: User ID for OAuth tokens (defaults to BATCH_EMAIL_OAUTH_USER_ID env var)
    
    Returns:
        True if email sent successfully
    """
    if not OAUTH_EMAIL_AVAILABLE:
        logger.error("OAuth email not available (missing dependencies)")
        return False
    
    # Get the user ID for OAuth tokens - use provided value or fall back to env var
    if oauth_user_id is not None:
        oauth_user_id_int = oauth_user_id
        logger.info(f"Using provided oauth_user_id={oauth_user_id_int} (from batch initiated_by)")
    else:
        oauth_user_id_str = os.getenv("BATCH_EMAIL_OAUTH_USER_ID", "10")
        try:
            oauth_user_id_int = int(oauth_user_id_str)
            logger.info(f"Using fallback BATCH_EMAIL_OAUTH_USER_ID={oauth_user_id_int} from env var")
        except ValueError:
            logger.error(f"Invalid BATCH_EMAIL_OAUTH_USER_ID: {oauth_user_id_str}")
            return False
    
    logger.info(f"Sending email via OAuth using user_id={oauth_user_id_int}")
    
    try:
        # Resolve credentials using existing infrastructure
        resolver = GoogleCredentialResolver()
        credentials = await resolver.load_credentials(oauth_user_id_int)
        
        # Ensure token is valid
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        
        # Build the email message
        msg = MIMEMultipart("mixed")
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject
        
        # Create alternative part for text/html
        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(text_body, "plain"))
        alt_part.attach(MIMEText(html_body, "html"))
        msg.attach(alt_part)
        
        # Add attachment if provided
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(attachment_path)}'
            )
            msg.attach(part)
        
        # Encode and send via Gmail API
        encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        response = service.users().messages().send(
            userId="me",
            body={"raw": encoded_message}
        ).execute()
        
        logger.info(f"OAuth email sent successfully to {len(to_emails)} recipient(s), message_id={response.get('id')}")
        return True
        
    except GoogleCredentialError as e:
        logger.error(f"Failed to load OAuth credentials for user {oauth_user_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send OAuth email: {e}", exc_info=True)
        return False


def _send_email_via_smtp(
    to_emails: List[str],
    subject: str,
    html_body: str,
    text_body: str,
    attachment_path: Optional[str] = None,
) -> bool:
    """
    Send email using SMTP.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject
        html_body: HTML email body
        text_body: Plain text email body (fallback)
        attachment_path: Optional path to file attachment
    
    Returns:
        True if email sent successfully
    """
    if not EMAIL_AVAILABLE:
        logger.error("SMTP email libraries not available")
        return False
    
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL") or smtp_user
    
    if not smtp_user or not smtp_password:
        logger.warning("SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD in .env")
        return False
    
    if not to_emails:
        logger.warning("No email recipients specified")
        return False
    
    if attachment_path and not os.path.exists(attachment_path):
        logger.error(f"Attachment file not found: {attachment_path}")
        return False
    
    try:
        msg = MIMEMultipart("mixed")
        msg['From'] = from_email
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = subject
        
        # Create alternative part for text/html
        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(text_body, "plain"))
        alt_part.attach(MIMEText(html_body, "html"))
        msg.attach(alt_part)
        
        if attachment_path:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(attachment_path)}'
            )
            msg.attach(part)
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        
        server.sendmail(from_email, to_emails, msg.as_string())
        server.quit()
        
        logger.info(f"SMTP email sent successfully to {len(to_emails)} recipient(s)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send SMTP email: {e}", exc_info=True)
        return False


async def send_report_email(
    to_emails: List[str],
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
    batch_info: Optional[Dict] = None,
    calls_with_analysis: int = 0,
    calls_without_analysis: int = 0,
    excel_path: Optional[str] = None,
    oauth_user_id: Optional[int] = None,
) -> bool:
    """
    Send email with optional attachment using configured method (SMTP or OAuth).
    
    The email method is determined by BATCH_EMAIL_METHOD environment variable:
    - "smtp" (default): Use traditional SMTP
    - "oauth": Use Google Gmail API with OAuth tokens
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject
        body: Email body text (plain text fallback)
        attachment_path: Optional path to file attachment
        batch_info: Optional batch metadata for HTML email
        calls_with_analysis: Number of calls with analysis
        calls_without_analysis: Number of calls without analysis
        excel_path: Path to Excel file for HTML email
        oauth_user_id: User ID for OAuth tokens (used when BATCH_EMAIL_METHOD=oauth)
    
    Returns:
        True if email sent successfully
    """
    if not to_emails:
        logger.warning("No email recipients specified")
        return False
    
    # Build HTML email body (enhanced formatting)
    html_body = _build_html_email_body(
        batch_info=batch_info,
        calls_with_analysis=calls_with_analysis,
        calls_without_analysis=calls_without_analysis,
        has_excel=excel_path is not None,
    )
    
    email_method = _get_email_method()
    logger.info(f"Sending batch report email via {email_method.upper()} to {len(to_emails)} recipient(s)")
    
    if email_method == "oauth":
        return await _send_email_via_oauth(
            to_emails=to_emails,
            subject=subject,
            html_body=html_body,
            text_body=body,
            attachment_path=attachment_path,
            oauth_user_id=oauth_user_id,
        )
    else:
        return _send_email_via_smtp(
            to_emails=to_emails,
            subject=subject,
            html_body=html_body,
            text_body=body,
            attachment_path=attachment_path,
        )


def _build_html_email_body(
    batch_info: Optional[Dict] = None,
    calls_with_analysis: int = 0,
    calls_without_analysis: int = 0,
    has_excel: bool = True,
) -> str:
    """
    Build a professional HTML email body for batch reports.
    
    Args:
        batch_info: Batch metadata dictionary
        calls_with_analysis: Number of calls with analysis
        calls_without_analysis: Number of calls without analysis  
        has_excel: Whether Excel report is attached
    
    Returns:
        HTML email body string
    """
    if batch_info is None:
        batch_info = {}
    
    # Get current time in GST
    gst_now = datetime.now(GST)
    report_time = gst_now.strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate statistics
    total_calls = batch_info.get('total_calls', 0)
    completed_calls = batch_info.get('completed_calls', 0)
    failed_calls = batch_info.get('failed_calls', 0)
    cancelled_calls = batch_info.get('cancelled_calls', 0)
    
    # Status color
    status = batch_info.get('status', 'unknown')
    status_colors = {
        'completed': '#28a745',  # Green
        'stopped': '#ffc107',    # Yellow
        'cancelled': '#dc3545',  # Red
        'running': '#17a2b8',    # Blue
        'pending': '#6c757d',    # Gray
    }
    status_color = status_colors.get(status.lower(), '#6c757d')
    
    # Success rate calculation
    success_rate = round((completed_calls / total_calls * 100), 1) if total_calls > 0 else 0
    analysis_rate = round((calls_with_analysis / total_calls * 100), 1) if total_calls > 0 else 0
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f4;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #f4f4f4;">
        <tr>
            <td style="padding: 20px 0;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" align="center" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 40px; border-radius: 8px 8px 0 0;">
                            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 600;">
                                📊 Batch Call Report
                            </h1>
                            <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 14px;">
                                Automated Voice Agent Report
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Status Badge -->
                    <tr>
                        <td style="padding: 25px 40px 15px 40px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                <tr>
                                    <td>
                                        <span style="display: inline-block; background-color: {status_color}; color: #ffffff; padding: 8px 20px; border-radius: 20px; font-size: 14px; font-weight: 600; text-transform: uppercase;">
                                            {status.title()}
                                        </span>
                                    </td>
                                    <td style="text-align: right; color: #666666; font-size: 13px;">
                                        Generated: {report_time} GST
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Batch ID -->
                    <tr>
                        <td style="padding: 0 40px 20px 40px;">
                            <p style="margin: 0; color: #888888; font-size: 12px;">BATCH ID</p>
                            <p style="margin: 5px 0 0 0; color: #333333; font-size: 14px; font-family: 'Courier New', monospace; word-break: break-all;">
                                {batch_info.get('id', 'N/A')}
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Stats Grid -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                <tr>
                                    <!-- Total Calls -->
                                    <td width="25%" style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                                        <p style="margin: 0; font-size: 32px; font-weight: 700; color: #333333;">{total_calls}</p>
                                        <p style="margin: 5px 0 0 0; font-size: 11px; color: #666666; text-transform: uppercase; letter-spacing: 0.5px;">Total Calls</p>
                                    </td>
                                    <td width="2%"></td>
                                    <!-- Completed -->
                                    <td width="23%" style="text-align: center; padding: 15px; background-color: #d4edda; border-radius: 8px;">
                                        <p style="margin: 0; font-size: 32px; font-weight: 700; color: #28a745;">{completed_calls}</p>
                                        <p style="margin: 5px 0 0 0; font-size: 11px; color: #155724; text-transform: uppercase; letter-spacing: 0.5px;">Completed</p>
                                    </td>
                                    <td width="2%"></td>
                                    <!-- Failed -->
                                    <td width="23%" style="text-align: center; padding: 15px; background-color: #f8d7da; border-radius: 8px;">
                                        <p style="margin: 0; font-size: 32px; font-weight: 700; color: #dc3545;">{failed_calls}</p>
                                        <p style="margin: 5px 0 0 0; font-size: 11px; color: #721c24; text-transform: uppercase; letter-spacing: 0.5px;">Failed</p>
                                    </td>
                                    <td width="2%"></td>
                                    <!-- Cancelled -->
                                    <td width="23%" style="text-align: center; padding: 15px; background-color: #fff3cd; border-radius: 8px;">
                                        <p style="margin: 0; font-size: 32px; font-weight: 700; color: #856404;">{cancelled_calls}</p>
                                        <p style="margin: 5px 0 0 0; font-size: 11px; color: #856404; text-transform: uppercase; letter-spacing: 0.5px;">Cancelled</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Progress Bars -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                <!-- Success Rate -->
                                <tr>
                                    <td style="padding-bottom: 15px;">
                                        <p style="margin: 0 0 8px 0; font-size: 13px; color: #333333;">
                                            <strong>Success Rate:</strong> {success_rate}%
                                        </p>
                                        <div style="background-color: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden;">
                                            <div style="background-color: #28a745; height: 100%; width: {success_rate}%; border-radius: 10px;"></div>
                                        </div>
                                    </td>
                                </tr>
                                <!-- Analysis Rate -->
                                <tr>
                                    <td>
                                        <p style="margin: 0 0 8px 0; font-size: 13px; color: #333333;">
                                            <strong>Analysis Coverage:</strong> {analysis_rate}% ({calls_with_analysis} of {total_calls} calls)
                                        </p>
                                        <div style="background-color: #e9ecef; border-radius: 10px; height: 10px; overflow: hidden;">
                                            <div style="background-color: #667eea; height: 100%; width: {analysis_rate}%; border-radius: 10px;"></div>
                                        </div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Timestamps -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #f8f9fa; border-radius: 8px; padding: 15px;">
                                <tr>
                                    <td style="padding: 15px;">
                                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                                            <tr>
                                                <td width="50%">
                                                    <p style="margin: 0; font-size: 12px; color: #666666;">Started</p>
                                                    <p style="margin: 5px 0 0 0; font-size: 14px; color: #333333;">
                                                        {str(batch_info.get('created_at', 'N/A'))[:19]}
                                                    </p>
                                                </td>
                                                <td width="50%">
                                                    <p style="margin: 0; font-size: 12px; color: #666666;">Completed</p>
                                                    <p style="margin: 5px 0 0 0; font-size: 14px; color: #333333;">
                                                        {str(batch_info.get('completed_at', 'N/A'))[:19] if batch_info.get('completed_at') else 'N/A'}
                                                    </p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Attachment Notice -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #e7f1ff; border-left: 4px solid #667eea; border-radius: 4px;">
                                <tr>
                                    <td style="padding: 15px 20px;">
                                        <p style="margin: 0; font-size: 14px; color: #333333;">
                                            📎 {"<strong>Excel report attached</strong> - Contains detailed call analysis organized by time periods (GST timezone)." if has_excel else "<em>Note: Excel report could not be generated (missing dependencies).</em>"}
                                        </p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #f8f9fa; padding: 25px 40px; border-radius: 0 0 8px 8px; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #888888;">
                                This is an automated report from the Voice Agent System.
                            </p>
                            <p style="margin: 10px 0 0 0; font-size: 11px; color: #aaaaaa;">
                                © {gst_now.year} Voice Agent • All times shown in Gulf Standard Time (GST/UTC+4)
                            </p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    return html


async def generate_batch_report(
    batch_id: str,
    send_email: bool = False,
    email_recipients: Optional[List[str]] = None,
    wait_for_analysis: bool = True,
    max_wait_seconds: int = 120,
) -> Dict[str, Any]:
    """
    Generate a batch report with Excel export and optional email
    
    This is the main entry point for integration with main.py
    Call this when a batch completes/stops/cancels.
    
    Args:
        batch_id: Batch UUID
        send_email: Whether to send email with report
        email_recipients: Optional list of email addresses (defaults to env var)
        wait_for_analysis: Whether to wait for post-call analysis to complete (default True)
        max_wait_seconds: Maximum time to wait for analysis (default 120s)
    
    Returns:
        Dict with status, file_path, email_sent, etc.
    """
    logger.info(f"Generating batch report for batch_id={batch_id}")
    
    # Get batch info
    batch_info = await get_batch_info(batch_id)
    if not batch_info:
        logger.error(f"Batch {batch_id} not found")
        return {
            "status": "error",
            "message": f"Batch {batch_id} not found",
            "file_path": None,
            "email_sent": False,
            "call_count": 0,
        }
    
    logger.info(f"Batch found: status={batch_info['status']}, total={batch_info['total_calls']}")
    
    # Wait for post-call analysis to complete before fetching data
    if wait_for_analysis:
        analysis_counts = await wait_for_analysis_completion(
            batch_id, 
            max_wait_seconds=max_wait_seconds,
            poll_interval_seconds=10
        )
        logger.info(f"Analysis complete: {analysis_counts['analyzed_calls']}/{analysis_counts['total_calls']} calls analyzed")
    
    # Fetch call data (now with analysis data populated)
    call_data = await fetch_batch_call_data(batch_id)
    
    calls_with_analysis = len([c for c in call_data if c.get('summary')])
    calls_without_analysis = len(call_data) - calls_with_analysis
    
    logger.info(f"Fetched {len(call_data)} calls ({calls_with_analysis} with analysis)")
    
    if not call_data:
        logger.warning(f"No calls found for batch {batch_id}")
        return {
            "status": "warning",
            "message": f"No calls found for batch {batch_id}",
            "file_path": None,
            "email_sent": False,
            "call_count": 0,
        }
    
    # Export to Excel
    excel_path = None
    try:
        excel_path = export_to_excel(call_data, batch_id, batch_info)
        logger.info(f"Excel report created: {excel_path}")
    except ImportError as e:
        logger.warning(f"Excel export not available: {e}")
        # Continue without Excel - we can still send a text-only email
    except Exception as e:
        logger.error(f"Failed to create Excel report: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to create Excel report: {e}",
            "file_path": None,
            "email_sent": False,
            "call_count": len(call_data),
        }
    
    # Get current time in GST for report
    gst_now = datetime.now(GST)
    
    # Send email if requested
    email_sent = False
    if send_email:
        recipients = email_recipients or await get_email_recipients_async(
            initiated_by=batch_info.get('initiated_by')
        )
        if recipients:
            subject = f"Batch Call Report - {batch_info['status'].title()} - {len(call_data)} Calls"
            body = f"""
Batch Call Report

Batch ID: {batch_id}
Status: {batch_info['status']}
Total Calls: {batch_info['total_calls']}
Completed: {batch_info['completed_calls']}
Failed: {batch_info['failed_calls']}
Cancelled: {batch_info['cancelled_calls']}
Created: {batch_info.get('created_at', 'N/A')}
Completed: {batch_info.get('completed_at', 'N/A')}

Calls with Analysis: {calls_with_analysis}
Calls without Analysis: {calls_without_analysis}

Report Generated: {gst_now.strftime('%Y-%m-%d %H:%M:%S')} GST (Gulf Standard Time)

{"The attached Excel report contains detailed call analysis organized by time periods (GST)." if excel_path else "Note: Excel report could not be generated (missing dependencies)."}

Best regards,
Voice Agent Automated Report System
            """
            
            email_sent = await send_report_email(
                to_emails=recipients,
                subject=subject,
                body=body,
                attachment_path=excel_path,
                batch_info=batch_info,
                calls_with_analysis=calls_with_analysis,
                calls_without_analysis=calls_without_analysis,
                excel_path=excel_path,
                oauth_user_id=batch_info.get('initiated_by'),  # Use batch initiator for OAuth email
            )
        else:
            logger.info("No email recipients configured - skipping email")
    
    return {
        "status": "success",
        "message": "Batch report generated successfully",
        "file_path": excel_path,
        "email_sent": email_sent,
        "call_count": len(call_data),
        "calls_with_analysis": calls_with_analysis,
        "calls_without_analysis": calls_without_analysis,
        "batch_info": batch_info,
    }


async def print_batch_list():
    """Print list of available batches"""
    batches = await list_batches(50)
    
    if not batches:
        print("No batches found in database")
        return
    
    print(f"\n{'='*100}")
    print("AVAILABLE BATCHES (Last 50)")
    print(f"{'='*100}")
    print(f"{'Batch ID':<40} {'Status':<12} {'Total':<8} {'Completed':<10} {'Failed':<8} {'Created'}")
    print("-" * 100)
    
    for batch in batches:
        created = batch['created_at'].strftime('%Y-%m-%d %H:%M') if batch['created_at'] else 'N/A'
        print(f"{batch['id']:<40} {batch['status']:<12} {batch['total_calls']:<8} "
              f"{batch['completed_calls']:<10} {batch['failed_calls']:<8} {created}")
    
    print("-" * 100)
    print(f"\nTo generate a report, use:")
    print(f"  uv run python -m post_call_analysis.batch_report_service --batch-id <BATCH_ID>")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Call Report Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all batches
    uv run python -m post_call_analysis.batch_report_service --list-batches
    
    # Generate report for a batch
    uv run python -m post_call_analysis.batch_report_service --batch-id <UUID>
    
    # Generate and email report
    uv run python -m post_call_analysis.batch_report_service --batch-id <UUID> --send-email
        """
    )
    
    parser.add_argument('--list-batches', action='store_true', help='List available batches')
    parser.add_argument('--batch-id', help='Batch UUID to generate report for')
    parser.add_argument('--send-email', action='store_true', help='Send report via email')
    parser.add_argument('--email', help='Override email recipient')
    
    args = parser.parse_args()
    
    if args.list_batches:
        asyncio.run(print_batch_list())
        return 0
    
    if not args.batch_id:
        parser.error("Either --list-batches or --batch-id is required")
    
    email_recipients = [args.email] if args.email else None
    
    result = asyncio.run(generate_batch_report(
        args.batch_id,
        send_email=args.send_email,
        email_recipients=email_recipients,
    ))
    
    print(f"\n{'='*60}")
    print("REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"File Path: {result.get('file_path', 'N/A')}")
    print(f"Email Sent: {'Yes' if result.get('email_sent') else 'No'}")
    print(f"Call Count: {result.get('call_count', 0)}")
    print(f"With Analysis: {result.get('calls_with_analysis', 0)}")
    print(f"Without Analysis: {result.get('calls_without_analysis', 0)}")
    print(f"{'='*60}\n")
    
    return 0 if result['status'] == 'success' else 1


if __name__ == "__main__":
    exit(main())
