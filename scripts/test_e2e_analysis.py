"""
End-to-End Test for Structured Output Migration
Runs the actual analysis files on a real call ID from the database.
"""
import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import RealDictCursor

# Test call ID that had parsing issues with old implementation  
TEST_CALL_ID = "bc7815e3-04da-4a6c-84c3-0ef6f2165f1e"


def get_db_connection():
    """Get a database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def get_call_log(call_id: str) -> dict:
    """Fetch call log from database using correct schema/table."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Using lad_dev.voice_call_logs table
            cur.execute("""
                SELECT id, transcripts, duration_seconds, started_at, ended_at, status, tenant_id
                FROM lad_dev.voice_call_logs
                WHERE id = %s
            """, (call_id,))
            result = cur.fetchone()
            return dict(result) if result else None
    finally:
        conn.close()


def get_call_analysis(call_id: str) -> dict:
    """Fetch call analysis from database (to check if it was saved)."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Try lad_dev.call_analysis first
            cur.execute("""
                SELECT call_log_id, sentiment_score, sentiment_category, llm_reasoning, 
                       call_summary, summary_generated_at, llm_cost_inr
                FROM lad_dev.call_analysis
                WHERE call_log_id = %s
            """, (call_id,))
            result = cur.fetchone()
            return dict(result) if result else None
    except psycopg2.errors.UndefinedTable:
        # Table might not exist
        return None
    finally:
        conn.close()


def get_lead_info(call_id: str) -> dict:
    """Fetch lead info extracted for this call."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT call_log_id, first_name, email, phone, program_interested,
                       available_time, additional_notes, extracted_at
                FROM lad_dev.lead_info
                WHERE call_log_id = %s
            """, (call_id,))
            result = cur.fetchone()
            return dict(result) if result else None
    except psycopg2.errors.UndefinedTable:
        return None
    finally:
        conn.close()


async def test_full_analysis_pipeline():
    """Test the full analysis pipeline with a real call from DB."""
    print("\n" + "="*70)
    print("END-TO-END TEST: Full Analysis Pipeline with Real Call ID")
    print("="*70)
    print(f"Call ID: {TEST_CALL_ID}")
    
    # Step 1: Fetch call log from DB
    print("\n[STEP 1] Fetching call log from database...")
    call_log = get_call_log(TEST_CALL_ID)
    
    if not call_log:
        print(f"   [FAIL] Call log not found for ID: {TEST_CALL_ID}")
        return False
    
    print(f"   [OK] Call log found:")
    print(f"     Duration: {call_log.get('duration_seconds')} seconds")
    print(f"     Status: {call_log.get('status')}")
    print(f"     Tenant: {call_log.get('tenant_id')}")
    
    transcripts = call_log.get('transcripts')
    if not transcripts:
        print("   [FAIL] No transcripts found in call log")
        return False
    
    print(f"   [OK] Transcripts length: {len(str(transcripts))} chars")
    
    # Step 2: Run the full analysis pipeline
    print("\n[STEP 2] Running full analysis pipeline...")
    
    from analysis.runner import run_post_call_analysis
    
    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", 5432),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    
    try:
        success = await run_post_call_analysis(
            call_log_id=TEST_CALL_ID,
            transcription_json=transcripts,
            duration_seconds=call_log.get('duration_seconds'),
            call_details=call_log,
            db_config=db_config,
            tenant_id=call_log.get('tenant_id'),
        )
        
        if success:
            print("   [OK] Analysis pipeline completed successfully")
        else:
            print("   [WARN] Analysis pipeline returned False (may be expected)")
    except Exception as e:
        print(f"   [FAIL] Analysis pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify results were saved to DB
    print("\n[STEP 3] Verifying analysis results in database...")
    
    analysis = get_call_analysis(TEST_CALL_ID)
    if analysis:
        print("   [OK] Call analysis found in database:")
        print(f"     Sentiment Category: {analysis.get('sentiment_category')}")
        print(f"     Sentiment Score: {analysis.get('sentiment_score')}")
        if analysis.get('llm_reasoning'):
            reasoning = str(analysis.get('llm_reasoning'))[:100]
            print(f"     LLM Reasoning: {reasoning}...")
        print(f"     Summary Generated At: {analysis.get('summary_generated_at')}")
        print(f"     LLM Cost: {analysis.get('llm_cost_inr')}")
    else:
        print("   [WARN] No call analysis found in database")
    
    # Step 4: Check lead info extraction
    print("\n[STEP 4] Checking lead info extraction...")
    
    lead_info = get_lead_info(TEST_CALL_ID)
    if lead_info:
        print("   [OK] Lead info found in database:")
        for key, value in lead_info.items():
            if value and value not in ["null", "None", ""] and key != 'call_log_id':
                print(f"     {key}: {value}")
    else:
        print("   [INFO] No lead info in database (may be expected if table doesn't exist)")
    
    print("\n" + "="*70)
    print("[OK] END-TO-END TEST COMPLETED")
    print("="*70 + "\n")
    
    return True


async def test_individual_modules():
    """Test individual analysis modules with the same call."""
    print("\n" + "="*70)
    print("INDIVIDUAL MODULE TESTS")
    print("="*70)
    
    # Get call log first
    call_log = get_call_log(TEST_CALL_ID)
    if not call_log:
        print("[FAIL] Could not fetch call log")
        return False
    
    transcripts = call_log.get('transcripts')
    if not transcripts:
        print("[FAIL] No transcripts in call log")
        return False
    
    # Build conversation text from transcripts
    import json
    if isinstance(transcripts, str):
        transcripts = json.loads(transcripts)
    
    segments = transcripts.get('segments', [])
    conversation_text = ""
    for seg in segments:
        speaker = seg.get('speaker', 'Unknown')
        text = seg.get('text', '')
        if text:
            conversation_text += f"[{speaker}]: {text}\n"
    
    print(f"Conversation length: {len(conversation_text)} chars")
    
    # Test 1: Lead Info Extractor
    print("\n[TEST 1] LeadInfoExtractor...")
    try:
        from analysis.lead_info_extractor import LeadInfoExtractor
        extractor = LeadInfoExtractor()
        result = await extractor.extract_lead_information(conversation_text)
        if result:
            print("   [OK] Lead info extracted:")
            non_empty = {k: v for k, v in result.items() if v and v not in ["", "null", "None"]}
            for k, v in list(non_empty.items())[:5]:
                print(f"     {k}: {v}")
        else:
            print("   [WARN] No lead info extracted")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 2: Lead Bookings Extractor
    print("\n[TEST 2] LeadBookingsExtractor...")
    try:
        from analysis.lead_bookings_extractor import LeadBookingsExtractor
        extractor = LeadBookingsExtractor()
        result = await extractor.extract_booking_info(conversation_text)
        if result:
            print("   [OK] Booking info extracted:")
            print(f"     Booking Type: {result.get('booking_type')}")
            print(f"     Scheduled At: {result.get('scheduled_at')}")
        else:
            print("   [WARN] No booking info extracted")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test 3: Merged Analytics Sentiment
    print("\n[TEST 3] CallAnalytics - Sentiment...")
    try:
        from analysis.merged_analytics import CallAnalytics
        analytics = CallAnalytics()
        
        # Extract user text only
        user_text = " ".join([seg.get('text', '') for seg in segments if seg.get('speaker', '').lower() == 'user'])
        
        result = await analytics._calculate_sentiment_with_llm(user_text, conversation_text)
        if result:
            print("   [OK] Sentiment calculated:")
            print(f"     Category: {result.get('category')}")
            print(f"     Score: {result.get('sentiment_score')}")
            print(f"     Confidence: {result.get('confidence')}")
        else:
            print("   [WARN] No sentiment result")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    return True


async def main():
    """Main entry point."""
    print("\n" + "#"*70)
    print("# STRUCTURED OUTPUT MIGRATION - END-TO-END TESTING")
    print("#"*70)
    
    # Run full pipeline test
    await test_full_analysis_pipeline()
    
    # Run individual module tests
    await test_individual_modules()


if __name__ == "__main__":
    asyncio.run(main())
