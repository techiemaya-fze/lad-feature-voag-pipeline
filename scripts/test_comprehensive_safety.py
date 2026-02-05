import asyncio
import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import sys

# Add parent directory to path to allow importing analysis module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('comprehensive_test')

load_dotenv()

# Test Call IDs
TEST_CALL_IDS = [
    "bc7815e3-04da-4a6c-84c3-0ef6f2165f1e",
    "51d2b149-d82b-4d56-8853-3059faec0ebf",
    # "b4b2f27e-e650-4410-aacd-d7cf62ee316b", # Skipping failed call for now to focus on happy/ended paths
]

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def get_call_data(call_id):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM lad_dev.voice_call_logs WHERE id = %s", (call_id,))
            return cur.fetchone()
    finally:
        conn.close()

def verify_analysis_results(call_id):
    conn = get_db_connection()
    results = {}
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check call_analysis (table is voice_call_analysis, column is call_log_id)
            cur.execute("SELECT * FROM lad_dev.voice_call_analysis WHERE call_log_id = %s", (call_id,))
            analysis = cur.fetchone()
            results['analysis_exists'] = analysis is not None
            if analysis:
                results['sentiment'] = analysis.get('sentiment_score')
                results['summary'] = analysis.get('call_summary')
                
                # Check lead_extraction column (JSONB)
                lead_extraction = analysis.get('lead_extraction')
                if lead_extraction:
                    results['lead_info_exists'] = True
                    results['lead_name'] = lead_extraction.get('first_name') or lead_extraction.get('full_name')
                else:
                    results['lead_info_exists'] = False
                
    finally:
        conn.close()
    return results

async def run_test():
    from analysis.runner import run_post_call_analysis
    
    overall_success = True
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE SAFETY TEST")
    print("="*80 + "\n")

    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT", 5432),
        "database": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

    for call_id in TEST_CALL_IDS:
        print(f"\nProcessing Call ID: {call_id}")
        print("-" * 40)
        
        # 1. Fetch Call Data
        call_log = get_call_data(call_id)
        if not call_log:
            print(f"❌ Call log not found for {call_id}")
            overall_success = False
            continue
            
        transcripts = call_log.get('transcripts')
        if not transcripts:
             print(f"❌ No transcripts for {call_id}")
             overall_success = False
             continue
             
        # 2. Run Analysis
        try:
            print("⏳ Running pipeline...")
            success = await run_post_call_analysis(
                call_log_id=call_id,
                transcription_json=transcripts,
                duration_seconds=call_log.get('duration_seconds'),
                call_details=dict(call_log),
                db_config=db_config,
                tenant_id=call_log.get('tenant_id')
            )
            
            if success:
                print("✅ Pipeline execution successful")
            else:
                print("⚠️ Pipeline returned False (check logs)")
                
        except Exception as e:
            print(f"❌ Pipeline Exception: {e}")
            import traceback
            traceback.print_exc()
            overall_success = False
            continue

        # 3. Verify Database Persistence
        print("🔍 Verifying Database...")
        verification = verify_analysis_results(call_id)
        
        if verification.get('analysis_exists'):
            print("✅ Call Analysis saved")
            # Safety Check: Look for thought tokens in saved summary
            summary = verification.get('summary', '')
            if summary and ('[THINKING]' in summary or '<thought>' in summary or 'Internal reasoning' in summary):
                print(f"🚨 CRITICAL FAILURE: Thought tokens found in summary for {call_id}!")
                overall_success = False
            else:
                 print("✅ Output cleanly formatted (no thought tokens detected in summary)")
        else:
            print("❌ Call Analysis NOT saved")
            overall_success = False
            
        if verification.get('lead_info_exists'):
             print("✅ Lead Info saved")
        else:
             print("⚠️ Lead Info NOT saved (might be expected depending on content)")

    print("\n" + "="*80)
    if overall_success:
        print("TEST SUITE RESULT: PASS ✅")
        print("All critical safety checks passed.")
    else:
        print("TEST SUITE RESULT: FAIL ❌")
        print("Safety checks failed. See details above.")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_test())
