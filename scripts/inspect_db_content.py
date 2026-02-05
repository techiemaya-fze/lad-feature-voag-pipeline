import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

TEST_CALL_IDS = [
    "bc7815e3-04da-4a6c-84c3-0ef6f2165f1e",
    "51d2b149-d82b-4d56-8853-3059faec0ebf"
]

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )

def inspect_content():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for call_id in TEST_CALL_IDS:
                print(f"\n{'='*80}")
                print(f"INSPECTING CALL ID: {call_id}")
                print(f"{'='*80}")
                
                cur.execute("""
                    SELECT 
                        a.summary, 
                        a.lead_extraction, 
                        a.sentiment,
                        l.transcripts
                    FROM lad_dev.voice_call_analysis a
                    JOIN lad_dev.voice_call_logs l ON a.call_log_id = l.id
                    WHERE a.call_log_id = %s
                """, (call_id,))
                
                row = cur.fetchone()
                if not row:
                    print("❌ No record found in lad_dev.voice_call_analysis")
                    continue
                
                summary = row.get('summary', '')
                lead_extraction = row.get('lead_extraction')
                transcripts = row.get('transcripts')
                
                print("\n--- [TRANSCRIPT PREVIEW] ---")
                if transcripts:
                    try:
                        # Parse if it's a string
                        if isinstance(transcripts, str):
                            data = json.loads(transcripts)
                        else:
                            data = transcripts
                            
                        # Extract segments if available
                        segments = data.get('segments', []) if isinstance(data, dict) else []
                        
                        if segments:
                             for i, segment in enumerate(segments[:20]): # First 20 turns
                                 role = segment.get('speaker', 'unknown')
                                 text = segment.get('text', '')
                                 print(f"{role.upper()}: {text}")
                             if len(segments) > 20:
                                 print("... (truncated) ...")
                        else:
                             print("No segments found in transcript JSON")
                             print(str(data)[:500])
                             
                    except Exception as e:
                        print(f"Error parsing transcript: {e}")
                        print(str(transcripts)[:500])
                else:
                    print("⚠️ NO TRANSCRIPT FOUND")

                print("\n--- [SUMMARY INSPECTION] ---")
                if summary:
                    print(f"Length: {len(summary)} chars")
                    print(f"Ends with punctuation? {summary.strip()[-1] in '.!?\"' if summary else False}")
                    print("-" * 20)
                    print(summary)
                    print("-" * 20)
                    
                    if len(summary) > 0 and summary.strip()[-1] not in '.!?"':
                        print("⚠️ WARNING: Summary might be truncated (does not end in punctuation)")
                    else:
                        print("✅ Summary appears complete")
                else:
                    print("⚠️ Summary is EMPTY")

                print("\n--- [LEAD EXTRACTION JSON INSPECTION] ---")
                if lead_extraction:
                    print(f"Type: {type(lead_extraction)}")
                    print(json.dumps(lead_extraction, indent=2))
                    
                    # Check specific fields
                    required = ['first_name', 'phone', 'requirements']
                    missing = [f for f in required if f not in lead_extraction and f not in str(lead_extraction)] # logic loose for str check
                    
                    if isinstance(lead_extraction, dict):
                         print("✅ Valid JSON Object")
                    else:
                         print("⚠️ WARNING: lead_extraction is not a dict")
                else:
                    print("⚠️ lead_extraction is NULL/EMPTY")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    inspect_content()
