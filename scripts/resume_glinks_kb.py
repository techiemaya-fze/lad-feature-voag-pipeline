"""Resume uploading failed Glinks PDFs - one by one with delay to avoid rate limits."""
import os, sys, json, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import httpx

FRONTEND_API_KEYS = json.loads(os.getenv("FRONTEND_API_KEYS", "{}"))
frontend_id = list(FRONTEND_API_KEYS.keys())[0]
api_key = FRONTEND_API_KEYS[frontend_id]
HEADERS = {
    "X-Frontend-ID": frontend_id, 
    "X-API-Key": api_key,
}

STORE_ID = "339e1f15-3206-4450-9b6d-d1766472a866"
FOLDER_PATH = r"D:\vonage\vonage-voice-agent\Glinks_pdf"

# Files that were already successfully uploaded
ALREADY_UPLOADED = {
    "Acadia University-Canada.pdf",
    "Alexander College -Canada.pdf",
    "Alfred University -Usa.pdf",
    "Arts University Bournemouth UK.pdf",
    "Bangor University UK.pdf",
    "Bond University Australia.pdf",
    "Brock University -Canada.pdf",
    "California Miramar University -usa.pdf",
    "California State University-usa.pdf",
    "CAMPUS SPAIN.pdf",
    "Capilano University-Canada.pdf",
    "Columbia-College-Canada.pdf",
    "De Montfort University  UK.pdf",
    "Drury University -usa.pdf",
    "Dusemond Study Programmes UK.pdf",
    "EU Business School Germany.pdf",
    "EU Business School-spain.pdf",
    "EURASIA Germany Study  Germany.pdf",
    "Glinks International Data INDEX.pdf",
    "Huron at Western University-Canada.pdf",
    "INTI International University -malaysisa.pdf",
    "ISM Germany Professional Germany.pdf",
    "International Cultural Exchange Services -usa.pdf",
    "John von Neumann University Hungary.pdf",
    "Justice Institute of British Columbia-Canada.pdf",
    "Kings at  Western University-canada.pdf",
    "Kingston University UK.pdf",
    "Lake Washington Institute of Technology -usa.pdf",
    "Lakehead University-canada.pdf",
    "Laurentian-University-Canada.pdf",
}

async def main():
    print("="*60)
    print("Resuming Glinks KB Upload - Failed Files Only")
    print("="*60)
    print(f"Store ID: {STORE_ID}")
    print(f"Already uploaded: {len(ALREADY_UPLOADED)} files")
    
    # Get all files in folder
    all_files = []
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith('.pdf') and filename not in ALREADY_UPLOADED:
            all_files.append(filename)
    
    print(f"Remaining to upload: {len(all_files)} files\n")
    
    successful = 0
    failed = 0
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, filename in enumerate(all_files):
            file_path = os.path.join(FOLDER_PATH, filename)
            display_name = os.path.splitext(filename)[0]
            
            print(f"[{i+1}/{len(all_files)}] Uploading: {filename[:50]}...", end=" ", flush=True)
            
            try:
                with open(file_path, 'rb') as f:
                    files = {'file': (filename, f, 'application/pdf')}
                    data = {'display_name': display_name}
                    resp = await client.post(
                        f"http://localhost:8000/knowledge-base/stores/{STORE_ID}/documents",
                        headers=HEADERS,
                        files=files,
                        data=data,
                    )
                
                if resp.status_code == 200:
                    print("✅")
                    successful += 1
                else:
                    print(f"❌ ({resp.status_code})")
                    failed += 1
                
                # Small delay to avoid rate limits
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                failed += 1
    
    print("\n" + "="*60)
    print(f"Resume Complete: {successful} succeeded, {failed} failed")
    print(f"Total in store: {len(ALREADY_UPLOADED) + successful} documents")
    print("="*60)

asyncio.run(main())
