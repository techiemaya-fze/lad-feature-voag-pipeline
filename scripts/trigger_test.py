"""Trigger a small test batch (10 calls) and print the result."""
from dotenv import load_dotenv
load_dotenv()
import httpx, json

r = httpx.post(
    "http://localhost:8000/batch/trigger-test-batch",
    json={
        "total": 10,
        "fail_count": 7,
        "stuck_count": 0,
        "dropped_count": 0,
        "voice_id": "IjAVmHJFAoVacLb7raEE",
        "initiated_by": "81f1decc-7ee5-4093-b55c-95ac9b7c9f45",
    },
    headers={
        "X-Frontend-ID": "dev",
        "X-API-Key": "kMQgGRDAa8t5CvmkfqFYuGiXIXgNYC1EEGjYs5v8_NU",
    },
    timeout=120,
)
data = json.loads(r.text)
print(f"STATUS: {r.status_code}")
print(f"JOB_ID: {data.get('job_id')}")
print(f"MESSAGE: {data.get('message')}")
