"""Test Ultravox API with the exact same payload the LiveKit plugin sends."""
import aiohttp, asyncio, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("ULTRAVOX_API_KEY", "")
print(f"Key: {key[:10]}...{key[-6:]}")

async def test():
    async with aiohttp.ClientSession() as s:
        # Exact payload the LiveKit plugin sends (from realtime_model.py lines 604-630)
        payload = {
            "systemPrompt": "test",
            "model": "fixie-ai/ultravox",
            "voice": "Mark",
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": 16000,
                    "outputSampleRate": 24000,
                    "clientBufferSizeMs": 30000,
                }
            },
            "selectedTools": [],
            "temperature": 0.4,
            "languageHint": "en",
            "maxDuration": "30m",
            "firstSpeaker": "FIRST_SPEAKER_USER",
        }
        
        url = "https://api.ultravox.ai/api/calls?enableGreetingPrompt=false"
        headers = {
            "User-Agent": "LiveKit Agents",
            "X-API-Key": key,
            "Content-Type": "application/json",
        }
        
        print(f"\n=== Request ===")
        print(f"URL: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        async with s.post(url, json=payload, headers=headers) as r:
            print(f"\n=== Response ===")
            print(f"Status: {r.status}")
            body = await r.text()
            print(f"Body: {body[:1000]}")

asyncio.run(test())
