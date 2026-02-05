
import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.gemini_client import generate_with_schema, generate_text
from google import genai
from google.genai import types

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_usage_extraction():
    logger.info("Testing usage extraction from Gemini Client...")
    
    # Test text generation
    logger.info("\n1. Testing generate_text(include_usage=True)...")
    try:
        text, usage = generate_text(
            prompt="Say hello in 5 words.", 
            include_usage=True
        )
        if text:
            print(f"Generated Text: {text}")
            print(f"Usage Metadata: {usage}")
            if usage and 'total_token_count' in usage:
                print("✅ Text generation usage extraction SUCCESS")
            else:
                print("❌ Text generation usage extraction FAILED (Empty usage)")
        else:
            print("❌ Text generation FAILED (No text)")
            
    except Exception as e:
        print(f"❌ Text generation FAILED with error: {e}")

    # Test structured generation
    logger.info("\n2. Testing generate_with_schema...")
    try:
        # Define a simple schema using google.genai.types
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "message": types.Schema(type=types.Type.STRING),
                "count": types.Schema(type=types.Type.INTEGER)
            },
            required=["message", "count"]
        )
        
        result = generate_with_schema(
            prompt="Return a JSON with message='Hello' and count=1",
            schema=schema
        )
        
        if result:
            print(f"Generated JSON: {json.dumps(result, indent=2)}")
            if '_usage_metadata' in result:
                print(f"Usage Metadata (injected): {result['_usage_metadata']}")
                print("✅ Structured generation usage extraction SUCCESS")
            else:
                print("❌ Structured generation usage extraction FAILED (Missing _usage_metadata)")
        else:
            print("❌ Structured generation FAILED (No result)")
            
    except Exception as e:
        print(f"❌ Structured generation FAILED with error: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_usage_extraction()
