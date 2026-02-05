"""
Test script for verifying Gemini structured output migration
Tests the new implementation against a real call that was problematic with old HTTP calls
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

# Test call ID that had parsing issues with old implementation
TEST_CALL_ID = "bc7815e3-04da-4a6c-84c3-0ef6f2165f1e"

# Sample conversation from the logs (the one that failed before)
SAMPLE_CONVERSATION = """
[17:27:07] Agent: Hi, this is Nithya from Sasya Spaces, Dubai. I hope I'm catching you at a good time?
[17:27:08] User: Yes.
[17:27:16] User: Okay.
[17:27:20] Agent: We got a list of profiles, and I think... your profile might be eligible for Dubai Golden Visa. We are currently assisting investors with Dubai real estate opportunities that qualify for the UAE Golden Visa.
[17:27:22] User: Okay.
[17:27:30] Agent: Just to confirm, are you currently exploring investment opportunities in Dubai real estate, or is this something you'd consider?
[17:27:33] User: Yeah. I will discuss about that tomorrow.
[17:27:36] User: Can you call me tomorrow? I am busy now.
[17:27:48] Agent: I completely understand, no problem at all.
To make sure I send you the right information beforehand, what budget range were you comfortable with for a Dubai investment, and would you prefer ready property or off-plan?
[17:27:52] User: Will discuss about that tomorrow.
[17:28:00] User: Yeah. That's fine.
[17:28:00] Agent: Understood. I respect your time.
Would tomorrow evening, say around 6:00 PM GST, be a better time for a brief follow-up call?
[17:28:09] User: Thank you.
[17:28:09] Agent: That sounds perfect. I'll tentatively mark that for a follow-up call tomorrow at 6:00 PM GST.
[17:28:12] Agent: You are most welcome. Thank you for your time, and have a great day!
"""


def test_gemini_client():
    """Test the gemini_client module directly"""
    print("\n" + "="*60)
    print("TEST 1: Gemini Client - Structured Output")
    print("="*60)
    
    from analysis.gemini_client import (
        generate_with_schema_retry,
        SENTIMENT_SCHEMA,
        LEAD_INFO_SCHEMA,
    )
    
    # Test 1: Sentiment analysis with structured output
    print("\n[1a] Testing SENTIMENT_SCHEMA...")
    sentiment_prompt = f"""Analyze the sentiment of this sales conversation and provide a sentiment score.

CONVERSATION:
{SAMPLE_CONVERSATION[:1000]}

TASK: Analyze the prospect's sentiment and provide:
1. Sentiment category: Positive, Neutral, Negative, or Very Interested
2. Sentiment score: -1.0 (very negative) to +1.0 (very positive)
3. Confidence: 0.0 to 1.0
4. Brief reasoning for the sentiment"""

    result = generate_with_schema_retry(
        prompt=sentiment_prompt,
        schema=SENTIMENT_SCHEMA,
        temperature=0.1,
        max_output_tokens=300,  # Sentiment has 4 fields - 300 tokens is plenty
    )
    
    if result:
        print(f"   [OK] SUCCESS - Got structured sentiment response:")
        print(f"     Category: {result.get('category')}")
        print(f"     Score: {result.get('sentiment_score')}")
        print(f"     Confidence: {result.get('confidence')}")
        print(f"     Reasoning: {result.get('reasoning', '')[:100]}...")
    else:
        print("   [FAIL] FAILED - No result from sentiment analysis")
        return False
    
    # Test 2: Lead info extraction with structured output
    print("\n[1b] Testing LEAD_INFO_SCHEMA...")
    lead_prompt = f"""Extract lead information from this conversation.

CONVERSATION:
{SAMPLE_CONVERSATION}

Extract any available information about the lead/prospect."""

    result = generate_with_schema_retry(
        prompt=lead_prompt,
        schema=LEAD_INFO_SCHEMA,
        temperature=0.2,
        max_output_tokens=2000,  # Generous limit - 24 fields need plenty of tokens
    )
    
    if result:
        print(f"   [OK] SUCCESS - Got structured lead info response:")
        # Show non-null fields
        for key, value in result.items():
            if value and value not in ["null", "None", ""]:
                print(f"     {key}: {value}")
    else:
        print("   [FAIL] FAILED - No result from lead info extraction")
        return False
    
    return True


async def test_merged_analytics():
    """Test merged_analytics with new structured output"""
    print("\n" + "="*60)
    print("TEST 2: Merged Analytics - Sentiment Analysis")
    print("="*60)
    
    from analysis.merged_analytics import CallAnalytics
    
    analytics = CallAnalytics()
    
    # Test sentiment analysis (this is what was failing before)
    print("\n[2a] Testing _calculate_sentiment_with_llm...")
    result = await analytics._calculate_sentiment_with_llm(
        user_text="Yes. Okay. Yeah I will discuss about that tomorrow. Can you call me tomorrow?",
        conversation_text=SAMPLE_CONVERSATION
    )
    
    if result:
        print(f"   [OK] SUCCESS - Got sentiment result:")
        print(f"     Category: {result.get('category')}")
        print(f"     Score: {result.get('sentiment_score')}")
        print(f"     Confidence: {result.get('confidence')}")
        print(f"     Reasoning: {result.get('llm_reasoning', '')[:100]}...")
    else:
        print("   [FAIL] FAILED - No sentiment result")
        return False
    
    return True


async def test_lead_info_extractor():
    """Test lead_info_extractor with new structured output"""
    print("\n" + "="*60)
    print("TEST 3: Lead Info Extractor - Structured Output")
    print("="*60)
    
    from analysis.lead_info_extractor import LeadInfoExtractor
    
    extractor = LeadInfoExtractor()
    
    print("\n[3a] Testing extract_lead_information...")
    # extract_lead_information is async
    result = await extractor.extract_lead_information(SAMPLE_CONVERSATION)
    
    if result:
        print(f"   [OK] SUCCESS - Got lead info result:")
        for key, value in result.items():
            if value and value not in ["null", "None", ""]:
                print(f"     {key}: {value}")
    else:
        print("   [WARN] No lead info extracted (may be expected for this conversation)")
    
    return True


async def test_lead_bookings_extractor():
    """Test lead_bookings_extractor with new structured output"""
    print("\n" + "="*60)
    print("TEST 4: Lead Bookings Extractor - Structured Output")
    print("="*60)
    
    from analysis.lead_bookings_extractor import LeadBookingsExtractor
    
    extractor = LeadBookingsExtractor()
    
    print("\n[4a] Testing extract_booking_info...")
    result = await extractor.extract_booking_info(SAMPLE_CONVERSATION)
    
    if result:
        print(f"   [OK] SUCCESS - Got booking info result:")
        print(f"     Booking Type: {result.get('booking_type')}")
        print(f"     Scheduled At: {result.get('scheduled_at')}")
        print(f"     Has Future Discussion: {result.get('has_future_discussion')}")
    else:
        print("   [FAIL] FAILED - No booking info result")
        return False
    
    return True


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("STRUCTURED OUTPUT MIGRATION TEST SUITE")
    print("Testing with call that previously had JSON parsing failures")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Gemini client directly
    if not test_gemini_client():
        all_passed = False
    
    # Test 2: Merged analytics
    if not await test_merged_analytics():
        all_passed = False
    
    # Test 3: Lead info extractor
    if not await test_lead_info_extractor():
        all_passed = False
    
    # Test 4: Lead bookings extractor
    if not await test_lead_bookings_extractor():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL TESTS PASSED - Structured output migration successful!")
    else:
        print("[FAIL] SOME TESTS FAILED - Review output above")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(run_all_tests())
