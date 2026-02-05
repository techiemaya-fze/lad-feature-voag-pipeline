"""Search agent instructions for bad hangup patterns that cause LLM to mix text with tool calls."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import asyncio
from db.storage.agents import AgentStorage

# Patterns that might cause LLM to output hangup_call as text instead of tool call
BAD_PATTERNS = [
    "call hangup_call",
    "call the hangup",
    "say goodbye and hangup",
    "immediately hangup",
    "immediately call hangup",
    "then hangup_call",
    "followed by hangup",
    "and hangup_call",
]

async def main():
    storage = AgentStorage()
    agents = await storage.list_agents(limit=100, active_only=False)
    
    print("=" * 70)
    print("SEARCHING FOR BAD HANGUP INSTRUCTIONS IN AGENTS")
    print("=" * 70)
    
    for agent in agents:
        agent_id = agent.get('id')
        instructions = agent.get('instructions', '') or ''
        outbound_starter = agent.get('outbound_starter_prompt', '') or ''
        
        combined = (instructions + outbound_starter).lower()
        
        found_patterns = []
        for pattern in BAD_PATTERNS:
            if pattern.lower() in combined:
                found_patterns.append(pattern)
        
        # Also check for inline tool call patterns like "hangup_call(..."
        if 'hangup_call(' in combined or 'hangup_call"' in combined:
            found_patterns.append("inline hangup_call()")
        
        if found_patterns:
            print(f"\n⚠️ Agent {agent_id} ({agent.get('name', 'Unknown')}):")
            print(f"   Patterns found: {found_patterns}")
            # Show context around the pattern
            for pattern in found_patterns[:1]:  # Just show first match
                idx = combined.find(pattern.lower())
                if idx != -1:
                    start = max(0, idx - 50)
                    end = min(len(combined), idx + len(pattern) + 100)
                    print(f"   Context: ...{combined[start:end]}...")
        else:
            # Check if agent has any hangup mention at all
            if 'hangup' in combined:
                print(f"✓ Agent {agent_id} ({agent.get('name', 'Unknown')}): has hangup instruction (OK)")

if __name__ == '__main__':
    asyncio.run(main())
