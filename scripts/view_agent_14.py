"""Fetch and display agent 14 instructions to find bad hangup pattern."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import asyncio
from db.storage.agents import AgentStorage

async def main():
    storage = AgentStorage()
    agent = await storage.get_agent_by_id(14)
    
    if not agent:
        print("Agent 14 not found!")
        return
    
    print("=" * 80)
    print("AGENT 14 INSTRUCTIONS")
    print("=" * 80)
    
    print("\n### Name:", agent.get('name'))
    print("\n### Instructions:")
    print("-" * 40)
    instructions = agent.get('instructions', '') or ''
    print(instructions)
    
    print("\n" + "=" * 80)
    print("### Outbound Starter Prompt:")
    print("-" * 40)
    starter = agent.get('outbound_starter_prompt', '') or ''
    print(starter)
    
    # Search for hangup patterns
    print("\n" + "=" * 80)
    print("### Hangup-related lines:")
    print("-" * 40)
    combined = instructions + "\n" + starter
    for i, line in enumerate(combined.split('\n')):
        if 'hangup' in line.lower():
            print(f"Line {i}: {line}")

if __name__ == '__main__':
    asyncio.run(main())
