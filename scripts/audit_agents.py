"""Simplified audit for agent 4 vs 14 hangup tool differences."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from db.storage.agents import AgentStorage
import asyncio

async def main():
    storage = AgentStorage()
    
    for agent_id in [4, 14]:
        agent = await storage.get_agent_by_id(agent_id)
        if not agent:
            print(f"Agent {agent_id}: NOT FOUND")
            continue
            
        print(f"\n=== AGENT {agent_id} ===")
        print(f"Name: {agent.get('name')}")
        print(f"Tenant: {agent.get('tenant_id')}")
        
        # Instructions hangup analysis
        instr = agent.get('instructions', '') or ''
        print(f"Instructions length: {len(instr)}")
        
        # Find hangup mentions with context
        for pattern in ['hangup_call', 'hangup', 'end call', 'terminate']:
            count = instr.lower().count(pattern.lower())
            if count > 0:
                print(f"  '{pattern}': {count} times")
                # Show first occurrence context
                idx = instr.lower().find(pattern.lower())
                if idx != -1:
                    start = max(0, idx - 30)
                    end = min(len(instr), idx + len(pattern) + 50)
                    print(f"    Context: ...{instr[start:end]}...")
        
        # Show starter prompt (often where hangup is instructed)
        starter = agent.get('outbound_starter_prompt', '')
        if starter:
            print(f"Starter prompt ({len(starter)} chars):")
            print(f"  {starter[:500]}")

if __name__ == '__main__':
    asyncio.run(main())
