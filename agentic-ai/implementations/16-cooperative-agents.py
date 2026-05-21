"""
Auto-generated from 16-cooperative-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Cooperative Agents
# Objectives: Peer collaboration, blackboard pattern, negotiation, conflict resolution, deadlock avoidance
# ======================================================================

import asyncio
from typing import Dict, List, Any
from enum import Enum
import time

# Level 1: Basic Cooperative Agents with Shared Blackboard

class SimpleAgent:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    async def contribute(self, goal: str) -> Dict:
        """Contribute solution from agent's specialty."""
        await asyncio.sleep(0.2)
        return {
            'agent': self.name,
            'specialty': self.specialty,
            'contribution': f'Insight from {self.specialty}'
        }

class SimpleBlackboard:
    def __init__(self):
        self.state = {}
    
    def write(self, agent_name: str, contribution: str):
        self.state[agent_name] = contribution
    
    def read(self, agent_name: str) -> str:
        return self.state.get(agent_name, None)
    
    def get_all(self) -> Dict:
        return self.state.copy()

class SimpleCooperativeSystem:
    def __init__(self, agents: List[SimpleAgent]):
        self.agents = agents
        self.blackboard = SimpleBlackboard()
    
    async def collaborate(self, goal: str):
        print(f'Shared Goal: "{goal}"')
        print(f'Agents: {[a.name for a in self.agents]}\n')
        
        # Phase 1: All agents contribute in parallel
        tasks = [agent.contribute(goal) for agent in self.agents]
        contributions = await asyncio.gather(*tasks)
        
        # Phase 2: Write to blackboard
        print('Blackboard Updates:')
        for contrib in contributions:
            self.blackboard.write(contrib['agent'], contrib['contribution'])
            print(f"  {contrib['agent']}: {contrib['contribution']}")
        
        # Phase 3: Read from blackboard
        print(f'\nFinal Shared State:')
        for agent_name, contribution in self.blackboard.get_all().items():
            print(f"  {agent_name}: {contribution}")

# Test Level 1
async def test_level1():
    agents = [
        SimpleAgent('Alice', 'data'),
        SimpleAgent('Bob', 'engineering'),
        SimpleAgent('Carol', 'product')
    ]
    system = SimpleCooperativeSystem(agents)
    await system.collaborate('Build real-time analytics')

await test_level1()


# Level 2: Cooperative Agents with Negotiation and Timeouts

class NegotiatingAgent:
    def __init__(self, name: str, specialty: str, needs: List[str] = None):
        self.name = name
        self.specialty = specialty
        self.needs = needs or []
        self.acquired_data = {}
    
    async def negotiate_with_agent(self, other_agent: 'NegotiatingAgent', request: str) -> bool:
        """Negotiate with another agent."""
        try:
            # Simulate negotiation with timeout
            async def negotiation():
                await asyncio.sleep(0.1)
                return True
            
            result = await asyncio.wait_for(negotiation(), timeout=0.5)
            if result:
                print(f"  {self.name} ← {other_agent.name}: agreed on '{request}'")
                self.acquired_data[other_agent.name] = request
            return result
        except asyncio.TimeoutError:
            print(f"  {self.name} ✗ {other_agent.name}: timeout on '{request}'")
            return False
    
    async def negotiate_phase(self, other_agents: List['NegotiatingAgent']):
        """Negotiate with other agents."""
        print(f"  {self.name}: negotiating...")
        tasks = [
            self.negotiate_with_agent(agent, f"data from {agent.specialty}")
            for agent in other_agents if agent.name != self.name
        ]
        await asyncio.gather(*tasks)
    
    async def solve(self, goal: str) -> Dict:
        """Solve task with acquired data."""
        await asyncio.sleep(0.2)
        data_str = ', '.join(self.acquired_data.values()) if self.acquired_data else 'no data'
        return {
            'agent': self.name,
            'result': f'{self.specialty} solution using {data_str}'
        }

class NegotiatingCooperativeSystem:
    def __init__(self, agents: List[NegotiatingAgent]):
        self.agents = agents
        self.blackboard = {}
    
    async def run(self, goal: str):
        print(f'Goal: "{goal}"\n')
        
        # Phase 1: Negotiation
        print('Phase 1: Negotiation')
        tasks = [
            agent.negotiate_phase([a for a in self.agents if a.name != agent.name])
            for agent in self.agents
        ]
        await asyncio.gather(*tasks)
        
        # Phase 2: Execution
        print('\nPhase 2: Execution')
        tasks = [agent.solve(goal) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        # Phase 3: Blackboard updates
        print('\nResults on Blackboard:')
        for result in results:
            self.blackboard[result['agent']] = result['result']
            print(f"  {result['agent']}: {result['result']}")

# Test Level 2
async def test_level2():
    agents = [
        NegotiatingAgent('Alice', 'data', needs=['metadata']),
        NegotiatingAgent('Bob', 'engineering', needs=['schema']),
        NegotiatingAgent('Carol', 'strategy', needs=['metrics'])
    ]
    system = NegotiatingCooperativeSystem(agents)
    await system.run('Design production system')

await test_level2()


# Example 1: Conflict Resolution via Voting

class VotingAgent:
    def __init__(self, name: str, specialty: str, confidence: float = 0.5):
        self.name = name
        self.specialty = specialty
        self.confidence = confidence
    
    def propose_solution(self, goal: str) -> Dict:
        """Propose solution with confidence score."""
        return {
            'agent': self.name,
            'proposal': f'{self.specialty} approach',
            'confidence': self.confidence
        }

class VotingCooperativeSystem:
    def __init__(self, agents: List[VotingAgent]):
        self.agents = agents
    
    def resolve_by_confidence(self, proposals: List[Dict]) -> Dict:
        """Choose proposal from highest-confidence agent."""
        if not proposals:
            return None
        best = max(proposals, key=lambda p: p['confidence'])
        return best
    
    async def run(self, goal: str):
        print(f'Goal: "{goal}"\n')
        
        # All agents propose solutions
        proposals = [agent.propose_solution(goal) for agent in self.agents]
        
        print('Proposals:')
        for p in proposals:
            confidence_bar = '█' * int(p['confidence'] * 10) + '░' * (10 - int(p['confidence'] * 10))
            print(f"  {p['agent']:6} | {confidence_bar} | {p['proposal']}")
        
        # Resolve conflict
        winner = self.resolve_by_confidence(proposals)
        print(f"\nResolution: Accept {winner['agent']}'s proposal (confidence: {winner['confidence']})")

# Test Example 1
async def test_example1():
    agents = [
        VotingAgent('Alice', 'ML', confidence=0.95),
        VotingAgent('Bob', 'Infra', confidence=0.70),
        VotingAgent('Carol', 'Product', confidence=0.85)
    ]
    system = VotingCooperativeSystem(agents)
    await system.run('Choose recommendation algorithm')

await test_example1()


# Example 2: Deadlock Detection and Resolution

class DeadlockDetector:
    def __init__(self):
        self.wait_graph = {}  # agent -> list of agents it's waiting for
        self.start_time = {}
    
    def add_wait(self, waiter: str, waiting_for: str):
        """Record that waiter is waiting for waiting_for."""
        if waiter not in self.wait_graph:
            self.wait_graph[waiter] = []
        self.wait_graph[waiter].append(waiting_for)
        self.start_time[f"{waiter}->{waiting_for}"] = time.time()
    
    def detect_deadlock(self, timeout: float = 1.0) -> List[tuple]:
        """Detect if any wait has exceeded timeout (indicates deadlock)."""
        current_time = time.time()
        deadlocks = []
        
        for edge, start in self.start_time.items():
            if current_time - start > timeout:
                waiter, waiting_for = edge.split('->')
                deadlocks.append((waiter, waiting_for))
        
        return deadlocks

class DeadlockResistantAgent:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    async def solve_with_timeout(self, goal: str, timeout: float = 0.5) -> Dict:
        """Solve with timeout; if exceeded, return partial result."""
        try:
            async def solve():
                await asyncio.sleep(0.3)
                return {'status': 'complete', 'result': f'{self.specialty} solution'}
            
            result = await asyncio.wait_for(solve(), timeout=timeout)
            return {'agent': self.name, **result}
        except asyncio.TimeoutError:
            print(f"  ⚠️  {self.name}: timeout, returning partial result")
            return {
                'agent': self.name,
                'status': 'timeout',
                'result': f'{self.specialty} partial solution'
            }

class DeadlockAwareSystem:
    def __init__(self, agents: List[DeadlockResistantAgent]):
        self.agents = agents
        self.detector = DeadlockDetector()
    
    async def run_with_deadlock_detection(self, goal: str):
        print(f'Goal: "{goal}"\n')
        
        # Execute with timeout
        tasks = [agent.solve_with_timeout(goal, timeout=0.5) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        # Check for deadlock
        timeouts = [r for r in results if r.get('status') == 'timeout']
        
        print(f'\nResults:')
        for r in results:
            status_symbol = '✓' if r.get('status') == 'complete' else '⏱'
            print(f"  {status_symbol} {r['agent']}: {r['result']}")
        
        if timeouts:
            print(f"\n⚠️  Deadlock detected: {len(timeouts)} agents timed out")
            print("   → Fallback: use partial results and escalate")

# Test Example 2
async def test_example2():
    agents = [
        DeadlockResistantAgent('Alice', 'data'),
        DeadlockResistantAgent('Bob', 'engineering'),
        DeadlockResistantAgent('Carol', 'strategy')
    ]
    system = DeadlockAwareSystem(agents)
    await system.run_with_deadlock_detection('Solve complex problem')

await test_example2()


# Example 3: Multi-Team Cooperation (Teams as Super-Agents)

class CooperativeTeam:
    def __init__(self, name: str, members: List[SimpleAgent]):
        self.name = name
        self.members = members
    
    async def solve_as_team(self, goal: str) -> Dict:
        """Team internally collaborates, returns single result."""
        tasks = [member.contribute(goal) for member in self.members]
        contributions = await asyncio.gather(*tasks)
        
        # Team synthesizes
        synthesis = f"Team {self.name}: " + "; ".join(
            [c['contribution'] for c in contributions]
        )
        
        return {
            'team': self.name,
            'result': synthesis,
            'member_count': len(self.members)
        }

class MultiTeamCooperativeSystem:
    def __init__(self, teams: List[CooperativeTeam]):
        self.teams = teams
    
    async def run(self, goal: str):
        print(f'Goal: "{goal}"\n')
        print(f'Teams: {[t.name for t in self.teams]}\n')
        
        # Each team solves internally
        tasks = [team.solve_as_team(goal) for team in self.teams]
        team_results = await asyncio.gather(*tasks)
        
        print('Team Results:')
        for tr in team_results:
            print(f"  {tr['team']} ({tr['member_count']} members):")
            print(f"    {tr['result']}")
        
        print(f"\nFinal: Aggregated insights from {len(self.teams)} teams")

# Test Example 3
async def test_example3():
    engineering_team = CooperativeTeam('Engineering', [
        SimpleAgent('Alice', 'backend'),
        SimpleAgent('Bob', 'frontend')
    ])
    data_team = CooperativeTeam('Data', [
        SimpleAgent('Carol', 'analytics'),
        SimpleAgent('David', 'ml')
    ])
    
    system = MultiTeamCooperativeSystem([engineering_team, data_team])
    await system.run('Scale platform for 1M users')

await test_example3()


# ======================================================================
# ## Key Takeaways
# **Core Cooperative Pattern:**
# 1. Agents are peers (no hierarchy)
# 2. Shared blackboard = central state
# ======================================================================
