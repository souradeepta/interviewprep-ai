"""
Auto-generated from 13-agent-communication.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Communication
# Learning objectives:
# - Understand synchronous and asynchronous communication patterns
# - Implement message queues and structured protocols
# ======================================================================

import os
import json
import queue
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Dict, List
from collections import defaultdict

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for agent communication!")


# ======================================================================
# ## Level 1: Synchronous Direct Communication
# Agents call each other directly (blocking).
# ======================================================================

class Agent:
    """Simple agent that can communicate with others."""
    def __init__(self, name: str):
        self.name = name
        self.client = Anthropic()
    
    def receive_message(self, message: str) -> str:
        """Receive and respond to message."""
        print(f"[{self.name}] Received: {message[:40]}...")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"Respond as {self.name}: {message}"
            }]
        )
        
        reply = response.content[0].text
        print(f"[{self.name}] Responded: {reply[:40]}...")
        return reply

class Team:
    """Team of agents that communicate directly."""
    def __init__(self):
        self.agents = {
            "Alice": Agent("Alice"),
            "Bob": Agent("Bob")
        }
    
    def ask_agent(self, agent_name: str, message: str) -> str:
        """Ask agent a question, get response synchronously."""
        return self.agents[agent_name].receive_message(message)
    
    def coordinate(self, task: str) -> Dict:
        """Alice asks Bob for help on a task."""
        print(f"\nTask: {task}")
        
        # Alice initiates
        alice_analysis = self.ask_agent("Alice", f"Analyze: {task}")
        
        # Alice asks Bob for input
        bob_perspective = self.ask_agent("Bob", f"Based on this: {alice_analysis[:100]}, what's your take?")
        
        return {
            "task": task,
            "alice_analysis": alice_analysis,
            "bob_perspective": bob_perspective
        }

# Test
team = Team()
result = team.coordinate("Should we expand to new markets?")
print(f"\nCoordination complete!")


# ======================================================================
# ## Level 2: Asynchronous Message Queue
# Agents send and receive messages via queue (non-blocking).
# ======================================================================

class QueueBasedAgent:
    """Agent that processes messages from queue."""
    def __init__(self, name: str):
        self.name = name
        self.client = Anthropic()
        self.inbox = queue.Queue()
        self.outbox_callbacks = {}  # Callbacks to other agents' inboxes
    
    def send_message(self, recipient_name: str, message: str):
        """Send message to another agent's inbox."""
        if recipient_name in self.outbox_callbacks:
            self.outbox_callbacks[recipient_name]({
                "from": self.name,
                "content": message,
                "timestamp": time.time()
            })
            print(f"[{self.name}] → {recipient_name}: {message[:40]}...")
    
    def process_inbox(self) -> List[Dict]:
        """Process all messages in inbox."""
        responses = []
        
        while not self.inbox.empty():
            msg = self.inbox.get()
            print(f"[{self.name}] Processing from {msg['from']}: {msg['content'][:40]}...")
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": f"You are {self.name}. Respond to: {msg['content']}"
                }]
            )
            
            reply = response.content[0].text
            responses.append({"from": msg["from"], "reply": reply})
        
        return responses

class AsyncTeam:
    """Team with async message passing."""
    def __init__(self):
        self.agents = {
            "Alice": QueueBasedAgent("Alice"),
            "Bob": QueueBasedAgent("Bob")
        }
        
        # Wire up message routing
        self.agents["Alice"].outbox_callbacks["Bob"] = self.agents["Bob"].inbox.put
        self.agents["Bob"].outbox_callbacks["Alice"] = self.agents["Alice"].inbox.put
    
    def coordinate_async(self, initial_task: str):
        """Async coordination: send messages, then process."""
        print(f"\nAsync Task: {initial_task}")
        
        # Alice sends message
        self.agents["Alice"].send_message("Bob", initial_task)
        
        # Bob processes and responds
        bob_responses = self.agents["Bob"].process_inbox()
        
        # Bob sends back
        for resp in bob_responses:
            self.agents["Bob"].send_message("Alice", resp["reply"][:100])
        
        # Alice processes
        alice_responses = self.agents["Alice"].process_inbox()
        
        return {"alice_responses": alice_responses, "bob_responses": bob_responses}

# Test
team = AsyncTeam()
result = team.coordinate_async("What's the best AI framework to learn?")
print(f"\nAsync coordination complete!")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Structured Message Protocol
# ======================================================================

class StructuredAgent:
    """Agent with structured message protocol (JSON schemas)."""
    def __init__(self, name: str):
        self.name = name
        self.client = Anthropic()
    
    def create_structured_message(self, msg_type: str, **fields) -> Dict:
        """Create message with specific schema."""
        return {
            "message_id": f"{self.name}_{int(time.time()*1000)}",
            "from": self.name,
            "type": msg_type,
            "fields": fields,
            "timestamp": time.time()
        }
    
    def process_structured_message(self, message: Dict) -> Dict:
        """Process message of specific type."""
        msg_type = message["type"]
        fields = message["fields"]
        
        print(f"[{self.name}] Processing {msg_type}")
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"As {self.name}, handle {msg_type}: {json.dumps(fields)}"
            }]
        )
        
        return self.create_structured_message(
            f"{msg_type}_response",
            result=response.content[0].text
        )

# Test
analyst = StructuredAgent("Analyst")
query = analyst.create_structured_message(
    "analysis_request",
    topic="market trends",
    depth="detailed"
)
print(f"\nMessage: {json.dumps(query, indent=2, default=str)[:200]}...")

response = analyst.process_structured_message(query)
print(f"Response type: {response['type']}")


# ======================================================================
# ### Example 2: Retry Logic with Timeouts
# ======================================================================

class ResilientAgent:
    """Agent with retry and timeout logic."""
    def __init__(self, name: str, max_retries: int = 3, timeout: float = 5.0):
        self.name = name
        self.client = Anthropic()
        self.max_retries = max_retries
        self.timeout = timeout
    
    def send_with_retry(self, message: str) -> Dict:
        """Send message with automatic retry on failure."""
        for attempt in range(self.max_retries):
            try:
                print(f"[{self.name}] Attempt {attempt+1}/{self.max_retries}")
                
                # Simulate timeout for first attempt
                if attempt < 1:
                    time.sleep(0.5)  # Simulated delay
                
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=256,
                    messages=[{"role": "user", "content": message}]
                )
                
                return {
                    "success": True,
                    "response": response.content[0].text,
                    "attempts": attempt + 1
                }
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1
                    }
                time.sleep(0.5)  # Wait before retry

# Test
agent = ResilientAgent("Resilient", max_retries=2)
result = agent.send_with_retry("What is the meaning of life?")
print(f"\nResult: success={result['success']}, attempts={result['attempts']}")
if result["success"]:
    print(f"Response: {result['response'][:60]}...")


# ======================================================================
# ### Example 3: Broadcast Communication
# ======================================================================

class CommulityAgent:
    """Agent that can broadcast to multiple recipients."""
    def __init__(self, name: str):
        self.name = name
        self.client = Anthropic()
        self.subscribers = []  # Agents listening to this agent
    
    def subscribe(self, agent):
        """Add subscriber to receive broadcasts."""
        self.subscribers.append(agent)
    
    def broadcast(self, message: str) -> Dict:
        """Send message to all subscribers."""
        print(f"\n[{self.name}] Broadcasting: {message[:40]}...")
        
        responses = []
        for subscriber in self.subscribers:
            response = subscriber.receive_broadcast(self.name, message)
            responses.append({"from": subscriber.name, "response": response})
        
        return responses
    
    def receive_broadcast(self, sender_name: str, message: str) -> str:
        """Receive broadcast from another agent."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"As {self.name}, react to {sender_name}'s message: {message}"
            }]
        )
        return response.content[0].text

# Test: One broadcaster, multiple listeners
broadcaster = CommulityAgent("NewsAgent")
listener1 = CommulityAgent("ListenerA")
listener2 = CommulityAgent("ListenerB")

broadcaster.subscribe(listener1)
broadcaster.subscribe(listener2)

responses = broadcaster.broadcast("Breaking: New AI breakthrough announced!")
print(f"\nBroadcast received by {len(responses)} agents")


# ======================================================================
# ## Key Takeaways
# 1. **Synchronous vs Asynchronous.** Synchronous (direct calls) is simpler but blocking. Asynchronous (queues) scales better and tolerates delays.
# 2. **Message passing enables coordination.** Instead of shared state, agents communicate via messages. Decouples agents, enables parallel execution.
# ======================================================================
