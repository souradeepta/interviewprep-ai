"""
Auto-generated from 11-memory-types.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Memory Types
# Objectives: Short-term vs long-term memory, episodic memory (action logs), semantic memory (knowledge bases), retrieval strategies, consistency management
# ======================================================================

from collections import deque
from datetime import datetime
from typing import Dict, List

# Level 1: Basic Memory Types

class BasicMemory:
    def __init__(self, short_term_size=10):
        # Short-term: immediate context (limited capacity)
        self.short_term = deque(maxlen=short_term_size)

        # Long-term: persistent history
        self.long_term = []

        # Episodic: action logs
        self.episodic = []

        # Semantic: facts and knowledge
        self.semantic = {
            "Paris": "capital of France",
            "London": "capital of UK"
        }

    def observe(self, event: str):
        '''Add observation to short-term memory.'''
        self.short_term.append(f"[{datetime.now().strftime('%H:%M:%S')}] {event}")

    def remember_action(self, action: str, outcome: str):
        '''Log action to episodic memory.'''
        self.episodic.append({
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.now()
        })

    def get_fact(self, key: str):
        '''Lookup fact from semantic memory.'''
        return self.semantic.get(key, "Unknown")

    def get_short_term_context(self) -> str:
        '''Retrieve current short-term context.'''
        return "\n".join(self.short_term)

# Test Level 1
print('Level 1 - Basic Memory Types:\n')
memory = BasicMemory(short_term_size=5)

# Observe events
memory.observe("User asked: What is Paris?")
memory.observe("Agent queried knowledge base")
memory.remember_action("query_kb", "Found: capital of France")

# Query
print(f"Fact lookup: {memory.get_fact('Paris')}")
print(f"Short-term context:\n{memory.get_short_term_context()}\n")
print(f"Episodic log: {len(memory.episodic)} actions logged\n")


# Level 2: Retrieval-Augmented Memory

class RetrievalAugmentedMemory:
    def __init__(self, max_context_tokens=100):
        self.short_term = deque(maxlen=5)
        self.long_term = []
        self.episodic = []
        self.semantic = {}
        self.max_context_tokens = max_context_tokens
        self.context_tokens = 0

    def add_short_term(self, text: str):
        '''Add to immediate context. If full, move to long-term.'''
        tokens = len(text.split())
        self.context_tokens += tokens

        if self.context_tokens > self.max_context_tokens:
            # Compress and move old items
            if len(self.short_term) > 0:
                self.long_term.append(f"[Summary] Compressed {len(self.short_term)} items")
            self.short_term.clear()
            self.context_tokens = tokens

        self.short_term.append(text)

    def retrieve_from_long_term(self, query: str) -> List[str]:
        '''Semantic search in long-term memory.'''
        # Simple keyword matching (in practice: embedding similarity)
        results = [item for item in self.long_term if any(word in item.lower() for word in query.lower().split())]
        return results[:3]  # Top 3 results

    def build_context(self, query: str = None) -> str:
        '''Build complete context: short-term + retrieved long-term.'''
        context = "SHORT-TERM:\n" + "\n".join(self.short_term)

        if query and self.long_term:
            retrieved = self.retrieve_from_long_term(query)
            if retrieved:
                context += "\nLONG-TERM (RETRIEVED):\n" + "\n".join(retrieved)

        return context

    def log_action(self, action: str, params: dict, outcome: str):
        '''Log to episodic memory for learning.'''
        self.episodic.append({
            "action": action,
            "params": params,
            "outcome": outcome,
            "success": outcome != "error"
        })

# Test Level 2
print('Level 2 - Retrieval-Augmented Memory:\n')
memory = RetrievalAugmentedMemory(max_context_tokens=50)

# Add observations
memory.add_short_term("User: Explain AI")
memory.add_short_term("Agent: AI is machine learning")
memory.add_short_term("User: How does it learn?")

# Log actions
memory.log_action("respond", {"topic": "AI"}, "success")
memory.log_action("retrieve", {"query": "machine learning"}, "success")

# Retrieve with context
context = memory.build_context(query="machine learning")
print(f"Built context ({memory.context_tokens} tokens):")
print(context)
print(f"\nEpisodic log: {len(memory.episodic)} actions\n")


# Example 1: Learning from Episodic Memory

class LearningAgent:
    def __init__(self):
        self.episodic = []
        self.learned_strategies = {}

    def execute_action(self, action: str, params: dict) -> bool:
        '''Execute action, log result.'''
        # Simulate: some actions succeed, some fail
        success = hash(f"{action}{params}") % 3 != 0  # Pseudo-random

        self.episodic.append({
            "action": action,
            "params": params,
            "success": success,
            "timestamp": datetime.now()
        })

        return success

    def analyze_past_actions(self, action: str) -> Dict:
        '''Analyze success rate for action type.'''
        relevant = [ep for ep in self.episodic if ep["action"] == action]

        if not relevant:
            return {"history": "none"}

        successes = sum(1 for ep in relevant if ep["success"])
        success_rate = successes / len(relevant)

        return {
            "attempts": len(relevant),
            "successes": successes,
            "success_rate": success_rate,
            "recommendation": "keep using" if success_rate > 0.6 else "try different approach"
        }

    def reflect_and_improve(self):
        '''Learn from past actions.'''
        actions = set(ep["action"] for ep in self.episodic)

        for action in actions:
            analysis = self.analyze_past_actions(action)
            self.learned_strategies[action] = analysis["recommendation"]

# Test Example 1
print('Example 1 - Learning from Episodic Memory:\n')
agent = LearningAgent()

# Execute actions
for _ in range(10):
    agent.execute_action("call_api", {"endpoint": "/search"})
    agent.execute_action("parse_response", {})

# Reflect
agent.reflect_and_improve()

for action, strategy in agent.learned_strategies.items():
    analysis = agent.analyze_past_actions(action)
    print(f"{action}: {analysis['success_rate']:.0%} success → {strategy}")
print()


# Example 2: Semantic Memory and Grounding

class GroundedAgent:
    def __init__(self):
        self.semantic = {
            "valid_endpoints": ["/search", "/api/data", "/analyze"],
            "error_codes": {
                "404": "Not found",
                "500": "Server error",
                "timeout": "Request timed out"
            },
            "facts": {
                "max_retries": 3,
                "timeout_seconds": 30
            }
        }

    def is_valid_endpoint(self, endpoint: str) -> bool:
        '''Check if endpoint is allowed.'''
        return endpoint in self.semantic["valid_endpoints"]

    def explain_error(self, error_code: str) -> str:
        '''Get error explanation from semantic memory.'''
        return self.semantic["error_codes"].get(error_code, "Unknown error")

    def get_config(self, key: str):
        '''Get configuration fact.'''
        return self.semantic["facts"].get(key)

    def decide_action(self, endpoint: str, error: str = None) -> str:
        '''Grounded decision-making using semantic memory.'''
        # Check endpoint validity
        if not self.is_valid_endpoint(endpoint):
            return f"ERROR: Endpoint {endpoint} not in allowed list"

        # Explain error if present
        if error:
            explanation = self.explain_error(error)
            max_retries = self.get_config("max_retries")
            return f"Error ({error}): {explanation}. Retry up to {max_retries} times."

        return f"OK: Endpoint {endpoint} is valid and allowed."

# Test Example 2
print('Example 2 - Semantic Memory for Grounding:\n')
agent = GroundedAgent()

# Check endpoint validity
print(agent.decide_action("/search"))
print(agent.decide_action("/invalid"))
print()

# Error explanation
print(agent.decide_action("/search", error="500"))
print(agent.decide_action("/search", error="404"))
print()


# Example 3: Maintaining Memory Consistency

class ConsistentMemory:
    def __init__(self):
        self.semantic = {}  # Source of truth
        self.semantic_history = []  # Version history
        self.episodic = []

    def update_fact(self, fact: str, value: str) -> None:
        '''Update semantic memory with versioning.'''
        # Record old value
        old_value = self.semantic.get(fact)

        # Update
        self.semantic[fact] = value

        # Log to history
        self.semantic_history.append({
            "fact": fact,
            "old_value": old_value,
            "new_value": value,
            "timestamp": datetime.now()
        })

        print(f"Updated '{fact}': {old_value} → {value}")

    def query_fact(self, fact: str, as_of_date=None) -> str:
        '''Query fact, optionally at historical timestamp.'''
        if as_of_date:
            # Find value at that date
            relevant = [h for h in self.semantic_history if h["timestamp"] <= as_of_date and h["fact"] == fact]
            if relevant:
                return relevant[-1]["new_value"]

        return self.semantic.get(fact, "Unknown")

    def log_decision_based_on_fact(self, fact: str, decision: str) -> None:
        '''Record decision tied to fact value.'''
        current_value = self.semantic.get(fact)
        self.episodic.append({
            "fact": fact,
            "fact_value_at_time": current_value,
            "decision": decision,
            "timestamp": datetime.now()
        })

# Test Example 3
print('Example 3 - Memory Consistency:\n')
memory = ConsistentMemory()

# Initial fact
memory.update_fact("API_TIMEOUT", "30")
memory.log_decision_based_on_fact("API_TIMEOUT", "Wait 30 seconds before retry")

# Fact changes
memory.update_fact("API_TIMEOUT", "60")
memory.log_decision_based_on_fact("API_TIMEOUT", "Wait 60 seconds before retry")

# Query historical value
print(f"\nCurrent timeout: {memory.query_fact('API_TIMEOUT')} seconds")
print(f"\nDecisions made with old timeout value:")
for ep in memory.episodic:
    print(f"  {ep['decision']} (timeout was {ep['fact_value_at_time']}s)")



# ======================================================================
# ## Key Takeaways
# **Memory Types and Their Roles:**
# - **Short-term (context window):** Limited capacity, immediate access, holds current reasoning state
# - **Long-term (external storage):** Unlimited capacity, retrieval latency, holds historical information
# ======================================================================
