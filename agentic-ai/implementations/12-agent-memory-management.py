"""
Auto-generated from 12-agent-memory-management.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Agent Memory Management
# Learning objectives:
# - Understand three memory layers: working, long-term, episodic
# - Implement sliding window and summarization strategies
# ======================================================================

import os
import json
from collections import deque
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("Setup complete. Ready for memory management!")


# ======================================================================
# ## Level 1: Sliding Window Memory
# Simplest approach: keep only last N messages.
# ======================================================================

class SlidingWindowAgent:
    """Agent with fixed-size message history (sliding window)"""
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.client = Anthropic()
    
    def chat(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=list(self.history)
        )
        
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        
        print(f"Memory: {len(self.history)}/{self.window_size}")
        return answer

# Test
agent = SlidingWindowAgent(window_size=4)
print("Q1:", agent.chat("What's your name?")[:50])
print("\nQ2:", agent.chat("What did you say your name was?")[:50])


# ======================================================================
# ## Level 2: Multi-Layer Memory
# Three layers: working memory (recent), long-term (facts), episodic (log).
# ======================================================================

class MultiLayerAgent:
    def __init__(self, working_size: int = 6):
        self.working = deque(maxlen=working_size)
        self.long_term = {}  # {topic: [facts]}
        self.episodic = []  # [{query, response, timestamp}]
        self.client = Anthropic()
    
    def learn_fact(self, topic: str, fact: str):
        if topic not in self.long_term:
            self.long_term[topic] = []
        self.long_term[topic].append(fact)
        print(f"Learned [{topic}]: {fact}")
    
    def get_context(self, query: str) -> str:
        relevant = []
        for topic, facts in self.long_term.items():
            if any(w in query.lower() for w in topic.lower().split()):
                relevant.extend(facts)
        return "\n".join(relevant) if relevant else None
    
    def chat(self, user_msg: str) -> str:
        context = self.get_context(user_msg)
        system_msg = f"Context: {context}" if context else None
        
        self.working.append({"role": "user", "content": user_msg})
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            system=system_msg,
            messages=list(self.working)
        )
        
        answer = response.content[0].text
        self.working.append({"role": "assistant", "content": answer})
        
        self.episodic.append({
            "query": user_msg[:50],
            "response": answer[:50]
        })
        
        return answer

# Test
agent = MultiLayerAgent()
agent.learn_fact("user_preferences", "User prefers concise answers")
agent.learn_fact("user_preferences", "User works in AI")

print("Q1:", agent.chat("Explain neural networks briefly")[:60])
print("\nQ2:", agent.chat("What's one key application?")[:60])
print(f"\nEpisodic log: {len(agent.episodic)} interactions")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Memory Summarization
# ======================================================================

class SummarizingAgent:
    """Compress old messages into summaries"""
    def __init__(self, summarize_every: int = 5):
        self.history = []
        self.summarize_every = summarize_every
        self.summary = None
        self.client = Anthropic()
    
    def maybe_summarize(self):
        if len(self.history) >= self.summarize_every:
            # Compress oldest 5 messages into summary
            to_summarize = self.history[:self.summarize_every]
            text = "\n".join([m.get("content", "") for m in to_summarize])
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": f"Summarize in 1-2 sentences: {text[:200]}"
                }]
            )
            
            self.summary = response.content[0].text
            self.history = self.history[self.summarize_every:]
            print(f"Summarized: {self.summary[:50]}...")
    
    def chat(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})
        
        messages = []
        if self.summary:
            messages.append({"role": "user", "content": f"[Prior: {self.summary}]"})
        messages.extend(self.history)
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=messages
        )
        
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        
        self.maybe_summarize()
        return answer

# Test
agent = SummarizingAgent(summarize_every=2)
for i in range(4):
    print(f"Turn {i+1}:", agent.chat(f"Question {i+1}")[:40])


# ======================================================================
# ### Example 2: Semantic Memory with Retrieval
# ======================================================================

class SemanticMemoryAgent:
    """Store and retrieve facts semantically"""
    def __init__(self):
        self.history = []
        self.facts = []  # [{text, topic}]
        self.client = Anthropic()
    
    def store_fact(self, fact: str, topic: str = "general"):
        self.facts.append({"text": fact, "topic": topic})
        print(f"Stored fact on {topic}")
    
    def retrieve_facts(self, query: str, top_k: int = 2) -> str:
        """Simple retrieval: match keywords"""
        query_words = query.lower().split()
        scored = []
        
        for fact in self.facts:
            score = sum(1 for w in query_words if w in fact["text"].lower())
            scored.append((fact["text"], score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return "\n".join([f for f, _ in scored[:top_k]])
    
    def chat(self, user_msg: str) -> str:
        facts = self.retrieve_facts(user_msg)
        context = f"[Facts: {facts}]\n" if facts else ""
        
        self.history.append({"role": "user", "content": context + user_msg})
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=self.history
        )
        
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        return answer

# Test
agent = SemanticMemoryAgent()
agent.store_fact("User works as a data scientist", "user_profile")
agent.store_fact("User is interested in ML model deployment", "interests")

print(agent.chat("What should I focus on for my data science role?")[:80])


# ======================================================================
# ### Example 3: Memory with Cost Tracking
# ======================================================================

class CostAwareMemoryAgent:
    """Track memory cost (tokens used)"""
    def __init__(self, max_tokens: int = 2000):
        self.history = deque()
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.metrics = {"queries": 0, "total_tokens": 0}
        self.client = Anthropic()
    
    def estimate_tokens(self, messages: list) -> int:
        """Rough estimate: 1 token per 4 chars"""
        total = sum(len(m.get("content", "")) for m in messages) // 4
        return max(50, total)  # Min 50 tokens
    
    def chat(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})
        
        # Estimate cost before query
        estimated_tokens = self.estimate_tokens(list(self.history))
        if estimated_tokens > self.max_tokens:
            print(f"⚠️  Context full ({estimated_tokens} tokens). Trimming...")
            self.history = deque(list(self.history)[-4:])  # Keep last 4
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=256,
            messages=list(self.history)
        )
        
        # Track actual tokens
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        self.current_tokens += tokens_used
        self.metrics["queries"] += 1
        self.metrics["total_tokens"] += tokens_used
        
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        
        print(f"Tokens: {tokens_used} (total: {self.metrics['total_tokens']})")
        return answer

# Test
agent = CostAwareMemoryAgent(max_tokens=1500)
for i in range(3):
    print(f"\nQuery {i+1}:")
    agent.chat(f"Tell me about memory management in AI agents - question {i+1}")

print(f"\nFinal metrics: {json.dumps(agent.metrics, indent=2)}")


# ======================================================================
# ## Key Takeaways
# 1. **Three-layer memory:** Working (recent context), Long-term (facts), Episodic (logs). Each serves different purpose.
# 2. **Sliding window is simplest:** Keep last N messages. When full, drop oldest. Works for short conversations.
# ======================================================================
