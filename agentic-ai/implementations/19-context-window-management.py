"""
Auto-generated from 19-context-window-management.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Context Window Management in Agents
# Learning objectives:
# - Understand why context management matters for long conversations
# - Implement sliding window to keep recent messages and discard old ones
# ======================================================================

# ======================================================================
# ## Setup
# ======================================================================

import os
import json
from datetime import datetime
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from anthropic import Anthropic

# Configuration
API_KEY = os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")
client = Anthropic(api_key=API_KEY)

print("✓ Setup complete: anthropic SDK ready")


# ======================================================================
# ## Level 1: Basic Implementation
# Core concept: Sliding window—keep recent N messages, discard old ones
# ======================================================================

class SimpleSlidingWindow:
    """Keep only last N messages, discard older ones"""
    
    def __init__(self, window_size: int = 10):
        self.window = deque(maxlen=window_size)  # Auto-discards old when full
        self.all_messages_count = 0  # Track total for stats
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to window"""
        self.window.append({"role": role, "content": content})
        self.all_messages_count += 1
    
    def get_context(self) -> list:
        """Get messages to send to LLM"""
        return list(self.window)
    
    def stats(self) -> dict:
        return {
            "messages_total": self.all_messages_count,
            "messages_in_window": len(self.window),
            "discarded": self.all_messages_count - len(self.window),
            "compression_ratio": f"{100 * len(self.window) / self.all_messages_count:.1f}%"
        }

# Demo: simulate long conversation
manager = SimpleSlidingWindow(window_size=5)

for i in range(20):
    manager.add_message("user", f"Question {i+1}?")
    manager.add_message("assistant", f"Answer to question {i+1}")

print(f"After 20 exchanges:")
print(json.dumps(manager.stats(), indent=2))
print(f"\nCurrent context (what LLM sees):")
for msg in manager.get_context():
    print(f"  {msg['role'].upper()}: {msg['content'][:40]}...")


# ======================================================================
# ## Level 2: Advanced Implementation
# Hierarchical memory: recent messages + compressed history + importance ranking
# ======================================================================

@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    importance: float = 1.0  # 1.0 = normal, >1.0 = important

class HierarchicalMemoryManager:
    """Keep recent + summarized + indexed older messages"""
    
    def __init__(self, client, recent_count: int = 5):
        self.client = client
        self.recent_count = recent_count
        self.messages = []  # All messages (with metadata)
        self.summaries = []  # Compressed history blocks
        self.important_facts = []  # Extracted key facts
    
    def add_message(self, role: str, content: str, importance: float = 1.0) -> None:
        """Add message with importance score"""
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            importance=importance
        )
        self.messages.append(msg)
        
        # Extract important facts from important messages
        if importance > 1.0:
            self.important_facts.append(content[:100])
    
    def should_compress(self) -> bool:
        """Check if we have enough old messages to compress"""
        return len(self.messages) > self.recent_count + 10
    
    def compress_history(self) -> str:
        """Create summary of old messages (without API call for demo)"""
        # In production: call client.messages.create with summarization prompt
        messages_to_compress = self.messages[:-self.recent_count]
        
        # Simulate summarization
        summary = f"Compressed {len(messages_to_compress)} old messages. " \
                  f"Key facts: {', '.join(self.important_facts[-3:])}"
        
        # Keep only recent messages (discard old)
        self.messages = self.messages[-self.recent_count:]
        self.summaries.append(summary)
        
        return summary
    
    def get_context(self) -> list:
        """Build context for LLM: summaries + recent messages"""
        context = []
        
        # Add compressed history as system message if available
        if self.summaries:
            context.append({
                "role": "user",
                "content": f"[Earlier context: {self.summaries[-1]}]"
            })
        
        # Add recent messages
        context.extend([
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ])
        
        return context
    
    def query(self, user_message: str, importance: float = 1.0) -> str:
        """Query and compress history if needed"""
        self.add_message("user", user_message, importance)
        
        # Auto-compress if needed
        if self.should_compress():
            print(f"[Compressing history: {len(self.messages)} → {self.recent_count} recent + summary]")
            self.compress_history()
        
        # For demo, return placeholder (avoids API calls)
        response = f"Response to: {user_message[:50]}..."
        self.add_message("assistant", response)
        
        return response
    
    def stats(self) -> dict:
        return {
            "total_messages": len(self.messages),
            "summaries": len(self.summaries),
            "important_facts": len(self.important_facts),
            "messages_in_context": len(self.get_context())
        }

# Demo: long conversation with importance marking
manager = HierarchicalMemoryManager(client, recent_count=5)

for i in range(30):
    # Mark every 10th message as important
    importance = 2.0 if i % 10 == 0 else 1.0
    manager.query(f"Question {i+1}?", importance=importance)

print("After 30 exchanges:")
print(json.dumps(manager.stats(), indent=2))
print(f"\nContext sent to LLM has {len(manager.get_context())} items (recent + summaries)")


# ======================================================================
# ## Level 3: Real-World Examples
# ### Example 1: Token-Budget-Aware Context Manager
# ======================================================================

class TokenBudgetContextManager:
    """Manage context by token budget, not message count"""
    
    def __init__(self, client, max_context_tokens: int = 4000, reserve_tokens: int = 500):
        self.client = client
        self.max_tokens = max_context_tokens
        self.reserve = reserve_tokens  # Leave room for response
        self.messages = []
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars = 1 token"""
        return len(text) // 4
    
    def get_context_tokens(self) -> int:
        """Count total tokens in current context"""
        total = 0
        for msg in self.messages:
            total += self.estimate_tokens(msg["content"]) + 4  # 4 tokens overhead
        return total
    
    def add_message(self, role: str, content: str) -> None:
        """Add message and trim if over budget"""
        self.messages.append({"role": role, "content": content})
        
        # Trim old messages if over budget
        while self.get_context_tokens() > (self.max_tokens - self.reserve):
            if len(self.messages) > 1:
                removed = self.messages.pop(0)
                print(f"[Trimmed oldest message: {removed['content'][:40]}...]")
            else:
                break
    
    def query(self, user_message: str) -> str:
        """Query with token budget awareness"""
        self.add_message("user", user_message)
        
        tokens = self.get_context_tokens()
        remaining = self.max_tokens - tokens
        percent = 100 * tokens / self.max_tokens
        
        print(f"Context: {tokens}/{self.max_tokens} tokens ({percent:.1f}%) | Remaining: {remaining}")
        
        # For demo, return placeholder
        response = f"Response to query {len(self.messages)}"
        self.add_message("assistant", response)
        
        return response
    
    def stats(self) -> dict:
        return {
            "messages": len(self.messages),
            "tokens_used": self.get_context_tokens(),
            "percent_full": f"{100 * self.get_context_tokens() / self.max_tokens:.1f}%",
            "avg_tokens_per_message": self.get_context_tokens() // len(self.messages) if self.messages else 0
        }

# Demo: simulate conversation that grows but stays within budget
print("Token-Budget-Aware Manager (max 4000 tokens):")
manager = TokenBudgetContextManager(client, max_context_tokens=4000, reserve_tokens=500)

for i in range(50):
    long_question = f"Question {i+1}? " * 10  # Make questions longer
    manager.query(long_question)

print(f"\nFinal stats:")
print(json.dumps(manager.stats(), indent=2))


# ======================================================================
# ### Example 2: Importance-Ranked Context Pruning
# ======================================================================

class ImportanceRankedContextManager:
    """Keep messages ranked by importance when pruning"""
    
    def __init__(self, client, max_messages: int = 20):
        self.client = client
        self.max_messages = max_messages
        self.messages = []
    
    def score_importance(self, role: str, content: str, turn_number: int, recent_mentions: int) -> float:
        """Score message importance (higher = keep longer)"""
        score = 1.0
        
        # Recent messages are more important
        recency_bonus = 0.1 * (1 - (turn_number / 100.0))
        
        # Messages with explicit markers ("IMPORTANT", "CRITICAL") are important
        if "IMPORTANT" in content or "CRITICAL" in content:
            score += 3.0
        elif "DECISION" in content or "CONSTRAINT" in content:
            score += 2.0
        
        # Messages referenced recently are important
        score += recent_mentions * 0.5
        
        # User messages slightly more important than assistant
        if role == "user":
            score += 0.2
        
        return score + recency_bonus
    
    def add_message(self, role: str, content: str, turn_number: int) -> None:
        """Add message with importance score"""
        score = self.score_importance(role, content, turn_number, recent_mentions=0)
        self.messages.append({
            "role": role,
            "content": content,
            "importance": score,
            "turn": turn_number
        })
        
        # Prune if over limit
        if len(self.messages) > self.max_messages:
            # Sort by importance and keep top N
            self.messages.sort(key=lambda x: x["importance"], reverse=True)
            self.messages = self.messages[:self.max_messages]
            # Re-sort by turn order for coherent context
            self.messages.sort(key=lambda x: x["turn"])
    
    def get_context(self) -> list:
        """Get context (sorted by turn order)"""
        return [{
            "role": msg["role"],
            "content": msg["content"]
        } for msg in self.messages]
    
    def stats(self) -> dict:
        if not self.messages:
            return {}
        
        importance_scores = [m["importance"] for m in self.messages]
        return {
            "messages_kept": len(self.messages),
            "avg_importance": f"{sum(importance_scores)/len(importance_scores):.2f}",
            "min_importance": f"{min(importance_scores):.2f}",
            "max_importance": f"{max(importance_scores):.2f}"
        }

# Demo: long conversation, some messages marked IMPORTANT
manager = ImportanceRankedContextManager(client, max_messages=10)

for i in range(30):
    if i % 10 == 0:
        message = f"IMPORTANT DECISION for turn {i}: Keep this!"
    else:
        message = f"Regular question {i}"
    
    role = "user" if i % 2 == 0 else "assistant"
    manager.add_message(role, message, turn_number=i)

print("After 30 turns, keeping 10 most important:")
print(json.dumps(manager.stats(), indent=2))
print(f"\nContext retained:")
for msg in manager.get_context():
    print(f"  {msg['role'].upper()}: {msg['content'][:50]}...")


# ======================================================================
# ### Example 3: Sparse Retrieval for On-Demand Context Recovery
# ======================================================================

class SparseRetrievalContextManager:
    """Keep compressed context, retrieve specific topics on demand"""
    
    def __init__(self, client):
        self.client = client
        self.all_messages = []  # Everything (indexed)
        self.recent_window = deque(maxlen=5)  # Recent messages only
        self.topic_index = {}  # topic -> [message_indices]
    
    def extract_topics(self, content: str) -> List[str]:
        """Simple topic extraction (in production: use embedding model)"""
        topics = []
        keywords = ["database", "api", "user", "query", "response", "error", "latency", "cost"]
        for kw in keywords:
            if kw.lower() in content.lower():
                topics.append(kw)
        return topics
    
    def add_message(self, role: str, content: str) -> None:
        """Add to both recent window and full index"""
        msg_idx = len(self.all_messages)
        self.all_messages.append({
            "role": role,
            "content": content,
            "index": msg_idx
        })
        self.recent_window.append(msg_idx)
        
        # Index by topic
        for topic in self.extract_topics(content):
            if topic not in self.topic_index:
                self.topic_index[topic] = []
            self.topic_index[topic].append(msg_idx)
    
    def retrieve_by_topic(self, query: str) -> list:
        """Retrieve messages about specific topic"""
        topics = self.extract_topics(query)
        relevant_indices = set()
        
        for topic in topics:
            if topic in self.topic_index:
                relevant_indices.update(self.topic_index[topic])
        
        # Return messages in order
        return sorted([
            self.all_messages[idx]
            for idx in relevant_indices
        ], key=lambda x: x["index"])
    
    def get_context_for_query(self, user_query: str) -> list:
        """Get context: recent + relevant retrieved messages"""
        context = []
        
        # Add recent messages
        for idx in self.recent_window:
            msg = self.all_messages[idx]
            context.append({"role": msg["role"], "content": msg["content"]})
        
        # Add relevant old messages
        retrieved = self.retrieve_by_topic(user_query)
        for msg in retrieved:
            if msg["index"] not in self.recent_window:
                context.append({"role": msg["role"], "content": msg["content"]})
        
        return context
    
    def query(self, user_query: str) -> str:
        """Query with retrieved context if relevant"""
        context = self.get_context_for_query(user_query)
        retrieved = self.retrieve_by_topic(user_query)
        
        return f"Response using {len(context)} context messages (retrieved {len(retrieved)} old)"

# Demo: long conversation with sparse retrieval
manager = SparseRetrievalContextManager(client)

conversations = [
    ("user", "How do I optimize database queries?"),
    ("assistant", "Use indexes and batch requests to database."),
    ("user", "What about API latency?"),
    ("assistant", "Reduce API calls with caching."),
    ("user", "Tell me about response times."),
    ("assistant", "Monitor response latency with metrics."),
    ("user", "Help with cost optimization."),
    ("assistant", "Reduce API calls and use cheaper models."),
    ("user", "Database performance again?"),  # Reference to old topic
]

for role, content in conversations:
    manager.add_message(role, content)

# User asks about database (references old conversation)
query = "Give me all database optimization tips"
print(f"Query: '{query}'")
print(f"Retrieved old messages about 'database': {len(manager.retrieve_by_topic(query))}")
print(f"Total context for this query: {len(manager.get_context_for_query(query))} messages")


# ======================================================================
# ## Key Takeaways
# 1. **Sliding Window is the Simplest Starting Point** — Keep last N messages (typically 5-20), discard older. Token usage is predictable, no complex logic. Works well for most conversations.
# 2. **Context Growth is Inevitable** — In multi-turn conversations, message history grows exponentially. Without management, you'll hit token limits after 50-100 turns. Plan for it from the start.
# ======================================================================
