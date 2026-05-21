"""
Auto-generated from 04-persistent-ai-memory.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Persistent AI Memory
# ## Learning Objectives
# 1. Build episodic memory (raw transcript storage) and semantic memory (embeddings + search) systems
# 2. Implement memory CRUD operations with staleness detection and pruning
# ======================================================================

import numpy as np
import torch
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Device setup for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ======================================================================
# ## Level 1: Basic Episodic + Semantic Memory with Manual Cosine Similarity
# ======================================================================

# Level 1: Simple episodic (raw) and semantic (embeddings-based) memory from scratch
@dataclass
class MemoryEntry:
    """A single memory entry with timestamp and embedding"""
    text: str
    timestamp: float
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(10))
    entry_id: str = field(default_factory=lambda: str(time.time()))
    access_count: int = 0

class SimpleMemoryBank:
    """Basic memory with episodic storage and manual semantic search"""
    
    def __init__(self, embedding_dim: int = 10):
        self.episodic_memory = []  # Raw transcripts
        self.semantic_embeddings = []  # Embeddings for search
        self.embedding_dim = embedding_dim
        self.access_stats = {}
    
    def create_dummy_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding (hash-based for demo)"""
        # In practice, use sentence-transformers
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def store_episodic(self, text: str) -> str:
        """Store raw text in episodic memory"""
        entry = MemoryEntry(
            text=text,
            timestamp=time.time(),
            embedding=self.create_dummy_embedding(text)
        )
        self.episodic_memory.append(entry)
        self.semantic_embeddings.append(entry.embedding)
        return entry.entry_id
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search episodic memory by semantic similarity"""
        query_embedding = self.create_dummy_embedding(query)
        
        # Compute similarity to all memories
        similarities = []
        for i, memory in enumerate(self.episodic_memory):
            sim = self.cosine_similarity(query_embedding, memory.embedding)
            similarities.append(('sim', sim, i))
        
        # Return top-k most similar
        top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for _, sim, idx in top_results:
            memory = self.episodic_memory[idx]
            memory.access_count += 1
            results.append({
                'text': memory.text,
                'similarity': sim,
                'age_seconds': time.time() - memory.timestamp
            })
        
        return results

# Test basic memory
memory = SimpleMemoryBank(embedding_dim=10)

# Store some conversations
memory.store_episodic("User asked about machine learning fundamentals")
memory.store_episodic("Discussed gradient descent and backpropagation")
memory.store_episodic("Explained neural network architectures")
memory.store_episodic("Reviewed data preprocessing techniques")
memory.store_episodic("Talked about model evaluation metrics")

# Search memory
results = memory.semantic_search("neural networks and deep learning", top_k=2)

print("Memory Storage and Semantic Search:")
print(f"  Total memories stored: {len(memory.episodic_memory)}")
print(f"  Top results for 'neural networks and deep learning':")
for i, result in enumerate(results, 1):
    print(f"    {i}. {result['text'][:50]}... (sim: {result['similarity']:.3f})")


# ======================================================================
# ## Level 2: Advanced Memory with CRUD, Pruning, and Statistics
# ======================================================================

class AdvancedMemorySystem:
    """Production memory system with CRUD, staleness detection, and pruning"""
    
    def __init__(self, max_age_seconds: float = 3600, max_size: int = 1000):
        self.memory_store = {}  # id -> MemoryEntry
        self.max_age_seconds = max_age_seconds
        self.max_size = max_size
        self.stats = {
            'creates': 0,
            'reads': 0,
            'updates': 0,
            'deletes': 0,
            'pruned': 0
        }
    
    def create(self, text: str, metadata: Dict = None) -> str:
        """Create new memory entry"""
        entry_id = str(time.time())
        entry = {
            'text': text,
            'metadata': metadata or {},
            'created': time.time(),
            'updated': time.time(),
            'access_count': 0,
            'last_accessed': None
        }
        self.memory_store[entry_id] = entry
        self.stats['creates'] += 1
        
        # Check if need to prune
        if len(self.memory_store) > self.max_size:
            self._prune_oldest()
        
        return entry_id
    
    def read(self, entry_id: str) -> Dict:
        """Read memory entry and update access stats"""
        if entry_id in self.memory_store:
            entry = self.memory_store[entry_id]
            entry['access_count'] += 1
            entry['last_accessed'] = time.time()
            self.stats['reads'] += 1
            return entry
        return None
    
    def update(self, entry_id: str, text: str, metadata: Dict = None) -> bool:
        """Update existing memory entry"""
        if entry_id in self.memory_store:
            entry = self.memory_store[entry_id]
            entry['text'] = text
            if metadata:
                entry['metadata'].update(metadata)
            entry['updated'] = time.time()
            self.stats['updates'] += 1
            return True
        return False
    
    def delete(self, entry_id: str) -> bool:
        """Delete memory entry"""
        if entry_id in self.memory_store:
            del self.memory_store[entry_id]
            self.stats['deletes'] += 1
            return True
        return False
    
    def is_stale(self, entry_id: str) -> bool:
        """Check if memory entry is stale (old and unused)"""
        if entry_id not in self.memory_store:
            return False
        
        entry = self.memory_store[entry_id]
        age = time.time() - entry['created']
        
        # Stale if older than max_age and rarely accessed
        return age > self.max_age_seconds and entry['access_count'] < 2
    
    def _prune_oldest(self):
        """Remove oldest unused memories when store is full"""
        stale_entries = []
        for entry_id, entry in self.memory_store.items():
            age = time.time() - entry['created']
            # Score: older and less accessed = higher score (more pruneable)
            score = age / (entry['access_count'] + 1)
            stale_entries.append((score, entry_id))
        
        # Remove top 10% oldest
        stale_entries.sort(reverse=True)
        remove_count = max(1, len(self.memory_store) // 10)
        
        for _, entry_id in stale_entries[:remove_count]:
            self.delete(entry_id)
            self.stats['pruned'] += 1
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics"""
        access_counts = [e['access_count'] for e in self.memory_store.values()]
        ages = [time.time() - e['created'] for e in self.memory_store.values()]
        
        return {
            'current_size': len(self.memory_store),
            'stats': self.stats,
            'avg_access_count': np.mean(access_counts) if access_counts else 0,
            'avg_age_seconds': np.mean(ages) if ages else 0,
            'max_age_seconds': max(ages) if ages else 0
        }

# Test advanced memory
adv_memory = AdvancedMemorySystem(max_age_seconds=3600, max_size=100)

# Perform CRUD operations
ids = []
for i in range(10):
    entry_id = adv_memory.create(f"Conversation turn {i}", {'turn': i})
    ids.append(entry_id)

# Read some entries to increase access count
for i in range(0, 5):
    adv_memory.read(ids[i])
    adv_memory.read(ids[i])  # Read twice

# Update an entry
adv_memory.update(ids[0], "Updated conversation turn 0")

# Get statistics
stats = adv_memory.get_statistics()

print("\nAdvanced Memory System Statistics:")
print(f"  Current size: {stats['current_size']} entries")
print(f"  Creates: {stats['stats']['creates']}, Reads: {stats['stats']['reads']}, Updates: {stats['stats']['updates']}")
print(f"  Average access count: {stats['avg_access_count']:.2f}")
print(f"  Average age: {stats['avg_age_seconds']:.3f}s")


# ======================================================================
# ## Real-World Example 1: Semantic Search with Sentence Transformers
# ======================================================================

# Example 1: Use sentence embeddings for semantic memory search
class SemanticMemoryStore:
    """Memory store with sentence-transformer-style embeddings"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embeddings = {}
        self.texts = {}
        self.embedding_dim = embedding_dim
    
    def mock_encode(self, text: str) -> np.ndarray:
        """Mock sentence-transformers.encode() - in reality use pretrained model"""
        # Deterministic embedding based on text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.embedding_dim)
        # Normalize like real encoders do
        return embedding / np.linalg.norm(embedding)
    
    def store(self, text: str, entry_id: str = None) -> str:
        """Store text and create embedding"""
        if entry_id is None:
            entry_id = f'mem_{len(self.texts)}'
        
        self.texts[entry_id] = text
        self.embeddings[entry_id] = self.mock_encode(text)
        return entry_id
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search by semantic similarity"""
        query_embedding = self.mock_encode(query)
        
        scores = []
        for entry_id, embedding in self.embeddings.items():
            # Cosine similarity
            sim = np.dot(query_embedding, embedding)
            scores.append((entry_id, sim))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for entry_id, sim in scores[:top_k]:
            results.append({
                'id': entry_id,
                'text': self.texts[entry_id],
                'similarity': float(sim)
            })
        
        return results

# Test semantic memory with conversations
semantic_mem = SemanticMemoryStore(embedding_dim=384)

conversations = [
    "User asked about Python programming best practices",
    "Discussed database optimization techniques",
    "Explained machine learning model evaluation metrics",
    "Reviewed REST API design patterns",
    "Talked about cloud deployment strategies",
    "Covered testing frameworks and unit testing",
    "Discussed Python data structures and algorithms",
]

for conv in conversations:
    semantic_mem.store(conv)

# Search for related conversations
query = "What did we discuss about python and coding?"
results = semantic_mem.search(query, top_k=3)

print("\nSemantic Memory Search Results:")
print(f"Query: '{query}'")
print(f"Total conversations: {len(conversations)}")
print(f"Top matches:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['text'][:60]}... (similarity: {result['similarity']:.3f})")


# ======================================================================
# ## Real-World Example 2: Conversation System with Automatic Retrieval
# ======================================================================

# Example 2: Build conversation system that retrieves relevant context automatically
class ConversationalMemoryAgent:
    """Agent that maintains memory and retrieves context for responses"""
    
    def __init__(self, context_window: int = 5):
        self.semantic_mem = SemanticMemoryStore(embedding_dim=384)
        self.conversation_history = []
        self.context_window = context_window
    
    def add_user_message(self, user_input: str) -> None:
        """Add user message to history and memory"""
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': time.time()
        })
        # Store in semantic memory for future retrieval
        self.semantic_mem.store(user_input)
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant past context for current query"""
        results = self.semantic_mem.search(query, top_k=top_k)
        return [r['text'] for r in results]
    
    def generate_response(self, user_input: str) -> Dict:
        """Generate response with retrieved context"""
        # Retrieve relevant past interactions
        context = self.retrieve_context(user_input, top_k=2)
        
        # Build response (mock)
        response = f"Based on our conversation history, I see we've discussed: "
        response += "; ".join(context[:2])
        response += ". Now, regarding your question..."
        
        # Log this interaction
        self.add_user_message(user_input)
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'retrieved_context': context,
            'timestamp': time.time()
        })
        
        return {
            'response': response,
            'retrieved_context': context,
            'context_count': len(context)
        }

# Test conversation agent
agent = ConversationalMemoryAgent(context_window=5)

# Simulate conversation
user_inputs = [
    "What's the best way to optimize database queries?",
    "Tell me about Python performance tips",
    "How do I scale APIs for high traffic?",
]

print("\nConversational Memory Agent:")
for i, user_input in enumerate(user_inputs, 1):
    result = agent.generate_response(user_input)
    print(f"\nTurn {i}:")
    print(f"  User: {user_input}")
    print(f"  Retrieved context count: {result['context_count']}")
    if result['retrieved_context']:
        print(f"  Context: {result['retrieved_context'][0][:50]}...")

print(f"\nTotal conversation turns: {len(agent.conversation_history)}")


# ======================================================================
# ## Real-World Example 3: Memory Management with Staleness Detection and Pruning
# ======================================================================

# Example 3: Memory system with automatic pruning and statistics
class ManagedMemorySystem:
    """Memory system with staleness detection and automatic cleanup"""
    
    def __init__(self, max_age_hours: int = 24, max_entries: int = 10000):
        self.entries = {}
        self.max_age_seconds = max_age_hours * 3600
        self.max_entries = max_entries
        self.pruning_history = []
    
    def add_entry(self, text: str) -> str:
        """Add entry and auto-prune if needed"""
        entry_id = f'entry_{len(self.entries)}_{int(time.time() * 1000)}'
        self.entries[entry_id] = {
            'text': text,
            'created': time.time(),
            'accessed': 0,
            'importance': np.random.uniform(0.3, 1.0)  # Mock importance score
        }
        
        # Auto-prune if over limit
        if len(self.entries) > self.max_entries:
            self._prune()
        
        return entry_id
    
    def _prune(self):
        """Remove stale/low-importance entries"""
        now = time.time()
        scores = []
        
        for entry_id, entry in self.entries.items():
            age_hours = (now - entry['created']) / 3600
            # Staleness score: high if old and unaccessed and low importance
            staleness = (age_hours / 24.0) * (1 - entry['importance']) / (entry['accessed'] + 1)
            scores.append((staleness, entry_id))
        
        # Remove bottom 20% by importance
        scores.sort(reverse=True)
        remove_count = max(1, len(self.entries) // 5)
        
        removed = []
        for _, entry_id in scores[:remove_count]:
            removed.append(entry_id)
            del self.entries[entry_id]
        
        self.pruning_history.append({
            'timestamp': now,
            'removed_count': len(removed),
            'remaining': len(self.entries)
        })
    
    def get_memory_stats(self) -> Dict:
        """Get detailed memory statistics"""
        now = time.time()
        ages = [(now - e['created']) / 3600 for e in self.entries.values()]
        importances = [e['importance'] for e in self.entries.values()]
        
        return {
            'total_entries': len(self.entries),
            'avg_age_hours': np.mean(ages) if ages else 0,
            'max_age_hours': max(ages) if ages else 0,
            'avg_importance': np.mean(importances) if importances else 0,
            'total_pruned': sum(p['removed_count'] for p in self.pruning_history),
            'prune_count': len(self.pruning_history)
        }

# Test managed memory
managed_mem = ManagedMemorySystem(max_age_hours=24, max_entries=100)

# Add entries
for i in range(120):
    managed_mem.add_entry(f"Memory entry {i}")

stats = managed_mem.get_memory_stats()

print("\nManaged Memory System Statistics:")
print(f"  Total entries (after pruning): {stats['total_entries']}")
print(f"  Average age: {stats['avg_age_hours']:.3f} hours")
print(f"  Average importance: {stats['avg_importance']:.3f}")
print(f"  Total pruned: {stats['total_pruned']} entries")
print(f"  Prune operations: {stats['prune_count']}")


# ======================================================================
# ## Comparison: Memory Growth and Retrieval Accuracy
# ======================================================================

# Benchmark memory growth and retrieval accuracy
import matplotlib.pyplot as plt

# Simulate memory growth and pruning
scenarios = ['No Pruning', 'Age-Based Pruning', 'Importance-Based Pruning']
memory_sizes = [[100 + i*50 for i in range(20)],  # Unbounded growth
               [min(100, 50 + i*2) for i in range(20)],  # Stabilizes at limit
               [min(100, 30 + i*1.5) for i in range(20)]]

retrieval_quality = [[0.9 - i*0.02 for i in range(20)],  # Degrades with time
                     [0.9 - i*0.005 for i in range(20)],  # Stable with pruning
                     [0.92 - i*0.001 for i in range(20)]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Memory growth
for scenario, sizes in zip(scenarios, memory_sizes):
    ax1.plot(range(20), sizes, marker='o', label=scenario, linewidth=2)

ax1.set_xlabel('Time (arbitrary units)')
ax1.set_ylabel('Memory Entries')
ax1.set_title('Memory Growth Over Time')
ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Size Limit')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Retrieval quality
for scenario, quality in zip(scenarios, retrieval_quality):
    ax2.plot(range(20), quality, marker='s', label=scenario, linewidth=2)

ax2.set_xlabel('Time (arbitrary units)')
ax2.set_ylabel('Retrieval Accuracy')
ax2.set_title('Retrieval Quality Over Time')
ax2.set_ylim([0.85, 0.95])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('memory_growth_and_quality.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nMemory System Comparison:")
print(f"{'Scenario':<30} {'Final Size':<15} {'Final Quality':<15}")
print("-" * 60)
for scenario, sizes, quality in zip(scenarios, memory_sizes, retrieval_quality):
    print(f"{scenario:<30} {sizes[-1]:<15.0f} {quality[-1]:<15.3f}")


# ======================================================================
# ## Key Takeaways
# ### Core Concept
# Persistent memory systems combine episodic (raw transcript) and semantic (embedding-based) storage. Episodic memory preserves exact conversations; semantic memory enables similarity-based retrieval. Together they allow agents to retrieve relevant past context automatically.
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Implement real embeddings**: Replace mock embeddings with `sentence-transformers` or OpenAI embeddings.
# 2. **Add multi-turn context**: When retrieving context, return previous/next entries to preserve narrative flow.
# ======================================================================
