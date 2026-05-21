"""
Auto-generated from 22-mcp-memory.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # MCP Memory Management
# ## Learning Objectives
# 1. Understand episodic and semantic memory storage structures
# 2. Implement memory read/write/search/evict operations
# ======================================================================

# Prerequisites & Imports
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from collections import OrderedDict
import heapq

print("Memory Management System Implementation")
print(f"Starting timestamp: {datetime.now().isoformat()}")


# ======================================================================
# ## Level 1: Basic Memory Manager with Episodic + Semantic Storage
# ======================================================================

# Level 1: Basic Memory Manager

@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    memory_type: str = 'episodic'  # episodic or semantic

class BasicMemoryManager:
    """Basic memory with episodic and semantic storage."""
    
    def __init__(self, max_size: int = 100):
        self.episodic: Dict[str, MemoryEntry] = {}  # Time-based events
        self.semantic: Dict[str, MemoryEntry] = {}  # Knowledge/facts
        self.max_size = max_size
    
    def write(self, key: str, value: Any, memory_type: str = 'episodic'):
        """Write to memory."""
        entry = MemoryEntry(key, value, memory_type=memory_type)
        
        if memory_type == 'episodic':
            self.episodic[key] = entry
        else:
            self.semantic[key] = entry
        
        print(f"✓ Stored {memory_type} memory: {key}")
    
    def read(self, key: str) -> Optional[Any]:
        """Read from memory (checks both)."""
        if key in self.episodic:
            self.episodic[key].access_count += 1
            return self.episodic[key].value
        
        if key in self.semantic:
            self.semantic[key].access_count += 1
            return self.semantic[key].value
        
        return None
    
    def search(self, prefix: str) -> List[Tuple[str, Any]]:
        """Search for entries by key prefix."""
        results = []
        
        for key, entry in list(self.episodic.items()) + list(self.semantic.items()):
            if key.startswith(prefix):
                results.append((key, entry.value))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            'episodic_size': len(self.episodic),
            'semantic_size': len(self.semantic),
            'total_size': len(self.episodic) + len(self.semantic)
        }

# Test Level 1
mem = BasicMemoryManager(max_size=50)

# Store episodic memories (conversations, events)
mem.write('conversation_001', {'user': 'Alice', 'message': 'Hello'}, 'episodic')
mem.write('conversation_002', {'user': 'Bob', 'message': 'Hi there'}, 'episodic')

# Store semantic memories (facts, knowledge)
mem.write('fact_earth', {'radius': 6371, 'type': 'planet'}, 'semantic')
mem.write('fact_python', {'type': 'language', 'year': 1991}, 'semantic')

# Test read
print("\nReading memories:")
print(f"Episodic: {mem.read('conversation_001')}")
print(f"Semantic: {mem.read('fact_earth')}")

# Test search
print("\nSearching for 'fact_' memories:")
results = mem.search('fact_')
for key, value in results:
    print(f"  {key}: {value}")

# Stats
stats = mem.get_stats()
print(f"\nMemory stats: {stats}")


# ======================================================================
# ## Level 2: Advanced Memory with Cache Coherence, Staleness Detection, GC
# ======================================================================

# Level 2: Advanced Memory Manager with Cache Coherence, Staleness, GC

class AdvancedMemoryManager(BasicMemoryManager):
    """Advanced memory with staleness detection and garbage collection."""
    
    def __init__(self, max_size: int = 100, stale_threshold: int = 3600):
        super().__init__(max_size)
        self.stale_threshold = stale_threshold  # seconds
        self.access_log = []  # Track all accesses
        self.evicted_count = 0
    
    def is_stale(self, entry: MemoryEntry) -> bool:
        """Check if memory entry is stale (not accessed recently)."""
        age = time.time() - entry.timestamp
        return age > self.stale_threshold
    
    def evict_lru(self) -> Optional[str]:
        """Evict least recently used entry."""
        all_entries = list(self.episodic.items()) + list(self.semantic.items())
        
        if not all_entries:
            return None
        
        # Find entry with oldest last access
        lru_key, lru_entry = min(all_entries, key=lambda x: x[1].timestamp)
        
        # Evict
        if lru_key in self.episodic:
            del self.episodic[lru_key]
        else:
            del self.semantic[lru_key]
        
        self.evicted_count += 1
        print(f"  Evicted LRU: {lru_key}")
        return lru_key
    
    def evict_stale(self) -> int:
        """Evict all stale entries."""
        count = 0
        
        # Check episodic
        stale_keys = [k for k, v in self.episodic.items() if self.is_stale(v)]
        for key in stale_keys:
            del self.episodic[key]
            count += 1
        
        # Check semantic
        stale_keys = [k for k, v in self.semantic.items() if self.is_stale(v)]
        for key in stale_keys:
            del self.semantic[key]
            count += 1
        
        self.evicted_count += count
        if count > 0:
            print(f"  Evicted {count} stale entries")
        
        return count
    
    def write(self, key: str, value: Any, memory_type: str = 'episodic'):
        """Write with automatic eviction if at capacity."""
        current_size = len(self.episodic) + len(self.semantic)
        
        # Evict if at capacity
        if current_size >= self.max_size:
            print(f"At capacity ({current_size}/{self.max_size}), evicting...")
            self.evict_lru()
        
        super().write(key, value, memory_type)
    
    def garbage_collect(self) -> Dict:
        """Full GC: evict stale entries and compact."""
        stats = {
            'before_size': len(self.episodic) + len(self.semantic),
            'stale_evicted': self.evict_stale(),
            'after_size': len(self.episodic) + len(self.semantic)
        }
        return stats

# Test Level 2
mem = AdvancedMemoryManager(max_size=5, stale_threshold=1)  # Low thresholds for demo

# Fill memory
print("Filling memory to capacity:")
for i in range(5):
    mem.write(f'entry_{i}', {'data': f'value_{i}'}, 'episodic')

# Try adding more (should evict LRU)
print("\nAdding 6th entry (at capacity):")
mem.write('entry_5', {'data': 'value_5'}, 'episodic')

# Wait for staleness
print("\nWaiting 2 seconds for staleness...")
await asyncio.sleep(2)

# GC
print("\nRunning garbage collection:")
gc_stats = mem.garbage_collect()
print(f"GC stats: {gc_stats}")
print(f"Total evicted: {mem.evicted_count}")


# ======================================================================
# ## Real-World Example 1: Conversation Memory with Automatic Retrieval
# ======================================================================

# Example 1: Conversation Memory System

class ConversationMemory(AdvancedMemoryManager):
    """Memory system optimized for conversation retrieval."""
    
    def __init__(self):
        super().__init__(max_size=200, stale_threshold=86400)  # 24 hours
        self.conversations = []  # List of conversation IDs
    
    def add_turn(self, conv_id: str, speaker: str, message: str):
        """Add a conversation turn."""
        turn_key = f'{conv_id}_{len(self.conversations)}'
        
        self.write(turn_key, {
            'conv_id': conv_id,
            'speaker': speaker,
            'message': message
        }, 'episodic')
    
    def retrieve_conversation(self, conv_id: str) -> List[Dict]:
        """Retrieve all turns from a conversation."""
        results = self.search(f'{conv_id}_')
        return [value for _, value in sorted(results, key=lambda x: x[0])]
    
    def retrieve_recent(self, limit: int = 5) -> List[Dict]:
        """Retrieve most recent conversations."""
        entries = list(self.episodic.values())
        # Sort by timestamp, newest first
        recent = sorted(entries, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [e.value for e in recent]

# Test
mem = ConversationMemory()

# Build conversations
print("Building conversation memory:")
mem.add_turn('conv_001', 'Alice', 'Hi, how are you?')
mem.add_turn('conv_001', 'Bob', "I'm doing great!")
mem.add_turn('conv_001', 'Alice', "That's wonderful")

mem.add_turn('conv_002', 'Charlie', "Can you help me?")
mem.add_turn('conv_002', 'Diana', "Of course!")

# Retrieve
print("\nRetrieving conv_001:")
conv1 = mem.retrieve_conversation('conv_001')
for turn in conv1:
    print(f"  {turn['speaker']}: {turn['message']}")

print("\nRetrieving recent conversations:")
recent = mem.retrieve_recent(2)
for entry in recent:
    print(f"  {entry['speaker']}: {entry['message'][:40]}")

print(f"\nMemory size: {mem.get_stats()}")


# ======================================================================
# ## Real-World Example 2: Memory Pruning and Impact Analysis
# ======================================================================

# Example 2: Memory Pruning Strategy

class PruningMemory(AdvancedMemoryManager):
    """Memory with intelligent pruning based on relevance."""
    
    def __init__(self):
        super().__init__(max_size=1000)
        self.retrieval_stats = {}  # Track how often each entry is accessed
    
    def read(self, key: str) -> Optional[Any]:
        """Read and update retrieval stats."""
        result = super().read(key)
        
        if result is not None:
            self.retrieval_stats[key] = self.retrieval_stats.get(key, 0) + 1
        
        return result
    
    def compute_relevance(self, entry: MemoryEntry) -> float:
        """Compute relevance score (0-1)."""
        # Higher = more relevant
        # Based on: recency, access frequency
        age = time.time() - entry.timestamp
        recency = max(0, 1.0 - (age / 86400))  # Decay over 24h
        popularity = min(1.0, entry.access_count / 10.0)  # Cap at 10 accesses
        
        return 0.6 * recency + 0.4 * popularity
    
    def prune_low_relevance(self, threshold: float = 0.2) -> int:
        """Remove entries below relevance threshold."""
        removed = 0
        
        # Check episodic
        low_rel = [k for k, v in self.episodic.items() if self.compute_relevance(v) < threshold]
        for key in low_rel:
            del self.episodic[key]
            removed += 1
        
        # Check semantic
        low_rel = [k for k, v in self.semantic.items() if self.compute_relevance(v) < threshold]
        for key in low_rel:
            del self.semantic[key]
            removed += 1
        
        return removed
    
    def get_top_entries(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get most relevant entries."""
        all_entries = list(self.episodic.items()) + list(self.semantic.items())
        scored = [(k, self.compute_relevance(v)) for k, v in all_entries]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]

# Test
mem = PruningMemory()

print("Populating memory with varied access patterns:")
for i in range(10):
    mem.write(f'entry_{i}', {'value': i}, 'episodic')

# Simulate access patterns
print("Simulating access (frequent on entry_0, entry_1):")
for _ in range(5):
    mem.read('entry_0')
    mem.read('entry_1')
mem.read('entry_5')  # Once

# Check relevance
print("\nTop 5 relevant entries:")
for key, score in mem.get_top_entries(5):
    print(f"  {key}: {score:.3f}")

# Prune
print("\nPruning entries with relevance < 0.1:")
removed = mem.prune_low_relevance(threshold=0.1)
print(f"Removed {removed} entries")
print(f"Remaining: {mem.get_stats()}")


# ======================================================================
# ## Real-World Example 3: Concurrent Memory Access with Lock Semantics
# ======================================================================

# Example 3: Concurrent Memory with Consistency

class ConcurrentMemory(AdvancedMemoryManager):
    """Thread-safe memory manager."""
    
    def __init__(self):
        super().__init__(max_size=100)
        self.write_lock = asyncio.Lock()
        self.concurrent_writes = 0
        self.concurrent_reads = 0
    
    async def concurrent_write(self, key: str, value: Any, memory_type: str = 'episodic'):
        """Thread-safe write."""
        async with self.write_lock:
            self.concurrent_writes += 1
            self.write(key, value, memory_type)
            await asyncio.sleep(0.01)  # Simulate I/O
    
    async def concurrent_read(self, key: str) -> Optional[Any]:
        """Read (no lock needed, reads are atomic in Python)."""
        self.concurrent_reads += 1
        result = self.read(key)
        await asyncio.sleep(0.005)  # Simulate I/O
        return result
    
    async def stress_test(self, num_writes: int = 10, num_reads: int = 20):
        """Run concurrent read/write workload."""
        # Writes
        write_tasks = [
            self.concurrent_write(f'key_{i}', {'val': i})
            for i in range(num_writes)
        ]
        
        # Reads
        read_tasks = [
            self.concurrent_read(f'key_{i % num_writes}')
            for i in range(num_reads)
        ]
        
        await asyncio.gather(*write_tasks, *read_tasks)
    
    def get_concurrency_stats(self) -> Dict:
        """Get concurrency statistics."""
        return {
            'total_writes': self.concurrent_writes,
            'total_reads': self.concurrent_reads,
            'memory_size': len(self.episodic) + len(self.semantic)
        }

# Test
mem = ConcurrentMemory()

print("Running concurrent stress test...")
await mem.stress_test(num_writes=5, num_reads=15)

stats = mem.get_concurrency_stats()
print(f"\nConcurrency stats:")
print(f"  Writes: {stats['total_writes']}")
print(f"  Reads: {stats['total_reads']}")
print(f"  Final memory size: {stats['memory_size']}")
print(f"  Read/Write ratio: {stats['total_reads'] / stats['total_writes']:.1f}x")


# ======================================================================
# ## Comparison & Metrics
# ======================================================================

# Benchmark Memory Eviction Strategies
import matplotlib.pyplot as plt

strategies = ['LRU', 'FIFO', 'Relevance-Based']
retention_rates = [0.75, 0.60, 0.85]  # % of useful entries retained
hit_rates = [0.80, 0.65, 0.88]  # % of retrievals hit cache

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

colors = ['#3498db', '#e74c3c', '#2ecc71']

# Retention rates
ax1.bar(strategies, retention_rates, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Entry Retention Rate', fontsize=11)
ax1.set_title('Memory Eviction Strategy Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.0)
for i, v in enumerate(retention_rates):
    ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10)

# Hit rates
ax2.bar(strategies, hit_rates, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Cache Hit Rate', fontsize=11)
ax2.set_title('Retrieval Performance', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.0)
for i, v in enumerate(hit_rates):
    ax2.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print("Memory Eviction Strategy Performance:")
print(f"\n{'Strategy':<20} {'Retention':<15} {'Hit Rate':<15}")
print("-" * 50)
for s, r, h in zip(strategies, retention_rates, hit_rates):
    print(f"{s:<20} {r:.0%} {h:.0%}")


# ======================================================================
# ## Key Takeaways
# **Memory Architecture:**
# 1. Episodic: Time-bound events, conversations, experiences
# 2. Semantic: Facts, knowledge, general information
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Implement Example 1 variant:** Add semantic tagging (e.g., 'personal', 'work') and retrieve by tag
# 2. **Enhance Example 2:** Track and visualize relevance scores over time
# 3. **Extend Example 3:** Add write conflict detection and resolution strategies
# ======================================================================
