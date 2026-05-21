"""
Auto-generated from 23-rag-memory-store.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # RAG Memory Store / Vector Database
# ## Learning Objectives
# 1. Implement vector store with cosine similarity search
# 2. Add FAISS-style indexing for fast retrieval
# ======================================================================

# Prerequisites & Imports
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import time
from collections import defaultdict

print("RAG Memory Store (Vector Database) Implementation")
print(f"NumPy version: {np.__version__}")
np.random.seed(42)


# ======================================================================
# ## Level 1: Basic Vector Store with Cosine Similarity
# ======================================================================

# Level 1: Basic Vector Store

@dataclass
class VectorRecord:
    """A record in the vector store."""
    id: str
    text: str
    vector: np.ndarray
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class BasicVectorStore:
    """Basic vector store with cosine similarity."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.records: Dict[str, VectorRecord] = {}
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def add(self, id: str, text: str, vector: np.ndarray, metadata: Dict = None):
        """Add a record to the store."""
        record = VectorRecord(id, text, vector, metadata or {})
        self.records[id] = record
        self.vectors.append(vector)
        self.ids.append(id)
        print(f"✓ Added record: {id}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float, str]]:
        """Search for top-k similar records."""
        if not self.vectors:
            return []
        
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = self.cosine_similarity(query_vector, vec)
            similarities.append((self.ids[i], sim, self.records[self.ids[i]].text))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            'total_records': len(self.records),
            'embedding_dim': self.embedding_dim,
            'memory_mb': sum(v.nbytes for v in self.vectors) / 1024 / 1024
        }

# Test Level 1
print("Creating vector store...")
store = BasicVectorStore(embedding_dim=64)

# Add sample records
docs = [
    'Machine learning is about training models on data',
    'Deep learning uses neural networks with multiple layers',
    'Natural language processing handles text',
    'Computer vision processes images',
    'Reinforcement learning learns from rewards'
]

for i, doc in enumerate(docs):
    # Simulate embedding (random vectors for demo)
    vector = np.random.randn(64)
    vector = vector / np.linalg.norm(vector)  # Normalize
    store.add(f'doc_{i}', doc, vector)

print(f"\nStore stats: {store.get_stats()}")

# Query
print("\nQuerying for 'neural network':")
query_vec = np.random.randn(64)
query_vec = query_vec / np.linalg.norm(query_vec)
results = store.search(query_vec, k=3)
for id, sim, text in results:
    print(f"  {id}: {sim:.3f} - {text[:40]}...")


# ======================================================================
# ## Level 2: Advanced Vector Store with FAISS-style Indexing, Chunking, Filtering
# ======================================================================

# Level 2: Advanced Vector Store with Indexing, Chunking, Filtering

class AdvancedVectorStore(BasicVectorStore):
    """Vector store with efficient indexing and filtering."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__(embedding_dim)
        self.metadata_index = defaultdict(list)  # Index by metadata keys
        self.chunk_size = 512  # Tokens per chunk
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or self.chunk_size
        overlap = chunk_size // 4  # 25% overlap
        
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > chunk_size // 2:  # Only keep substantial chunks
                chunks.append(chunk)
        
        return chunks
    
    def add_with_chunks(self, id: str, text: str, vector: np.ndarray, metadata: Dict = None):
        """Add document with automatic chunking."""
        chunks = self.chunk_text(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f'{id}_chunk_{chunk_idx}'
            # Simulate chunk embedding (in practice, use actual embedder)
            chunk_vec = vector + np.random.randn(len(vector)) * 0.01  # Slight variation
            chunk_vec = chunk_vec / np.linalg.norm(chunk_vec)
            
            chunk_meta = {**(metadata or {}), 'chunk_idx': chunk_idx, 'parent_id': id}
            self.add(chunk_id, chunk, chunk_vec, chunk_meta)
    
    def add_indexed_metadata(self, id: str, text: str, vector: np.ndarray, metadata: Dict = None):
        """Add with metadata indexing for filtering."""
        self.add(id, text, vector, metadata)
        
        if metadata:
            for key, value in metadata.items():
                self.metadata_index[f'{key}:{value}'].append(id)
    
    def search_with_filter(self, query_vector: np.ndarray, k: int = 5, 
                          metadata_filter: Dict = None) -> List[Tuple[str, float, str]]:
        """Search with metadata filtering."""
        if not self.vectors:
            return []
        
        # Get matching IDs if filter provided
        allowed_ids = None
        if metadata_filter:
            allowed_ids = set()
            for key, value in metadata_filter.items():
                matching = self.metadata_index.get(f'{key}:{value}', [])
                allowed_ids.update(matching)
        
        similarities = []
        for i, vec in enumerate(self.vectors):
            id = self.ids[i]
            
            # Skip if doesn't match filter
            if allowed_ids is not None and id not in allowed_ids:
                continue
            
            sim = self.cosine_similarity(query_vector, vec)
            similarities.append((id, sim, self.records[id].text))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_index_stats(self) -> Dict:
        """Get indexing statistics."""
        stats = self.get_stats()
        stats['metadata_keys'] = len(self.metadata_index)
        return stats

# Test Level 2
print("Testing advanced vector store with chunking and filtering...")
store = AdvancedVectorStore(embedding_dim=64)

# Add documents with metadata
docs = [
    ('doc_0', 'Machine learning fundamentals including supervised and unsupervised learning techniques', {'source': 'textbook', 'chapter': '1'}),
    ('doc_1', 'Deep learning architectures like CNNs and Transformers for various tasks', {'source': 'paper', 'chapter': '2'}),
    ('doc_2', 'Natural language processing for text analysis and generation', {'source': 'textbook', 'chapter': '3'}),
]

for id, text, meta in docs:
    vec = np.random.randn(64)
    vec = vec / np.linalg.norm(vec)
    store.add_indexed_metadata(id, text, vec, meta)

print(f"Store stats: {store.get_index_stats()}")

# Search with filter
print("\nSearching for 'learning' in textbooks:")
query_vec = np.random.randn(64)
query_vec = query_vec / np.linalg.norm(query_vec)
results = store.search_with_filter(query_vec, k=3, metadata_filter={'source': 'textbook'})
for id, sim, text in results:
    print(f"  {id}: {sim:.3f} - {text[:40]}...")


# ======================================================================
# ## Real-World Example 1: Document Retrieval with Sentence Transformers Simulation
# ======================================================================

# Example 1: Realistic Document Retrieval System

class DocumentRAG(AdvancedVectorStore):
    """RAG system for document retrieval."""
    
    def __init__(self):
        super().__init__(embedding_dim=384)  # BERT-like size
        self.query_log = []
    
    def add_document_collection(self, documents: List[Tuple[str, str, Dict]]):
        """Add multiple documents to the store."""
        print(f"Indexing {len(documents)} documents...")
        
        for doc_id, text, metadata in documents:
            # Simulate embedding (in production: use sentence-transformers)
            # Here we use a simple hash-based pseudo-embedding for reproducibility
            vector = np.zeros(384)
            for i, char in enumerate(text[:100]):
                vector[i % 384] += ord(char) / 256
            vector = vector / (np.linalg.norm(vector) + 1e-8)
            
            self.add_indexed_metadata(doc_id, text, vector, metadata)
        
        print(f"✓ Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, k: int = 5, source_filter: str = None) -> List[Dict]:
        """Retrieve documents relevant to query."""
        # Simulate query embedding
        query_vec = np.zeros(384)
        for i, char in enumerate(query[:100]):
            query_vec[i % 384] += ord(char) / 256
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Search with optional filter
        metadata_filter = {'source': source_filter} if source_filter else None
        results = self.search_with_filter(query_vec, k=k, metadata_filter=metadata_filter)
        
        # Format results
        formatted = [
            {
                'id': id,
                'text': text[:100],
                'relevance': float(sim),
                'source': self.records[id].metadata.get('source', 'unknown')
            }
            for id, sim, text in results
        ]
        
        # Log
        self.query_log.append({'query': query, 'results': len(formatted)})
        
        return formatted
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics."""
        return {
            'total_queries': len(self.query_log),
            'avg_results': np.mean([r['results'] for r in self.query_log]) if self.query_log else 0,
            'documents': len(self.records)
        }

# Test
print("Example 1: Document RAG System\n")

docs = [
    ('arxiv_001', 'Attention Is All You Need - Transformer architecture for NLP', {'source': 'arxiv', 'year': 2017}),
    ('arxiv_002', 'BERT: Pre-training of Deep Bidirectional Transformers', {'source': 'arxiv', 'year': 2018}),
    ('blog_001', 'Understanding Transformer Attention Mechanisms', {'source': 'blog', 'year': 2023}),
    ('arxiv_003', 'Vision Transformer for Image Classification', {'source': 'arxiv', 'year': 2021}),
]

rag = DocumentRAG()
rag.add_document_collection(docs)

print("\nQuery 1: 'transformer attention mechanism'")
results = rag.retrieve('transformer attention mechanism', k=3)
for r in results:
    print(f"  {r['id']}: {r['relevance']:.3f} - {r['source']}")

print("\nQuery 2: 'BERT pre-training' (from arxiv only)")
results = rag.retrieve('BERT pre-training', k=2, source_filter='arxiv')
for r in results:
    print(f"  {r['id']}: {r['relevance']:.3f} - {r['source']}")

print(f"\nRAG stats: {rag.get_retrieval_stats()}")


# ======================================================================
# ## Real-World Example 2: Chunk Size Impact Analysis
# ======================================================================

# Example 2: Chunk Size vs Retrieval Quality Trade-off

class ChunkingExperiment(AdvancedVectorStore):
    """Experiment to measure chunking impact."""
    
    def __init__(self):
        super().__init__()
        self.chunk_experiments = {}
    
    def experiment_chunk_sizes(self, text: str, sizes: List[int]) -> Dict:
        """Test different chunk sizes and measure fragmentation."""
        results = {}
        
        for size in sizes:
            chunks = self.chunk_text(text, chunk_size=size)
            results[size] = {
                'num_chunks': len(chunks),
                'avg_chunk_len': np.mean([len(c) for c in chunks]),
                'min_chunk_len': min([len(c) for c in chunks]) if chunks else 0,
                'max_chunk_len': max([len(c) for c in chunks]) if chunks else 0
            }
        
        return results

# Test
print("Example 2: Chunk Size Impact\n")

long_text = """Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
Machine learning focuses on developing computer programs that can access data and use it 
to learn for themselves. The process starts with observations about data. To build a machine 
learning model, the algorithm searches for patterns in data. The data set used for training 
needs to be large and comprehensive."""

experiment = ChunkingExperiment()
chunk_sizes = [50, 100, 200, 400]

print(f"Analyzing chunking for document with {len(long_text)} characters:")
results = experiment.experiment_chunk_sizes(long_text, chunk_sizes)

for size, stats in results.items():
    print(f"\nChunk size {size}:")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Avg length: {stats['avg_chunk_len']:.1f}")
    print(f"  Range: {stats['min_chunk_len']}-{stats['max_chunk_len']}")

print(f"\nTrade-off: Larger chunks = fewer docs, smaller chunks = more retrieval points")
print(f"Optimal ~200 chars: balance between granularity and memory")

# Visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

chunks_count = [results[s]['num_chunks'] for s in chunk_sizes]
ax1.plot(chunk_sizes, chunks_count, marker='o', linewidth=2, markersize=8, color='#3498db')
ax1.set_xlabel('Chunk Size (chars)', fontsize=11)
ax1.set_ylabel('Number of Chunks', fontsize=11)
ax1.set_title('Chunks vs Chunk Size', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, (cs, cc) in enumerate(zip(chunk_sizes, chunks_count)):
    ax1.text(cs, cc + 0.1, str(int(cc)), ha='center', fontsize=10)

avg_lens = [results[s]['avg_chunk_len'] for s in chunk_sizes]
ax2.plot(chunk_sizes, avg_lens, marker='s', linewidth=2, markersize=8, color='#e74c3c')
ax2.set_xlabel('Chunk Size (chars)', fontsize=11)
ax2.set_ylabel('Average Chunk Length', fontsize=11)
ax2.set_title('Chunk Utilization', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
for i, (cs, al) in enumerate(zip(chunk_sizes, avg_lens)):
    ax2.text(cs, al + 5, f'{al:.0f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# ======================================================================
# ## Real-World Example 3: Metadata-Aware Filtering
# ======================================================================

# Example 3: Metadata Filtering

class FilteredRAG(DocumentRAG):
    """RAG with advanced filtering."""
    
    def __init__(self):
        super().__init__()
        self.source_stats = defaultdict(int)
    
    def add_with_filtering_stats(self, doc_id: str, text: str, metadata: Dict):
        """Track filtering statistics."""
        vec = np.random.randn(384)
        vec = vec / np.linalg.norm(vec)
        self.add_indexed_metadata(doc_id, text, vec, metadata)
        
        source = metadata.get('source', 'unknown')
        self.source_stats[source] += 1
    
    def multi_filter_search(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search with multiple filters applied."""
        query_vec = np.zeros(384)
        for i, char in enumerate(query[:100]):
            query_vec[i % 384] += ord(char) / 256
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Build metadata filter
        metadata_filter = filters or {}
        results = self.search_with_filter(query_vec, k=5, metadata_filter=metadata_filter)
        
        return [
            {'id': id, 'text': text[:80], 'relevance': float(sim)}
            for id, sim, text in results
        ]
    
    def get_source_coverage(self) -> Dict:
        """Get coverage by source."""
        total = sum(self.source_stats.values())
        return {source: count/total for source, count in self.source_stats.items()}

# Test
print("Example 3: Metadata-Aware Filtering\n")

rag = FilteredRAG()

# Add diverse documents
docs = [
    ('doc_1', 'Deep learning for medical imaging', {'source': 'paper', 'domain': 'healthcare', 'year': 2022}),
    ('doc_2', 'Transformers in NLP applications', {'source': 'blog', 'domain': 'nlp', 'year': 2023}),
    ('doc_3', 'Machine learning ops best practices', {'source': 'paper', 'domain': 'mlops', 'year': 2023}),
    ('doc_4', 'Vision transformers explained', {'source': 'blog', 'domain': 'cv', 'year': 2023}),
    ('doc_5', 'Healthcare AI systems', {'source': 'paper', 'domain': 'healthcare', 'year': 2023}),
]

for id, text, meta in docs:
    rag.add_with_filtering_stats(id, text, meta)

print("Source coverage:")
for source, pct in rag.get_source_coverage().items():
    print(f"  {source}: {pct:.0%}")

print("\nQuery 1: All results for 'learning'")
results = rag.multi_filter_search('learning')
for r in results:
    print(f"  {r['id']}: {r['relevance']:.3f}")

print("\nQuery 2: Healthcare domain only")
results = rag.multi_filter_search('deep learning', filters={'domain': 'healthcare'})
for r in results:
    print(f"  {r['id']}: {r['relevance']:.3f}")

print("\nQuery 3: Recent papers (2023+)")
results = rag.multi_filter_search('transformers', filters={'source': 'paper', 'year': 2023})
for r in results:
    print(f"  {r['id']}: {r['relevance']:.3f}")


# ======================================================================
# ## Comparison & Metrics
# ======================================================================

# Retrieval Quality Metrics
import matplotlib.pyplot as plt

# Simulate precision/recall at different chunk sizes
chunk_sizes = [64, 128, 256, 512]
precision = [0.92, 0.88, 0.85, 0.79]
recall = [0.65, 0.75, 0.82, 0.88]
query_latency = [0.002, 0.003, 0.005, 0.010]  # seconds

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Precision vs Recall
ax1.plot(chunk_sizes, precision, marker='o', label='Precision', linewidth=2, color='#2ecc71')
ax1.plot(chunk_sizes, recall, marker='s', label='Recall', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Chunk Size', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# F1 Score
f1 = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
ax2.bar(range(len(chunk_sizes)), f1, color='#3498db', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(chunk_sizes)))
ax2.set_xticklabels(chunk_sizes)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.set_title('Overall Quality (F1)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1)
for i, v in enumerate(f1):
    ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

# Query Latency
ax3.plot(chunk_sizes, query_latency, marker='^', linewidth=2, markersize=8, color='#e67e22')
ax3.set_xlabel('Chunk Size', fontsize=11)
ax3.set_ylabel('Latency (seconds)', fontsize=11)
ax3.set_title('Query Latency', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
for cs, latency in zip(chunk_sizes, query_latency):
    ax3.text(cs, latency + 0.0005, f'{latency*1000:.1f}ms', ha='center', fontsize=9)

# Index Size (estimated)
index_sizes = [10, 18, 32, 60]  # MB
ax4.bar(range(len(chunk_sizes)), index_sizes, color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(chunk_sizes)))
ax4.set_xticklabels(chunk_sizes)
ax4.set_ylabel('Index Size (MB)', fontsize=11)
ax4.set_title('Memory Usage', fontsize=12, fontweight='bold')
for i, v in enumerate(index_sizes):
    ax4.text(i, v + 1, f'{v}MB', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print("Retrieval Quality Analysis:")
print(f"\n{'Chunk':<10} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Latency':<12} {'Index Size':<12}")
print("-" * 70)
for cs, p, r, lat, idx in zip(chunk_sizes, precision, recall, query_latency, index_sizes):
    f1_score = 2 * (p * r) / (p + r)
    print(f"{cs:<10} {p:.2f} {r:.2f} {f1_score:.2f} {lat*1000:.1f}ms {idx}MB")

print("\nOptimal: 256 chars - best balance of quality and performance")



# ======================================================================
# ## Key Takeaways
# **Vector Store Architecture:**
# 1. Vectors normalized for cosine similarity
# 2. Metadata index enables efficient filtering
# ======================================================================

# ======================================================================
# ## Try It Yourself
# 1. **Extend Example 1:** Add BM25 keyword search alongside semantic search
# 2. **Modify Example 2:** Test different overlap strategies (25%, 50%, none)
# 3. **Enhance Example 3:** Add hierarchical filtering (source → domain → date)
# ======================================================================
