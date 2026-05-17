# Vector Databases

## TL;DR
Specialized databases for storing, indexing, and searching embeddings at scale. Enable fast similarity search (ANN), support high-dimensional vectors (300D+), handle millions-billions of vectors. Critical for RAG systems, semantic search, recommendation engines.

## Core Intuition
Regular databases optimize for exact key lookups (e.g., "get user ID 42"). Vector DBs optimize for approximate nearest neighbors (e.g., "find 10 vectors similar to [0.5, -0.2, ...]"). They answer "which documents are similar to this query?" efficiently.

## How It Works

**Core Operations:**

1. **Insert/Upsert:** Add vectors (and metadata) to the database
   ```
   db.upsert(id="doc_1", vector=[0.1, 0.2, ...], metadata={"title": "..."})
   ```

2. **Search:** Find k nearest neighbors
   ```
   results = db.query(vector=query_vec, k=10)
   → returns [(id, distance, metadata), ...]
   ```

3. **Delete/Update:** Remove or modify vectors
   ```
   db.delete(id="doc_1")
   ```

**Indexing Strategies:**

**Flat (Exhaustive Search):**
- Compare query to all vectors (no index)
- Pros: exact nearest neighbors, simple
- Cons: slow (O(n) time), for <1M vectors only

**HNSW (Hierarchical Navigable Small World):**
- Graph-based, navigates layers (fast)
- Fast search: O(log n)
- Memory overhead: ~20% extra
- Default for production (Pinecone, Weaviate use this)

**IVF (Inverted File):**
- Partition space into clusters
- Search only nearby clusters
- Fast but approximate
- Good for billions of vectors

**FAISS (Facebook AI Similarity Search):**
- Quantization + IVF combo
- Reduces memory 4-16x
- Trade: accuracy for memory

**Filtering:**
- Pre-filter by metadata: "find similar docs where author='John'"
- Post-filter: retrieve k+buffer, filter, return top-k
- Metadata filtering: exact match, range, boolean ops

## Key Properties / Trade-offs

| Database | Latency | Scale | Cost | Ease |
|----------|---------|-------|------|------|
| FAISS (local) | <1ms | 100M-1B | Free | Medium |
| Pinecone | 10-100ms | Unlimited | $$$ | Easy |
| Weaviate | 10-100ms | Billions | $$ | Medium |
| Milvus | 10-100ms | Billions | $$ | Medium |
| Qdrant | 10-100ms | Billions | $$ | Easy |
| Redis | 1-10ms | <100M | $ | Easy |

**Dimension vs Speed:**
- 100D: very fast, lower quality
- 384D (Sentence-BERT): balanced
- 768D (BERT-large): slower, better quality
- 1536D (GPT-3 embeddings): slowest, best quality

**Latency vs Accuracy:**
- Flat index: O(n), exact (slow)
- HNSW: O(log n), 99%+ accuracy (medium)
- IVF: ~1-10ms, 90-95% accuracy (fast)
- FAISS: <1ms, 80-90% accuracy (very fast)

## Common Mistakes / Gotchas

- **Wrong vector dimension:** If indexed at 384D but query is 768D, incompatible. Standardize embeddings.
- **Stale vectors:** If documents change but embeddings don't, outdated search results. Re-embed when docs update.
- **Unscaled dimensions:** Mixing 100D and 1000D vectors confuses distance metrics. Normalize or scale.
- **No metadata filtering:** If querying millions of vectors, add metadata filters to reduce search space.
- **Latency surprises:** FAISS is fast locally, but network latency (cloud) adds 50-100ms. Check end-to-end latency.
- **Memory explosion:** Storing billions of vectors uses lots of RAM. Use quantization (int8, int4) to compress.
- **Not handling deletes:** Soft deletes (mark deleted) vs hard deletes (remove). Soft deletes faster, hard deletes save memory.

## Code Example

```python
import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-east1-aws")
index = pinecone.Index("documents")

# 2. Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    {"id": "doc_1", "text": "Python is a programming language"},
    {"id": "doc_2", "text": "Snakes are reptiles"},
]

vectors_to_upsert = [
    (doc["id"], model.encode(doc["text"]).tolist(), {"text": doc["text"]})
    for doc in documents
]

# 3. Upsert vectors
index.upsert(vectors=vectors_to_upsert)

# 4. Search for similar documents
query = "What is Python?"
query_vec = model.encode(query)

results = index.query(query_vec.tolist(), top_k=3, include_metadata=True)
for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']:.3f}, Text: {match['metadata']['text']}")

# 5. Delete document
index.delete(ids=["doc_1"])

# 6. Local FAISS (alternative to cloud)
import faiss
dimension = 384
faiss_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 10)

# Add vectors
vectors = np.array([model.encode(doc["text"]) for doc in documents]).astype('float32')
faiss_index.train(vectors)
faiss_index.add(vectors)

# Search
query_vec = model.encode(query).astype('float32').reshape(1, -1)
distances, indices = faiss_index.search(query_vec, k=3)
print(indices, distances)
```

## Interview Quick-Reference

| Question | What to say |
|---|---|
| "Vector DB vs regular DB?" | Regular: exact key lookups. Vector DB: nearest neighbor search at scale. Optimized for similarity. |
| "HNSW vs IVF?" | HNSW: graph-based, balanced (10-100ms, 99% accurate). IVF: partition-based, fast (1-10ms, 90% accurate). |
| "Scaling to billions?" | Use approximate methods (IVF, FAISS). Quantization (int8/int4) for memory. Cloud DBs (Pinecone) for unlimited scale. |
| "Latency?" | Local FAISS: <1ms. Network latency adds 10-100ms. Cloud DBs: 10-100ms depending on index type. |
| "Filtering?" | Pre-filter metadata before search (faster) or post-filter results (more accurate). Trade latency vs precision. |

## Related Topics
- [Embeddings](embeddings.md) — what gets stored in vector DBs
- [Semantic Search](semantic-search.md) — uses vector DBs for retrieval
- [RAG](rag.md) — vector DB is the retrieval component
- [Inference Caching](../system-design/patterns/inference-caching.md) — cache vector search results

## Resources
- [Pinecone: Vector Database](https://www.pinecone.io/)
- [Weaviate: Open-source Vector Database](https://weaviate.io/)
- [Qdrant: Vector Database for Similarity Search](https://qdrant.tech/)
- [FAISS: Efficient Similarity Search and Clustering](https://github.com/facebookresearch/faiss)
- [Milvus: Open-source Vector Database](https://milvus.io/)
