# AI Semantic Search Engine - Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as Search API
    participant Embedder as Embedding Service
    participant VectorStore as Vector Store
    participant Reranker as Re-ranker
    participant Filter as Business Filter

    U->>API: Search query
    API->>API: Normalize query (lowercase, trim)
    API->>Embedder: Encode query to vector
    Embedder-->>API: Query embedding vector
    API->>VectorStore: ANN search (top-100 candidates)
    VectorStore-->>API: Top-100 results with scores
    API->>Reranker: Re-rank top-100 (query + doc pairs)
    Reranker-->>API: Re-ranked top-20 results
    API->>Filter: Apply business rules (ACL, policy)
    Filter-->>API: Filtered top-10 results
    API-->>U: Ranked search results with snippets
```

**Key Decision Points:**
1. **Query Normalization**: Lowercase, trim, optionally spell-correct before embedding
2. **ANN Recall vs Precision**: Retrieve top-100 to ensure relevant docs in candidate set before re-ranking
3. **Re-ranking**: Cross-encoder improves precision from bi-encoder recall; adds ~50ms latency
4. **Business Filter**: Post-retrieval ACL enforcement to remove unauthorized results

**Optimization Points:**
- Embedding cache (Redis) for popular queries reduces P50 latency by 30-50%
- HNSW index (FAISS) gives sub-10ms ANN search for up to 100M vectors
- Async index updates so new documents appear in search within 5 minutes of ingestion
- Batch embedding during indexing (256+ chunks per batch) for 10x throughput improvement
