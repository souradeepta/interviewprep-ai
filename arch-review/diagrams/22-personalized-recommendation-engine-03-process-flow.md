## Process Flow (User Request to Personalized Feed)

```mermaid
sequenceDiagram
    participant User as User
    participant FeedAPI as Feed API
    participant ABRouter as A/B Router
    participant EmbLookup as Embedding Lookup
    participant Retrieval as Two-Tower Retrieval
    participant MLRanker as ML Ranker
    participant LLMReranker as LLM Reranker
    participant DivFilter as Diversity Filter
    participant Cache as Result Cache

    User->>FeedAPI: GET /feed (user_id, context)
    FeedAPI->>Cache: Check cache (user_id + context hash)
    alt Cache hit (60% of requests)
        Cache-->>User: Return cached recommendations
    else Cache miss
        FeedAPI->>ABRouter: Allocate experiment bucket
        ABRouter->>EmbLookup: Fetch user embedding (Redis, 5ms)
        alt Cold start - no user history
            EmbLookup->>Retrieval: Use popularity-based fallback
        else Warm user
            EmbLookup->>Retrieval: User embedding vector
        end
        Retrieval->>Retrieval: ANN search (Faiss HNSW, top-100, 50ms)
        Retrieval->>MLRanker: Top-100 candidates + user features
        MLRanker->>MLRanker: Score all 100 candidates (100ms)
        MLRanker->>LLMReranker: Top-10 candidates for LLM reranking
        LLMReranker->>LLMReranker: Rerank with diversity + context (200ms)
        LLMReranker->>DivFilter: Ranked top-10
        DivFilter->>DivFilter: Enforce 30pct novel items minimum
        DivFilter->>Cache: Store result (TTL 5 minutes)
        DivFilter-->>User: Return final top-10 recommendations
        User->>FeedAPI: Send click/skip events
        FeedAPI->>MLRanker: Update feedback loop
    end
```

**Key Decision Points:**
1. **Cache Check**: 60% hit rate avoids full pipeline for common user/context pairs
2. **Cold Start Fallback**: New users without embeddings receive popularity-based recommendations
3. **Two-Stage Ranking**: Retrieval narrows from full catalog to 100, ranker narrows to 10
4. **LLM Selectivity**: Only reranks the final top-10, not all 100 candidates
5. **Diversity Enforcement**: At least 30% novel or diverse items injected to prevent filter bubbles

**Error Paths:**
- LLM reranker timeout: fall back to ML ranker output directly
- Faiss index unavailable: fall back to popularity + content-based retrieval
- Embedding stale (older than 7 days): serve stale with diversity boosted

**Optimization Points:**
- Cache at user-context granularity (device type, time-of-day bucket)
- Parallelize user embedding lookup and item embedding prefetch
- Pre-compute trending item boosts once per hour, not per request
