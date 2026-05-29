## Process Flow (User Session to Personalized Feed with Summaries)

```mermaid
sequenceDiagram
    participant User as Reader
    participant FeedAPI as Feed API
    participant EmbLookup as User Embedding Lookup
    participant Retrieval as ANN Retrieval
    participant Ranker as LambdaMART Ranker
    participant DivFilter as Diversity Injector
    participant SumCache as Summary Cache
    participant LLM as LLM Summarizer
    participant Delivery as Feed Delivery

    User->>FeedAPI: GET /feed (user_id, context)
    FeedAPI->>EmbLookup: Fetch user embedding (Redis, 5ms)
    alt Cold start (no embedding)
        EmbLookup->>Retrieval: Use popularity-based candidates
    else Warm user
        EmbLookup->>Retrieval: User embedding vector (256-d)
    end
    Retrieval->>Retrieval: ANN search Faiss (top-100 articles, 50ms)
    Retrieval->>Ranker: Top-100 candidate articles
    Ranker->>Ranker: Score by user affinity and freshness (100ms)
    Ranker->>DivFilter: Top-20 ranked articles
    DivFilter->>DivFilter: Check topic diversity
    alt Diversity below 20pct threshold
        DivFilter->>DivFilter: Inject novel articles to 20pct minimum
    end
    DivFilter->>Delivery: Return top-10 ranked articles
    Delivery-->>User: Feed headlines (immediate, 150ms total)
    Delivery->>SumCache: Check cache for each top-10 article
    loop For each uncached article (top-10)
        SumCache->>LLM: Generate 50-100 word summary
        LLM-->>SumCache: Cache summary (article-level key)
        SumCache->>Delivery: Stream summary to client
        Delivery-->>User: Article summary arrives (streaming)
    end
    User->>FeedAPI: Click, skip, or read time signal
    FeedAPI->>Ranker: Feedback signal for retraining
```

**Key Decision Points:**
1. **Headlines First**: Feed headlines delivered immediately (150ms), summaries stream in background
2. **Cold Start**: No user embedding falls back to popularity-based retrieval
3. **ANN Retrieval**: Faiss approximate nearest neighbor limits candidate pool to 100 relevant articles
4. **Diversity Injection**: If top-20 candidates lack variety, novel articles forcibly inserted at 20% minimum
5. **Selective Summarization**: LLM only generates summaries for final top-10, not all 100 candidates

**Error Paths:**
- LLM summarizer unavailable: serve feed without summaries, retry async
- User embedding stale (older than 7 days): serve stale with diversity boosted as safety mechanism
- Faiss index unavailable: fall back to Elasticsearch keyword-based retrieval

**Optimization Points:**
- Cache summaries at article level (not user level) for high reuse across users
- Pre-generate summaries for top-1000 daily trending articles during low-traffic periods
- Streaming headline delivery decouples latency from summary generation time
