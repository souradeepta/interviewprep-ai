## Process Flow (Search Query to Personalized Results with Price)

```mermaid
sequenceDiagram
    participant User as Shopper
    participant SearchAPI as Search API
    participant NLU as Query Understanding
    participant Retrieval as Product Retrieval
    participant LTR as LTR Ranker
    participant PersonLayer as Personalization Layer
    participant Pricing as Dynamic Pricing
    participant ABRouter as A/B Router
    participant Display as Result Page

    User->>SearchAPI: GET /search?q=blue running shoes
    SearchAPI->>ABRouter: Allocate experiment bucket
    ABRouter->>NLU: Parse query intent and attributes
    NLU->>NLU: Extract: category=shoes, color=blue, use=running
    NLU->>Retrieval: Structured query with filters
    Retrieval->>Retrieval: BM25 + semantic search (Elasticsearch)
    Retrieval->>LTR: Top-100 candidate products
    LTR->>LTR: Score by relevance (LambdaMART)
    LTR->>PersonLayer: Top-50 scored candidates
    PersonLayer->>PersonLayer: Lookup user embedding (Redis, 5ms)
    alt User has purchase history
        PersonLayer->>PersonLayer: Re-rank by affinity score
    else Cold start user
        PersonLayer->>PersonLayer: Use popularity-based ranking
    end
    PersonLayer->>Pricing: Top-20 products + user segment
    Pricing->>Pricing: Compute elasticity-adjusted price per user
    Pricing->>Pricing: Apply margin constraints (min price floor)
    Pricing->>Display: Final top-20 with personalized prices
    Display-->>User: Search results page (target 100ms)
    User->>SearchAPI: Click product
    SearchAPI->>LTR: Record click signal (feedback)
    LTR->>LTR: Accumulate for weekly retraining
```

**Key Decision Points:**
1. **Query Understanding**: Extracts category, color, use-case to filter catalog before ranking
2. **Two-Stage Ranking**: BM25 retrieval (top-100) then LambdaMART re-ranking (top-50) for efficiency
3. **Cold Start Check**: No user history falls back to popularity-based ordering
4. **Dynamic Pricing**: Elasticity model adjusts price per user segment, constrained by margin floor
5. **A/B Routing**: Traffic split for testing new ranking models before full rollout

**Error Paths:**
- Query understanding failure: fall back to keyword search
- Personalization service timeout: serve unranked results within SLA
- Pricing service unavailable: serve catalog price as fallback

**Optimization Points:**
- Cache popular query results (30-minute TTL for top-1000 queries)
- Pre-compute user affinity scores for top-10K most active users
- Warm pricing cache with elasticity estimates computed overnight
