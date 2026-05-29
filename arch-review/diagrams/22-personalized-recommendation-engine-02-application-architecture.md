## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        FeedAPI["Feed API\n(REST, GET /feed)"]
        EventAPI["Event API\n(POST /events)"]
        ExplainAPI["Explain API\n(POST /explain)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        RecOrchestrator["Rec Orchestrator\n(Pipeline coordinator)"]
        FallbackHandler["Fallback Handler\n(Cold start)"]
        ABRouter["A/B Router\n(Experiment allocation)"]
    end

    subgraph RetrievalServices["Retrieval Services"]
        UserEmbeddingLookup["User Embedding Lookup\n(Redis, 5ms)"]
        TwoTowerRetriever["Two-Tower Retriever\n(Faiss ANN, 50ms)"]
        PopularityFallback["Popularity Fallback\n(Cold start)"]
    end

    subgraph RankingServices["Ranking Services"]
        FeatureAssembler["Feature Assembler\n(User+Item+Context)"]
        MLRanker["ML Ranker\n(LambdaMART, 100ms)"]
        LLMReranker["LLM Reranker\n(GPT-3.5, top-10, 200ms)"]
    end

    subgraph PostProcessing["Post-Processing Services"]
        DiversityFilter["Diversity Filter\n(Category spread)"]
        BusinessFilter["Business Filter\n(Inventory, eligibility)"]
        RealtimeBooster["Realtime Booster\n(Trending items)"]
    end

    subgraph DataServices["Data Services"]
        UserEmbeddingClient["User Embedding Client\n(Redis)"]
        ItemEmbeddingClient["Item Embedding Client\n(Faiss)"]
        FeedbackWriter["Feedback Writer\n(Kafka)"]
        CacheClient["Cache Client\n(Redis, 60% hit)"]
    end

    FeedAPI --> ABRouter
    ABRouter --> RecOrchestrator
    RecOrchestrator --> UserEmbeddingLookup
    UserEmbeddingLookup --> FallbackHandler
    UserEmbeddingLookup --> TwoTowerRetriever
    TwoTowerRetriever --> FeatureAssembler
    FeatureAssembler --> MLRanker
    MLRanker --> LLMReranker
    LLMReranker --> DiversityFilter
    DiversityFilter --> BusinessFilter
    BusinessFilter --> RealtimeBooster
    RealtimeBooster --> CacheClient
    EventAPI --> FeedbackWriter
    UserEmbeddingClient --> UserEmbeddingLookup
    ItemEmbeddingClient --> TwoTowerRetriever
```

**Layer Breakdown:**
- **Presentation**: Feed, event ingestion, and explanation APIs
- **Orchestration**: Pipeline coordination, cold-start fallback, A/B experiment routing
- **Retrieval Services**: Redis user embedding lookup (5ms), Faiss ANN retrieval (50ms), popularity fallback
- **Ranking Services**: Feature assembly, LambdaMART ML ranker (100ms), selective LLM reranker (top-10 only)
- **Post-Processing**: Diversity enforcement, business eligibility filter, trending item boost
- **Data Services**: User/item embedding stores, Kafka feedback writer, result cache
