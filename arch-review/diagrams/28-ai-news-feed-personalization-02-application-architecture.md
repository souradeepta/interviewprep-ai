## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        FeedAPI["Feed API\n(GET /feed)"]
        ArticleAPI["Article API\n(GET /article)"]
        SignalAPI["Signal API\n(POST /signal)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        FeedOrchestrator["Feed Orchestrator\n(Ranking pipeline)"]
        SummaryOrchestrator["Summary Orchestrator\n(Selective LLM)"]
        RetrainingOrchestrator["Retraining Orchestrator\n(Weekly schedule)"]
    end

    subgraph ContentServices["Content Services"]
        ContentEmbedderSvc["Content Embedder Svc\n(OpenAI API)"]
        FreshnessScorer["Freshness Scorer\n(Time decay)"]
        CategoryTagger["Category Tagger\n(Multi-label)"]
    end

    subgraph UserProfilingServices["User Profiling Services"]
        UserEmbeddingUpdater["User Embedding Updater\n(Daily batch)"]
        RealtimeSignalProcessor["Realtime Signal Processor\n(Flink)"]
        ColdStartHandler["Cold Start Handler\n(Popularity fallback)"]
    end

    subgraph RankingServices["Ranking Services"]
        ANCRetriever["ANN Retriever\n(Faiss, top-100)"]
        LambdaMARTRanker["LambdaMART Ranker\n(CTR optimize)"]
        DiversityInjector["Diversity Injector\n(20pct novel minimum)"]
    end

    subgraph SummarizationServices["Summarization Services"]
        SummaryCacheChecker["Summary Cache Checker\n(Article-level Redis)"]
        LLMSummarizer["LLM Summarizer\n(GPT-3.5, top-10)"]
        StreamingDelivery["Streaming Delivery\n(Partial headline first)"]
    end

    subgraph DataServices["Data Services"]
        UserEmbeddingClient["User Embedding Client\n(Redis)"]
        ContentIndexClient["Content Index Client\n(Faiss)"]
        FeedbackWriter["Feedback Writer\n(Kafka)"]
    end

    FeedAPI --> FeedOrchestrator
    FeedOrchestrator --> UserEmbeddingClient
    UserEmbeddingClient --> ColdStartHandler
    UserEmbeddingClient --> ANCRetriever
    ContentIndexClient --> ANCRetriever
    ANCRetriever --> LambdaMARTRanker
    FreshnessScorer --> LambdaMARTRanker
    LambdaMARTRanker --> DiversityInjector
    DiversityInjector --> SummaryOrchestrator
    SummaryOrchestrator --> SummaryCacheChecker
    SummaryCacheChecker --> LLMSummarizer
    LLMSummarizer --> StreamingDelivery
    StreamingDelivery --> FeedAPI
    SignalAPI --> RealtimeSignalProcessor
    RealtimeSignalProcessor --> UserEmbeddingUpdater
    FeedbackWriter --> RetrainingOrchestrator
```

**Layer Breakdown:**
- **Presentation**: Feed retrieval, article detail, and user signal ingestion APIs
- **Orchestration**: Feed ranking pipeline, selective LLM summarization, weekly retraining
- **Content Services**: OpenAI content embedding, freshness decay scoring, multi-label category tagging
- **User Profiling**: Daily batch embedding updates, real-time Flink signal processing, cold start fallback
- **Ranking Services**: Faiss ANN retrieval (top-100), LambdaMART CTR optimization, diversity injection
- **Summarization Services**: Article-level Redis cache check, GPT-3.5 top-10 summarization, streaming headline delivery
