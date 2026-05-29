## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Content and Signal Ingestion"]
        ArticleIngest["Article Ingest\n(RSS, crawlers)"]
        UserSignals["User Signals\n(Kafka, click/time/skip)"]
        ContentEmbedder["Content Embedder\n(OpenAI API)"]
    end

    subgraph UserProfiling["User Profiling Infrastructure"]
        UserProfiler["User Profiler\n(Embedding update daily)"]
        UserEmbeddingStore["User Embedding Store\n(Redis, 1M users)"]
        RealtimeBooster["Realtime Signal Booster\n(Recent interactions)"]
    end

    subgraph RankingCluster["Ranking Cluster (GPU)"]
        CandidateRetrieval["Candidate Retrieval\n(ANN, top-100)"]
        LambdaMARTRanker["LambdaMART Ranker\n(CTR optimize)"]
        DiversityReranker["Diversity Reranker\n(20pct novel inject)"]
    end

    subgraph SummarizationLayer["Summarization Layer"]
        LLMSummarizer["LLM Summarizer\n(GPT-3.5, top-10 only)"]
        SummaryCache["Summary Cache\n(Redis, article-level)"]
        StreamingSummary["Streaming Summary\n(Partial delivery)"]
    end

    subgraph DeliveryLayer["Delivery and Feedback"]
        FeedDelivery["Feed Delivery API\n(REST)"]
        FeedbackCollector["Feedback Collector\n(Kafka)"]
        RetrainingPipeline["Retraining Pipeline\n(Weekly)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(CTR, engagement)"]
        DiversityTracker["Diversity Tracker\n(Topic coverage)"]
        ABDashboard["A/B Dashboard\n(Grafana)"]
    end

    ArticleIngest --> ContentEmbedder
    ContentEmbedder --> CandidateRetrieval
    UserSignals --> UserProfiler
    UserProfiler --> UserEmbeddingStore
    UserEmbeddingStore --> CandidateRetrieval
    UserEmbeddingStore --> RealtimeBooster
    CandidateRetrieval --> LambdaMARTRanker
    RealtimeBooster --> LambdaMARTRanker
    LambdaMARTRanker --> DiversityReranker
    DiversityReranker --> SummaryCache
    SummaryCache --> LLMSummarizer
    LLMSummarizer --> StreamingSummary
    StreamingSummary --> FeedDelivery
    FeedDelivery --> FeedbackCollector
    FeedbackCollector --> RetrainingPipeline
    LambdaMARTRanker --> Prom
    DiversityReranker --> DiversityTracker
```

**Infrastructure Components:**
- **Compute**: GPU cluster for LambdaMART ranker, async LLM summarization workers
- **Storage**: Redis (user embeddings for 1M users, article summary cache), Kafka (event streams)
- **Summarization**: LLM only for top-10 ranked articles per feed (not all candidates)
- **Monitoring**: CTR tracking, diversity score (topic coverage percentage), A/B test dashboards
