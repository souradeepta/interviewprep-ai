## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Event Ingestion Layer"]
        UserEvents["User Events\n(Click, Purchase, View)"]
        KafkaTopic["Kafka Topic\n(user-events)"]
        RealtimeProcessor["Realtime Processor\n(Flink)"]
    end

    subgraph EmbeddingLayer["Embedding Infrastructure"]
        UserEmbeddings["User Embedding Store\n(Redis, 500M users)"]
        ItemEmbeddings["Item Embedding Store\n(Faiss HNSW)"]
        EmbeddingRefresh["Embedding Refresh\n(Batch, weekly)"]
    end

    subgraph RetrievalLayer["Retrieval and Ranking Cluster"]
        TwoTowerRetrieval["Two-Tower Retrieval\n(ANN, top-100)"]
        MLRanker["ML Ranker\n(TorchServe, 8x V100)"]
        LLMReranker["LLM Reranker\n(GPT-3.5, top-10 only)"]
        DiversityFilter["Diversity Filter\n(Post-process)"]
    end

    subgraph FeatureStore["Feature Store"]
        UserFeatures["User Features\n(Redis, real-time)"]
        ItemFeatures["Item Features\n(PostgreSQL)"]
        ContextFeatures["Context Features\n(Time, device)"]
    end

    subgraph DataLayer["Data Layer"]
        FeedbackStore["Feedback Store\n(Kafka + S3)"]
        RetrainingPipeline["Retraining Pipeline\n(Weekly)"]
        ABTestStore["A/B Test Store\n(PostgreSQL)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(CTR, latency)"]
        ELK["ELK Stack\n(Rec logs)"]
        ABDashboard["A/B Dashboard\n(Grafana)"]
    end

    UserEvents --> KafkaTopic
    KafkaTopic --> RealtimeProcessor
    RealtimeProcessor --> UserEmbeddings
    UserEmbeddings --> TwoTowerRetrieval
    ItemEmbeddings --> TwoTowerRetrieval
    TwoTowerRetrieval --> MLRanker
    UserFeatures --> MLRanker
    ItemFeatures --> MLRanker
    ContextFeatures --> MLRanker
    MLRanker --> LLMReranker
    LLMReranker --> DiversityFilter
    DiversityFilter --> FeedbackStore
    FeedbackStore --> RetrainingPipeline
    RetrainingPipeline --> EmbeddingRefresh
    MLRanker --> Prom
    Prom --> ABDashboard
```

**Infrastructure Components:**
- **Compute**: GPU cluster (8x V100) for ML ranker, async LLM reranker workers
- **Storage**: Redis (user embeddings, 500M users), Faiss (item embeddings, HNSW index), PostgreSQL (item features, A/B results)
- **Pipeline**: Kafka for event streaming, Flink for real-time processing, weekly batch retraining
- **Monitoring**: Real-time CTR tracking, A/B test dashboards, embedding staleness alerts
