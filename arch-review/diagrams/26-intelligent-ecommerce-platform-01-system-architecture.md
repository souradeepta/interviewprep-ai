## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph UserLayer["User Interaction Layer"]
        SearchUI["Search UI\n(React)"]
        PDPUI["Product Detail Page\n(React)"]
        CartUI["Cart and Checkout\n(React)"]
        APIGW["API Gateway\n(Kong, rate limiting)"]
    end

    subgraph AIServicesCluster["AI Services Cluster (Kubernetes)"]
        QueryUnderstanding["Query Understanding\n(NLU Pod x 5)"]
        SearchRanker["Search Ranker\n(LambdaMART Pod x 10)"]
        PersonLayer["Personalization Layer\n(User2Vec Pod x 8)"]
        DynPricing["Dynamic Pricing\n(Elasticity Model Pod x 4)"]
    end

    subgraph ABLayer["A/B Routing Layer"]
        ABRouter["A/B Router\n(Feature flags)"]
        ExperimentTracker["Experiment Tracker\n(MLflow)"]
    end

    subgraph DataLayer["Data Layer"]
        UserProfileDB["User Profile DB\n(Redis + PostgreSQL)"]
        ProductCatalogDB["Product Catalog\n(Elasticsearch)"]
        InventoryDB["Inventory DB\n(PostgreSQL)"]
        FeatureStore["Feature Store\n(Redis, real-time)"]
    end

    subgraph FeedbackLoop["Feedback and Learning Loop"]
        EventStream["Event Stream\n(Kafka)"]
        RetrainingPipeline["Retraining Pipeline\n(Weekly)"]
        ModelRegistry["Model Registry\n(MLflow)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(GMV, CTR, latency)"]
        ELK["ELK Stack\n(Search logs)"]
        RevenueDashboard["Revenue Dashboard\n(Grafana)"]
    end

    SearchUI --> APIGW
    PDPUI --> APIGW
    CartUI --> APIGW
    APIGW --> QueryUnderstanding
    QueryUnderstanding --> SearchRanker
    SearchRanker --> PersonLayer
    UserProfileDB --> PersonLayer
    FeatureStore --> PersonLayer
    InventoryDB --> DynPricing
    PersonLayer --> DynPricing
    DynPricing --> ABRouter
    ABRouter --> ExperimentTracker
    ABRouter --> EventStream
    EventStream --> RetrainingPipeline
    RetrainingPipeline --> ModelRegistry
    ModelRegistry --> SearchRanker
    SearchRanker --> Prom
    RevenueDashboard --> Prom
```

**Infrastructure Components:**
- **Compute**: Kubernetes cluster with separate pods for query understanding, search ranking, personalization, dynamic pricing
- **Storage**: Elasticsearch (product catalog, 100K+ SKUs), Redis (user profiles, feature store), PostgreSQL (inventory, transaction history)
- **Learning**: Kafka event stream, weekly retraining pipeline, MLflow model registry
- **Monitoring**: Real-time GMV tracking, CTR dashboards, A/B experiment analytics
