## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        SearchAPI["Search API\n(GET /search)"]
        RecsAPI["Recs API\n(GET /recommendations)"]
        PricingAPI["Pricing API\n(GET /price)"]
        EventAPI["Event API\n(POST /events)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        SearchOrchestrator["Search Orchestrator\n(Query to ranked results)"]
        PersonOrchestrator["Personalization Orchestrator\n(User-aware ranking)"]
        PricingOrchestrator["Pricing Orchestrator\n(Dynamic price compute)"]
    end

    subgraph SearchServices["Search Services"]
        QueryUnderstanding["Query Understanding\n(NLU + intent)"]
        ProductRetrieval["Product Retrieval\n(Elasticsearch BM25)"]
        LTRRanker["LTR Ranker\n(LambdaMART)"]
    end

    subgraph PersonalizationServices["Personalization Services"]
        UserEmbeddingLookup["User Embedding Lookup\n(Redis, User2Vec)"]
        PersonLayer["Personalization Filter\n(Score by affinity)"]
        CollabFilter["Collaborative Filter\n(Similar users)"]
    end

    subgraph PricingServices["Pricing Services"]
        ElasticityModel["Elasticity Model\n(Price-demand curve)"]
        CompetitorPriceFeed["Competitor Price Feed\n(External API)"]
        PriceConstraints["Price Constraints\n(Min margin, max delta)"]
    end

    subgraph DataServices["Data Services"]
        UserProfileClient["User Profile Client\n(Redis + PostgreSQL)"]
        CatalogClient["Catalog Client\n(Elasticsearch)"]
        InventoryClient["Inventory Client\n(PostgreSQL)"]
        FeedbackWriter["Feedback Writer\n(Kafka)"]
    end

    SearchAPI --> SearchOrchestrator
    RecsAPI --> PersonOrchestrator
    PricingAPI --> PricingOrchestrator
    SearchOrchestrator --> QueryUnderstanding
    QueryUnderstanding --> ProductRetrieval
    ProductRetrieval --> LTRRanker
    LTRRanker --> PersonLayer
    UserEmbeddingLookup --> PersonLayer
    CollabFilter --> PersonLayer
    PersonLayer --> PricingOrchestrator
    PricingOrchestrator --> ElasticityModel
    ElasticityModel --> CompetitorPriceFeed
    ElasticityModel --> PriceConstraints
    UserProfileClient --> UserEmbeddingLookup
    CatalogClient --> ProductRetrieval
    InventoryClient --> PricingOrchestrator
    EventAPI --> FeedbackWriter
```

**Layer Breakdown:**
- **Presentation**: Separate APIs for search, recommendations, pricing, and event ingestion
- **Orchestration**: Per-capability orchestrators for search, personalization, and dynamic pricing
- **Search Services**: NLU query understanding, BM25 product retrieval, LambdaMART re-ranking
- **Personalization Services**: User2Vec embedding lookup, personalization filter, collaborative filtering
- **Pricing Services**: Elasticity model, competitor price feed, margin constraint enforcement
- **Data Services**: User profiles, product catalog, inventory, event stream writer
