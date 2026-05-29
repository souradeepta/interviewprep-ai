## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        TransactionAPI["Transaction API\n(REST, POST /score)"]
        WebhookEmitter["Webhook Emitter\n(Decision events)"]
        ReviewDashboard["Review Dashboard\n(Analyst UI)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        ScoringOrchestrator["Scoring Orchestrator\n(Critical path)"]
        ExplanationScheduler["Explanation Scheduler\n(Async off-path)"]
        AlertRouter["Alert Router\n(Threshold rules)"]
    end

    subgraph FeatureServices["Feature Services"]
        FeatureExtractor["Feature Extractor\n(50 features, TensorRT)"]
        VelocityChecker["Velocity Checker\n(Redis sliding window)"]
        DeviceFingerprinter["Device Fingerprinter\n(Geolocation + UA)"]
    end

    subgraph ScoringServices["Scoring Services"]
        RuleEngine["Rule Engine\n(Hard rules, 1ms)"]
        MLScorer["ML Scorer\n(XGBoost, 20ms)"]
        GraphScorer["Graph Scorer\n(GNN, 100ms async)"]
        DecisionFusion["Decision Fusion\n(Weighted ensemble)"]
    end

    subgraph ActionServices["Action Services"]
        ApprovalEmitter["Approval Emitter"]
        BlockEmitter["Block Emitter"]
        ReviewQueueWriter["Review Queue Writer"]
    end

    subgraph DataServices["Data Services"]
        FeatureStoreClient["Feature Store Client\n(Redis, 5ms)"]
        TransactionLogWriter["Transaction Log Writer\n(Kafka)"]
        ExplanationDBWriter["Explanation DB Writer\n(PostgreSQL)"]
    end

    TransactionAPI --> ScoringOrchestrator
    ScoringOrchestrator --> FeatureExtractor
    FeatureExtractor --> VelocityChecker
    FeatureExtractor --> DeviceFingerprinter
    VelocityChecker --> FeatureStoreClient
    FeatureExtractor --> RuleEngine
    FeatureExtractor --> MLScorer
    RuleEngine --> DecisionFusion
    MLScorer --> DecisionFusion
    GraphScorer --> DecisionFusion
    DecisionFusion --> ApprovalEmitter
    DecisionFusion --> BlockEmitter
    DecisionFusion --> ReviewQueueWriter
    DecisionFusion --> TransactionLogWriter
    DecisionFusion --> ExplanationScheduler
    ExplanationScheduler --> ExplanationDBWriter
    AlertRouter --> ReviewDashboard
    WebhookEmitter --> TransactionLogWriter
```

**Layer Breakdown:**
- **Presentation**: Transaction scoring API, webhook decision emitter, analyst review dashboard
- **Orchestration**: Critical-path scoring orchestrator, async explanation scheduler, alert router
- **Feature Services**: TensorRT feature extraction, Redis velocity windows, device fingerprinting
- **Scoring Services**: Rule engine (1ms), XGBoost ML scorer (20ms), GNN graph scorer (100ms async), weighted fusion
- **Action Services**: Approval, block, and review queue emitters
- **Data Services**: Redis feature store, Kafka transaction log, PostgreSQL explanation store
