## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Transaction Ingestion Layer"]
        PaymentGateway["Payment Gateway\n(Stripe/Internal)"]
        KafkaQueue["Kafka Topic\n(txn-stream)"]
        FeatureStore["Feature Store\n(Redis, 10ms)"]
    end

    subgraph MLCluster["ML Scoring Cluster (GPU)"]
        FeatureExtractor["Feature Extractor\n(TensorRT, 10ms)"]
        MLModel["ML Scorer\n(XGBoost, 20ms)"]
        GraphModel["Graph Model\n(GNN, 100ms)"]
        DecisionEngine["Decision Engine\n(Rule fusion)"]
    end

    subgraph ActionLayer["Action Layer"]
        ApproveService["Approve Service\n(Low risk)"]
        BlockService["Block Service\n(High risk)"]
        ReviewQueue["Review Queue\n(Medium risk)"]
    end

    subgraph LLMLayer["LLM Explanation Layer (Async)"]
        ExplanationQueue["Explanation Queue\n(Kafka async)"]
        LLMExplainer["LLM Explainer\n(GPT-4 Turbo)"]
        ExplanationStore["Explanation Store\n(PostgreSQL)"]
    end

    subgraph DataLayer["Data Layer"]
        TransactionLog["Transaction Log\n(Kafka + S3)"]
        FraudPatternDB["Fraud Pattern DB\n(PostgreSQL)"]
        RetrainingPipeline["Retraining Pipeline\n(Weekly)"]
    end

    subgraph Monitoring["Observability"]
        Prom["Prometheus\n(FPR, TPR, Latency)"]
        DriftDetector["Drift Detector\n(KS-test)"]
        AlertSystem["Alert System\n(PagerDuty)"]
    end

    PaymentGateway --> KafkaQueue
    KafkaQueue --> FeatureExtractor
    FeatureStore --> FeatureExtractor
    FeatureExtractor --> MLModel
    MLModel --> DecisionEngine
    GraphModel --> DecisionEngine
    DecisionEngine --> ApproveService
    DecisionEngine --> BlockService
    DecisionEngine --> ReviewQueue
    DecisionEngine --> ExplanationQueue
    ExplanationQueue --> LLMExplainer
    LLMExplainer --> ExplanationStore
    DecisionEngine --> TransactionLog
    TransactionLog --> RetrainingPipeline
    RetrainingPipeline --> MLModel
    MLModel --> Prom
    Prom --> DriftDetector
    DriftDetector --> AlertSystem
```

**Infrastructure Components:**
- **Compute**: GPU cluster for XGBoost and GNN scoring (8x A100), async LLM explanation workers
- **Storage**: Kafka (transaction stream), Redis (feature store, 10ms lookup), PostgreSQL (fraud patterns, explanations)
- **Latency Budget**: Feature extraction 10ms + ML scoring 20ms + decision 10ms = 40ms critical path
- **Monitoring**: Real-time FPR/TPR tracking, KS-test drift detection, weekly retraining triggers
