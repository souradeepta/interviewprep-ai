## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph Ingestion["Metrics Ingestion Layer"]
        MetricsAgents["Metrics Agents\n(100 services)"]
        KafkaTopic["Kafka Topic\n(metrics-stream)"]
        StreamProcessor["Stream Processor\n(Flink, 10K/sec)"]
    end

    subgraph DetectionCluster["Anomaly Detection Cluster"]
        FeatureExtractor["Feature Extractor\n(Rolling windows)"]
        IsolationForest["Isolation Forest\n(Primary detector)"]
        AutoencoderModel["Autoencoder Model\n(Deep anomaly)"]
        AnomalyScorer["Anomaly Scorer\n(Ensemble fusion)"]
    end

    subgraph AnalysisLayer["Analysis Layer"]
        ThresholdDetector["Threshold Detector\n(Confidence-based)"]
        RCAAgent["RCA Agent\n(LLM + rules)"]
        SeasonalBaseline["Seasonal Baseline\n(STL, 2yr history)"]
    end

    subgraph ActionLayer["Action and Alert Layer"]
        AlertRouter["Alert Router\n(Severity 1-4)"]
        PagerDuty["PagerDuty\n(High severity)"]
        SlackNotifier["Slack Notifier\n(Medium severity)"]
        AutoRemediation["Auto Remediation\n(Playbooks)"]
    end

    subgraph DataLayer["Data Layer"]
        TimeSeriesDB["Time-Series DB\n(InfluxDB)"]
        BaselineStore["Baseline Store\n(Historical norms)"]
        IncidentLog["Incident Log\n(PostgreSQL)"]
    end

    subgraph Monitoring["Self-monitoring"]
        Prom["Prometheus\n(FP rate, MTTR)"]
        FeedbackStore["Feedback Store\n(Engineer ack)"]
        ModelRetrainer["Model Retrainer\n(Weekly on feedback)"]
    end

    MetricsAgents --> KafkaTopic
    KafkaTopic --> StreamProcessor
    StreamProcessor --> FeatureExtractor
    FeatureExtractor --> IsolationForest
    FeatureExtractor --> AutoencoderModel
    IsolationForest --> AnomalyScorer
    AutoencoderModel --> AnomalyScorer
    SeasonalBaseline --> AnomalyScorer
    AnomalyScorer --> ThresholdDetector
    ThresholdDetector --> RCAAgent
    RCAAgent --> AlertRouter
    AlertRouter --> PagerDuty
    AlertRouter --> SlackNotifier
    AlertRouter --> AutoRemediation
    RCAAgent --> IncidentLog
    FeatureExtractor --> TimeSeriesDB
    AnomalyScorer --> BaselineStore
    FeedbackStore --> ModelRetrainer
    ModelRetrainer --> IsolationForest
```

**Infrastructure Components:**
- **Compute**: Flink stream processors, Isolation Forest and Autoencoder GPU inference workers
- **Storage**: InfluxDB (time-series metrics), PostgreSQL (incident log), baseline historical store (2 years)
- **Detection**: Isolation Forest primary detector (50ms), Autoencoder for deep patterns (200ms), seasonal STL baseline
- **Action**: Confidence-tiered alerts (log only, dashboard, Slack, PagerDuty, auto-remediate)
