## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph ModelFleet["ML Model Fleet (100+ models)"]
        ModelA["Production Model A\n(Predictions + features)"]
        ModelB["Production Model B\n(Predictions + features)"]
        ModelN["Production Model N\n(Predictions + features)"]
    end

    subgraph LogPipeline["Metrics Ingestion Pipeline"]
        KafkaTopic["Kafka Topic\n(50K metrics/sec)"]
        MetricsCollector["Metrics Collector\n(Lightweight agent)"]
        StreamProcessor["Stream Processor\n(Flink)"]
    end

    subgraph MonitoringServices["Monitoring Services"]
        FeatureMonitor["Feature Monitor\n(Distribution stats)"]
        PredMonitor["Prediction Monitor\n(Accuracy, AUROC)"]
        PerfMonitor["Performance Monitor\n(Latency, cost)"]
        DriftDetector["Drift Detector\n(KL-divergence, KS-test)"]
    end

    subgraph AnalysisLayer["Analysis Layer"]
        RCAAgent["RCA Agent\n(Root cause LLM)"]
        BiasDetector["Bias Detector\n(Fairness metrics)"]
        CostAnalyzer["Cost Analyzer\n(Budget tracking)"]
    end

    subgraph AlertLayer["Alert and Action Layer"]
        AlertRouter["Alert Router\n(Severity routing)"]
        PagerDuty["PagerDuty\n(On-call)"]
        AutoRollback["Auto Rollback\n(High confidence)"]
    end

    subgraph DataLayer["Data Layer"]
        TimeSeriesDB["Time-Series DB\n(InfluxDB)"]
        MetricStore["Metric Store\n(Hourly rollups)"]
        IncidentLog["Incident Log\n(PostgreSQL)"]
    end

    ModelA --> MetricsCollector
    ModelB --> MetricsCollector
    ModelN --> MetricsCollector
    MetricsCollector --> KafkaTopic
    KafkaTopic --> StreamProcessor
    StreamProcessor --> FeatureMonitor
    StreamProcessor --> PredMonitor
    StreamProcessor --> PerfMonitor
    FeatureMonitor --> DriftDetector
    PredMonitor --> DriftDetector
    DriftDetector --> RCAAgent
    DriftDetector --> AlertRouter
    RCAAgent --> IncidentLog
    BiasDetector --> AlertRouter
    CostAnalyzer --> AlertRouter
    AlertRouter --> PagerDuty
    AlertRouter --> AutoRollback
    FeatureMonitor --> TimeSeriesDB
    PredMonitor --> MetricStore
```

**Infrastructure Components:**
- **Ingestion**: Lightweight agents in each model pod, Kafka for 50K metrics/sec throughput
- **Monitoring**: Feature distribution, prediction accuracy, latency/cost monitors with Flink stream processing
- **Analysis**: LLM-based root cause analysis, fairness/bias detection, budget tracking
- **Action**: Tiered alerting (dashboard only, Slack, PagerDuty, auto-rollback) based on drift confidence
