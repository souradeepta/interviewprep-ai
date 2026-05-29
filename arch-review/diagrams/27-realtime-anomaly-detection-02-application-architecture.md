## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        MetricsIngestionAPI["Metrics Ingestion API\n(REST, push)"]
        AlertDashboardAPI["Alert Dashboard API\n(REST, read)"]
        FeedbackAPI["Feedback API\n(Engineer ack)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        AnomalyOrchestrator["Anomaly Orchestrator\n(Detection pipeline)"]
        AlertOrchestrator["Alert Orchestrator\n(Triage and route)"]
        RemediationOrchestrator["Remediation Orchestrator\n(Playbook executor)"]
    end

    subgraph DetectionServices["Detection Services"]
        FeatureWindowizer["Feature Windowizer\n(Sliding window stats)"]
        IsolationForestSvc["Isolation Forest Svc\n(Primary, 50ms)"]
        AutoencoderSvc["Autoencoder Svc\n(Deep, 200ms)"]
        SeasonalAdjuster["Seasonal Adjuster\n(STL baseline)"]
    end

    subgraph AnalysisServices["Analysis Services"]
        EnsembleScorer["Ensemble Scorer\n(Fusion of detectors)"]
        ConfidenceCalibrator["Confidence Calibrator\n(0.0 to 1.0)"]
        RCATool["RCA Tool\n(LLM + rule engine)"]
    end

    subgraph ActionServices["Action Services"]
        SeverityClassifier["Severity Classifier\n(1 to 4)"]
        NotificationEmitter["Notification Emitter\n(Slack, PD, email)"]
        PlaybookRunner["Playbook Runner\n(Auto-remediation)"]
    end

    subgraph DataServices["Data Services"]
        TimeSeriesClient["Time-Series Client\n(InfluxDB write)"]
        BaselineClient["Baseline Client\n(Historical norms)"]
        IncidentDBClient["Incident DB Client\n(PostgreSQL)"]
    end

    MetricsIngestionAPI --> AnomalyOrchestrator
    AnomalyOrchestrator --> FeatureWindowizer
    FeatureWindowizer --> IsolationForestSvc
    FeatureWindowizer --> AutoencoderSvc
    SeasonalAdjuster --> EnsembleScorer
    IsolationForestSvc --> EnsembleScorer
    AutoencoderSvc --> EnsembleScorer
    EnsembleScorer --> ConfidenceCalibrator
    ConfidenceCalibrator --> RCATool
    RCATool --> SeverityClassifier
    SeverityClassifier --> AlertOrchestrator
    AlertOrchestrator --> NotificationEmitter
    AlertOrchestrator --> RemediationOrchestrator
    RemediationOrchestrator --> PlaybookRunner
    FeatureWindowizer --> TimeSeriesClient
    BaselineClient --> SeasonalAdjuster
    RCATool --> IncidentDBClient
    FeedbackAPI --> IncidentDBClient
```

**Layer Breakdown:**
- **Presentation**: Metrics push API, alert read API, engineer feedback API
- **Orchestration**: Detection pipeline coordinator, alert triage, remediation executor
- **Detection Services**: Sliding window featurization, Isolation Forest (50ms), Autoencoder (200ms), STL seasonal baseline
- **Analysis Services**: Ensemble fusion, confidence calibration, LLM+rule-based root cause analysis
- **Action Services**: Severity 1-4 classification, multi-channel notifications, automated playbook execution
- **Data Services**: InfluxDB time-series, historical baseline store, PostgreSQL incident log
