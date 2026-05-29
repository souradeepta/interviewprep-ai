## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        DashboardAPI["Dashboard API\n(REST)"]
        AlertWebhookAPI["Alert Webhook API\n(Outbound)"]
        ReportAPI["Report API\n(PDF/JSON)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        MonitoringOrchestrator["Monitoring Orchestrator\n(Per-model schedule)"]
        IncidentCoordinator["Incident Coordinator\n(Alert triage)"]
        ReportScheduler["Report Scheduler\n(Daily/weekly)"]
    end

    subgraph MonitoringServices["Monitoring Services"]
        FeatureMonitorSvc["Feature Monitor\n(KL-divergence, PSI)"]
        PredMonitorSvc["Prediction Monitor\n(AUROC, accuracy)"]
        LatencyMonitor["Latency Monitor\n(P50, P99, P999)"]
        CostMonitorSvc["Cost Monitor\n(Per-model budget)"]
    end

    subgraph AnalysisServices["Analysis Services"]
        DriftDetectorSvc["Drift Detector\n(KS-test, statistical)"]
        RCAEngine["RCA Engine\n(LLM + rule-based)"]
        BiasAnalyzer["Bias Analyzer\n(Demographic parity)"]
        TrendForecaster["Trend Forecaster\n(Cost projection)"]
    end

    subgraph ActionServices["Action Services"]
        AlertClassifier["Alert Classifier\n(Severity 1-4)"]
        AlertRouter["Alert Router\n(Slack, PD, dashboard)"]
        AutoRemediation["Auto Remediation\n(Rollback, scale)"]
    end

    subgraph DataServices["Data Services"]
        TimeSeriesClient["Time-Series Client\n(InfluxDB)"]
        IncidentDBClient["Incident DB Client\n(PostgreSQL)"]
        ModelRegistryClient["Model Registry Client\n(MLflow)"]
    end

    DashboardAPI --> MonitoringOrchestrator
    MonitoringOrchestrator --> FeatureMonitorSvc
    MonitoringOrchestrator --> PredMonitorSvc
    MonitoringOrchestrator --> LatencyMonitor
    MonitoringOrchestrator --> CostMonitorSvc
    FeatureMonitorSvc --> DriftDetectorSvc
    PredMonitorSvc --> DriftDetectorSvc
    DriftDetectorSvc --> RCAEngine
    DriftDetectorSvc --> AlertClassifier
    RCAEngine --> IncidentDBClient
    BiasAnalyzer --> AlertClassifier
    TrendForecaster --> AlertClassifier
    AlertClassifier --> AlertRouter
    AlertRouter --> AutoRemediation
    AutoRemediation --> ModelRegistryClient
    FeatureMonitorSvc --> TimeSeriesClient
    IncidentCoordinator --> IncidentDBClient
    ReportAPI --> ReportScheduler
```

**Layer Breakdown:**
- **Presentation**: Dashboard, alert webhook, and report APIs
- **Orchestration**: Per-model monitoring schedule, incident triage, report scheduling
- **Monitoring Services**: Feature distribution, prediction accuracy, latency, and cost monitors
- **Analysis Services**: KS-test drift detection, LLM root cause analysis, bias detection, cost forecasting
- **Action Services**: Severity classification (1-4), tiered routing, automatic rollback
- **Data Services**: InfluxDB time-series, PostgreSQL incident log, MLflow model registry
