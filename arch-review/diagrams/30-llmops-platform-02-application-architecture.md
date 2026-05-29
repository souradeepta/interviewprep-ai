## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        DevPortalAPI["Dev Portal API\n(REST, multi-tenant)"]
        InferenceAPI["Inference API\n(POST /complete)"]
        AdminAPI["Admin API\n(Budget, governance)"]
    end

    subgraph Orchestration["Orchestration Layer"]
        RequestOrchestrator["Request Orchestrator\n(Route to model)"]
        FineTuneOrchestrator["Fine-tune Orchestrator\n(Job lifecycle)"]
        EvalOrchestrator["Eval Orchestrator\n(Benchmark runner)"]
    end

    subgraph RoutingServices["Routing and Inference Services"]
        ModelRouterSvc["Model Router Svc\n(Cost + latency + accuracy)"]
        CacheChecker["Cache Checker\n(Redis, 60pct hit)"]
        InferenceDispatcher["Inference Dispatcher\n(Select tier)"]
        RateLimiter["Rate Limiter\n(Per team budget)"]
    end

    subgraph LifecycleServices["Model Lifecycle Services"]
        SafetyGate["Safety Gate\n(Content policy check)"]
        TrainingJobRunner["Training Job Runner\n(Ray Train)"]
        EvalBenchmarker["Eval Benchmarker\n(Auto-score vs baseline)"]
        DeployApproval["Deploy Approval\n(Human + auto gate)"]
    end

    subgraph MonitoringServices["Monitoring Services"]
        DriftMonitor["Drift Monitor\n(Production accuracy)"]
        CostMonitor["Cost Monitor\n(Per-model, per-team)"]
        QualityLeaderboard["Quality Leaderboard\n(Model comparison)"]
    end

    subgraph DataServices["Data Services"]
        ModelRegistryClient["Model Registry Client\n(MLflow)"]
        TrainingDataClient["Training Data Client\n(S3, per-tenant)"]
        InferenceLogClient["Inference Log Client\n(S3 + PostgreSQL)"]
        CostDBClient["Cost DB Client\n(PostgreSQL)"]
    end

    DevPortalAPI --> RequestOrchestrator
    InferenceAPI --> RateLimiter
    RateLimiter --> ModelRouterSvc
    ModelRouterSvc --> CacheChecker
    CacheChecker --> InferenceDispatcher
    InferenceDispatcher --> ModelRegistryClient
    FineTuneOrchestrator --> SafetyGate
    SafetyGate --> TrainingJobRunner
    TrainingJobRunner --> EvalBenchmarker
    EvalBenchmarker --> DeployApproval
    DeployApproval --> ModelRegistryClient
    InferenceDispatcher --> InferenceLogClient
    InferenceLogClient --> DriftMonitor
    InferenceLogClient --> CostMonitor
    CostMonitor --> CostDBClient
    DriftMonitor --> QualityLeaderboard
    TrainingDataClient --> TrainingJobRunner
```

**Layer Breakdown:**
- **Presentation**: Developer portal, inference, and admin APIs (multi-tenant)
- **Orchestration**: Inference routing, fine-tuning lifecycle, evaluation benchmark runner
- **Routing Services**: Cost+latency+accuracy-aware model router, 60% cache hit layer, per-team rate limiter
- **Lifecycle Services**: Safety content gate, Ray Train job runner, auto-benchmark evaluator, human+auto deploy gate
- **Monitoring Services**: Production drift monitor, per-model/per-team cost monitor, model quality leaderboard
- **Data Services**: MLflow model registry, per-tenant S3 training data, inference logs, cost database
