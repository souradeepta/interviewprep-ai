## System Architecture (Infrastructure and Deployment)

```mermaid
graph TD
    subgraph UserLayer["User and Team Layer"]
        DevPortal["Developer Portal\n(Web UI)"]
        PlatformAPI["Platform API\n(REST, multi-tenant)"]
        RBAC["RBAC System\n(Team isolation)"]
    end

    subgraph LifecycleCluster["LLM Lifecycle Management"]
        FineTuningJob["Fine-Tuning Job\n(Ray Train, 2hr max)"]
        EvalPipeline["Eval Pipeline\n(Auto-benchmark)"]
        ModelRegistry["Model Registry\n(MLflow)"]
        DeployGate["Deploy Gate\n(Safety + quality check)"]
    end

    subgraph InferenceCluster["Inference Cluster (GPU)"]
        ModelRouter["Model Router\n(Cost + latency aware)"]
        GPT35Svc["GPT-3.5 Service\n(Default tier)"]
        FineTunedSvc["Fine-tuned Service\n(Custom tier, 200ms)"]
        GPT4Svc["GPT-4 Service\n(Premium tier)"]
        ResponseCache["Response Cache\n(Redis, 60pct hit)"]
    end

    subgraph ABLayer["A/B and Rollout Layer"]
        ABRouter["A/B Router\n(Traffic split)"]
        ExperimentTracker["Experiment Tracker\n(MLflow)"]
        StagedRollout["Staged Rollout\n(5pct to 100pct)"]
    end

    subgraph DataLayer["Data Layer"]
        TrainingDataStore["Training Data Store\n(S3, per-tenant)"]
        InferenceLogStore["Inference Log Store\n(S3 + PostgreSQL)"]
        CostDB["Cost DB\n(Per-model, per-team)"]
    end

    subgraph Monitoring["Observability"]
        DriftDetector["Drift Detector\n(Accuracy monitoring)"]
        CostTracker["Cost Tracker\n(Budget alerts)"]
        Leaderboard["Model Leaderboard\n(Grafana)"]
    end

    DevPortal --> PlatformAPI
    PlatformAPI --> RBAC
    RBAC --> FineTuningJob
    RBAC --> EvalPipeline
    FineTuningJob --> ModelRegistry
    EvalPipeline --> DeployGate
    DeployGate --> StagedRollout
    StagedRollout --> ABRouter
    ABRouter --> ModelRouter
    ModelRouter --> GPT35Svc
    ModelRouter --> FineTunedSvc
    ModelRouter --> GPT4Svc
    GPT35Svc --> ResponseCache
    FineTunedSvc --> ResponseCache
    GPT4Svc --> ResponseCache
    ResponseCache --> InferenceLogStore
    InferenceLogStore --> DriftDetector
    InferenceLogStore --> CostTracker
    CostTracker --> CostDB
    DriftDetector --> Leaderboard
```

**Infrastructure Components:**
- **Compute**: Ray Train for fine-tuning (2hr job limit), GPU cluster for inference, async evaluation workers
- **Storage**: S3 per-tenant training data (isolation), MLflow model registry, PostgreSQL cost database
- **Routing**: Cost+latency-aware model router (GPT-3.5 default, fine-tuned custom, GPT-4 premium)
- **Governance**: RBAC per team, safety gate before deployment, staged rollout (5% to 100%)
