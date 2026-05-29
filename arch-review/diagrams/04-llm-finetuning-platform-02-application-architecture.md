# LLM Finetuning Platform - Application Architecture

```mermaid
graph TD
    subgraph Ingestion["Ingestion Layer"]
        UploadAPI["Upload API<br/>(REST)"]
        DataValidator["Data Validator<br/>(Format and Quality)"]
    end

    subgraph Preparation["Data Preparation"]
        Preprocessor["Preprocessor<br/>(Tokenizer)"]
        DataSplitter["Train/Val Splitter<br/>(Stratified)"]
    end

    subgraph JobManagement["Job Management"]
        JobScheduler["Job Scheduler<br/>(Priority Queue)"]
        ResourceMgr["Resource Manager<br/>(GPU Allocator)"]
    end

    subgraph TrainingWorkers["Training Workers"]
        TrainWorker["Training Worker<br/>(Trainer Loop)"]
        MetricsLogger["Metrics Logger<br/>(MLflow Client)"]
    end

    subgraph PostTraining["Post-Training"]
        EvalRunner["Eval Runner<br/>(Benchmark Suite)"]
        RegistryAPI["Registry API<br/>(Model Push)"]
        NotifyService["Notify Service<br/>(Email/Slack)"]
    end

    UploadAPI --> DataValidator
    DataValidator --> Preprocessor
    Preprocessor --> DataSplitter
    DataSplitter --> JobScheduler
    JobScheduler --> ResourceMgr
    ResourceMgr --> TrainWorker
    TrainWorker --> MetricsLogger
    TrainWorker --> EvalRunner
    EvalRunner -->|Pass| RegistryAPI
    EvalRunner -->|Fail| NotifyService
    RegistryAPI --> NotifyService
    MetricsLogger --> EvalRunner
```

**Layer Breakdown:**
- **Ingestion**: REST upload endpoint with format validation (JSONL, CSV) and quality checks
- **Data Preparation**: Tokenization, train/validation split with stratified sampling
- **Job Management**: Priority queue scheduling with GPU resource allocation
- **Training Workers**: Distributed training loop with live metric emission to MLflow
- **Post-Training**: Automated eval against benchmark suite, registry push on pass, notifications
