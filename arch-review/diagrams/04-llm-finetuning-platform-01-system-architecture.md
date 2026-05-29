# LLM Finetuning Platform - System Architecture

```mermaid
graph TD
    subgraph DataSources["Data Sources"]
        S3Data["S3 Bucket<br/>(Training Data)"]
        DBData["Database<br/>(Structured Data)"]
        APIData["External API<br/>(Data Feed)"]
    end

    subgraph DataPipeline["Data Pipeline"]
        Validator["Data Validator<br/>(Schema Check)"]
        Preprocessor["Preprocessor<br/>(Tokenizer)"]
        DataStore["Feature Store<br/>(Processed Data)"]
    end

    subgraph TrainingCluster["Training Cluster (GPU)"]
        Scheduler["Job Scheduler<br/>(Queue)"]
        Worker1["Training Worker<br/>(GPU A100 x4)"]
        Worker2["Training Worker<br/>(GPU A100 x4)"]
    end

    subgraph Tracking["Experiment Tracking"]
        MLflow["MLflow<br/>(Metrics and Params)"]
        ArtifactStore["Artifact Store<br/>(Checkpoints)"]
    end

    subgraph EvalDeploy["Eval and Deploy"]
        EvalEngine["Eval Engine<br/>(Benchmarks)"]
        ModelRegistry["Model Registry<br/>(Versioned)"]
        DeployService["Deploy Service<br/>(Inference API)"]
    end

    S3Data --> Validator
    DBData --> Validator
    APIData --> Validator
    Validator --> Preprocessor
    Preprocessor --> DataStore
    DataStore --> Scheduler
    Scheduler --> Worker1
    Scheduler --> Worker2
    Worker1 --> MLflow
    Worker2 --> MLflow
    Worker1 --> ArtifactStore
    Worker2 --> ArtifactStore
    ArtifactStore --> EvalEngine
    EvalEngine -->|Pass| ModelRegistry
    EvalEngine -->|Fail| Scheduler
    ModelRegistry --> DeployService
    MLflow --> ModelRegistry
```

**Infrastructure Components:**
- **Data Pipeline**: Multi-source ingestion, schema validation, tokenization preprocessing
- **Training Cluster**: GPU workers (A100) with distributed training (DDP/FSDP)
- **Experiment Tracking**: MLflow for params, metrics, artifact versioning
- **Eval Engine**: Automated benchmarks (perplexity, task-specific evals) before registration
- **Model Registry**: Versioned model storage with promotion gates (staging to production)
- **Deploy Service**: Serving infrastructure with auto-scaling inference endpoints
