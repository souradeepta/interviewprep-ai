# LLM Finetuning Platform - Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant Platform as Platform API
    participant Storage as Data Storage
    participant Scheduler as Job Scheduler
    participant Workers as GPU Workers
    participant Tracker as Experiment Tracker
    participant Registry as Model Registry

    U->>Platform: Submit finetune job (dataset + config)
    Platform->>Storage: Validate and store dataset
    alt Validation failed
        Storage-->>Platform: Validation error
        Platform-->>U: Error details
    else Validation passed
        Storage-->>Platform: Dataset stored
        Platform->>Scheduler: Queue training job
        Scheduler-->>Platform: Job ID assigned
        Platform-->>U: Job ID and status URL
        Scheduler->>Workers: Dispatch job to GPU workers
        loop Training epochs
            Workers->>Tracker: Log loss and metrics
        end
        Workers->>Tracker: Log final checkpoint
        Workers->>Registry: Register candidate model
        Registry-->>Workers: Eval triggered
        Registry-->>Platform: Model registered
        Platform-->>U: Job complete, model ready
    end
```

**Key Decision Points:**
1. **Data Validation**: Schema, format, and quality checks before any GPU resources allocated
2. **Job Queuing**: Priority-based scheduling; high-priority jobs preempt lower-priority ones
3. **Metric Logging**: Real-time training metrics allow early stopping if loss diverges
4. **Eval Gate**: Models must pass benchmark thresholds before registry promotion

**Optimization Points:**
- Mixed precision (BF16) training reduces memory by 2x and speeds training 30-40%
- Gradient checkpointing trades compute for memory on long-context fine-tunes
- Checkpoint every N steps to enable resume from failure without full restart
