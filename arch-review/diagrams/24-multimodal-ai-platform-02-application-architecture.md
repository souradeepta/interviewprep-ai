## Application Architecture (Components and Layers)

```mermaid
graph TD
    subgraph Presentation["Presentation Layer"]
        UnifiedAPI["Unified API\n(POST /understand)"]
        AsyncAPI["Async API\n(POST /understand/async)"]
        StatusAPI["Status API\n(GET /status/{id})"]
    end

    subgraph Orchestration["Orchestration Layer"]
        ModalityParser["Modality Parser\n(Detect present modalities)"]
        ParallelDispatcher["Parallel Dispatcher\n(Async encoder fanout)"]
        FusionCoordinator["Fusion Coordinator\n(Wait for encoders)"]
    end

    subgraph EncoderServices["Encoder Services"]
        VisionEncoderSvc["Vision Encoder Service\n(ViT-B, GPU)"]
        TextEncoderSvc["Text Encoder Service\n(RoBERTa)"]
        AudioEncoderSvc["Audio Encoder Service\n(Whisper-Large)"]
        MissingModalityHandler["Missing Modality Handler\n(Zero embedding)"]
    end

    subgraph FusionServices["Fusion and Interpretation Services"]
        ProjectionLayer["Projection Layer\n(Normalize to 768-d)"]
        AttentionFusion["Attention Fusion\n(Learned weights)"]
        LLMInterpretation["LLM Interpretation\n(LLaMA-7B)"]
        ConflictResolver["Conflict Resolver\n(Modality disagreement)"]
    end

    subgraph CacheServices["Cache Services"]
        EmbeddingCache["Embedding Cache\n(Content hash, Redis)"]
        ResultCache["Result Cache\n(Request hash, Redis)"]
        BatchAccumulator["Batch Accumulator\n(100 requests)"]
    end

    subgraph DataServices["Data Services"]
        ResultWriter["Result Writer\n(PostgreSQL)"]
        AuditWriter["Audit Writer\n(S3)"]
        MetricsEmitter["Metrics Emitter\n(Prometheus)"]
    end

    UnifiedAPI --> ModalityParser
    AsyncAPI --> ModalityParser
    ModalityParser --> EmbeddingCache
    EmbeddingCache --> ParallelDispatcher
    ParallelDispatcher --> VisionEncoderSvc
    ParallelDispatcher --> TextEncoderSvc
    ParallelDispatcher --> AudioEncoderSvc
    MissingModalityHandler --> FusionCoordinator
    VisionEncoderSvc --> FusionCoordinator
    TextEncoderSvc --> FusionCoordinator
    AudioEncoderSvc --> FusionCoordinator
    FusionCoordinator --> ProjectionLayer
    ProjectionLayer --> AttentionFusion
    AttentionFusion --> ConflictResolver
    ConflictResolver --> LLMInterpretation
    LLMInterpretation --> ResultCache
    LLMInterpretation --> ResultWriter
    ResultWriter --> AuditWriter
    MetricsEmitter --> ResultWriter
```

**Layer Breakdown:**
- **Presentation**: Synchronous, async, and status-check APIs for flexible latency handling
- **Orchestration**: Modality detection, parallel encoder dispatch, fusion coordination
- **Encoder Services**: Independent GPU-backed services per modality with graceful missing-modality handling
- **Fusion Services**: Dimension projection, attention-based fusion, LLM interpretation, conflict resolution
- **Cache Services**: Embedding-level and request-level caches, batch accumulator for cost efficiency
- **Data Services**: Result persistence, audit log, metrics emission
