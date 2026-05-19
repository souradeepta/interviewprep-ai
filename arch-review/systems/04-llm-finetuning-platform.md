# LLM Fine-tuning Infrastructure at Scale

## TL;DR
Multi-tenant fine-tuning platform supporting 10K concurrent fine-tuning jobs on 100+ model variants (Mistral, Llama, Claude, GPT). Auto-scaling on GPUs, cost <$0.50 per training hour, validation in <1 hour.

## Problem Statement
Organizations want to fine-tune LLMs on proprietary data but lack infrastructure. Manual setup is complex, costs unpredictable, compute underutilized.

## Requirements

### Functional
- Support multiple base models (Mistral, Llama 2/3, Claude, GPT)
- LoRA, QLoRA, and full fine-tuning
- Distributed training (multi-GPU/multi-node)
- Automatic hyperparameter tuning
- Validation dataset evaluation
- Model versioning and rollback
- Cost allocation per customer

### Non-Functional (Scale Targets)
- Max training time: 24 hours (long-tail)
- GPU utilization: >70% (cost efficiency)
- Cost: <$0.50/hour for small jobs, <$0.10/hour for large batches
- Concurrent jobs: 10K
- Model registry size: 100+ variants

## Envelope Calculation

### Workload Estimation
- 10K jobs/day average, peak 1K concurrent jobs
- Avg training: 2 hours, 8 GPUs per job = 16K GPU-hours/day
- GPU cost (A100 on-demand): $2.50/hour × 16K = $40K/day
- Revenue target: $50K/day → $1.50/job average

### Cost Breakdown (1 job, 8 GPUs, 2 hours)
- GPU compute: 8 × $2.50 × 2 = $40
- Storage (data + model): ~$2
- Networking: ~$1
- Overhead (monitoring, support): ~$7
- Total: $50, charge: $75 (margin: 50%)

### Revenue Model
- Small job (1 GPU, 1h): $15
- Medium job (4 GPUs, 2h): $40
- Large job (32 GPUs, 4h): $200
- Enterprise (dedicated): $5K/month

### Storage
- Dataset storage: 100K customers × 1GB avg = 100TB (S3)
- Model checkpoints: 10K jobs × 3 checkpoints × 1GB = 30TB
- Total: 130TB yearly growth


## High-Level Architecture

```mermaid
graph TD
    A[User Upload Data] -->|REST API| B[Job Queue]
    B -->|Scheduler| C[Resource Allocator]
    C -->|Provisioner| D[GPU Provisioning<br/>K8s + Karpenter]
    D -->|Container| E[Training Pod<br/>Transformers + PyTorch]
    E -->|Checkpoint| F[(Model Storage<br/>S3)]
    E -->|Metrics| G[Monitoring<br/>Prometheus]
    G -->|Alert| H{Validation}
    H -->|Pass| I[Model Registry<br/>MLflow]
    H -->|Fail| J[Notify User]
    I -->|Ready| K[Deploy Options<br/>Endpoint/Export]
    C -->|Utilization| L[Cost Optimizer<br/>Bin Packing]
```


## Component Breakdown

### Job Queue & Scheduler
- Queue: Kafka + PostgreSQL for job persistence
- Scheduler: ML-based job prediction (duration, compute needs)
- Priority: spot GPU > on-demand GPU (cost optimization)
- Latency: job queued → training started <5 minutes

### GPU Resource Manager
- Infra: K8s + Karpenter (auto-scaling)
- GPU types: A100, H100, L4 (cost vs performance trade-offs)
- Bin packing: fit jobs into nodes to minimize idle GPU
- Preemption: low-priority jobs paused if high-priority arrives

### Training Container
- Base image: PyTorch + Transformers
- LoRA: PEFT library (4-32 rank, efficient)
- QLoRA: 4-bit quantization (3x cost reduction)
- Multi-GPU: FSDP (fully sharded data parallel)
- Gradient accumulation: reduce memory, no speed loss

### Validation Engine
- Eval set: 10% of training data (held out)
- Metrics: perplexity, F1, custom metrics
- Automated threshold checks: stop if validation loss increases
- Comparison: fine-tuned vs base model performance

### Model Registry
- Storage: HuggingFace Hub + internal MLflow
- Versioning: track all checkpoints, rollback capability
- Provenance: lineage from data → training → eval
- Deployment: export ONNX, vLLM, or vllm-serve

### Cost Optimizer
- Bin packing: fit small jobs on shared GPU nodes
- Spot instances: 70% discount, 5-minute termination warning
- Batch training: accumulate jobs, run together (cost amortization)
- Time-of-day pricing: cheaper during off-peak (adjust queue priority)


## AI/ML Integration Points

1. **Automatic Hyperparameter Tuning**:
   - Meta-learning: predict good hyperparams based on dataset
   - AutoML sweep: Bayesian optimization over learning rate, batch size, rank
   - Early stopping: stop if val loss plateaus

2. **Distributed Fine-tuning**:
   - FSDP across 8-32 GPUs for large models
   - Gradient checkpointing (trade compute for memory)
   - LoRA: train 1-5% of parameters, 10x memory savings

3. **Quality Control**:
   - Automated validation against base model
   - Detect overfitting: if train loss >> val loss, notify user
   - Benchmark: compare fine-tuned vs similar open models

4. **Cost Optimization**:
   - Recommend cheaper model if performance is similar
   - Suggest LoRA instead of full fine-tuning (10x cheaper)
   - Consolidate similar jobs for batch training


## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Scheduler
    participant GPU
    participant Storage

    User->>API: Upload data + config
    API->>Scheduler: Create job
    Scheduler->>GPU: Allocate 8 A100s
    GPU->>GPU: Load base model
    GPU->>Storage: Load training data
    GPU->>GPU: Fine-tune (2 hours)
    GPU->>Storage: Save checkpoint every 30min
    GPU->>GPU: Validate on eval set
    GPU-->>API: Training complete, metrics
    API-->>User: Model ready for deploy
```


## Key Trade-offs

| Decision | Full FT | LoRA | QLoRA | Choice |
|----------|---------|------|-------|--------|
| Cost/hour | $50 | $15 | $5 | QLoRA default (cost), upgrade on request |
| Speed | 1x | 0.9x | 0.8x | LoRA sweet spot |
| Quality | Best | 95% quality | 90% quality | LoRA (good quality, 70% cheaper) |
| Memory | 80GB | 20GB | 8GB | QLoRA for large models |
| Supported models | All | Llama/Mistral | All | LoRA as default |


## Interview Q&A

**Q1: How do you prevent runaway costs if user sets bad hyperparameters?**

A: Cost guardrails: max budget per job ($500 default), max hours (24h), max GPUs (32). User can override with approval. Overspend alerts at 50%, 80%, 100% of budget. Auto-termination if exceeds budget by 10%.

**Q2: A user uploads 500GB dataset. How do you handle storage costs?**

A: Charge storage: $0.023/GB/month (S3 pricing). 500GB = $12/month. For expensive datasets, recommend S3 RI (reserved capacity) or Glacier for cold storage. Compression: encourage users to tokenize offline (saves 50-80% storage).

**Q3: How do you bin-pack 1K jobs onto limited GPU capacity?**

A: ML-based scheduler: predict job duration and memory needs. Sort jobs by job_hours/memory (efficiency ratio). Fit into GPU clusters using first-fit-decreasing bin packing. Can fit 4-8 LoRA jobs per A100. Target: >80% GPU utilization.

**Q4: LoRA vs QLoRA cost/quality trade-off at scale?**

A: LoRA: 100K jobs/month × $15/job = $1.5M revenue, 98% model quality. QLoRA: 100K jobs × $5 = $500K revenue, 92% quality. Recommendation: default LoRA, offer QLoRA for cost-sensitive customers. 20% pick QLoRA.

**Q5: Distributed training across 8 GPUs: communication overhead?**

A: FSDP reduces communication overhead to ~10-15% (well-optimized). 8 GPUs → ~7.2x speedup (not 8x due to overhead). Training 8h on 1 GPU → 1.1h on 8 GPUs. Cost: 1 × $40 = $40 vs 8 × $5 = $40 (same cost, 7x faster).

**Q6: How do you validate fine-tuned model quality before user deployment?**

A: Automated eval: run on held-out 10% of user's data. Compute F1, perplexity. Compare to base model: if fine-tuned is <5% worse, reject and suggest hyperparameter tuning. User can override validation step.

**Q7: Multi-tenant GPU sharing: isolation and fair scheduling?**

A: Use K8s resource quotas (CPU, memory limits). Preemption: low-priority job pauses if high-priority arrives. Fairness: each customer gets share of GPU based on spending. High-paying customers get priority.

**Q8: Model versioning: how many checkpoints to keep?**

A: Keep all checkpoints for 30 days (cost ~$100/user). After 30 days, keep only best checkpoint + latest. Users can request checkpoints at any step. Automatic cleanup for stale jobs (>90 days old).

## Interview Quick-Reference

| Metric | Value |
|--------|-------|
| **Cost/hour** | LoRA: $15, QLoRA: $5, Full FT: $50 |
| **Training time** | Avg 2h, max 24h |
| **GPU utilization** | Target >75%, achieve 80% with bin-packing |
| **Concurrent jobs** | 10K, peak 1K |
| **Quality drop** | LoRA: 2%, QLoRA: 8% vs full FT |
| **Validation** | Auto eval on 10% holdout set |


## Related Systems
- 03-llm-api-gateway.md (serves fine-tuned models)
- 25-ai-observability.md (monitors training)
- 30-llmops-platform.md (part of broader MLOps)
