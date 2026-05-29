## Process Flow (Training Job to Production Deployment)

```mermaid
sequenceDiagram
    participant Dev as ML Engineer
    participant Portal as Dev Portal
    participant Safety as Safety Gate
    participant Trainer as Training Job Runner
    participant Eval as Eval Benchmarker
    participant Registry as Model Registry
    participant DeployGate as Deploy Gate
    participant ABRouter as A/B Router
    participant Monitor as Drift Monitor
    participant OnCall as On-call Engineer

    Dev->>Portal: Submit fine-tuning job (data + config)
    Portal->>Safety: Check training data (PII, policy)
    alt Safety check fails
        Safety-->>Dev: Rejected - policy violation
    else Safety check passes
        Safety->>Trainer: Start training job (Ray Train, 2hr max)
        Trainer->>Trainer: Fine-tune with validation split
        Trainer->>Eval: Submit trained model for benchmark
        Eval->>Eval: Run auto-evaluation (100-sample suite)
        Eval->>Eval: Compare vs baseline model
        alt Beats baseline (quality above threshold)
            Eval->>Registry: Register model version in MLflow
            Registry->>DeployGate: Propose for deployment
            DeployGate->>DeployGate: Auto-check latency and cost
            alt Auto-check passes
                DeployGate->>Dev: Request deployment approval
                Dev->>DeployGate: Approve
                DeployGate->>ABRouter: Staged rollout (5pct traffic)
                ABRouter->>Monitor: Watch for quality or latency degradation
                alt No degradation after 24h
                    ABRouter->>ABRouter: Expand to 25pct then 100pct
                    ABRouter-->>Dev: Deployment complete
                else Degradation detected
                    Monitor->>ABRouter: Trigger rollback
                    ABRouter->>Registry: Revert to previous version
                    Monitor->>OnCall: Alert with regression details
                end
            end
        else Does not beat baseline
            Eval-->>Dev: Evaluation failed - model not promoted
        end
    end
```

**Key Decision Points:**
1. **Safety Gate**: Checks training data for PII and policy violations before any compute is allocated
2. **Evaluation Gate**: Auto-benchmark against baseline model before registry registration
3. **Human Approval**: Engineer must explicitly approve before traffic is shifted
4. **Staged Rollout**: 5% traffic for 24 hours before expanding to minimize blast radius
5. **Auto-Rollback**: Drift monitor triggers automatic revert if quality degrades during rollout

**Error Paths:**
- Training job runs over 2-hour budget: checkpoint and pause, alert engineer
- Eval benchmark fails to complete: block promotion, notify engineer with partial results
- A/B rollout degrades latency P99: immediate rollback, investigate before retry

**Optimization Points:**
- Reuse evaluation results for identical model versions (hash-based dedup)
- Cache common inference requests during staged rollout to reduce cost of parallel model serving
- Pre-warm fine-tuned model containers before shifting traffic to avoid cold-start latency spike
