## Process Flow (Transaction to Decision)

```mermaid
sequenceDiagram
    participant Gateway as Payment Gateway
    participant Kafka as Kafka Queue
    participant FeatExt as Feature Extractor
    participant Rules as Rule Engine
    participant ML as ML Scorer
    participant Graph as Graph Model
    participant Decision as Decision Engine
    participant Action as Action Service
    participant LLM as LLM Explainer

    Gateway->>Kafka: Publish transaction event
    Kafka->>FeatExt: Consume transaction (target 10ms)
    FeatExt->>FeatExt: Extract 50 features (velocity, geo, device)
    FeatExt->>Rules: Feature vector (target 1ms)
    Rules->>Rules: Hard rule evaluation (block list, velocity)
    alt Hard rule match
        Rules->>Decision: Immediate block signal
        Decision->>Action: Block transaction
        Action-->>Gateway: Decline response (20ms total)
    else No hard rule match
        Rules->>ML: Feature vector (target 20ms)
        ML->>ML: XGBoost ensemble scoring
        ML->>Decision: Fraud probability score
        Decision->>Decision: Threshold check (0.8 approve, 0.2 block)
        alt High risk (score above 0.8)
            Decision->>Action: Block
            Decision->>LLM: Async explanation request
            Action-->>Gateway: Decline (40ms total)
        else Medium risk (0.2 to 0.8)
            Decision->>Graph: Request graph scoring (async)
            Graph->>Decision: Graph model score
            Decision->>Action: Manual review queue
            Decision->>LLM: Async explanation request
            Action-->>Gateway: Approve with flag (45ms total)
        else Low risk (below 0.2)
            Decision->>Action: Approve
            Action-->>Gateway: Approve response (40ms total)
        end
        LLM->>LLM: Generate explanation (background, 1000ms)
    end
```

**Key Decision Points:**
1. **Hard Rules (1ms)**: Block list, impossible velocity (two cities in 30 min) - no ML needed
2. **ML Threshold (20ms)**: XGBoost score above 0.8 = auto-block, below 0.2 = auto-approve
3. **Medium Risk Zone**: Graph model adds accuracy for borderline cases (async, non-blocking)
4. **LLM Explanation**: Always async - never on the critical path to avoid latency impact
5. **SLA Budget**: 40ms total for approve/block, explanation delivered asynchronously

**Error Paths:**
- Feature extraction timeout: fall back to rule-based decision only
- ML model unavailable: fall back to conservative rules (approve only known users)
- Kafka lag: alert ops team, scale consumer group

**Optimization Points:**
- Pre-compute user velocity windows in Redis (sliding window counter)
- Cache device fingerprints per user (avoid recomputation)
- Batch LLM explanations during low-traffic periods
