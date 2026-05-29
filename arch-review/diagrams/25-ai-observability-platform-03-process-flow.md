## Process Flow (Metric Ingestion to Alert)

```mermaid
sequenceDiagram
    participant Model as Production Model
    participant Agent as Metrics Agent
    participant Kafka as Kafka Topic
    participant Flink as Stream Processor
    participant Drift as Drift Detector
    participant RCA as RCA Engine
    participant Alert as Alert Router
    participant Engineer as On-call Engineer

    Model->>Agent: Emit prediction + feature metrics
    Agent->>Kafka: Publish to metrics topic (50K/sec)
    Kafka->>Flink: Stream consume and aggregate
    Flink->>Flink: Hourly rollup and windowing
    Flink->>Drift: Feature distribution statistics
    Drift->>Drift: Run KS-test and KL-divergence
    alt No drift detected
        Drift->>Flink: Continue monitoring baseline
    else Drift detected (confidence above 0.70)
        Drift->>RCA: Pass drift signal + correlated metrics
        RCA->>RCA: Classify issue type (data drift, model drift, infra)
        RCA->>RCA: Generate LLM root cause hypothesis
        alt Very high confidence (above 0.99)
            RCA->>Alert: Severity 1 - Auto-rollback and notify
            Alert->>Engineer: PagerDuty phone call
        else High confidence (0.90 to 0.99)
            RCA->>Alert: Severity 2 - Immediate Slack + email
            Alert->>Engineer: Slack alert with RCA context
            Engineer->>Alert: Acknowledge within 5 minutes
        else Medium confidence (0.70 to 0.90)
            RCA->>Alert: Severity 3 - Dashboard alert
            Alert->>Engineer: Dashboard notification
        end
    end
```

**Key Decision Points:**
1. **Hourly Aggregation**: Raw metrics aggregated to reduce alert noise from transient spikes
2. **Statistical Tests**: KS-test for distribution comparison, KL-divergence for feature drift
3. **RCA Classification**: LLM + rule-based system classifies root cause before alerting
4. **Confidence-based Escalation**: Higher confidence = faster, more aggressive response
5. **Auto-Rollback Gate**: Only triggered above 0.99 confidence to avoid false-positive rollbacks

**Error Paths:**
- Kafka lag growing: alert ops team, scale consumer group
- Drift detector false positive surge: tighten statistical threshold, add multi-day confirmation
- RCA engine unavailable: route to manual review queue with raw drift signal

**Optimization Points:**
- Downsample normal metrics (log 1% of healthy predictions, 100% of anomalies)
- Archive metrics older than 30 days to cold storage (reduce DB cost 80%)
- Cache baseline distributions per model to speed up KS-test comparisons
