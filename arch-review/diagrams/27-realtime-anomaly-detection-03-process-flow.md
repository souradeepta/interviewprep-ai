## Process Flow (Metrics Stream to Alert and Remediation)

```mermaid
sequenceDiagram
    participant Service as Monitored Service
    participant Agent as Metrics Agent
    participant Kafka as Kafka Topic
    participant Flink as Stream Processor
    participant IsoForest as Isolation Forest
    participant Seasonal as Seasonal Baseline
    participant Ensemble as Ensemble Scorer
    participant RCA as RCA Agent
    participant Alert as Alert Router
    participant Engineer as On-call Engineer

    Service->>Agent: Emit metrics (CPU, latency, error rate)
    Agent->>Kafka: Publish metrics event
    Kafka->>Flink: Consume and window (30s rolling)
    Flink->>Seasonal: Request seasonal baseline for this time window
    Seasonal-->>Flink: Expected range (mean +/- 2 sigma)
    Flink->>IsoForest: Feature vector with seasonal adjustment
    IsoForest->>IsoForest: Score anomaly likelihood (50ms)
    IsoForest->>Ensemble: Primary anomaly score
    Flink->>Ensemble: Seasonal deviation score
    Ensemble->>Ensemble: Combine scores with weights
    alt Ensemble score below 0.70
        Ensemble->>Flink: No alert, update baseline
    else Score 0.70 to 0.85
        Ensemble->>Alert: Severity 3 (dashboard only)
        Alert-->>Engineer: Dashboard notification
    else Score 0.85 to 0.99
        Ensemble->>RCA: Analyze correlated metrics
        RCA->>RCA: LLM generates root cause hypothesis
        RCA->>Alert: Severity 2 with RCA context
        Alert-->>Engineer: Slack alert with explanation
    else Score above 0.99
        Ensemble->>RCA: Immediate RCA analysis
        RCA->>Alert: Severity 1 - auto-remediate
        Alert->>Alert: Execute remediation playbook
        Alert-->>Engineer: PagerDuty phone call
        Engineer->>Alert: Acknowledge and confirm action
    end
```

**Key Decision Points:**
1. **Seasonal Adjustment**: Baselines adjust for known patterns (peak hours, weekly cycles) to reduce false positives
2. **Isolation Forest**: Unsupervised outlier detection without labeled anomaly data
3. **Ensemble Fusion**: Combines statistical and ML scores for higher precision
4. **Confidence Tiers**: Four response levels (ignore, dashboard, Slack, PagerDuty+auto-remediate)
5. **Auto-Remediation Gate**: Only at 0.99+ confidence to prevent false-positive rollbacks

**Error Paths:**
- Kafka consumer lag: alert ops team, scale Flink workers
- Isolation Forest unavailable: fall back to pure threshold-based detection
- RCA engine timeout: send alert without explanation, flag for manual investigation

**Optimization Points:**
- Pre-compute seasonal baselines once per hour for each metric/service
- Cache Isolation Forest inference for identical feature vectors (within 60-second window)
- Batch low-severity anomalies into digest notifications (hourly dashboard summary)
