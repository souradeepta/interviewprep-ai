# Data Pipelines

## TL;DR
ETL: Extract data, Transform (clean, join, aggregate), Load to warehouse/lake. Batch (daily) or streaming (continuous). Foundation of ML systems.

## Core Intuition
Raw data → useful features. Pipeline automates: extract from sources, transform, load.

## How It Works
```
Data sources (databases, APIs, logs)
  ↓
Extract (scheduled jobs, streaming)
  ↓
Transform (clean, join, aggregate)
  ↓
Load (to warehouse, feature store)
  ↓
Available for training, serving, analytics
```

**Batch:** nightly, daily (latency OK, cost-efficient)
**Streaming:** continuous, real-time (low latency, expensive)

## Key Properties / Trade-offs
- Latency: batch (hours), streaming (seconds)
- Cost: batch cheap, streaming expensive
- Complexity: batch simple, streaming complex

## Common Mistakes / Gotchas
- **No error handling:** bad data propagates → broken models
- **Slow transforms:** bottleneck. Optimize with distributed processing (Spark).
- **Data quality:** no validation → garbage data in
- **Lineage:** can't trace data origin → hard to debug

## Interview Q&A

**Q: How do you decide between batch processing and stream processing for an ML data pipeline?**
A: Batch when: features don't need to be real-time (end-of-day reports, daily retraining), data volumes are large and latency requirements are loose (hours to days), simplicity is valued over freshness. Stream when: features need to be up-to-date within seconds (fraud detection, real-time recommendations), you're processing continuously arriving events, or decisions need the latest context. Many production systems use both: batch for expensive features (complex aggregations), stream for simple low-latency features (recent activity counts).

**Q: What are the most common causes of data pipeline failures in production?**
A: Schema changes upstream: a source system adds/removes/renames a column without notifying the ML team. Upstream system outages: the pipeline fails silently when source data is missing. Data volume spikes: pipeline times out or runs out of memory during unusually large batches. Silent data quality degradation: pipeline succeeds but produces wrong features (e.g., NULL values increase, outliers appear). Infrastructure changes: a library upgrade breaks serialization. All of these require monitoring, not just code fixes.

**Q: How do you implement data validation in an ML pipeline?**
A: Validate at every boundary: raw data ingestion (check expected columns, types, row counts), feature computation (check distributions against historical baselines), training data preparation (check class balance, feature correlations). Tools: Great Expectations, TFX Data Validation, or custom statistical tests. Set thresholds based on historical variation (flag if metric deviates >3σ from rolling average). Fail the pipeline and alert rather than silently proceeding with bad data.

**Q: What is data lineage and why does it matter for ML?**
A: Data lineage tracks how each piece of data was transformed from raw source to model input. It enables: debugging (trace a model prediction back to the raw data that produced it), compliance (prove what data was used to train a model), impact analysis (understand which models are affected when a data source changes), and reproducibility (recreate exact training datasets). Without lineage, investigating production issues is extremely difficult—you can't answer "why did this prediction change?"

**Q: How do you handle late-arriving data in time-based features?**
A: Late data is common in distributed systems: events with timestamps from yesterday that arrive today. Strategies: (1) watermarks—define a maximum lateness threshold (e.g., 6 hours), process data within the window, reject or ignore later arrivals; (2) reprocessing—allow late data to trigger recomputation of affected time windows; (3) approximate processing—treat late data as "close enough" and accept small errors in time-sensitive features. For model training, ensure your training data has the same late-arrival characteristics as production.

## Interview Quick-Reference
**Data pipeline?** Extract, transform, load. Batch or streaming. Automate, monitor, validate.

## Related Topics
- [Feature Store](03-feature-store.md) — stores transformed data
- [Data Governance](26-data-governance.md) — data quality

## Resources
- [Apache Airflow](https://airflow.apache.org/) — orchestration
- [Spark](https://spark.apache.org/) — distributed processing
